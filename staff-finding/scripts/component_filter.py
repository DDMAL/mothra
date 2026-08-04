"""
Component filter for Stage 1 of the staff detection pipeline.

Given a cropped, BGR-preprocessed image of a detected staffline bounding box,
isolate the connected component most likely to be the staffline itself.
Output is consumed by the curve-fitting step (not implemented here).

See ADR-001 for design decisions and rationale.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunable constants (see ADR-001 §9)
# ---------------------------------------------------------------------------

# Aspect-ratio floor: components must be at least this many times wider than tall.
ASPECT_RATIO_FLOOR = 2.0

# Minimum component size, in multiples of the scale unit h (page-level median
# staffline thickness). Components smaller than MIN_COMPONENT_SIZE_MULTIPLIER * h
# pixels are discarded as noise.
MIN_COMPONENT_SIZE_MULTIPLIER = 5

# Connectivity for connected-components analysis. 8 = diagonal neighbors connected.
CONNECTIVITY = 8

# Scoring weights (must sum to 1.0). See ADR-001 §4.
WEIGHT_HORIZONTAL_EXTENT = 0.5
WEIGHT_VERTICAL_CENTER_PROXIMITY = 0.5

# A second-place score within this fraction of first-place flags the ambiguity.
COMPARABLE_SCORE_THRESHOLD = 0.20

# Default scale unit if the caller does not provide one. A conservative middle
# value across the calibration corpus. Callers should override per page.
DEFAULT_SCALE_UNIT = 10.0

# --- Merge step (experimental; see Option B run for comparison) ---
# Two components are candidates for merging if their y-center distance is below
# this multiple of h.
#
# Original default: 0.5 — conservative, intended to prevent bridging to
# neighbouring stafflines. Raised to 1.0 because page arch causes distant
# fragments of the same staffline to have y-centers > 0.5h apart, causing them
# to land in separate merge clusters and ultimately be discarded. Adjacent
# stafflines are typically separated by much more than 1h, so the risk of a
# false cross-line merge is low; the x-gap check provides a second guard.
# Revert toward 0.5 if false merges appear between parallel lines.
MERGE_Y_CENTER_DISTANCE_MULTIPLIER = 1.0

# Two components are candidates for merging if the horizontal gap between them
# is below this multiple of h.
#
# Original default: 5.0 — sufficient for small neume interruptions. Raised to
# 10.0 because large decorated initials and majuscule letters can create gaps
# well beyond 5h between the left and right fragments of a staffline (at h≈15
# a gap of 10h = 150 px; at h≈20 a gap of 10h = 200 px). These fragments are
# still clearly part of the same line and should merge. Revert toward 5.0 if
# unrelated blobs from adjacent staves start merging horizontally.
MERGE_X_GAP_MULTIPLIER = 10.0

# Minimum score for a discarded "not_top_scoring" component to be retained as
# a companion. Set conservatively so only plausible line fragments are kept,
# not noise blobs. The companion must also have a non-overlapping x-range with
# the active winner (i.e. it must be a horizontal continuation, not a parallel
# duplicate or adjacent staffline fragment).
#
# Original default: n/a — companion retention did not exist. Added because even
# after loosening merge thresholds, some staffline fragments end up in separate
# clusters and only the highest-scoring cluster survives. The companion floor
# ensures high-scoring continuations are still included in coords/mask even
# when the merge logic cannot unite them.
COMPANION_SCORE_FLOOR = 0.25

# --- Sauvola binarization (see ADR-002) ---
# Window size for local adaptive thresholding, expressed as a multiple of h.
# Large enough to capture both line and surrounding parchment in local stats;
# small enough to remain locally adaptive. Rounded to nearest odd integer at
# use time.
SAUVOLA_WINDOW_MULTIPLIER = 9
# default was 15

# Threshold coefficient k. Lower k = more aggressive (more ink retained,
# including potential noise); higher k = more conservative. Literature
# default 0.2; lowered to 0.15 experimentally to recover more faint ink.
# further lowered to .1 for faintness test, but that may be too aggressive for general use.
SAUVOLA_K = 0.1

# Minimum Sauvola window size in pixels. Guards against degenerate tiny
# windows when scale_unit is very small.
# small test = 8; large test=15; 15 is safer for general use to avoid tiny windows that overfit noise.
# current use: 5, due to some very small scale_unit cases in the calibration corpus, but may be too aggressive for general use.
SAUVOLA_MIN_WINDOW = 5


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class ComponentFilterResult:
    """
    Output of the component filter.

    The primary fields (coords, mask, flags) reflect the *active* mode. When
    the filter is called with merge_components=True (default), they reflect
    the merged cluster; with merge_components=False, they reflect the single
    highest-scoring connected component.

    The no_merge_* fields always carry the no-merge result regardless of mode,
    for inspection and downstream comparison.

    Attributes:
        coords: List of (x, y) tuples of pixels belonging to the kept (active)
            component or cluster. Empty if no component survived filtering.
        mask: Boolean array, same height/width as the input crop. True where
            the kept pixels are.
        score_breakdown: Per-candidate scoring details (from no-merge scoring).
        discarded: List of dicts describing components that were considered
            but not kept by the no-merge scoring.
        flags: Strings indicating notable conditions, e.g.
            'multiple_components_kept', 'multiple_clusters_kept',
            'no_components_survived'.
        no_merge_coords: List of (x, y) tuples from the no-merge winner.
            Same as `coords` when merge_components=False.
        no_merge_mask: Boolean mask of the no-merge winner. Same as `mask`
            when merge_components=False.
        merged_cluster_labels: List of connected-component labels that were
            merged into the active cluster (when merge_components=True);
            empty list otherwise.
        companion_labels: List of connected-component labels retained as
            companions of the active winner/cluster (score above
            COMPANION_SCORE_FLOOR, non-overlapping x-range).
    """

    coords: list[tuple[int, int]] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    score_breakdown: dict = field(default_factory=dict)
    discarded: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    no_merge_coords: list[tuple[int, int]] = field(default_factory=list)
    no_merge_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool)
    )
    merged_cluster_labels: list[int] = field(default_factory=list)
    companion_labels: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def filter_components(
    crop: np.ndarray,
    scale_unit: float = DEFAULT_SCALE_UNIT,
    save_path: Optional[Path] = None,
    merge_components: bool = True,
    binarization: str = "sauvola",
) -> ComponentFilterResult:
    """
    Isolate the staffline-bearing connected component(s) from a BGR-preprocessed
    crop.

    Args:
        crop: BGR-preprocessed image of a single detected bounding box. RGB on a
            white background (output of the upstream BGR script). Shape (H, W, 3)
            or (H, W) if already grayscale.
        scale_unit: Page-level median staffline thickness in pixels. Used to
            size the minimum-component-size noise floor, the merge thresholds,
            and (when applicable) the Sauvola window. See ADR-001 §10.
        save_path: If provided, a diagnostic visualization is saved here as PNG.
            If None, no visualization is produced.
        merge_components: When True (default), the active output reflects the
            merged cluster (multiple fragments stitched into one). When False,
            the active output is the single highest-scoring connected component.
            The no-merge result is always recorded in the result's no_merge_*
            fields regardless of this setting.
        binarization: 'sauvola' (default) or 'otsu'. Sauvola is a local
            adaptive method better suited to faint or unevenly-contrasted
            ink; Otsu is retained for comparison and as a fallback. See
            ADR-002.

    Returns:
        ComponentFilterResult. If no component survives filtering, the result
        carries the 'no_components_survived' flag and empty coords/mask.
    """
    # --- Binarize ---
    binary = _binarize(crop, method=binarization, scale_unit=scale_unit)

    # --- Find connected components ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=CONNECTIVITY
    )

    # Label 0 is the background; skip it.
    candidate_labels = list(range(1, n_labels))

    # --- Filter and score ---
    min_size = MIN_COMPONENT_SIZE_MULTIPLIER * scale_unit
    box_height = binary.shape[0]
    box_width = binary.shape[1]

    survivors = []  # list of (label, score, sub_scores, stats_dict)
    discarded = []

    for label in candidate_labels:
        x, y, w, h, area = stats[label]
        component_stats = {
            "label": int(label),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
        }

        # Reject on minimum size.
        if area < min_size:
            discarded.append({**component_stats, "reason": "below_min_size"})
            continue

        # Reject on aspect-ratio floor. Guard against zero-height components.
        if h == 0 or (w / h) < ASPECT_RATIO_FLOOR:
            discarded.append({**component_stats, "reason": "below_aspect_floor"})
            continue

        # Score survivors.
        score, sub_scores = _score_component(
            x=x,
            y=y,
            w=w,
            h=h,
            box_width=box_width,
            box_height=box_height,
        )
        survivors.append((label, score, sub_scores, component_stats))

    # --- Handle empty result ---
    if not survivors:
        result = ComponentFilterResult(
            mask=np.zeros_like(binary, dtype=bool),
            no_merge_mask=np.zeros_like(binary, dtype=bool),
            discarded=discarded,
            flags=["no_components_survived"],
        )
        if save_path is not None:
            _save_diagnostic(
                crop,
                binary,
                labels,
                kept_label=None,
                discarded_labels=[d["label"] for d in discarded],
                save_path=save_path,
                binarization_method=binarization,
            )
        return result

    # --- Rank and pick the no-merge winner ---
    survivors.sort(key=lambda s: s[1], reverse=True)
    winner_label, winner_score, winner_sub_scores, winner_stats = survivors[0]

    # Check for no-merge ambiguity (multiple_components_kept flag).
    no_merge_flags = []
    if len(survivors) > 1:
        runner_up_score = survivors[1][1]
        if winner_score > 0:
            relative_gap = (winner_score - runner_up_score) / winner_score
            if relative_gap < COMPARABLE_SCORE_THRESHOLD:
                no_merge_flags.append("multiple_components_kept")

    # Record all surviving candidates (including the winner) in score_breakdown.
    score_breakdown = {
        s[3]["label"]: {
            "total": s[1],
            "sub_scores": s[2],
            "stats": s[3],
            "kept": (s[0] == winner_label),
        }
        for s in survivors
    }

    # Non-winning survivors also go into 'discarded' with their scores.
    for s in survivors[1:]:
        discarded.append(
            {
                **s[3],
                "reason": "not_top_scoring",
                "score": s[1],
                "sub_scores": s[2],
            }
        )

    # --- Build the no-merge mask and coord list ---
    no_merge_mask = labels == winner_label
    ys, xs = np.where(no_merge_mask)
    no_merge_coords = list(zip(xs.tolist(), ys.tolist()))

    # --- Compute merge clusters and merged-winner data (always, cheap) ---
    merge_clusters = _compute_merge_groups(survivors, scale_unit)
    merged_scored = []
    for cluster in merge_clusters:
        m_score, m_sub_scores, m_stats = _score_merged_cluster(
            cluster,
            stats,
            box_width,
            box_height,
        )
        merged_scored.append((cluster, m_score, m_sub_scores, m_stats))
    merged_scored.sort(key=lambda m: m[1], reverse=True)
    merged_winner_labels = merged_scored[0][0] if merged_scored else []

    # Merge-ambiguity flag: do multiple clusters score comparably?
    merge_flags = []
    if len(merged_scored) > 1:
        m_top_score = merged_scored[0][1]
        m_runner_up = merged_scored[1][1]
        if m_top_score > 0:
            m_relative_gap = (m_top_score - m_runner_up) / m_top_score
            if m_relative_gap < COMPARABLE_SCORE_THRESHOLD:
                merge_flags.append("multiple_clusters_kept")

    # Build the merged-winner mask and coord list.
    merged_mask = np.zeros_like(labels, dtype=bool)
    for lbl in merged_winner_labels:
        merged_mask |= labels == lbl
    m_ys, m_xs = np.where(merged_mask)
    merged_coords = list(zip(m_xs.tolist(), m_ys.tolist(), strict=True))

    # --- Companion retention ---
    # Scan "not_top_scoring" losers. Keep any that:
    #   (a) score >= COMPANION_SCORE_FLOOR, and
    #   (b) have non-overlapping x-range with the active winner
    #       (horizontal continuation, not a parallel duplicate / adjacent line).
    # Companion pixels are added to both the no-merge and merge coord/mask sets
    # so the downstream fit sees the full staffline extent.

    no_merge_win_x_start = winner_stats["x"]
    no_merge_win_x_end = winner_stats["x"] + winner_stats["w"]

    if merged_scored:
        m_stats_top = merged_scored[0][3]  # (cluster, score, sub_scores, stats)
        merge_win_x_start = m_stats_top["x"]
        merge_win_x_end = m_stats_top["x"] + m_stats_top["w"]
    else:
        merge_win_x_start = no_merge_win_x_start
        merge_win_x_end = no_merge_win_x_end

    merged_winner_set = set(int(lbl) for lbl in merged_winner_labels)

    no_merge_companion_labels: list[int] = []
    no_merge_companion_coords: list[tuple[int, int]] = []
    no_merge_companion_mask = np.zeros_like(labels, dtype=bool)

    merge_companion_labels: list[int] = []
    merge_companion_coords: list[tuple[int, int]] = []
    merge_companion_mask = np.zeros_like(labels, dtype=bool)

    y_threshold = MERGE_Y_CENTER_DISTANCE_MULTIPLIER * scale_unit
    winner_y_center = winner_stats["y"] + winner_stats["h"] / 2.0

    for s in survivors[1:]:
        s_label, s_score, _s_sub, s_stats = s
        if s_score < COMPANION_SCORE_FLOOR:
            continue
        comp_y_center = s_stats["y"] + s_stats["h"] / 2.0
        if abs(comp_y_center - winner_y_center) > y_threshold:
            continue
        comp_x = s_stats["x"]
        comp_w = s_stats["w"]
        comp_mask = labels == s_label
        c_ys, c_xs = np.where(comp_mask)
        comp_pixels = list(zip(c_xs.tolist(), c_ys.tolist(), strict=True))

        # No-merge companion: non-overlapping with no-merge winner bbox.
        no_overlap_no_merge = (comp_x + comp_w <= no_merge_win_x_start) or (
            comp_x >= no_merge_win_x_end
        )
        if no_overlap_no_merge:
            no_merge_companion_labels.append(int(s_label))
            no_merge_companion_mask |= comp_mask
            no_merge_companion_coords.extend(comp_pixels)

        # Merge companion: must not already be in merged winner; non-overlapping
        # with merged winner's combined bbox.
        if int(s_label) not in merged_winner_set:
            no_overlap_merge = (comp_x + comp_w <= merge_win_x_start) or (
                comp_x >= merge_win_x_end
            )
            if no_overlap_merge:
                merge_companion_labels.append(int(s_label))
                merge_companion_mask |= comp_mask
                merge_companion_coords.extend(comp_pixels)

    # --- Choose active outputs based on the mode ---
    if merge_components:
        active_coords = merged_coords + merge_companion_coords
        active_mask = merged_mask | merge_companion_mask
        active_flags = merge_flags
        active_companion_labels = merge_companion_labels
    else:
        active_coords = no_merge_coords + no_merge_companion_coords
        active_mask = no_merge_mask | no_merge_companion_mask
        active_flags = no_merge_flags
        active_companion_labels = no_merge_companion_labels

    result = ComponentFilterResult(
        coords=active_coords,
        mask=active_mask,
        score_breakdown=score_breakdown,
        discarded=discarded,
        flags=active_flags,
        no_merge_coords=no_merge_coords,
        no_merge_mask=no_merge_mask,
        merged_cluster_labels=(
            [int(lbl) for lbl in merged_winner_labels] if merge_components else []
        ),
        companion_labels=active_companion_labels,
    )

    if save_path is not None:
        _save_diagnostic(
            crop=crop,
            binary=binary,
            labels=labels,
            kept_label=winner_label,
            discarded_labels=[d["label"] for d in discarded],
            companion_labels=active_companion_labels,
            merge_clusters=merge_clusters,
            merged_winner_labels=merged_winner_labels,
            save_path=save_path,
            binarization_method=binarization,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _binarize(
    crop: np.ndarray,
    method: str = "sauvola",
    scale_unit: float = DEFAULT_SCALE_UNIT,
) -> np.ndarray:
    """Convert RGB (or grayscale) crop to a binary image.

    Args:
        crop: RGB or grayscale image.
        method: 'sauvola' (default) or 'otsu'. Sauvola is a local adaptive
            method better suited to faint or unevenly-contrasted ink. Otsu
            is a single global threshold, retained for comparison and
            fallback. See ADR-002.
        scale_unit: Page-level h, used to size the Sauvola window when
            method='sauvola'. Ignored for Otsu.

    Returns:
        uint8 array with foreground=255, background=0, suitable for
        cv2.connectedComponentsWithStats.
    """
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop

    if method == "otsu":
        # THRESH_BINARY_INV: ink (dark) becomes foreground (255), parchment 0.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    if method == "sauvola":
        # Import locally so the module can be imported without scikit-image
        # in environments that don't need Sauvola.
        from skimage.filters import threshold_sauvola

        # Window size: scale-relative, odd, no smaller than a sane floor.
        raw_window = int(round(SAUVOLA_WINDOW_MULTIPLIER * scale_unit))
        window_size = max(SAUVOLA_MIN_WINDOW, raw_window)
        # Sauvola requires an odd window.
        if window_size % 2 == 0:
            window_size += 1

        thresh_map = threshold_sauvola(gray, window_size=window_size, k=SAUVOLA_K)
        # Ink (dark) is foreground; same convention as Otsu output above.
        binary = ((gray < thresh_map).astype(np.uint8)) * 255
        return binary

    raise ValueError(
        f"Unknown binarization method: {method!r}. " "Expected 'otsu' or 'sauvola'."
    )


def _score_component(
    x: int,
    y: int,
    w: int,
    h: int,
    box_width: int,
    box_height: int,
) -> tuple[float, dict]:
    """Score a component on horizontal extent and vertical-center proximity.

    Both sub-scores are normalized to [0, 1] before weighting.

    Returns (total_score, sub_scores_dict).
    """
    # Horizontal extent: fraction of box width spanned by the component.
    horizontal_extent_score = w / box_width if box_width > 0 else 0.0

    # Vertical-center proximity: 1.0 when the component's vertical center
    # coincides with the box center; 0.0 when it sits at the very top or bottom.
    component_center_y = y + h / 2.0
    box_center_y = box_height / 2.0
    if box_height > 0:
        distance_from_center = abs(component_center_y - box_center_y)
        # Normalize: max possible distance is half the box height.
        normalized_distance = distance_from_center / (box_height / 2.0)
        vertical_center_score = max(0.0, 1.0 - normalized_distance)
    else:
        vertical_center_score = 0.0

    sub_scores = {
        "horizontal_extent": horizontal_extent_score,
        "vertical_center_proximity": vertical_center_score,
    }
    total = (
        WEIGHT_HORIZONTAL_EXTENT * horizontal_extent_score
        + WEIGHT_VERTICAL_CENTER_PROXIMITY * vertical_center_score
    )
    return total, sub_scores


def _compute_merge_groups(
    survivors: list[tuple],
    scale_unit: float,
) -> list[list[int]]:
    """Group surviving components into merge clusters based on y-center
    proximity and x-gap.

    Args:
        survivors: list of (label, score, sub_scores, stats_dict). Only the
            stats_dict is used here; the other fields are ignored.
        scale_unit: page-level h, used to scale the merge thresholds.

    Returns:
        A list of clusters; each cluster is a list of component labels that
        should be merged together. Singletons (components that don't merge
        with anyone) are returned as one-element lists.
    """
    if not survivors:
        return []

    y_threshold = MERGE_Y_CENTER_DISTANCE_MULTIPLIER * scale_unit
    x_gap_threshold = MERGE_X_GAP_MULTIPLIER * scale_unit

    # Build a list of (label, x_start, x_end, y_center) tuples for clarity.
    items = []
    for _, _, _, stats_dict in survivors:
        x = stats_dict["x"]
        y = stats_dict["y"]
        w = stats_dict["w"]
        h = stats_dict["h"]
        items.append(
            {
                "label": stats_dict["label"],
                "x_start": x,
                "x_end": x + w,
                "y_center": y + h / 2.0,
            }
        )

    # Union-find for clustering. Index by position in items.
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Pairwise: check whether each pair should merge.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]

            # Y-center proximity check.
            if abs(a["y_center"] - b["y_center"]) > y_threshold:
                continue

            # X-gap check. Compute the gap between their x-ranges. A merge is
            # only valid when the ranges are genuinely disjoint (gap >= 0):
            # fragments of the same interrupted line never overlap in x,
            # whereas two parallel, distinct stafflines close in y typically
            # DO overlap in x (gap < 0, since right.x_start < left.x_end).
            # Excluding negative gaps is what stops two simultaneously-
            # present, distinct lines from being unioned just because they
            # sit close together vertically. Mirrors the x-disjointness
            # already required for companion retention below.
            if a["x_start"] <= b["x_start"]:
                left, right = a, b
            else:
                left, right = b, a
            gap = right["x_start"] - left["x_end"]
            if gap < 0 or gap > x_gap_threshold:
                continue

            union(i, j)

    # Collect clusters by root.
    clusters_by_root: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_by_root.setdefault(root, []).append(items[i]["label"])

    return list(clusters_by_root.values())


def _score_merged_cluster(
    cluster_labels: list[int],
    stats: np.ndarray,
    box_width: int,
    box_height: int,
) -> tuple[float, dict, dict]:
    """Score a cluster as if it were a single merged component.

    Computes the bounding-box stats of the union (x_start = min, x_end = max,
    y_start = min, y_end = max, area = sum), then runs the same scoring.

    Returns (score, sub_scores, merged_stats).
    """
    xs_start = [stats[lbl, cv2.CC_STAT_LEFT] for lbl in cluster_labels]
    ys_start = [stats[lbl, cv2.CC_STAT_TOP] for lbl in cluster_labels]
    xs_end = [
        stats[lbl, cv2.CC_STAT_LEFT] + stats[lbl, cv2.CC_STAT_WIDTH]
        for lbl in cluster_labels
    ]
    ys_end = [
        stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT]
        for lbl in cluster_labels
    ]
    areas = [stats[lbl, cv2.CC_STAT_AREA] for lbl in cluster_labels]

    merged_x = int(min(xs_start))
    merged_y = int(min(ys_start))
    merged_w = int(max(xs_end) - merged_x)
    merged_h = int(max(ys_end) - merged_y)
    merged_area = int(sum(areas))

    score, sub_scores = _score_component(
        x=merged_x,
        y=merged_y,
        w=merged_w,
        h=merged_h,
        box_width=box_width,
        box_height=box_height,
    )
    merged_stats = {
        "labels": [int(lbl) for lbl in cluster_labels],
        "x": merged_x,
        "y": merged_y,
        "w": merged_w,
        "h": merged_h,
        "area": merged_area,
    }
    return score, sub_scores, merged_stats


def _save_diagnostic(
    crop: np.ndarray,
    binary: np.ndarray,
    labels: np.ndarray,
    kept_label: Optional[int],
    discarded_labels: list[int],
    save_path: Path,
    companion_labels: Optional[list[int]] = None,
    merge_clusters: Optional[list[list[int]]] = None,
    merged_winner_labels: Optional[list[int]] = None,
    binarization_method: str = "sauvola",
) -> None:
    """Render and save a multi-panel diagnostic figure.

    Panels (without merge comparison):
        1. Original BGR-preprocessed crop.
        2. Binarized image (Otsu or Sauvola, as configured).
        3. Components: kept (green), discarded (red), other (gray).
        4. Kept-component mask alone.

    Additional panels when merge_clusters is provided:
        5. Components with merge clusters shown: each cluster a distinct color,
           the would-be merged winner highlighted in green.
        6. Merged-winner mask alone.
    """
    # Import locally so the module can be imported in environments without
    # matplotlib (e.g., production batch runs that pass save_path=None).
    import matplotlib.pyplot as plt

    has_merge_panels = merge_clusters is not None
    n_panels = 6 if has_merge_panels else 4

    # Vertical stack: each panel gets the full figure width, with a fixed
    # height per panel. Crops are wide-and-short, so this produces a tall
    # narrow figure that the user can scroll through. Panels are
    # readable per-panel without zooming.
    panel_height_in = 3.0
    figure_width_in = 16.0
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(figure_width_in, panel_height_in * n_panels),
    )

    # Panel 1: original
    if crop.ndim == 3:
        axes[0].imshow(crop)
    else:
        axes[0].imshow(crop, cmap="gray")
    axes[0].set_title("BGR-preprocessed crop")
    axes[0].axis("off")

    # Panel 2: binarized (method shown in title)
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title(f"Binarized ({binarization_method.capitalize()})")
    axes[1].axis("off")

    # Panel 3: components with kept/discarded coloring (no merge)
    # Color order matters: gray → red → cyan → green (each overwrites the last).
    color_img = np.zeros((*labels.shape, 3), dtype=np.uint8)
    color_img[labels > 0] = [128, 128, 128]  # gray for any survivors-not-evaluated
    for d_label in discarded_labels:
        color_img[labels == d_label] = [200, 60, 60]  # red for discarded
    for comp_label in companion_labels or []:
        color_img[labels == comp_label] = [0, 200, 200]  # cyan for companions
    if kept_label is not None:
        color_img[labels == kept_label] = [60, 180, 75]  # green for kept

    axes[2].imshow(color_img)
    axes[2].set_title("No-merge: kept (green), companions (cyan), discarded (red)")
    axes[2].axis("off")

    # Panel 4: kept + companion mask (no merge)
    if kept_label is not None:
        kept_mask = labels == kept_label
        for comp_label in companion_labels or []:
            kept_mask = kept_mask | (labels == comp_label)
        axes[3].imshow(kept_mask, cmap="gray")
        title_suffix = (
            f" + {len(companion_labels or [])} companion(s)" if companion_labels else ""
        )
        axes[3].set_title(f"No-merge: kept mask{title_suffix}")
    else:
        axes[3].imshow(np.zeros_like(binary), cmap="gray")
        axes[3].set_title("No-merge: no component kept")
    axes[3].axis("off")

    if has_merge_panels:
        # Panel 5: merge clusters, with the would-be merged winner in green
        # and other clusters in distinct non-green colors.
        merge_color_img = np.zeros((*labels.shape, 3), dtype=np.uint8)
        winner_set = set(merged_winner_labels or [])
        palette = [
            [80, 130, 200],  # blue
            [200, 160, 60],  # orange
            [160, 100, 200],  # purple
            [60, 160, 160],  # teal
            [200, 200, 80],  # yellow
            [180, 100, 130],  # magenta
        ]
        non_winner_idx = 0
        for cluster in merge_clusters or []:
            if set(cluster) == winner_set and winner_set:
                color = [60, 180, 75]  # green
            else:
                color = palette[non_winner_idx % len(palette)]
                non_winner_idx += 1
            for lbl in cluster:
                merge_color_img[labels == lbl] = color
        # Overlay companions in cyan on top of cluster colors.
        for comp_label in companion_labels or []:
            merge_color_img[labels == comp_label] = [0, 200, 200]
        axes[4].imshow(merge_color_img)
        axes[4].set_title("With-merge: clusters (green=winner, cyan=companions)")
        axes[4].axis("off")

        # Panel 6: merged-winner mask alone
        merged_winner_mask = np.zeros_like(labels, dtype=bool)
        for lbl in winner_set:
            merged_winner_mask |= labels == lbl
        axes[5].imshow(merged_winner_mask, cmap="gray")
        axes[5].set_title("With-merge: would-be kept mask")
        axes[5].axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
