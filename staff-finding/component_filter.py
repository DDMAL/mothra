"""
Component filter for Stage 1 of the staff detection pipeline.

Given a cropped, BGR-preprocessed image of a detected staffline bounding box,
isolate the connected component most likely to be the staffline itself.
Output is consumed by the curve-fitting step (not implemented here).

See ADR-001 for design decisions and rationale.

On my mini, use this path for the BGR model: 
`/Users/kyriebouressa/Documents/muscrat/layer_sep/scripts/quickstart_outputs/best_model_14april.pth`
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
ASPECT_RATIO_FLOOR = 3.0 #default was 2.0, relaxed to 3.0 to allow for more variability in staffline thickness and detection boxes.

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


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class ComponentFilterResult:
    """
    Output of the component filter.

    Attributes:
        coords: List of (x, y) tuples of pixels belonging to the kept component.
            Empty if no component survived filtering.
        mask: Boolean array, same height/width as the input crop. True where
            the kept component's pixels are. Empty array if no component survived.
        score_breakdown: Per-candidate scoring details for the kept component
            and all evaluated candidates. Keys map candidate ids to their
            sub-scores and total score.
        discarded: List of dicts describing components that were considered but
            not kept. Each includes the component's stats and reason for
            rejection or its (lower) score.
        flags: Strings indicating notable conditions, e.g.
            'multiple_components_kept', 'no_components_survived'.
    """
    coords: list[tuple[int, int]] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    score_breakdown: dict = field(default_factory=dict)
    discarded: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_components(
    crop: np.ndarray,
    scale_unit: float = DEFAULT_SCALE_UNIT,
    save_path: Optional[Path] = None,
) -> ComponentFilterResult:
    """
    Isolate the staffline-bearing connected component from a BGR-preprocessed crop.

    Args:
        crop: BGR-preprocessed image of a single detected bounding box. RGB on a
            white background (output of the upstream BGR script). Shape (H, W, 3)
            or (H, W) if already grayscale.
        scale_unit: Page-level median staffline thickness in pixels. Used to
            size the minimum-component-size noise floor. See ADR-001 §10.
        save_path: If provided, a diagnostic visualization is saved here as PNG.
            If None, no visualization is produced.

    Returns:
        ComponentFilterResult. If no component survives filtering, the result
        carries the 'no_components_survived' flag and empty coords/mask.
    """
    # --- Binarize ---
    binary = _binarize(crop)

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
            x=x, y=y, w=w, h=h,
            box_width=box_width,
            box_height=box_height,
        )
        survivors.append((label, score, sub_scores, component_stats))

    # --- Handle empty result ---
    if not survivors:
        result = ComponentFilterResult(
            mask=np.zeros_like(binary, dtype=bool),
            discarded=discarded,
            flags=["no_components_survived"],
        )
        if save_path is not None:
            _save_diagnostic(crop, binary, labels, kept_label=None,
                             discarded_labels=[d["label"] for d in discarded],
                             save_path=save_path)
        return result

    # --- Rank and pick the winner ---
    survivors.sort(key=lambda s: s[1], reverse=True)
    winner_label, winner_score, winner_sub_scores, winner_stats = survivors[0]

    # Check for ambiguity (multiple_components_kept flag).
    flags = []
    if len(survivors) > 1:
        runner_up_score = survivors[1][1]
        # Avoid division by zero on degenerate cases.
        if winner_score > 0:
            relative_gap = (winner_score - runner_up_score) / winner_score
            if relative_gap < COMPARABLE_SCORE_THRESHOLD:
                flags.append("multiple_components_kept")

    # Record all surviving candidates (including the winner) in score_breakdown
    # so downstream logging can inspect the full ranking.
    score_breakdown = {
        s[3]["label"]: {
            "total": s[1],
            "sub_scores": s[2],
            "stats": s[3],
            "kept": (s[0] == winner_label),
        }
        for s in survivors
    }

    # Non-winning survivors also go into 'discarded' with their scores, so the
    # caller has one consolidated record of everything that didn't win.
    for s in survivors[1:]:
        discarded.append({
            **s[3],
            "reason": "not_top_scoring",
            "score": s[1],
            "sub_scores": s[2],
        })

    # --- Build the kept-component mask and coord list ---
    mask = (labels == winner_label)
    ys, xs = np.where(mask)
    coords = list(zip(xs.tolist(), ys.tolist()))

    result = ComponentFilterResult(
        coords=coords,
        mask=mask,
        score_breakdown=score_breakdown,
        discarded=discarded,
        flags=flags,
    )

    if save_path is not None:
        _save_diagnostic(
            crop=crop,
            binary=binary,
            labels=labels,
            kept_label=winner_label,
            discarded_labels=[d["label"] for d in discarded],
            save_path=save_path,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binarize(crop: np.ndarray) -> np.ndarray:
    """Convert RGB (or grayscale) crop to a binary image via Otsu.

    Returns a uint8 array with foreground=255, background=0, suitable for
    cv2.connectedComponentsWithStats.
    """
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop

    # THRESH_BINARY_INV: ink (dark) becomes foreground (255), parchment becomes 0.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def _score_component(
    x: int, y: int, w: int, h: int,
    box_width: int, box_height: int,
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


def _save_diagnostic(
    crop: np.ndarray,
    binary: np.ndarray,
    labels: np.ndarray,
    kept_label: Optional[int],
    discarded_labels: list[int],
    save_path: Path,
) -> None:
    """Render and save a multi-panel diagnostic figure.

    Panels:
        1. Original BGR-preprocessed crop.
        2. Binarized image (Otsu output).
        3. All connected components, kept (green) vs. discarded (red) vs.
           other (gray).
        4. Kept-component mask alone.
    """
    # Import locally so the module can be imported in environments without
    # matplotlib (e.g., production batch runs that pass save_path=None).
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: original
    if crop.ndim == 3:
        axes[0].imshow(crop)
    else:
        axes[0].imshow(crop, cmap="gray")
    axes[0].set_title("BGR-preprocessed crop")
    axes[0].axis("off")

    # Panel 2: binarized
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Binarized (Otsu)")
    axes[1].axis("off")

    # Panel 3: components with kept/discarded coloring
    color_img = np.zeros((*labels.shape, 3), dtype=np.uint8)
    color_img[labels > 0] = [128, 128, 128]  # default gray for any survivors-not-evaluated
    for d_label in discarded_labels:
        color_img[labels == d_label] = [200, 60, 60]  # red for discarded
    if kept_label is not None:
        color_img[labels == kept_label] = [60, 180, 75]  # green for kept

    axes[2].imshow(color_img)
    axes[2].set_title("Components (green=kept, red=discarded)")
    axes[2].axis("off")

    # Panel 4: kept mask alone
    if kept_label is not None:
        kept_mask = (labels == kept_label)
        axes[3].imshow(kept_mask, cmap="gray")
        axes[3].set_title("Kept component mask")
    else:
        axes[3].imshow(np.zeros_like(binary), cmap="gray")
        axes[3].set_title("No component kept")
    axes[3].axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)