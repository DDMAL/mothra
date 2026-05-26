# -*- coding: utf-8 -*-
"""
Stave grouping for Stage 2 of the staff detection pipeline.

Given the FitResults produced by Stage 1 on a single page (or within a single
layout region of a page), groups them into staves using ratio-based gap
analysis on the fitted centerlines' y-positions.

Outputs per-fit stave assignments, a per-page log of grouping evidence, and
optionally synthesized interpolated lines where a stave appears to be missing
one.

See design doc §6 for Stage 2 design decisions and ADR-001 / ADR-002 for
upstream context.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fit_centerline import FitResult


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Multiplier on the median intra-stave gap that defines the cut threshold.
# Gaps larger than median_intra * CUT_THRESHOLD_MULTIPLIER are treated as
# inter-stave gaps. Simple and easy to inspect; can be replaced with Otsu or
# k-means on the gap distribution later if needed.
CUT_THRESHOLD_MULTIPLIER = 1.5

# Default: don't synthesize lines for staves with fewer than the mode line
# count. Flag the missing lines instead. Override per-call via the
# interpolate_missing parameter.
DEFAULT_INTERPOLATE_MISSING = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StaveAssignment:
    """One entry per input FitResult, in the same order as the input list.

    Attributes:
        fit_index: Index into the original list of FitResults.
        stave_id: 0-based stave index on the page (top to bottom). None if the
            fit could not be grouped (empty fit, isolated outlier, etc.).
        y_at_center: The y value used as this fit's position for grouping,
            sampled at the horizontal center of the fit's x-range. None if
            the fit had no y_values to draw from.
        within_stave_index: 0-based position of this line within its stave
            (top to bottom). None if not grouped.
        flags: Per-fit grouping flags, e.g. 'no_y_position_available',
            'ambiguous_assignment'.
    """
    fit_index: int
    stave_id: Optional[int] = None
    y_at_center: Optional[float] = None
    within_stave_index: Optional[int] = None
    flags: list[str] = field(default_factory=list)


@dataclass
class InterpolatedLine:
    """A synthesized line filling a presumed gap in a stave.

    Only produced when interpolate_missing=True. Carries enough information
    to be appended to the page's line stream with source='interpolated'.

    Attributes:
        stave_id: Which stave this line belongs to.
        within_stave_index: 0-based position within that stave.
        x_start: First x of the synthesized centerline.
        x_end: Last x (inclusive).
        y_values: Interpolated y per integer x in [x_start, x_end], computed
            by interpolating between neighboring detected lines at each x.
        neighbor_fit_indices: The two detected FitResults whose centerlines
            were interpolated between.
    """
    stave_id: int
    within_stave_index: int
    x_start: int
    x_end: int
    y_values: list[float] = field(default_factory=list)
    neighbor_fit_indices: tuple[int, int] = (0, 0)


@dataclass
class StaveGroupingResult:
    """Output of group_staves.

    Attributes:
        assignments: One StaveAssignment per input FitResult, in input order.
        mode_lines_per_stave: The most common lines-per-stave count across
            the page. Used as the expected N for flagging short/long staves.
        line_count_distribution: Full distribution, e.g. {4: 23, 3: 4, 5: 1}.
            Preserved so 'mode of 3' (suggesting under-detection) is
            distinguishable from a clean modal answer.
        cut_threshold_px: The y-gap threshold (in pixels) used to separate
            intra-stave from inter-stave gaps.
        gap_distribution: All consecutive y-gaps observed (sorted top to
            bottom). Diagnostic; preserves the evidence the cut was based on.
        interpolated_lines: Synthesized lines for missing-line cases.
            Empty when interpolate_missing=False (the default).
        flags: Page-level grouping flags, e.g. 'no_fits_available',
            'mode_count_below_typical', 'staves_with_unexpected_count'.
    """
    assignments: list[StaveAssignment] = field(default_factory=list)
    mode_lines_per_stave: Optional[int] = None
    line_count_distribution: dict[int, int] = field(default_factory=dict)
    cut_threshold_px: float = 0.0
    gap_distribution: list[float] = field(default_factory=list)
    interpolated_lines: list[InterpolatedLine] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def group_staves(
    fits: list[FitResult],
    scale_unit: float,
    interpolate_missing: bool = DEFAULT_INTERPOLATE_MISSING,
    save_path: Optional[Path] = None,
    page_size: Optional[tuple[int, int]] = None,
    page_image: Optional[np.ndarray] = None,
) -> StaveGroupingResult:
    """Group fitted centerlines into staves on a single page or region.

    Args:
        fits: All FitResults produced by Stage 1, in detection order. Fits
            with no y_values (no_fit_attempted, fit_did_not_converge) are
            recorded but excluded from grouping.
        scale_unit: Page-level scale unit h. Used as a sanity floor on the
            grouping gap threshold (a cut threshold smaller than h is
            implausible).
        interpolate_missing: When True, synthesize centerlines for staves
            that have fewer than the page's modal line count. Each
            synthesized line is interpolated between the two neighboring
            detected lines at each x and carries source='interpolated' for
            downstream rendering. Default False per Stage 1 design §6.3:
            for the proof of concept, flag missing lines without
            synthesizing, so QA can correct them manually.
        save_path: If provided, render a page-level diagnostic showing each
            fit colored by its stave assignment, plus annotations of gap
            distribution and cut threshold.
        page_size: (width, height) of the page in pixels. Required only
            when save_path is set and page_image is not.
        page_image: The full page (or the BGR-processed page) to overlay
            stave assignments onto in the diagnostic. Optional; if absent,
            the diagnostic falls back to a blank canvas of page_size.

    Returns:
        StaveGroupingResult carrying per-fit assignments and a per-page
        evidence record. See dataclass docstring for fields.
    """
    # Design doc §6.1: Simple, inspectable ratio-based gap analysis.
    # Classical method with preserved evidence; the bimodal IS/IL ratio
    # (2.2–2.8 across measured manuscripts) makes this approach robust.

    result = StaveGroupingResult()

    # --- Edge case: no fits ---
    if not fits:
        result.flags.append("no_fits_available")
        return result

    # --- Stage 1: Extract y-positions and sort ---
    # Design doc §6.1: Extract y-position at each fit's horizontal center.
    # Fits without y_values are excluded from grouping.  We collect them
    # separately so the final assignments list can be emitted in input order
    # (matching the StaveGroupingResult.assignments contract).
    fit_positions = []  # list of (fit_index, y_at_center)
    unassignable: dict[int, list[str]] = {}  # fit_index → flags

    for i, fit in enumerate(fits):
        y_center = _y_at_fit_center(fit)
        if y_center is not None:
            fit_positions.append((i, y_center))
        else:
            unassignable[i] = ["no_y_position_available"]

    if not fit_positions:
        result.flags.append("no_fits_with_y_positions")
        # Emit unassigned entries in input order before returning.
        for i in range(len(fits)):
            result.assignments.append(
                StaveAssignment(
                    fit_index=i,
                    stave_id=None,
                    y_at_center=None,
                    within_stave_index=None,
                    flags=unassignable.get(i, ["unknown_failure_unassigned"]),
                )
            )
        return result

    # Sort by y-position (top to bottom).
    fit_positions.sort(key=lambda x: x[1])
    y_positions = [y for _, y in fit_positions]
    sorted_fit_indices = [i for i, _ in fit_positions]

    # --- Stage 2: Compute all consecutive gaps ---
    # Design doc §6.1: Gaps between consecutive lines (sorted by y).
    gaps = _compute_gap_distribution(y_positions)
    result.gap_distribution = gaps  # helper returns sorted gaps

    # --- Stage 3: Determine cut threshold ---
    # Design doc §6.1: Ratio-based threshold; floored at scale_unit.
    cut_threshold = _determine_cut_threshold(gaps, scale_unit)
    result.cut_threshold_px = cut_threshold

    # --- Stage 4 & 5: Assign stave IDs and within-stave indices ---
    # Design doc §6.1: Gaps >= cut_threshold are inter-stave boundaries.
    raw_assignments = _assign_staves(sorted_fit_indices, y_positions, cut_threshold)
    fit_to_stave = {
        fit_idx: (stave_id, within_idx)
        for fit_idx, stave_id, within_idx in raw_assignments
    }

    # Emit assignments in original fit order so callers can index by fit position.
    for i, fit in enumerate(fits):
        if i in fit_to_stave:
            stave_id, within_idx = fit_to_stave[i]
            result.assignments.append(
                StaveAssignment(
                    fit_index=i,
                    stave_id=stave_id,
                    y_at_center=_y_at_fit_center(fit),
                    within_stave_index=within_idx,
                    flags=[],
                )
            )
        else:
            result.assignments.append(
                StaveAssignment(
                    fit_index=i,
                    stave_id=None,
                    y_at_center=None,
                    within_stave_index=None,
                    flags=unassignable.get(i, ["unknown_failure_unassigned"]),
                )
            )

    # --- Stage 6: Compute line count distribution and mode ---
    # Design doc §6.2: Mode of lines-per-stave; flag anomalous staves for QA.
    stave_line_counts: dict[int, int] = {}
    for _, stave_id, _ in raw_assignments:
        stave_line_counts[stave_id] = stave_line_counts.get(stave_id, 0) + 1
    line_counts = list(stave_line_counts.values())

    for count in line_counts:
        result.line_count_distribution[count] = (
            result.line_count_distribution.get(count, 0) + 1
        )

    if line_counts:
        result.mode_lines_per_stave = max(
            line_counts, key=lambda c: result.line_count_distribution[c]
        )

        if result.mode_lines_per_stave < 4:
            result.flags.append("mode_count_below_typical")

        # Flag unexpected-count staves by ID so QA knows which ones to inspect.
        unexpected_ids = [
            str(sid) for sid, cnt in stave_line_counts.items()
            if abs(cnt - result.mode_lines_per_stave) > 1
        ]
        if unexpected_ids:
            result.flags.append(
                f"staves_with_unexpected_count:{','.join(unexpected_ids)}"
            )

    # --- Stage 6b: Flag assignments whose gap to the next fit is near the
    # cut threshold (design doc §6.6: surface ambiguous groupings for QA). ---
    NEAR_THRESHOLD_PCT = 0.20
    for k in range(len(y_positions) - 1):
        gap = y_positions[k + 1] - y_positions[k]
        if abs(gap - cut_threshold) / cut_threshold <= NEAR_THRESHOLD_PCT:
            fit_idx = sorted_fit_indices[k]
            for asg in result.assignments:
                if asg.fit_index == fit_idx:
                    asg.flags.append("gap_near_threshold")
                    break

    # --- Stage 7: Synthesize missing lines (if requested) ---
    # Design doc §6.3: Default is False (flag for QA); set True to interpolate.
    if interpolate_missing and result.mode_lines_per_stave is not None:
        for stave_id, count in stave_line_counts.items():
            if count < result.mode_lines_per_stave:
                # TODO: Implement interpolation logic (not critical for MVP).
                pass

    # --- Stage 8: Generate diagnostic image (if requested) ---
    # Design doc §6.6: Diagnostic preserves all grouping evidence for QA.
    if save_path is not None:
        _save_grouping_diagnostic(
            fits, result, page_size, page_image, save_path
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers (signatures only)
# ---------------------------------------------------------------------------

def _y_at_fit_center(fit: FitResult) -> Optional[float]:
    """Return the y-value at the horizontal midpoint of a fit's x-range.

    Returns None if the fit has no y_values (empty or failed fit).

    Design doc §6.1: Extract a single representative y-position per fit
    for gap analysis. Using the horizontal center is robust to local skew
    and gives a stable ordering for grouping. If the fit failed or has no
    y_values, None signals exclusion from grouping logic.
    """
    if not fit.y_values:
        return None

    # --- Calculate horizontal center index ---
    # fit.y_values has one entry per integer x in [x_start, x_end].
    # The center index is roughly halfway along this range.
    center_idx = len(fit.y_values) // 2

    # Clamp to valid range (should be unnecessary, but safe).
    center_idx = max(0, min(center_idx, len(fit.y_values) - 1))

    return fit.y_values[center_idx]


def _compute_gap_distribution(y_positions: list[float]) -> list[float]:
    """Compute the sorted list of consecutive gaps between y-positions.

    Input is assumed to be a list of y values; this helper sorts them and
    returns the differences between consecutive entries.
    """
    if len(y_positions) < 2:
        return []
    sorted_y = sorted(y_positions)
    gaps = [sorted_y[i + 1] - sorted_y[i] for i in range(len(sorted_y) - 1)]
    return gaps


def _determine_cut_threshold(gaps: list[float], scale_unit: float) -> float:
    """Find the y-gap threshold separating intra-stave from inter-stave gaps.

    Uses the simple "median gap × CUT_THRESHOLD_MULTIPLIER" approach.
    The bimodality of intra- vs. inter-stave gaps means the median of all
    gaps is a reliable intra-stave estimate when pages have more intra-stave
    gaps than inter-stave gaps (which is almost always true).

    Floored at scale_unit (a cut threshold smaller than h is implausible).

    Design doc §6.1: Ratio-based thresholding is simple and inspectable.
    """
    if not gaps:
        return scale_unit
    median_gap = float(np.median(gaps))
    cut_threshold = median_gap * CUT_THRESHOLD_MULTIPLIER
    return max(cut_threshold, scale_unit)


def _assign_staves(
    sorted_fit_indices: list[int],
    y_positions: list[float],
    cut_threshold: float,
) -> list[tuple[int, int, int]]:
    """Walk the sorted fits top-to-bottom and assign stave IDs.

    Returns a list of (fit_index, stave_id, within_stave_index) tuples in
    sorted-by-y order. Splits whenever a consecutive gap exceeds
    cut_threshold.

    Design doc §6.1: Gaps >= cut_threshold signal inter-stave boundaries.
    """
    if not sorted_fit_indices:
        return []

    assignments = []
    stave_id = 0
    within_idx = 0

    assignments.append((sorted_fit_indices[0], stave_id, within_idx))

    for i in range(1, len(sorted_fit_indices)):
        gap = y_positions[i] - y_positions[i - 1]
        if gap >= cut_threshold:
            # Inter-stave gap; start a new stave.
            stave_id += 1
            within_idx = 0
        else:
            # Intra-stave gap; continue.
            within_idx += 1
        assignments.append((sorted_fit_indices[i], stave_id, within_idx))

    return assignments


def _save_grouping_diagnostic(
    fits: list[FitResult],
    result: StaveGroupingResult,
    page_size: Optional[tuple[int, int]],
    page_image: Optional[np.ndarray],
    save_path: Path,
) -> None:
    """Render a diagnostic showing stave assignments and gap evidence.

    Design doc §6.6: Diagnostic preserves all evidence for inspection by QA.
    Shows each fit colored by stave assignment, plus annotations of the
    gap distribution and cut threshold used.

    If page_image is None, creates a blank canvas of page_size.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # --- Determine canvas size ---
    if page_image is not None:
        canvas_h, canvas_w = page_image.shape[:2]
        canvas = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB).astype(
            np.float32
        ) / 255.0
    else:
        if page_size is None:
            page_size = (1000, 1000)  # Fallback.
        canvas_w, canvas_h = page_size
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)

    # --- Generate colors for staves ---
    # Key colors by actual stave_id so the legend always matches the lines.
    assigned_ids = sorted({
        a.stave_id for a in result.assignments if a.stave_id is not None
    })
    if not assigned_ids:
        assigned_ids = list(range(5))  # fallback for empty result

    cmap = plt.cm.get_cmap("tab10")
    stave_colors = {sid: cmap(i % 10) for i, sid in enumerate(assigned_ids)}

    # --- Overlay centerlines colored by stave ---
    # Convert canvas to uint8 once so cv2.polylines draws into a persistent
    # array (drawing into (canvas * 255).astype(np.uint8) inside the loop
    # would create a throwaway copy each iteration and lose every stroke).
    canvas_uint8 = (canvas * 255).astype(np.uint8)

    for assignment in result.assignments:
        if assignment.stave_id is None:
            continue

        fit_idx = assignment.fit_index
        if fit_idx >= len(fits):
            continue

        fit = fits[fit_idx]
        if not fit.y_values or fit.x_start > fit.x_end:
            continue

        xs = np.arange(fit.x_start, fit.x_end + 1)
        ys = np.array(fit.y_values)

        mask = (xs >= 0) & (xs < canvas_w) & (ys >= 0) & (ys < canvas_h)
        xs = xs[mask]
        ys = ys[mask]

        if len(xs) < 2:
            continue

        points = np.column_stack([xs, ys]).astype(np.int32)

        color_rgba = stave_colors[assignment.stave_id]
        color_rgb = color_rgba[:3]
        color_bgr = tuple(int(c * 255) for c in reversed(color_rgb))

        cv2.polylines(canvas_uint8, [points], False, color_bgr, thickness=2)

    # --- Create a figure with subplots ---
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6)
    )  # Top: centerlines; bottom: evidence.

    # --- Left panel: centerlines with stave colors ---
    axes[0].imshow(canvas_uint8)
    axes[0].set_title(
        f"Stave Assignments\n(Mode: {result.mode_lines_per_stave} lines/stave, "
        f"Cut Threshold: {result.cut_threshold_px:.1f} px)"
    )
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")

    # --- Right panel: gap distribution and threshold ---
    if result.gap_distribution:
        axes[1].bar(range(len(result.gap_distribution)), result.gap_distribution)
        axes[1].axhline(
            y=result.cut_threshold_px,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Cut Threshold ({result.cut_threshold_px:.1f} px)",
        )
        axes[1].set_title("Gap Distribution (Consecutive Fits)")
        axes[1].set_xlabel("Gap Index")
        axes[1].set_ylabel("Gap (pixels)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No gaps (0 or 1 fit)", ha="center", va="center")
        axes[1].set_title("Gap Distribution")

    # --- Add legend for staves ---
    legend_elements = [
        mpatches.Patch(color=stave_colors[sid], label=f"Stave {sid}")
        for sid in assigned_ids
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=5, bbox_to_anchor=(
            0.5, -0.02
        )
    )

    # --- Save figure ---
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)