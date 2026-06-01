# -*- coding: utf-8 -*-
"""
Staffline interpolation for Stage 2 of the staff detection pipeline.

After grouping (group_staves.py) has assigned detected FitResults to staves
and flagged any rhythm anomalies, this module synthesises the missing lines.

Two complementary triggers
--------------------------
(A) **In-stave gap fill** — if two consecutive detected lines within the same
    stave have a y-gap in ``[cut_threshold, max_threshold]``, one or more lines
    are interpolated between them.  The count is estimated from the gap divided
    by the stave's intra-line period h_est.

(B) **Edge extrapolation** — if a stave has at least ``mode_n // 2`` detected
    lines but fewer than ``mode_n``, a regular grid is anchored at the
    bottommost detected line and empty grid slots are filled by offsetting from
    the nearest detected neighbour.  Constrained by territory boundaries
    (midpoints to adjacent staves) so no synthesised line ever lands in
    inter-stave space.

Rhythm gate
-----------
Staves flagged ``under_populated`` or ``over_populated`` by the rhythm check in
group_staves are skipped entirely.  When the gap pattern deviates from the
page's expected periodicity we cannot synthesise lines safely; the detected
assignments are left unchanged and the stave is deferred to QA.

Public API
----------
``interpolate_missing_lines(fits, assignments, rhythm_anomalies,
                            scale_unit, min_threshold, cut_threshold,
                            mode_n, interpolation_max_gap, all_gaps)``
    Returns ``(list[InterpolatedLine], max_threshold_used_px)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fit_centerline import FitResult
    from group_staves import StaveAssignment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fallback upper-bound multiplier when no inter-stave gaps are observed on the
# page (single-stave or very sparse).  Adaptive computation (mean of observed
# inter-stave gaps) is preferred; this is only the emergency fallback.
INTERPOLATION_GAP_MULTIPLIER: float = 1.8


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class InterpolatedLine:
    """A synthesised staffline filling a presumed gap in a stave.

    Only produced when interpolate_missing=True.  Carries enough information
    to be appended to the page's line stream with source='interpolated'.

    Attributes:
        stave_id: Which stave this line belongs to.
        within_stave_index: 0-based position within that stave.  Set to a
            placeholder during synthesis; re-assigned by the re-indexing pass
            in group_staves after detected and interpolated lines are merged.
        x_start: First x of the synthesised centerline (page-absolute).
        x_end: Last x (inclusive).
        y_values: Interpolated y per integer x in [x_start, x_end], computed
            by bilinear interpolation between neighbouring detected lines.
        neighbor_fit_indices: The two detected FitResult indices used as
            spatial anchors.  Both entries are the same fit for edge lines
            synthesised via trigger B (only one neighbour available).
    """
    stave_id: int
    within_stave_index: int
    x_start: int
    x_end: int
    y_values: list[float] = field(default_factory=list)
    neighbor_fit_indices: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _y_at_center(fit: FitResult) -> Optional[float]:
    """Page-absolute y at the horizontal centre of a fit's x-range."""
    if not fit.y_values:
        return None
    idx = max(0, min(len(fit.y_values) // 2, len(fit.y_values) - 1))
    return fit.y_page_offset + fit.y_values[idx]


def _interpolate_between(
    above_fit: FitResult,
    below_fit: FitResult,
    x_start: int,
    x_end: int,
    y_target: float,
) -> list[float]:
    """Synthesise y-values for a missing line between two detected neighbours.

    At each x, reads the y-value from the line above and below (clamping to
    their endpoints if x falls outside their range), then linearly interpolates
    to y_target based on its fractional position between the two neighbours.

    Returns page-absolute y-values (y_page_offset already included).
    """
    def y_at_x(fit: FitResult, x: int) -> float:
        idx = x - fit.x_start
        idx = max(0, min(idx, len(fit.y_values) - 1))
        return fit.y_page_offset + fit.y_values[idx]

    ys: list[float] = []
    for x in range(x_start, x_end + 1):
        ya = y_at_x(above_fit, x)
        yb = y_at_x(below_fit, x)
        if yb > ya:
            t = max(0.0, min(1.0, (y_target - ya) / (yb - ya)))
            ys.append(ya + t * (yb - ya))
        else:
            ys.append((ya + yb) / 2.0)  # degenerate: lines crossed or identical
    return ys


def _compute_interpolation_max_gap(
    all_gaps: list[float],
    cut_threshold: float,
    fallback_multiplier: float = INTERPOLATION_GAP_MULTIPLIER,
) -> float:
    """Adaptive upper bound for the interpolation trigger.

    Uses the mean of observed inter-stave gaps so the window
    ``[cut_threshold, max_threshold]`` captures exactly the zone between normal
    intra-stave spacing and typical inter-stave spacing.  Territory boundaries
    are the safety net that prevents synthesised lines from landing in
    inter-stave space; this threshold only controls the trigger.

    Falls back to ``cut_threshold × fallback_multiplier`` when no inter-stave
    gaps are observed.
    """
    inter_stave = [g for g in all_gaps if g > cut_threshold]
    if inter_stave:
        return float(np.mean(inter_stave))
    return cut_threshold * fallback_multiplier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpolate_missing_lines(
    fits: list[FitResult],
    assignments: list[StaveAssignment],
    rhythm_anomalies: dict[int, dict],
    scale_unit: float,
    min_threshold: float,
    cut_threshold: float,
    mode_n: int,
    interpolation_max_gap: Optional[float] = None,
    all_gaps: Optional[list[float]] = None,
) -> tuple[list[InterpolatedLine], float]:
    """Synthesise missing stafflines for staves that appear incomplete.

    Args:
        fits:                All FitResults from Stage 1, in detection order.
        assignments:         StaveAssignment list from group_staves.
        rhythm_anomalies:    Output of _check_stave_rhythm.  Staves whose
                             status is not 'normal' are skipped entirely.
        scale_unit:          Median staffline-box height (pixels); fallback
                             for h_est when a stave has no clean intra-gaps.
        min_threshold:       Lower gap bound (noise floor) for h_est estimation.
        cut_threshold:       Gap separating intra-stave from inter-stave.
        mode_n:              Expected lines per stave (page-level mode).
        interpolation_max_gap: Override for the upper trigger bound (pixels).
                             When None, computed adaptively from all_gaps.
        all_gaps:            Full page gap distribution; used only when
                             interpolation_max_gap is None.

    Returns:
        ``(interpolated_lines, max_threshold_used)``
    """
    # Resolve max_threshold.
    if interpolation_max_gap is not None:
        max_threshold = float(interpolation_max_gap)
    elif all_gaps:
        max_threshold = _compute_interpolation_max_gap(all_gaps, cut_threshold)
    else:
        max_threshold = cut_threshold * INTERPOLATION_GAP_MULTIPLIER

    # Build stave_id → [(fit_index, FitResult)] sorted by y.
    stave_fits: dict[int, list[tuple[int, FitResult]]] = {}
    for asg in assignments:
        if asg.stave_id is None:
            continue
        stave_fits.setdefault(asg.stave_id, []).append(
            (asg.fit_index, fits[asg.fit_index])
        )

    # --- Territory boundaries ---
    # Each stave owns the y-range bounded by the midpoint to the stave above
    # and the midpoint to the stave below.  Top stave: upper bound clamped to
    # its own topmost detected line (no upward extrapolation).  Bottom stave:
    # lower bound clamped to its own bottommost line.
    stave_y_mins: dict[int, float] = {}
    stave_y_maxs: dict[int, float] = {}
    for sid, fp in stave_fits.items():
        ys = [_y_at_center(f) for _, f in fp if _y_at_center(f) is not None]
        if ys:
            stave_y_mins[sid] = min(ys)
            stave_y_maxs[sid] = max(ys)

    sorted_sids = sorted(stave_y_mins, key=lambda s: stave_y_mins[s])
    upper_bound: dict[int, float] = {}
    lower_bound: dict[int, float] = {}
    for rank, sid in enumerate(sorted_sids):
        if rank > 0:
            prev = sorted_sids[rank - 1]
            upper_bound[sid] = (stave_y_maxs[prev] + stave_y_mins[sid]) / 2.0
        else:
            upper_bound[sid] = stave_y_mins[sid]   # top stave: no upward extrapolation
        if rank < len(sorted_sids) - 1:
            nxt = sorted_sids[rank + 1]
            lower_bound[sid] = (stave_y_maxs[sid] + stave_y_mins[nxt]) / 2.0
        else:
            lower_bound[sid] = stave_y_maxs[sid]   # bottom stave: no downward extrapolation

    result: list[InterpolatedLine] = []

    for stave_id, fit_pairs in stave_fits.items():
        # Rhythm gate: skip staves whose gap pattern deviates from the page
        # periodicity.  We can't synthesise lines safely in these cases.
        if rhythm_anomalies.get(stave_id, {}).get("status", "normal") != "normal":
            continue

        fit_pairs_sorted = sorted(
            fit_pairs, key=lambda p: _y_at_center(p[1]) or 0.0
        )
        centers = [_y_at_center(f) for _, f in fit_pairs_sorted]
        centers = [c for c in centers if c is not None]
        if not centers:
            continue

        # Intra-stave period estimate from gaps in [min_threshold, cut_threshold].
        all_gaps_s = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        intra_gaps = [g for g in all_gaps_s if min_threshold <= g < cut_threshold]
        h_est = float(np.median(intra_gaps)) if intra_gaps else float(scale_unit)
        if h_est <= 0:
            h_est = float(scale_unit)

        ub = upper_bound.get(stave_id, float("-inf"))
        lb = lower_bound.get(stave_id, float("inf"))
        stave_lines: list[InterpolatedLine] = []

        # --- Trigger A: in-stave gap fill ---
        if len(centers) >= 2:
            for idx in range(len(fit_pairs_sorted) - 1):
                above_fi, above_f = fit_pairs_sorted[idx]
                below_fi, below_f = fit_pairs_sorted[idx + 1]
                y_above = _y_at_center(above_f)
                y_below = _y_at_center(below_f)
                if y_above is None or y_below is None:
                    continue
                gap = y_below - y_above
                if gap < cut_threshold or gap > max_threshold:
                    continue
                n_missing = max(1, round(gap / h_est) - 1)
                x_start = min(above_f.x_start, below_f.x_start)
                x_end   = max(above_f.x_end,   below_f.x_end)
                for i in range(1, n_missing + 1):
                    y_target = y_above + i * gap / (n_missing + 1)
                    stave_lines.append(InterpolatedLine(
                        stave_id=stave_id,
                        within_stave_index=0,  # placeholder; re-indexed later
                        x_start=x_start,
                        x_end=x_end,
                        y_values=_interpolate_between(
                            above_f, below_f, x_start, x_end, y_target),
                        neighbor_fit_indices=(above_fi, below_fi),
                    ))

        # --- Trigger B: edge extrapolation (territory-bounded) ---
        # Only fires when the stave is mostly complete (≥ mode_n // 2 detected),
        # and only places lines within the stave's territory.
        if len(fit_pairs) < mode_n and len(fit_pairs) >= mode_n // 2:
            y_bottom = centers[-1]
            candidate_ys = [y_bottom - (mode_n - 1 - k) * h_est for k in range(mode_n)]

            # Greedy slot matching: claim nearest unoccupied slot per detected line.
            remaining_slots = list(range(mode_n))
            for cy, (_, _f) in zip(centers, fit_pairs_sorted):
                best_k = min(remaining_slots, key=lambda k: abs(candidate_ys[k] - cy))
                remaining_slots.remove(best_k)

            for k in remaining_slots:
                y_target = candidate_ys[k]
                if y_target <= ub or y_target >= lb:
                    continue  # outside this stave's territory

                above_pairs = [
                    (fi, f) for fi, f in fit_pairs_sorted
                    if (_y_at_center(f) or 0.0) < y_target
                ]
                below_pairs = [
                    (fi, f) for fi, f in fit_pairs_sorted
                    if (_y_at_center(f) or 0.0) >= y_target
                ]

                if above_pairs and below_pairs:
                    continue  # inside detected range; trigger A handles this

                ref_fi, ref_f = (above_pairs[-1] if above_pairs else below_pairs[0])
                ref_center = _y_at_center(ref_f) or y_target
                offset = y_target - ref_center
                stave_lines.append(InterpolatedLine(
                    stave_id=stave_id,
                    within_stave_index=k,  # grid slot; overwritten by re-index
                    x_start=ref_f.x_start,
                    x_end=ref_f.x_end,
                    y_values=[ref_f.y_page_offset + y + offset for y in ref_f.y_values],
                    neighbor_fit_indices=(ref_fi, ref_fi),
                ))

        result.extend(stave_lines)

    return result, max_threshold
