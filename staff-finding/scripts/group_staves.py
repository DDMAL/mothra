# -*- coding: utf-8 -*-
"""
Stave grouping for Stage 2 of the staff detection pipeline.

Given the FitResults produced by Stage 1 on a single page (or within a single
layout region of a page), groups them into staves using ratio-based gap
analysis on the fitted centerlines' y-positions, then delegates line synthesis
to interpolate_staves.py.

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

# InterpolatedLine lives in interpolate_staves; re-exported here so existing
# callers (shared_utils, tests) can continue importing it from group_staves.
from interpolate_staves import (  # noqa: F401
    InterpolatedLine,
    INTERPOLATION_GAP_MULTIPLIER,
    interpolate_missing_lines,
)

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

# Minimum gap multiplier on scale_unit.  Gaps smaller than
# scale_unit * MIN_GAP_MULTIPLIER are considered sub-spacing noise (e.g. from
# near-duplicate detections or near-zero intra-stave overlaps) and are
# excluded from the periodicity / h_est calculation.  Also shown as the lower
# bound on the gap distribution chart so the "live" spacing zone is clear.
MIN_GAP_MULTIPLIER = 0.5

# --- Stave-level duplicate / split-fragment reconciliation ---
# Two fits are candidates for reconciliation (i.e. likely the same physical
# staffline, detected more than once by the YOLO stafflline detector) when
# their y-at-center values are within this multiple of scale_unit, floored at
# an absolute minimum (mirrors fit_centerline.py's REFIT_TRIGGER_ABS_FLOOR_PX
# pattern, since a pure ratio can shrink toward 0 on small-scale_unit pages).
#
# Calibrated against the layer_1_3801 x 5013 reference page (scale_unit=71px):
# confirmed duplicate/split-fragment pairs showed y-gaps of 0.0-7.2px, while
# the smallest confirmed genuinely-distinct adjacent-line gap on that page was
# 18.9px. 0.15*71=10.65px sits well inside the gap between those two
# populations, deliberately biased toward the duplicate side -- under-merging
# just leaves the existing rhythm_over_populated / staves_with_unexpected_count
# flags active for QA, whereas over-merging silently discards a real line, the
# worse failure mode. Starting point only; needs validation across more of
# addtl-gt. Do NOT reuse MIN_GAP_MULTIPLIER (0.5) here -- at this page's scale
# that's 35.5px, which would wrongly merge the confirmed-real 18.9px pair.
DEDUP_Y_PROXIMITY_MULTIPLIER = 0.15
DEDUP_Y_PROXIMITY_ABS_FLOOR_PX = 3.0

# Two fits with DISJOINT x-ranges are reconciliation candidates when the gap
# between them is within this multiple of scale_unit. Reuses
# component_filter.py's MERGE_X_GAP_MULTIPLIER value directly -- this is the
# same physical question (how wide a decorated-initial / layout interruption
# can be) whether the two fragments landed in one YOLO box or two.
DEDUP_X_GAP_MULTIPLIER = 10.0

# Curve-agreement tolerance for a disjoint-x (Pattern B) pair, evaluated at
# their shared gap-reference x instead of each fragment's own, different
# center. Remains looser than DEDUP_Y_PROXIMITY_MULTIPLIER/_ABS_FLOOR_PX
# (which is the coarse candidate-pool gate AND the fallback tolerance when
# the extrapolation guard below doesn't clear) would suggest was needed --
# see below for why.
#
# An initial, tighter value (max(5.0, 0.10*scale_unit)=7.1px at this page's
# scale_unit=71) was calibrated from only 2 confirmed real Pattern-B pairs
# (disagreement 6.1px and 5.8px) and looked like a clear improvement over
# the old 10.65px coarse gate. Full end-to-end validation surfaced a 3rd
# real pair the tighter value wrongly rejected: two independently-fit
# quadratics of the SAME real line (confirmed by the old single-pass system
# correctly merging them, and by their own-center comparison agreeing to
# 5.8px) disagreed by 16.72px when extrapolated ~200px to their shared gap
# midpoint. Their coefficients have OPPOSITE-signed curvature
# (concave-down vs. concave-up) -- consistent with each independently-fit
# fragment only capturing its own local piece of a wavy/arching real line
# (this codebase already documents "S-shaped waves (parchment warping in
# two spots)" as a real phenomenon on these manuscripts -- see
# fit_centerline.py's LINE_FOLLOW_POLY_DEGREE comment). A single quadratic
# fit to one local piece of an S-curve does not necessarily extrapolate
# consistently with an independent quadratic fit to a different piece, even
# for the same real line -- this is a structural limit of comparing
# independently-fit local polynomials, not just fit noise, and no amount of
# _extrapolation_guard_px width-based tightening addresses it (both
# fragments here were comfortably within their own guard).
#
# Loosened to max(15.0, 0.30*scale_unit)=21.3px at this page's scale, clearing
# the confirmed 16.72px case with ~27% headroom. This only ever applies on
# top of a pair that already passed the coarse DEDUP_Y_PROXIMITY_* pool gate
# (unchanged), so it's a narrower, still-additive check, just less strict
# than the first attempt. Calibrated against exactly one confirmed
# "must accept" real data point and zero confirmed real "must reject via
# curve-agreement specifically" data points -- treat as more tentative than
# the other DEDUP_* constants, and revisit once addtl-gt validation surfaces
# more examples either direction.
DEDUP_CURVE_AGREEMENT_MULTIPLIER = 0.30
DEDUP_CURVE_AGREEMENT_ABS_FLOOR_PX = 15.0

# How far past a fragment's OWN observed width (x_end - x_start) we trust its
# fitted polynomial to be evaluated, keyed by polynomial degree -- expressed
# as a fraction of the fragment's own width (the direct measure of how much
# support informed this polynomial), not scale_unit (the wrong axis -- a
# line-thickness quantity, not an x-distance one) and not a fixed pixel
# budget (doesn't adapt across manuscripts of different scan scale). Cubic
# fits (fit_centerline.py's line-following refit) get half the trusted
# radius of quadratic: a cubic's leading term grows with the CUBE rather
# than the square of distance from the fit's own support, so the same
# extrapolation ratio is riskier there. Falls back to the cubic (more
# conservative) multiplier for any degree not listed.
#
# No separate absolute ceiling: DEDUP_X_GAP_MULTIPLIER (10*scale_unit)
# already caps the total candidate gap, so required extrapolation (gap/2)
# is already bounded by 5*scale_unit; once a fragment's own width exceeds
# 10*scale_unit, this guard is non-binding for any gap the candidacy pool
# would even consider.
EXTRAP_GUARD_WIDTH_MULTIPLIER_BY_DEGREE = {2: 0.5, 3: 0.25}
EXTRAP_GUARD_DEFAULT_MULTIPLIER = 0.25
EXTRAP_GUARD_ABS_FLOOR_PX = 15.0


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
        duplicate_reconciliation: Maps a primary fit_index to the list of
            fit_indices absorbed into it as duplicate/split-fragment
            detections of the same physical line (see
            _reconcile_duplicate_fits). Absorbed fits still get a
            StaveAssignment (stave_id=None, within_stave_index=None, flagged
            'duplicate_of:<primary>' or 'companion_of:<primary>') -- this
            field exists for page-level diagnostics, not as the source of
            truth for which fits were absorbed.
        flags: Page-level grouping flags, e.g. 'no_fits_available',
            'mode_count_below_typical', 'staves_with_unexpected_count'.
    """

    assignments: list[StaveAssignment] = field(default_factory=list)
    mode_lines_per_stave: Optional[int] = None
    line_count_distribution: dict[int, int] = field(default_factory=dict)
    cut_threshold_px: float = 0.0
    min_threshold_px: float = 0.0
    gap_distribution: list[float] = field(default_factory=list)
    interpolated_lines: list[InterpolatedLine] = field(default_factory=list)
    interpolation_max_gap_px: Optional[float] = None
    rhythm_anomalies: dict[int, dict] = field(default_factory=dict)
    duplicate_reconciliation: dict[int, list[int]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def group_staves(
    fits: list[FitResult],
    scale_unit: float,
    interpolate_missing: bool = DEFAULT_INTERPOLATE_MISSING,
    interpolation_max_gap: Optional[float] = None,
    save_path: Optional[Path] = None,
    page_size: Optional[tuple[int, int]] = None,
    page_image: Optional[np.ndarray] = None,
    use_valley_threshold: bool = False,
) -> StaveGroupingResult:
    """Group fitted centerlines into staves on a single page or region.

    Args:
        fits: All FitResults produced by Stage 1, in detection order. Fits
            with no y_values (no_fit_attempted, fit_did_not_converge) are
            recorded but excluded from grouping.
        scale_unit: Page-level scale unit h. Used as a sanity floor on the
            grouping gap threshold (a cut threshold smaller than h is
            implausible).
        interpolate_missing: When True, synthesize centerlines where a gap
            between two consecutive detected lines in the same stave falls
            in [cut_threshold, interpolation_max_gap].  Each synthesized
            line is interpolated between its two detected neighbours and
            carries source='interpolated'.  Default False.
        interpolation_max_gap: Upper gap bound (pixels) for the
            interpolation trigger.  Gaps above this are treated as
            inter-stave spacing and are not filled.  When None it is
            derived adaptively from the page gap distribution (see
            interpolate_staves._compute_interpolation_max_gap), falling
            back to cut_threshold * INTERPOLATION_GAP_MULTIPLIER when no
            gaps are available.
        save_path: If provided, render a page-level diagnostic showing each
            fit colored by its stave assignment, plus annotations of gap
            distribution and cut threshold.
        page_size: (width, height) of the page in pixels. Required only
            when save_path is set and page_image is not.
        page_image: The full page (or the BGR-processed page) to overlay
            stave assignments onto in the diagnostic. Optional; if absent,
            the diagnostic falls back to a blank canvas of page_size.
        use_valley_threshold: When True, use _find_valley_threshold() instead
            of the default median-based _determine_cut_threshold().  The valley
            method finds the largest gap between consecutive distinct gap values
            and places the cut at the midpoint of that gap, making it robust when
            intra-stave line spacing is close to scale_unit h (which causes the
            median-based threshold to land inside the intra-stave cluster).
            Falls back to the median method if no clear valley is found.
            Default False so existing pipeline behaviour is unchanged.

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
        y_center = y_at_fit_center(fit)
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

    # --- Stage 1.5: Reconcile duplicate / split-fragment detections ---
    # Must run before the y-sort below: this is the only stage that needs
    # x-range data, and it removes absorbed fits from fit_positions so every
    # later stage (gaps, threshold, assignment, rhythm check) sees only
    # deduplicated positions with no further changes required.
    fit_positions, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        fits, fit_positions, scale_unit
    )
    unassignable.update(absorbed_flags)
    result.duplicate_reconciliation = duplicate_groups
    if duplicate_groups:
        n_absorbed = sum(len(v) for v in duplicate_groups.values())
        result.flags.append(f"reconciled_duplicate_fits:{n_absorbed}")

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
    if use_valley_threshold:
        cut_threshold = _find_valley_threshold(gaps, scale_unit)
    else:
        cut_threshold = _determine_cut_threshold(gaps, scale_unit)
    result.cut_threshold_px = cut_threshold

    # Minimum gap: sub-spacing noise floor.  Gaps below this are excluded from
    # the periodicity (h_est) calculation and displayed on the diagnostic chart.
    min_threshold = scale_unit * MIN_GAP_MULTIPLIER
    result.min_threshold_px = min_threshold

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
                    y_at_center=y_at_fit_center(fit),
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
            str(sid)
            for sid, cnt in stave_line_counts.items()
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

    # --- Stage 6c: Rhythm / periodicity anomaly detection ---
    # Checks whether each stave's intra-stave gap count and spacing consistency
    # match the page's expected periodicity (mode_lines_per_stave).  Staves that
    # fall outside the expected pattern are flagged as "under_populated" or
    # "over_populated" so downstream code and QA can treat them differently from
    # ordinary short staves.
    if result.mode_lines_per_stave is not None:
        result.rhythm_anomalies = _check_stave_rhythm(
            gaps=gaps,
            raw_assignments_sorted=raw_assignments,
            cut_threshold=cut_threshold,
            min_threshold=min_threshold,
            mode_n=result.mode_lines_per_stave,
        )
        # Propagate rhythm flags to individual assignments so per-line QA
        # records carry the stave-level anomaly label.
        for asg in result.assignments:
            if asg.stave_id is not None:
                ra = result.rhythm_anomalies.get(asg.stave_id, {})
                if ra.get("status", "normal") != "normal":
                    asg.flags.append(f"rhythm_{ra['status']}")

    # --- Stage 7: Synthesise missing lines (if requested) ---
    # Full algorithm in interpolate_staves.py.  This stage is the call site
    # only; see that module for trigger A/B details and territory logic.
    if interpolate_missing and result.mode_lines_per_stave is not None:
        interp_lines, max_gap = interpolate_missing_lines(
            fits=fits,
            assignments=result.assignments,
            rhythm_anomalies=result.rhythm_anomalies,
            scale_unit=scale_unit,
            min_threshold=min_threshold,
            cut_threshold=cut_threshold,
            mode_n=result.mode_lines_per_stave,
            interpolation_max_gap=interpolation_max_gap,
            all_gaps=gaps,
        )
        result.interpolated_lines.extend(interp_lines)
        result.interpolation_max_gap_px = max_gap
        _reindex_stave_lines(result.assignments, result.interpolated_lines)

    # --- Stage 8: Generate diagnostic image (if requested) ---
    # Design doc §6.6: Diagnostic preserves all grouping evidence for QA.
    if save_path is not None:
        _save_grouping_diagnostic(fits, result, page_size, page_image, save_path)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reindex_stave_lines(
    assignments: list[StaveAssignment],
    interpolated_lines: list[InterpolatedLine],
) -> None:
    """Re-sort within_stave_index for every stave after interpolation.

    Merges detected assignments and newly synthesised lines for each stave,
    sorts all by y-position, and assigns consecutive 0-based indices.  Called
    once at the end of Stage 7 so both detected and interpolated lines carry
    correct, non-overlapping within_stave_index values.
    """
    stave_ids = {a.stave_id for a in assignments if a.stave_id is not None}
    for stave_id in stave_ids:
        entries: list[tuple[float, str, int]] = []
        for asg in assignments:
            if asg.stave_id == stave_id:
                entries.append((asg.y_at_center or 0.0, "det", asg.fit_index))
        for k, il in enumerate(interpolated_lines):
            if il.stave_id == stave_id:
                yc = il.y_values[len(il.y_values) // 2] if il.y_values else 0.0
                entries.append((yc, "interp", k))
        entries.sort(key=lambda e: e[0])
        for new_idx, (_, kind, obj_idx) in enumerate(entries):
            if kind == "det":
                for asg in assignments:
                    if asg.fit_index == obj_idx:
                        asg.within_stave_index = new_idx
                        break
            else:
                interpolated_lines[obj_idx].within_stave_index = new_idx


def y_at_fit_center(fit: "FitResult") -> Optional[float]:
    """Return the page-absolute y at the horizontal midpoint of a fit's x-range.

    Returns None if the fit has no y_values (empty or failed fit).

    Design doc §6.1: Extract a single representative y-position per fit for
    gap analysis. Using the horizontal center is robust to local skew and gives
    a stable ordering for grouping. The result is page-absolute (fit.y_page_offset
    added) so that gaps between staves are real pixel distances on the page rather
    than crop-local values that cluster near 0.
    """
    if not fit.y_values:
        return None

    # --- Calculate horizontal center index ---
    # fit.y_values has one entry per integer x in [x_start, x_end].
    # The center index is roughly halfway along this range.
    center_idx = len(fit.y_values) // 2

    # Clamp to valid range (should be unnecessary, but safe).
    center_idx = max(0, min(center_idx, len(fit.y_values) - 1))

    # y_values are crop-local; add y_page_offset to get the page-absolute y.
    return fit.y_page_offset + fit.y_values[center_idx]


def page_absolute_x_range(fit: "FitResult") -> Optional[tuple[float, float]]:
    """Return the page-absolute (x_start, x_end) of a fit's horizontal extent.

    Mirrors y_at_fit_center's emptiness check: x_start/x_end are meaningless
    placeholders on a fit with no y_values (empty or failed fit), so this
    returns None in that case too.
    """
    if not fit.y_values:
        return None
    return (fit.x_start + fit.x_page_offset, fit.x_end + fit.x_page_offset)


def _fit_degree(fit: "FitResult") -> int:
    """Polynomial degree actually used for this fit's coefficients.

    Reads len(coefficients) - 1 directly rather than parsing fit.flags for
    'line_following_applied:deg3' -- this is robust to flag-string drift, and
    also correctly reports deg2 for the documented fit_centerline.py case
    where line-following triggers but falls back to POLY_DEGREE (fewer than
    2*(LINE_FOLLOW_POLY_DEGREE+1)=8 trace points available), which the flags
    list alone can't distinguish from "line-following never triggered."
    """
    return max(0, len(fit.coefficients) - 1)


def _fit_y_at_x(fit: "FitResult", x_page: float) -> Optional[float]:
    """Evaluate a fit's fitted polynomial at an arbitrary page-absolute x.

    Unlike y_at_fit_center (samples a precomputed y_values index), this
    evaluates fit.coefficients directly via np.polyval and is meant to be
    called at x_page positions OUTSIDE the fit's own [x_start, x_end] --
    genuine polynomial extrapolation, not interpolation/clamping.

    Coordinate-system note (see FitResult docstring, fit_centerline.py):
    coefficients are fit against CROP-LOCAL x, not page-absolute x:
        crop_local_x = x_page - fit.x_page_offset
        crop_local_y = np.polyval(fit.coefficients, crop_local_x)
        page_absolute_y = crop_local_y + fit.y_page_offset

    Returns None if fit.coefficients is empty (mirrors the emptiness
    convention already used by y_at_fit_center / page_absolute_x_range).
    """
    if not fit.coefficients:
        return None
    crop_local_x = x_page - fit.x_page_offset
    crop_local_y = float(np.polyval(fit.coefficients, crop_local_x))
    return crop_local_y + fit.y_page_offset


def _gap_reference_x(
    left_range: tuple[float, float], right_range: tuple[float, float]
) -> float:
    """Shared page-absolute x for comparing two disjoint fragments: the
    midpoint of the gap between them.

    This is the unique point that makes both fragments extrapolate the same,
    minimal distance (gap/2 each) -- any other shared point increases the
    worse of the two distances. A single physical line's slope creates
    spurious disagreement only when the two fragments are compared at
    DIFFERENT x's (which is what comparing each fragment's own
    y_at_fit_center does); evaluating both at the SAME x removes that bias.
    """
    return (left_range[1] + right_range[0]) / 2.0


def _extrapolation_guard_px(width_px: float, degree: int) -> float:
    """Max page-x distance past a fragment's own observed width we trust its
    fitted polynomial to be evaluated at, for the disjoint-pair curve-
    agreement check specifically. See EXTRAP_GUARD_* constants for the
    reasoning behind the per-degree multipliers and the absolute floor.
    """
    multiplier = EXTRAP_GUARD_WIDTH_MULTIPLIER_BY_DEGREE.get(
        degree, EXTRAP_GUARD_DEFAULT_MULTIPLIER
    )
    return max(EXTRAP_GUARD_ABS_FLOOR_PX, multiplier * width_px)


def _disjoint_curve_cost(
    left_fit: "FitResult",
    right_fit: "FitResult",
    left_own_range: tuple[float, float],
    right_own_range: tuple[float, float],
    ref_x: float,
) -> tuple[float, bool]:
    """Cost (px) and evaluation-mode flag for one disjoint-x candidate pair.

    When both fragments' own extrapolation distances (from their OWN
    observed range to ref_x) clear _extrapolation_guard_px for their own
    width/degree, cost is the curve-agreement disagreement at ref_x
    (compare against DEDUP_CURVE_AGREEMENT_*). Otherwise falls back to the
    coarse |own-center y difference| via y_at_fit_center -- i.e. exactly
    the pre-refinement comparison (compare against y_threshold). This
    fallback is what guarantees the curve-agreement machinery can only ever
    ADD strictness relative to the old behavior, never merge something the
    old coarse check wouldn't already have allowed.
    """
    left_extrap = max(0.0, ref_x - left_own_range[1])
    right_extrap = max(0.0, right_own_range[0] - ref_x)
    left_ok = left_extrap <= _extrapolation_guard_px(
        left_own_range[1] - left_own_range[0], _fit_degree(left_fit)
    )
    right_ok = right_extrap <= _extrapolation_guard_px(
        right_own_range[1] - right_own_range[0], _fit_degree(right_fit)
    )
    if left_ok and right_ok:
        y_left = _fit_y_at_x(left_fit, ref_x)
        y_right = _fit_y_at_x(right_fit, ref_x)
        if y_left is not None and y_right is not None:
            return abs(y_left - y_right), True

    y_left_c = y_at_fit_center(left_fit)
    y_right_c = y_at_fit_center(right_fit)
    if y_left_c is None or y_right_c is None:
        return float("inf"), False
    return abs(y_left_c - y_right_c), False


def _primary_and_absorbed(
    members: list[int], fits: list["FitResult"]
) -> tuple[int, list[int]]:
    """Pick the primary of a cluster and return (primary, absorbed_rest).

    Primary: most n_pixels_used (most real ink supporting the fit -- the
    same quantity fit_centerline.py already computes for this purpose), then
    lowest residual_mean (a sound tiebreaker once pixel count has already
    screened for substantial detections; on its own it can look spuriously
    good on a short/sparse fragment), then lowest fit_index (full
    determinism). Sorts a copy so the caller's own list order is untouched.
    """
    ordered = sorted(
        members, key=lambda fi: (-fits[fi].n_pixels_used, fits[fi].residual_mean, fi)
    )
    return ordered[0], ordered[1:]


def _reconcile_duplicate_fits(
    fits: list["FitResult"],
    fit_positions: list[tuple[int, float]],
    scale_unit: float,
) -> tuple[list[tuple[int, float]], dict[int, list[str]], dict[int, list[int]]]:
    """Collapse duplicate / split-fragment YOLO detections of one physical
    staffline into a single representative fit, before stave assignment.

    The YOLO staffline detector sometimes emits more than one box for the
    same physical line: either two heavily-overlapping boxes (a near-
    duplicate detection -- e.g. from an NMS IoU threshold that's slightly too
    permissive for that pair), or two x-disjoint boxes (a real line split by
    a page-layout interruption, e.g. a decorated initial). Neither shape is
    caught anywhere else in the pipeline: component_filter.py's merge logic
    only ever sees pixels from one YOLO box at a time, and _assign_staves
    groups purely by y-gap with no x-range awareness at all.

    Two passes, sharing one union-find structure:

    Pass A -- overlapping-x duplicates (Pattern A). Uses the same
    union-find clustering shape as component_filter._compute_merge_groups,
    but with INVERTED x-overlap polarity. Within a single YOLO box crop, two
    overlapping-x components are likely two DIFFERENT parallel lines caught
    in one too-tall box, so that function excludes overlap from merging.
    Here, comparing centerlines of SEPARATE YOLO detections, overlapping
    x-ranges at near-identical y is the opposite signal -- real adjacent
    stafflines are never that close in y -- so overlap plus y-proximity
    means "same line, detected twice."

    Pass B -- disjoint-x split fragments (Pattern B). Each Pass-A cluster's
    primary becomes a "representative"; representatives are matched via
    mutual-nearest-neighbor, evaluated per side (left/right) independently
    (so a three-fragment left-middle-right split can still resolve in one
    pass), using curve-agreement at the shared gap midpoint
    (_gap_reference_x / _fit_y_at_x / _disjoint_curve_cost) rather than each
    representative's own arbitrary center -- comparing two fragments at the
    SAME x removes the slope-driven disagreement a plain center-to-center
    comparison is prone to. The extrapolation-distance guard
    (_extrapolation_guard_px) falls back to the coarse own-center comparison
    when a fragment is too narrow to trust evaluating its curve that far
    from its own support; the fallback tolerance is the unchanged
    y_threshold, so this machinery can only ever ADD strictness relative to
    the coarse gate, never merge a pair the coarse gate wouldn't already
    have allowed.

    Args:
        fits: All FitResults for the page, in detection order (the same list
            group_staves() received).
        fit_positions: (fit_index, y_at_center) pairs for fits that have a
            valid y-position, in arbitrary order (as built by Stage 1, before
            sorting). Only these are eligible for clustering.
        scale_unit: Page-level scale unit h.

    Returns:
        kept_fit_positions: fit_positions with absorbed entries removed --
            same shape as the input, safe to feed directly into the existing
            sort-by-y step.
        absorbed_flags: fit_index -> single-element flags list
            (["duplicate_of:<primary>"] or ["companion_of:<primary>"]),
            meant to be folded directly into Stage 1's `unassignable` dict.
        duplicate_groups: primary fit_index -> [absorbed fit_index, ...], for
            StaveGroupingResult.duplicate_reconciliation (diagnostics only).
    """
    if len(fit_positions) < 2:
        return fit_positions, {}, {}

    y_threshold = max(
        DEDUP_Y_PROXIMITY_ABS_FLOOR_PX, DEDUP_Y_PROXIMITY_MULTIPLIER * scale_unit
    )
    x_gap_threshold = DEDUP_X_GAP_MULTIPLIER * scale_unit

    y_by_idx = dict(fit_positions)
    # fit_positions only contains fits with non-empty y_values (Stage 1's own
    # check), and page_absolute_x_range uses that identical emptiness check,
    # so every entry here is guaranteed non-None.
    x_ranges = {i: page_absolute_x_range(fits[i]) for i, _ in fit_positions}

    # Walk pairs in y-sorted order so the inner loop can break out early once
    # the y-gap exceeds threshold -- pages run 80-150+ fits, and this avoids
    # a full O(n^2) sweep (fine within one small crop in component_filter.py,
    # less so across a whole page here).
    order = sorted((i for i, _ in fit_positions), key=lambda i: y_by_idx[i])

    parent = {i: i for i in order}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # --- Pass A: overlapping-x duplicates only (Pattern A). Disjoint-x
    # candidates (Pattern B) are handled separately in Pass B below, via
    # curve-agreement rather than a flat y-gap comparison. ---
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            if abs(y_by_idx[i] - y_by_idx[j]) > y_threshold:
                break  # order is y-sorted; no later j can be closer
            xi, xj = x_ranges[i], x_ranges[j]
            left, right = (xi, xj) if xi[0] <= xj[0] else (xj, xi)
            gap = right[0] - left[1]  # negative => overlapping x-ranges
            if gap < 0:
                union(i, j)

    # --- Bridge: one representative + aggregate x-range per Pass-A cluster.
    # The aggregate range (not just the primary's own) matters because a
    # cluster's non-primary member can reach slightly further toward a
    # disjoint partner than the primary alone -- using only the primary's
    # own range for candidacy would silently miss connections the old,
    # single-pass full-pairwise sweep would have found via transitivity. ---
    pass_a_clusters: dict[int, list[int]] = {}
    for i in order:
        pass_a_clusters.setdefault(find(i), []).append(i)

    representatives: list[int] = []
    agg_x_ranges: dict[int, tuple[float, float]] = {}
    for members in pass_a_clusters.values():
        primary, _ = _primary_and_absorbed(members, fits)
        representatives.append(primary)
        agg_x_ranges[primary] = (
            min(x_ranges[m][0] for m in members),
            max(x_ranges[m][1] for m in members),
        )

    # --- Pass B: disjoint-x split fragments (Pattern B), mutual-nearest-
    # neighbor over representatives, evaluated per side (left/right)
    # independently so a three-fragment left-middle-right split can still
    # resolve in one pass (the middle fragment can simultaneously be the
    # mutual best match of its left neighbor and its right neighbor). ---
    rep_order = sorted(representatives, key=lambda i: y_by_idx[i])
    right_candidates: dict[int, list[tuple[float, int]]] = {r: [] for r in rep_order}
    left_candidates: dict[int, list[tuple[float, int]]] = {r: [] for r in rep_order}

    curve_tolerance = max(
        DEDUP_CURVE_AGREEMENT_ABS_FLOOR_PX, DEDUP_CURVE_AGREEMENT_MULTIPLIER * scale_unit
    )

    for a in range(len(rep_order)):
        for b in range(a + 1, len(rep_order)):
            ri, rj = rep_order[a], rep_order[b]
            if abs(y_by_idx[ri] - y_by_idx[rj]) > y_threshold:
                break  # rep_order is y-sorted; no later rj can be closer
            agg_i, agg_j = agg_x_ranges[ri], agg_x_ranges[rj]
            left_rep, right_rep = (ri, rj) if agg_i[0] <= agg_j[0] else (rj, ri)
            agg_left, agg_right = agg_x_ranges[left_rep], agg_x_ranges[right_rep]
            agg_gap = agg_right[0] - agg_left[1]
            if not (0 <= agg_gap <= x_gap_threshold):
                continue
            ref_x = _gap_reference_x(agg_left, agg_right)
            cost, used_curve = _disjoint_curve_cost(
                fits[left_rep], fits[right_rep],
                x_ranges[left_rep], x_ranges[right_rep], ref_x,
            )
            tolerance = curve_tolerance if used_curve else y_threshold
            if cost > tolerance:
                continue
            right_candidates[left_rep].append((cost, right_rep))
            left_candidates[right_rep].append((cost, left_rep))

    for r in rep_order:
        if right_candidates[r]:
            _, best_right = min(right_candidates[r])
            if left_candidates[best_right]:
                _, best_of_best = min(left_candidates[best_right])
                if best_of_best == r:
                    union(r, best_right)
        if left_candidates[r]:
            _, best_left = min(left_candidates[r])
            if right_candidates[best_left]:
                _, best_of_best = min(right_candidates[best_left])
                if best_of_best == r:
                    union(r, best_left)

    # --- Final clusters, reflecting both passes' unions ---
    final_clusters: dict[int, list[int]] = {}
    for i in order:
        final_clusters.setdefault(find(i), []).append(i)

    kept: list[tuple[int, float]] = []
    absorbed_flags: dict[int, list[str]] = {}
    duplicate_groups: dict[int, list[int]] = {}

    for members in final_clusters.values():
        if len(members) == 1:
            fi = members[0]
            kept.append((fi, y_by_idx[fi]))
            continue

        primary, absorbed = _primary_and_absorbed(members, fits)
        kept.append((primary, y_by_idx[primary]))
        duplicate_groups[primary] = absorbed

        xp = x_ranges[primary]
        for fi in absorbed:
            xi = x_ranges[fi]
            left, right = (xi, xp) if xi[0] <= xp[0] else (xp, xi)
            label = "duplicate_of" if (right[0] - left[1]) < 0 else "companion_of"
            absorbed_flags[fi] = [f"{label}:{primary}"]

    return kept, absorbed_flags, duplicate_groups


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


def _find_valley_threshold(gaps: list[float], scale_unit: float) -> float:
    """Find the cut threshold separating intra-stave from inter-stave gaps.

    Uses a 1-D Otsu criterion: sweep all candidate thresholds and pick the one
    that maximises inter-class variance between the "small gap" and "large gap"
    clusters.  This is equivalent to minimising the weighted within-class
    variance, and is robust to outlier gaps in the upper tail — unlike a simple
    "largest jump" heuristic, which is easily misled by a single outlier gap at
    the top of the inter-stave distribution.

    The gap distribution for a page with multiple staves is bimodal: a dense
    cluster of small intra-stave gaps (lines within the same stave) and a
    sparser cluster of larger inter-stave gaps (space between staves).  The
    Otsu criterion naturally finds this boundary regardless of whether the upper
    cluster has a tight or diffuse spread.

    Falls back to the median-based threshold when fewer than 4 gaps are
    available (not enough data to fit two meaningful clusters), or if the
    optimal split would leave only a single gap in the upper class (likely a
    lone outlier rather than a true inter-stave cluster).

    Floored at scale_unit for the same reason as _determine_cut_threshold.
    """
    if not gaps:
        return scale_unit
    if len(gaps) < 4:
        return _determine_cut_threshold(gaps, scale_unit)

    gaps_arr = np.array(sorted(gaps))
    n = len(gaps_arr)

    # Sweep every unique gap value as a candidate split point, pick the one
    # that maximises inter-class variance (Otsu 1-D).
    best_var = -1.0
    best_threshold = None

    for t in np.unique(gaps_arr)[:-1]:  # exclude the very last value
        below = gaps_arr[gaps_arr <= t]
        above = gaps_arr[gaps_arr > t]
        if len(below) == 0 or len(above) == 0:
            continue
        w1 = len(below) / n
        w2 = len(above) / n
        inter_var = w1 * w2 * (float(below.mean()) - float(above.mean())) ** 2
        if inter_var > best_var:
            best_var = inter_var
            best_threshold = float(t)

    if best_threshold is None:
        return _determine_cut_threshold(gaps, scale_unit)

    # Place the threshold just above the Otsu split point so that the split
    # value itself falls in the lower (intra-stave) class.
    threshold = best_threshold + 0.5

    # Require at least 2 gaps in the upper class; a single gap is more likely
    # a lone outlier than a genuine inter-stave cluster.
    if np.sum(gaps_arr > threshold) < 2:
        return _determine_cut_threshold(gaps, scale_unit)

    return max(threshold, scale_unit)


def _check_stave_rhythm(
    gaps: list[float],
    raw_assignments_sorted: list[tuple[int, int, int]],
    cut_threshold: float,
    min_threshold: float,
    mode_n: int,
) -> dict[int, dict]:
    """Detect staves whose gap pattern deviates from the page's expected periodicity.

    Walks the sorted gap sequence and, for each stave, measures:

    * **intra_count**: how many consecutive intra-stave gaps the stave produced
      (= detected_line_count − 1).  Expected value is ``mode_n − 1``.
    * **gap_cv**: coefficient of variation of intra-stave gap sizes, using only
      gaps in ``[min_threshold, cut_threshold]``.  Low = consistent period;
      high = chaotic spacing.

    Classification:
        ``"under_populated"`` — ``intra_count < mode_n // 2``.  The stave has
        fewer than half the expected lines.  Could be a split stave, a grouper
        error, or a non-musical section (rubric / lesson text).
        ``"over_populated"``  — ``intra_count > mode_n + 1``.  More lines than
        expected; two staves may have been merged.
        ``"normal"``          — within ±1 of ``mode_n − 1``.

    Returns:
        dict mapping stave_id → {
            "status":                 str,
            "lines_observed":         int,   # detected_count
            "lines_expected":         int,   # mode_n
            "intra_count_observed":   int,   # detected_count − 1
            "intra_count_expected":   int,   # mode_n − 1
            "gap_cv":                 float | None,
            "gap_index_range":        [int, int],  # indices into gap_distribution
        }
    """
    if not gaps or mode_n < 2 or not raw_assignments_sorted:
        return {}

    # Build stave → (start_pos, end_pos) in the sorted fit sequence.
    # start_pos and end_pos are 0-based indices into raw_assignments_sorted.
    stave_ranges: dict[int, tuple[int, int]] = {}
    for pos, (_, stave_id, _) in enumerate(raw_assignments_sorted):
        if stave_id not in stave_ranges:
            stave_ranges[stave_id] = (pos, pos)
        else:
            s, _ = stave_ranges[stave_id]
            stave_ranges[stave_id] = (s, pos)

    expected_intra = mode_n - 1
    result: dict[int, dict] = {}

    for stave_id, (start, end) in stave_ranges.items():
        intra_count = end - start  # = detected_line_count − 1

        # Collect intra-stave gap values for this stave's position range,
        # filtering out noise gaps below min_threshold.
        intra_gap_vals = [
            gaps[k]
            for k in range(start, end)
            if k < len(gaps) and min_threshold <= gaps[k] < cut_threshold
        ]

        if len(intra_gap_vals) >= 2:
            mean_g = float(np.mean(intra_gap_vals))
            std_g = float(np.std(intra_gap_vals))
            gap_cv = std_g / mean_g if mean_g > 0 else float("inf")
        elif len(intra_gap_vals) == 1:
            gap_cv = 0.0
        else:
            gap_cv = None

        # Classify rhythm status.
        if intra_count < mode_n // 2:
            status = "under_populated"
        elif intra_count > mode_n + 1:
            status = "over_populated"
        else:
            status = "normal"

        # Gap index range: intra-stave bars in the gap distribution chart.
        # Use [start-1, end] so the bounding inter-stave spikes are included
        # in the shaded region, making the anomaly visually obvious even when
        # a stave has zero intra-stave gaps.
        chart_start = max(0, start - 1)
        chart_end = min(len(gaps) - 1, end)

        result[stave_id] = {
            "status": status,
            "lines_observed": intra_count + 1,
            "lines_expected": mode_n,
            "intra_count_observed": intra_count,
            "intra_count_expected": expected_intra,
            "gap_cv": round(gap_cv, 3) if gap_cv is not None else None,
            "gap_index_range": [chart_start, chart_end],
        }

    return result


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
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # --- Determine canvas size ---
    if page_image is not None:
        canvas_h, canvas_w = page_image.shape[:2]
        canvas = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        if page_size is None:
            page_size = (1000, 1000)  # Fallback.
        canvas_w, canvas_h = page_size
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)

    # --- Generate colors for staves ---
    # Key colors by actual stave_id so the legend always matches the lines.
    assigned_ids = sorted(
        {a.stave_id for a in result.assignments if a.stave_id is not None}
    )
    if not assigned_ids:
        assigned_ids = list(range(5))  # fallback for empty result

    cmap = matplotlib.colormaps["tab10"]
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

        # Convert crop-local coords to page-absolute before drawing on the
        # full-page canvas.
        xs = np.arange(fit.x_start, fit.x_end + 1) + int(fit.x_page_offset)
        ys = np.array(fit.y_values) + fit.y_page_offset

        mask = (xs >= 0) & (xs < canvas_w) & (ys >= 0) & (ys < canvas_h)
        xs = xs[mask]
        ys = ys[mask]

        if len(xs) < 2:
            continue

        points = np.column_stack([xs, ys]).astype(np.int32)

        color_rgba = stave_colors[assignment.stave_id]
        color_rgb = color_rgba[:3]
        color_rgb_255 = tuple(int(c * 255) for c in color_rgb)

        cv2.polylines(canvas_uint8, [points], False, color_rgb_255, thickness=2)

    # --- Overlay interpolated lines (dashed, same stave color but lighter) ---
    for interp in result.interpolated_lines:
        if interp.stave_id not in stave_colors:
            continue
        if not interp.y_values:
            continue

        xs = np.arange(interp.x_start, interp.x_end + 1)
        ys = np.array(interp.y_values)
        mask = (xs >= 0) & (xs < canvas_w) & (ys >= 0) & (ys < canvas_h)
        xs = xs[mask].astype(np.int32)
        ys = ys[mask].astype(np.int32)
        if len(xs) < 2:
            continue

        color_rgba = stave_colors[interp.stave_id]
        color_rgb = color_rgba[:3]
        # Lighten the color by blending toward white (0.55 original + 0.45 white)
        light_rgb = tuple(min(1.0, c * 0.55 + 0.45) for c in color_rgb)
        color_rgb_255 = tuple(int(c * 255) for c in light_rgb)

        # Draw as a dashed line: alternate 10-pixel drawn / 6-pixel gap segments.
        DASH, GAP = 10, 6
        i = 0
        while i < len(xs) - 1:
            j = min(i + DASH, len(xs) - 1)
            seg = np.column_stack([xs[i : j + 1], ys[i : j + 1]])
            cv2.polylines(canvas_uint8, [seg], False, color_rgb_255, thickness=2)
            i += DASH + GAP

    # --- Draw stave bounding boxes (red rectangles) ---
    # Collect fits per stave, then compute the page-absolute bounding box
    # enclosing all lines in that stave.
    stave_fits: dict[int, list] = {}
    for assignment in result.assignments:
        if assignment.stave_id is None:
            continue
        fit_idx = assignment.fit_index
        if fit_idx >= len(fits):
            continue
        fit = fits[fit_idx]
        if not fit.y_values:
            continue
        stave_fits.setdefault(assignment.stave_id, []).append(fit)

    BOX_PAD = 12  # px of breathing room around each stave's extent
    for stave_id, stave_fit_list in stave_fits.items():
        left_x = int(min(f.x_page_offset + f.x_start for f in stave_fit_list)) - BOX_PAD
        right_x = int(max(f.x_page_offset + f.x_end for f in stave_fit_list)) + BOX_PAD
        top_y = (
            int(min(f.y_page_offset + min(f.y_values) for f in stave_fit_list))
            - BOX_PAD
        )
        bot_y = (
            int(max(f.y_page_offset + max(f.y_values) for f in stave_fit_list))
            + BOX_PAD
        )
        # Clamp to canvas.
        left_x = max(0, left_x)
        right_x = min(canvas_w - 1, right_x)
        top_y = max(0, top_y)
        bot_y = min(canvas_h - 1, bot_y)
        cv2.rectangle(
            canvas_uint8, (left_x, top_y), (right_x, bot_y), (220, 30, 30), thickness=3
        )
        cv2.putText(
            canvas_uint8,
            f"S{stave_id}",
            (left_x + 6, top_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(220, 30, 30),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

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

    # --- Right panel: gap distribution and thresholds ---
    if result.gap_distribution:
        # Color each bar by stave; anomalous staves get warning colors so the
        # rhythm irregularities are immediately visible in the bar chart.
        n_gaps = len(result.gap_distribution)
        bar_colors = ["steelblue"] * n_gaps

        if result.rhythm_anomalies:
            for sid, ra in result.rhythm_anomalies.items():
                if ra["status"] == "normal":
                    continue
                c = stave_colors.get(sid, None)
                if ra["status"] == "under_populated":
                    fill = "tomato"
                else:  # over_populated
                    fill = "mediumpurple"
                gi0, gi1 = ra["gap_index_range"]
                for k in range(gi0, min(gi1 + 1, n_gaps)):
                    bar_colors[k] = fill

        axes[1].bar(range(n_gaps), result.gap_distribution, color=bar_colors, zorder=2)

        # Shade the noise zone below min_threshold (gaps too small to be real
        # staffline spacing — excluded from periodicity calculations).
        if result.min_threshold_px > 0:
            axes[1].axhspan(
                0,
                result.min_threshold_px,
                alpha=0.12,
                color="gray",
                label="_nolegend_",
                zorder=1,
            )
            axes[1].axhline(
                y=result.min_threshold_px,
                color="gray",
                linestyle=":",
                linewidth=1.8,
                label=f"Min Gap / Noise Floor ({result.min_threshold_px:.1f} px)",
                zorder=3,
            )

        axes[1].axhline(
            y=result.cut_threshold_px,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Cut Threshold ({result.cut_threshold_px:.1f} px)",
            zorder=4,
        )

        if result.interpolation_max_gap_px is not None:
            # Shade the interpolation window between cut and max thresholds.
            axes[1].axhspan(
                result.cut_threshold_px,
                result.interpolation_max_gap_px,
                alpha=0.10,
                color="orange",
                label="_nolegend_",
                zorder=1,
            )
            axes[1].axhline(
                y=result.interpolation_max_gap_px,
                color="darkorange",
                linestyle="--",
                linewidth=2,
                label=f"Max Interp Gap / Mean Inter-Stave ({result.interpolation_max_gap_px:.1f} px)",
                zorder=4,
            )

        # --- Annotate anomalous staves ---
        if result.rhythm_anomalies and result.gap_distribution:
            y_max = max(result.gap_distribution)
            for sid, ra in result.rhythm_anomalies.items():
                if ra["status"] == "normal":
                    continue
                gi0, gi1 = ra["gap_index_range"]
                mid_x = (gi0 + gi1) / 2.0
                label_color = (
                    "tomato" if ra["status"] == "under_populated" else "mediumpurple"
                )
                symbol = "▼" if ra["status"] == "under_populated" else "▲"
                axes[1].text(
                    mid_x,
                    y_max * 1.04,
                    f"S{sid}{symbol}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=label_color,
                    fontweight="bold",
                    clip_on=False,
                )
                # Small note showing observed vs expected
                axes[1].text(
                    mid_x,
                    y_max * 1.01,
                    f"{ra['lines_observed']}/{ra['lines_expected']}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=label_color,
                    clip_on=False,
                )

        axes[1].set_title(
            "Gap Distribution (Consecutive Fits)\n"
            "Gray zone = noise floor  |  Orange zone = missing-line trigger  |  "
            "Red/purple bars = rhythm anomaly"
        )
        axes[1].set_xlabel("Gap Index")
        axes[1].set_ylabel("Gap (pixels)")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No gaps (0 or 1 fit)", ha="center", va="center")
        axes[1].set_title("Gap Distribution")

    # --- Add legend for staves ---
    legend_elements = [
        mpatches.Patch(color=stave_colors[sid], label=f"Stave {sid}")
        for sid in assigned_ids
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.02)
    )

    # --- Save full-resolution stave overlay ---
    hq_path = save_path.with_name(save_path.stem + "_hq.png")
    cv2.imwrite(str(hq_path), cv2.cvtColor(canvas_uint8, cv2.COLOR_RGB2BGR))

    # --- Save figure ---
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
