# -*- coding: utf-8 -*-
"""
Targeted missed-detection fallback for Stage 2 of the staff detection
pipeline.

After grouping (group_staves.py) has assigned detected FitResults to staves
and flagged rhythm anomalies, some staves may be genuinely under-detected --
the YOLO stafflline detector simply never emitted a box for one or more real
lines in that region (confirmed, on real pages, by direct inspection of the
raw YOLO .txt output: no box exists at all where a line is visibly missing).
This is a different failure mode from the duplicate/split-fragment problem
group_staves.py's _reconcile_duplicate_fits already handles -- there is
nothing to reconcile when no detection was made in the first place.

This module is pure logic: it identifies which stave regions are worth a
second, more targeted look (identify_probe_regions), and validates/selects
among whatever candidates a caller's own re-detection call turns up
(validate_and_select_candidates). It has no `ultralytics` import and no
knowledge of how re-detection is actually performed -- that orchestration
(loading the model, cropping the image, calling model.predict(), running the
recovered boxes through the same Stage-1 filter_components/fit_centerline
path as every other box) lives in run_page.py, which already holds the image
data and model weights this module deliberately doesn't need.

Public API
----------
``identify_probe_regions(fits, assignments, rhythm_anomalies,
                          gap_distribution, scale_unit, min_threshold,
                          cut_threshold, mode_n, page_width)``
    Returns ``list[ProbeRegion]``, one per stave worth re-probing.

``validate_and_select_candidates(region, candidates, sibling_widths)``
    Returns ``(accepted: list[FallbackCandidate], cap_exceeded: bool)``.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fit_centerline import FitResult
from group_staves import StaveAssignment, page_absolute_x_range, y_at_fit_center
from interpolate_staves import compute_stave_territories

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Padding added each side of a probe's x-range, beyond the union of the
# stave's own existing detections' x-ranges. Expressed as a multiple of
# scale_unit, not a fraction of page width: directly measured on a real page
# (Fribourg reference page, see plan), even clean/unflagged stave lines span
# only ~15-33% of the full scanned page width -- the writing column is much
# narrower than the page -- so anchoring padding to the stave's own detected
# ink (rather than page width) avoids pulling in an unrelated column/margin.
PROBE_X_PADDING_MULTIPLIER = 3.0
PROBE_X_PADDING_ABS_FLOOR_PX = 100.0

# A recovered candidate's x-span must be at least this fraction of the
# median x-span of its stave's own sibling lines (relative, not an absolute
# page-width fraction -- see PROBE_X_PADDING_MULTIPLIER's comment for why an
# absolute fraction would be wrong here) to be accepted as plausible rather
# than a stray fragment.
MIN_RELATIVE_WIDTH_RATIO = 0.4

# Minimum Stage-1 component-filter score (run_page.py's own _top_score_of)
# a recovered candidate must clear, in addition to the YOLO confidence
# already enforced by the fallback detection call itself (--fallback-conf).
# A second, independent signal: a low-confidence YOLO box that also scores
# poorly in Stage-1 (no clean, well-centered connected component found) is
# much more likely to be genuine noise than one that clears both.
STAGE1_SCORE_FLOOR = 0.3

# A candidate within this fraction of h_est of an already-detected line's
# own center is almost certainly re-detecting that same line, not a missing
# one -- rejected before ranking/capping so it can never consume a
# max_new_lines slot that a genuinely new line needed. group_staves.py's own
# _reconcile_duplicate_fits would eventually collapse such a duplicate too,
# but only after it has already displaced a real candidate at the cap; this
# rejects it earlier, before that displacement can happen.
DUPLICATE_EXISTING_LINE_TOLERANCE_RATIO = 0.5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProbeRegion:
    """One targeted region to re-run detection over, to recover a real
    staffline the primary detection pass appears to have missed entirely.

    One ProbeRegion per anomalous stave, not one per missing line: the
    territory bounds already provide the safety margin against bleeding
    into a neighboring stave, so a single crop is enough for the detector to
    find whatever's really there.

    Attributes:
        stave_id: Which (already-established) stave this probe belongs to.
        y_start: Top of the probe crop, page-absolute px (this stave's own
            territory upper bound -- see interpolate_staves.compute_stave_
            territories -- never extends into a neighboring stave's space).
        y_end: Bottom of the probe crop, page-absolute px (territory lower
            bound).
        x_start: Left edge of the probe crop, page-absolute px.
        x_end: Right edge of the probe crop, page-absolute px.
        h_est: This stave's estimated intra-line spacing (px). Diagnostic
            metadata (surfaced in fallback_redetect_report.txt), not itself
            used to shape the crop.
        lines_observed: How many lines are already detected in this stave.
        mode_n: Page-level expected lines per stave.
        max_new_lines: Cap on how many fallback-redetected lines this stave
            may accept (mode_n - lines_observed).
        existing_centers: Page-absolute y-centers of this stave's own
            already-detected lines, carried through so
            validate_and_select_candidates can reject a candidate that is
            really just a re-detection of one of these, before it ever
            competes for a max_new_lines slot.
    """

    stave_id: int
    y_start: float
    y_end: float
    x_start: float
    x_end: float
    h_est: float
    lines_observed: int
    mode_n: int
    max_new_lines: int
    existing_centers: list[float] = field(default_factory=list)


@dataclass
class FallbackCandidate:
    """One candidate fit recovered from a probe region, awaiting validation.

    Attributes:
        fit: The candidate's FitResult, produced by running its detected box
            through the exact same crop_with_padding -> filter_components ->
            fit_centerline sequence as every other box on the page (not a
            parallel path).
        yolo_confidence: The fallback detector's own confidence for this box.
        stage1_score: The Stage-1 component-filter's own top score for this
            box (run_page.py's _top_score_of) -- an independent signal from
            yolo_confidence, since it reflects a clean/well-centered
            connected component was actually found, not just that YOLO
            thought a line-shaped region was there.
    """

    fit: FitResult
    yolo_confidence: float
    stage1_score: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_h_est_global(
    gap_distribution: list[float],
    min_threshold: float,
    cut_threshold: float,
    scale_unit: float,
) -> float:
    """Page-level intra-stave spacing estimate: median of the page's own
    observed gaps in [min_threshold, cut_threshold) -- the same intra- vs.
    inter-stave gap classification group_staves.py already uses, just
    aggregated with a median instead of split per-stave. Deliberately not
    experiments/periodicity's autocorrelation approach (its own NOTES.md
    documents it as unreliable per-box on real manuscript pages) and not
    scale_unit directly (fit_centerline.py's own comments document a real
    page where scale_unit diverged sharply from true line spacing) -- an
    observed, page-derived gap estimate beats both. Falls back to
    scale_unit only if the page has no gaps in that band at all
    (pathological, e.g. a single-stave page).
    """
    intra = [g for g in gap_distribution if min_threshold <= g < cut_threshold]
    if intra:
        return float(np.median(intra))
    return float(scale_unit)


def _stave_h_est(
    centers: list[float],
    min_threshold: float,
    cut_threshold: float,
    h_est_global: float,
) -> float:
    """This stave's own intra-gap median if it has >= 2 detected lines
    (matching interpolate_staves.py's own h_est calculation); otherwise the
    page-level estimate.
    """
    if len(centers) < 2:
        return h_est_global
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    intra = [g for g in gaps if min_threshold <= g < cut_threshold]
    if intra:
        return float(np.median(intra))
    return h_est_global


def _fit_within_territory(fit: FitResult, region: ProbeRegion) -> bool:
    y = y_at_fit_center(fit)
    if y is None:
        return False
    return region.y_start <= y <= region.y_end


def _fit_width(fit: FitResult) -> float:
    rng = page_absolute_x_range(fit)
    if rng is None:
        return 0.0
    return rng[1] - rng[0]


def _duplicates_existing_line(fit: FitResult, region: ProbeRegion) -> bool:
    """True if fit's center falls within DUPLICATE_EXISTING_LINE_TOLERANCE_
    RATIO * h_est of one of the stave's own already-detected lines -- almost
    certainly a re-detection of that line, not a missing one."""
    y = y_at_fit_center(fit)
    if y is None or not region.existing_centers:
        return False
    tolerance = DUPLICATE_EXISTING_LINE_TOLERANCE_RATIO * region.h_est
    return any(abs(y - c) < tolerance for c in region.existing_centers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_probe_regions(
    fits: list[FitResult],
    assignments: list[StaveAssignment],
    rhythm_anomalies: dict[int, dict],
    gap_distribution: list[float],
    scale_unit: float,
    min_threshold: float,
    cut_threshold: float,
    mode_n: int,
    page_width: int,
) -> list["ProbeRegion"]:
    """Identify stave regions worth re-probing for missed detections.

    Only staves _check_stave_rhythm classified 'under_populated' are
    considered. 'over_populated' staves are explicitly excluded: that's a
    different failure mode (likely two staves merged into one), already
    group_staves._reconcile_duplicate_fits' concern -- adding more lines to
    an already-over-populated stave would make it worse, not better.

    A stave with zero detected lines at all is also excluded: there's
    nothing to anchor a territory or an h_est estimate on. (In practice this
    should be rare -- a stave has to have been assigned at least one line by
    _assign_staves to exist as a stave_id at all.)
    """
    stave_fits: dict[int, list[tuple[int, FitResult]]] = {}
    for asg in assignments:
        if asg.stave_id is None:
            continue
        stave_fits.setdefault(asg.stave_id, []).append((asg.fit_index, fits[asg.fit_index]))

    if not stave_fits:
        return []

    upper_bound, lower_bound = compute_stave_territories(stave_fits)
    h_est_global = _compute_h_est_global(gap_distribution, min_threshold, cut_threshold, scale_unit)

    regions: list[ProbeRegion] = []
    for stave_id, fit_pairs in stave_fits.items():
        status = rhythm_anomalies.get(stave_id, {}).get("status", "normal")
        if status != "under_populated":
            continue

        centers = sorted(c for _, f in fit_pairs if (c := y_at_fit_center(f)) is not None)
        lines_observed = len(centers)
        if lines_observed == 0:
            continue

        max_new_lines = mode_n - lines_observed
        if max_new_lines <= 0:
            continue

        x_ranges = [r for _, f in fit_pairs if (r := page_absolute_x_range(f)) is not None]
        if not x_ranges:
            continue

        h_est = _stave_h_est(centers, min_threshold, cut_threshold, h_est_global)
        padding = max(PROBE_X_PADDING_ABS_FLOOR_PX, PROBE_X_PADDING_MULTIPLIER * scale_unit)
        x_union_start = min(r[0] for r in x_ranges)
        x_union_end = max(r[1] for r in x_ranges)

        regions.append(
            ProbeRegion(
                stave_id=stave_id,
                y_start=upper_bound.get(stave_id, min(centers)),
                y_end=lower_bound.get(stave_id, max(centers)),
                x_start=max(0.0, x_union_start - padding),
                x_end=min(float(page_width), x_union_end + padding),
                h_est=h_est,
                lines_observed=lines_observed,
                mode_n=mode_n,
                max_new_lines=max_new_lines,
                existing_centers=centers,
            )
        )

    return regions


def is_plausible_width(candidate_width: float, sibling_widths: list[float]) -> bool:
    """A recovered candidate's x-span must be at least MIN_RELATIVE_WIDTH_
    RATIO times the median x-span of comparison lines (typically its
    stave's own siblings, or the page median among normal staves if the
    caller has no siblings to offer) -- not an absolute page-width fraction.
    Directly measured on a real page, even clean/unflagged stave lines span
    only ~15-33% of the full scanned page width, so a filter calibrated on
    "a normal line looks like ~90%+ of page width" would wrongly reject good
    detections.

    Returns True (does not block) when there is nothing to compare against.
    """
    if not sibling_widths:
        return True
    median_width = float(np.median(sibling_widths))
    if median_width <= 0:
        return True
    return candidate_width >= MIN_RELATIVE_WIDTH_RATIO * median_width


def validate_and_select_candidates(
    region: ProbeRegion,
    candidates: list[FallbackCandidate],
    sibling_widths: list[float],
) -> tuple[list[FallbackCandidate], bool]:
    """Filter a probe region's candidates to plausible ones, then cap at
    region.max_new_lines.

    Plausibility requires all of:
      - falls within the probe's own territory bounds (region.y_start/y_end)
        -- redundant with the crop itself in practice, but cheap to also
        check explicitly here since a caller could in principle hand in
        candidates from a looser crop;
      - not a re-detection of one of the stave's own already-detected lines
        (_duplicates_existing_line) -- rejected before ranking so a
        duplicate can never consume a max_new_lines slot and displace a
        genuinely missing line; group_staves._reconcile_duplicate_fits
        would eventually collapse such a duplicate too, but only after
        that displacement could already have happened;
      - relative-width plausible (is_plausible_width) against sibling_widths;
      - Stage-1 score clears STAGE1_SCORE_FLOOR.

    Accepted candidates are ranked by YOLO confidence (ties broken by
    Stage-1 score) and capped at region.max_new_lines.

    Returns:
        (accepted, cap_exceeded). cap_exceeded is True when more candidates
        passed the plausibility filters than max_new_lines allows -- the
        caller should flag this (e.g.
        f"fallback_found_more_than_expected:{region.stave_id}") rather than
        silently drop the excess.
    """
    plausible = [
        c
        for c in candidates
        if _fit_within_territory(c.fit, region)
        and not _duplicates_existing_line(c.fit, region)
        and is_plausible_width(_fit_width(c.fit), sibling_widths)
        and c.stage1_score >= STAGE1_SCORE_FLOOR
    ]
    plausible.sort(key=lambda c: (-c.yolo_confidence, -c.stage1_score))
    cap_exceeded = len(plausible) > region.max_new_lines
    return plausible[: region.max_new_lines], cap_exceeded
