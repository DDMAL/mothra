"""Sanity check for fallback_redetect on synthetic FitResult sequences.

No live model dependency -- identify_probe_regions and
validate_and_select_candidates are pure logic. Upstream StaveAssignment/
rhythm_anomalies/gap_distribution fixtures are produced by the real
group_staves() rather than hand-rolled, so tests exercise self-consistent
input exactly as the real pipeline would produce it.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure the scripts directory is on the path regardless of how the test is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fit_centerline import FitResult
from group_staves import group_staves
from fallback_redetect import (
    FallbackCandidate,
    ProbeRegion,
    identify_probe_regions,
    is_plausible_width,
    validate_and_select_candidates,
)


def make_fit_at_y(y_value: float, x_start: int = 0, x_end: int = 999) -> FitResult:
    """A simple straight-line FitResult at constant y across [x_start, x_end]."""
    return FitResult(
        x_start=x_start,
        x_end=x_end,
        y_values=[y_value] * (x_end - x_start + 1),
        coefficients=[0.0, 0.0, y_value],
        residual_mean=0.0,
        residual_max=0.0,
        n_pixels_used=x_end - x_start + 1,
    )


def probe_regions_for(fits: list[FitResult], scale_unit: float, page_width: int = 4000) -> list[ProbeRegion]:
    """Run the real group_staves() then identify_probe_regions on its output --
    keeps every test's upstream fixtures (assignments/rhythm_anomalies/
    gap_distribution) self-consistent with what the real pipeline would
    produce, rather than hand-rolled and possibly-inconsistent."""
    result = group_staves(fits, scale_unit=scale_unit)
    return identify_probe_regions(
        fits=fits,
        assignments=result.assignments,
        rhythm_anomalies=result.rhythm_anomalies,
        gap_distribution=result.gap_distribution,
        scale_unit=scale_unit,
        min_threshold=result.min_threshold_px,
        cut_threshold=result.cut_threshold_px,
        mode_n=result.mode_lines_per_stave,
        page_width=page_width,
    )


def test_under_populated_middle_stave_gets_one_probe_within_territory():
    # Stave A (4 lines) / Stave B (2 of an expected 4) / Stave C (4 lines).
    # Only B is under-populated (intra_count=1 < mode_n//2=2); expect exactly
    # one probe, and its y-range must sit strictly between A's and C's lines
    # (i.e. within B's own territory, never overlapping a neighbor stave).
    fits = [make_fit_at_y(y) for y in [100, 130, 160, 190, 310, 340, 460, 490, 520, 550]]
    regions = probe_regions_for(fits, scale_unit=10.0)
    print(f"middle-stave probe: {regions}")
    assert len(regions) == 1
    region = regions[0]
    assert region.lines_observed == 2
    assert region.mode_n == 4
    assert region.max_new_lines == 2
    # Territory: strictly between stave A's bottom (190) and stave C's top (460).
    assert 190 < region.y_start <= 310
    assert 340 <= region.y_end < 460


def test_single_line_stave_still_gets_a_probe():
    # Regression test for interpolate_staves.py's own edge-extrapolation
    # trigger, which requires len(fit_pairs) >= mode_n // 2 and therefore
    # gets nothing for a stave with just 1 line when mode_n=4 (1 < 4//2=2).
    # identify_probe_regions must not inherit that gate.
    fits = [make_fit_at_y(y) for y in [100, 130, 160, 190, 400]]
    regions = probe_regions_for(fits, scale_unit=10.0)
    print(f"single-line stave probe: {regions}")
    assert len(regions) == 1
    assert regions[0].lines_observed == 1
    assert regions[0].max_new_lines == 3


def test_probe_x_range_anchors_to_narrow_line_not_full_page_width():
    # Stave A is full-width (x=[0,3000]); the sparse stave's one known line
    # is narrow and off-center (x=[1800,2200]) on a much wider 4000px page.
    # The probe's x-range must anchor near the narrow line (plus padding),
    # not span the full page width.
    fits = [make_fit_at_y(y, x_start=0, x_end=3000) for y in [100, 130, 160, 190]]
    fits.append(make_fit_at_y(400, x_start=1800, x_end=2200))
    regions = probe_regions_for(fits, scale_unit=10.0, page_width=4000)
    print(f"narrow-anchor probe: {regions}")
    assert len(regions) == 1
    region = regions[0]
    # Padding is max(100, 3*scale_unit) = 100px at scale_unit=10.
    assert region.x_start == 1700.0
    assert region.x_end == 2300.0
    assert region.x_end - region.x_start < 3000  # nowhere near full page width


def test_no_probes_when_every_stave_hits_mode():
    fits = [make_fit_at_y(y) for y in [100, 130, 160, 190, 400, 430, 460, 490]]
    regions = probe_regions_for(fits, scale_unit=10.0)
    print(f"all-normal page probes: {regions}")
    assert regions == []


def test_over_populated_stave_gets_no_probe_only_under_populated_does():
    # Stave 0: 7 lines when the page mode is 4 (over-populated,
    # intra_count=6 > mode_n+1=5) -- a different failure mode (likely two
    # staves merged), already _reconcile_duplicate_fits' concern. Staves 1
    # and 2: normal 4-line staves (two of them, so the mode is unambiguous).
    # Stave 3: 1 line (under-populated). Only stave 3 should get a probe.
    stave_0_ys = [100 + 28 * i for i in range(7)]
    fits = [make_fit_at_y(y) for y in stave_0_ys]
    fits += [make_fit_at_y(y) for y in [700, 730, 760, 790]]
    fits += [make_fit_at_y(y) for y in [1100, 1130, 1160, 1190]]
    fits.append(make_fit_at_y(1500))
    regions = probe_regions_for(fits, scale_unit=10.0)
    print(f"mixed over/under populated probes: {[r.stave_id for r in regions]}, "
          f"lines_observed={[r.lines_observed for r in regions]}")
    assert len(regions) == 1
    assert regions[0].lines_observed == 1


def test_narrow_fragment_candidate_rejected_by_relative_width_filter():
    region = ProbeRegion(
        stave_id=0, y_start=0.0, y_end=1000.0, x_start=0.0, x_end=1000.0,
        h_est=30.0, lines_observed=1, mode_n=4, max_new_lines=3,
    )
    sibling_widths = [900.0, 950.0, 920.0]  # median ~920
    good_candidate = FallbackCandidate(
        fit=make_fit_at_y(500.0, x_start=50, x_end=900),  # width 850, >= 0.4*920
        yolo_confidence=0.4, stage1_score=0.6,
    )
    narrow_candidate = FallbackCandidate(
        fit=make_fit_at_y(600.0, x_start=500, x_end=580),  # width 80, << 0.4*920
        yolo_confidence=0.5, stage1_score=0.6,
    )
    accepted, cap_exceeded = validate_and_select_candidates(
        region, [good_candidate, narrow_candidate], sibling_widths
    )
    print(f"width-filter: accepted={[c.fit.y_values[0] for c in accepted]}, cap_exceeded={cap_exceeded}")
    assert len(accepted) == 1
    assert accepted[0] is good_candidate
    assert cap_exceeded is False
    assert not is_plausible_width(80.0, sibling_widths)
    assert is_plausible_width(850.0, sibling_widths)


def test_cap_exceeded_flag_when_more_candidates_than_allowed():
    region = ProbeRegion(
        stave_id=0, y_start=0.0, y_end=1000.0, x_start=0.0, x_end=1000.0,
        h_est=30.0, lines_observed=2, mode_n=4, max_new_lines=1,
    )
    candidates = [
        FallbackCandidate(fit=make_fit_at_y(100.0, x_start=0, x_end=900), yolo_confidence=0.3, stage1_score=0.5),
        FallbackCandidate(fit=make_fit_at_y(200.0, x_start=0, x_end=900), yolo_confidence=0.6, stage1_score=0.5),
    ]
    accepted, cap_exceeded = validate_and_select_candidates(region, candidates, sibling_widths=[900.0])
    print(f"cap test: accepted_ys={[c.fit.y_values[0] for c in accepted]}, cap_exceeded={cap_exceeded}")
    assert len(accepted) == 1
    assert accepted[0].yolo_confidence == 0.6  # higher-confidence candidate wins
    assert cap_exceeded is True


if __name__ == "__main__":
    test_under_populated_middle_stave_gets_one_probe_within_territory()
    test_single_line_stave_still_gets_a_probe()
    test_probe_x_range_anchors_to_narrow_line_not_full_page_width()
    test_no_probes_when_every_stave_hits_mode()
    test_over_populated_stave_gets_no_probe_only_under_populated_does()
    test_narrow_fragment_candidate_rejected_by_relative_width_filter()
    test_cap_exceeded_flag_when_more_candidates_than_allowed()
    print("\nAll fallback_redetect sanity checks passed.")
