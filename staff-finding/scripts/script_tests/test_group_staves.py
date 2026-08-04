"""Sanity check for group_staves on synthetic FitResult sequences."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure the scripts directory is on the path regardless of how the test is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fit_centerline import FitResult
from group_staves import (
    group_staves,
    StaveAssignment,
    InterpolatedLine,
    _compute_gap_distribution,
    _determine_cut_threshold,
    _assign_staves,
    _reconcile_duplicate_fits,
)


def make_fit_at_y(
    y_value: float,
    x_start: int = 0,
    x_end: int = 999,
    n_pixels_used: Optional[int] = None,
    residual_mean: float = 0.0,
) -> FitResult:
    """A simple straight-line FitResult at constant y across [x_start, x_end]."""
    return FitResult(
        x_start=x_start,
        x_end=x_end,
        y_values=[y_value] * (x_end - x_start + 1),
        coefficients=[0.0, 0.0, y_value],
        residual_mean=residual_mean,
        residual_max=residual_mean,
        n_pixels_used=(
            n_pixels_used if n_pixels_used is not None else x_end - x_start + 1
        ),
    )


def make_sloped_fit(
    coefficients: list[float],
    x_start: int,
    x_end: int,
    x_page_offset: float = 0.0,
    y_page_offset: float = 0.0,
    n_pixels_used: Optional[int] = None,
    residual_mean: float = 0.0,
) -> FitResult:
    """A FitResult whose y_values are sampled from `coefficients` (crop-local,
    matching a real fit_centerline.py output) -- for tests needing genuine
    slope/curvature rather than make_fit_at_y's flat line."""
    xs_local = np.arange(x_start, x_end + 1)
    y_values = [round(float(y), 1) for y in np.polyval(coefficients, xs_local)]
    return FitResult(
        x_start=x_start,
        x_end=x_end,
        y_values=y_values,
        coefficients=list(coefficients),
        residual_mean=residual_mean,
        residual_max=residual_mean,
        n_pixels_used=(
            n_pixels_used if n_pixels_used is not None else x_end - x_start + 1
        ),
        x_page_offset=x_page_offset,
        y_page_offset=y_page_offset,
    )


def test_compute_gap_distribution():
    gaps = _compute_gap_distribution([100.0, 130.0, 160.0, 190.0])
    print(f"gaps for 100,130,160,190: {gaps}")
    assert gaps == [30.0, 30.0, 30.0]


def test_cut_threshold_floor():
    # If gaps are tiny, threshold should be floored at scale_unit.
    thresh = _determine_cut_threshold([2.0, 2.0, 2.0], scale_unit=10.0)
    print(f"cut threshold with tiny gaps and h=10: {thresh}")
    assert thresh == 10.0


def test_cut_threshold_bimodal():
    # Intra-stave gaps ~30; inter-stave gap ~100. Threshold should fall
    # between them.
    gaps = [30.0, 30.0, 30.0, 100.0, 30.0, 30.0, 30.0, 100.0, 30.0, 30.0, 30.0]
    thresh = _determine_cut_threshold(gaps, scale_unit=5.0)
    print(f"cut threshold for bimodal gaps: {thresh}")
    assert 30.0 < thresh < 100.0


def test_assign_staves_split():
    # Three lines at 100,130,160, then gap of 100, then three more at
    # 260,290,320. Threshold 50 should split into two staves of three.
    triples = _assign_staves(
        sorted_fit_indices=[0, 1, 2, 3, 4, 5],
        y_positions=[100.0, 130.0, 160.0, 260.0, 290.0, 320.0],
        cut_threshold=50.0,
    )
    print(f"assignments: {triples}")
    stave_ids = [t[1] for t in triples]
    assert stave_ids == [0, 0, 0, 1, 1, 1]
    within = [t[2] for t in triples]
    assert within == [0, 1, 2, 0, 1, 2]


def test_full_grouping_two_staves_of_four():
    # Two staves of four lines each. Intra-stave 30, inter-stave 120.
    ys = [100, 130, 160, 190, 310, 340, 370, 400]  # stave 0  # stave 1
    fits = [make_fit_at_y(y) for y in ys]
    result = group_staves(fits, scale_unit=10.0)
    stave_ids = [a.stave_id for a in result.assignments]
    print(
        f"two staves of four: stave_ids={stave_ids}, "
        f"mode={result.mode_lines_per_stave}, "
        f"distribution={result.line_count_distribution}, "
        f"cut={result.cut_threshold_px:.1f}, flags={result.flags}"
    )
    assert stave_ids == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result.mode_lines_per_stave == 4
    assert result.line_count_distribution == {4: 2}


def test_grouping_with_missing_line_no_interpolation():
    # Two staves; the second one is missing one line in the middle.
    # NOTE: a missing-middle-line creates a gap roughly twice the intra-stave
    # gap, which exceeds the 1.5x threshold and causes the stave to split.
    # The mode-based flagging surfaces this as 'staves_with_unexpected_count'.
    # See known-limitations note in group_staves.py for the followup question
    # of whether the threshold heuristic should be more lenient.
    ys = [
        100,
        130,
        160,
        190,  # stave 0: 4 lines
        310,
        340,
        400,
    ]  # stave 1: 3 lines (one missing in middle)
    fits = [make_fit_at_y(y) for y in ys]
    result = group_staves(fits, scale_unit=10.0)
    stave_ids = [a.stave_id for a in result.assignments]
    print(
        f"missing-line (no interp): stave_ids={stave_ids}, "
        f"mode={result.mode_lines_per_stave}, "
        f"distribution={result.line_count_distribution}, "
        f"flags={result.flags}"
    )
    # Stave 0 (4 lines) is intact; the missing-line gap in the second stave
    # exceeds threshold and causes a split. The algorithm flags this via the
    # unexpected-count signal.
    assert stave_ids[:4] == [0, 0, 0, 0], "stave 0 should be intact"
    # Flag now includes the offending stave IDs: "staves_with_unexpected_count:N"
    assert any("staves_with_unexpected_count" in f for f in result.flags)
    assert result.interpolated_lines == []
    # Mode should still find 4 as the canonical count (stave 0 contributes 4).
    assert result.mode_lines_per_stave == 4


def test_grouping_with_missing_line_and_interpolation():
    # Stave 0 has 4 lines (mode), stave 1 has only 3 — one line is missing.
    # With interpolate_missing=True the pipeline should synthesise the gap.
    # The high scale_unit keeps the 50-px intra-stave gap below the cut
    # threshold so both staves stay intact.
    ys_stave_0 = [100, 130, 160, 190]
    ys_stave_1 = [400, 430, 480]  # widest internal gap is 50
    ys = ys_stave_0 + ys_stave_1
    fits = [make_fit_at_y(y) for y in ys]
    result = group_staves(fits, scale_unit=55.0, interpolate_missing=True)
    print(f"missing-line (interp ON): "
          f"stave_ids={[a.stave_id for a in result.assignments]}, "
          f"mode={result.mode_lines_per_stave}, "
          f"n_interpolated={len(result.interpolated_lines)}")
    stave_ids = [a.stave_id for a in result.assignments]
    assert stave_ids[:4] == [0, 0, 0, 0]
    assert stave_ids[4:] == [1, 1, 1]
    assert result.mode_lines_per_stave == 4
    # Stave 1 is short by one line; interpolation should produce exactly one
    # synthetic line for it.
    assert len(result.interpolated_lines) == 1
    assert result.interpolated_lines[0].stave_id == 1


def test_failed_fits_excluded_from_grouping():
    # Mix of usable and failed fits.
    ys = [100, 130, 160, 190]
    fits = [make_fit_at_y(y) for y in ys]
    # Insert a failed fit (no y_values).
    fits.insert(2, FitResult(flags=["fit_did_not_converge"]))
    result = group_staves(fits, scale_unit=10.0)
    print(
        f"with one failed fit: "
        f"stave_ids={[a.stave_id for a in result.assignments]}, "
        f"flags on failed={result.assignments[2].flags}"
    )
    # Failed fit has no stave_id.
    assert result.assignments[2].stave_id is None
    assert "no_y_position_available" in result.assignments[2].flags
    # The four usable fits still grouped as one stave.
    usable_assignments = [a for a in result.assignments if a.stave_id is not None]
    assert all(a.stave_id == 0 for a in usable_assignments)


def test_no_fits_available():
    # All fits are failed (no y_values); distinct from an empty list.
    # An empty list → 'no_fits_available'; non-empty all-failed → 'no_fits_with_y_positions'.
    fits = [
        FitResult(flags=["fit_did_not_converge"]),
        FitResult(flags=["no_fit_attempted"]),
    ]
    result = group_staves(fits, scale_unit=10.0)
    print(f"no usable fits: flags={result.flags}")
    assert "no_fits_with_y_positions" in result.flags


def test_overlapping_duplicate_pair_collapses_to_one():
    # Modeled on the real duplicate pair line0044/line0060 from the
    # layer_1_3801 x 5013 reference page (28July e2e fixture): two YOLO boxes
    # for the same physical line, heavily overlapping in x, y-centers 0.8px
    # apart. fit0 has more supporting pixels, so it should win the primary
    # tiebreak.
    fits = [
        make_fit_at_y(
            552.3, x_start=1322, x_end=3427, n_pixels_used=34618, residual_mean=24.515
        ),
        make_fit_at_y(
            553.1, x_start=1305, x_end=3427, n_pixels_used=33684, residual_mean=24.135
        ),
    ]
    fit_positions = [(0, 552.3), (1, 553.1)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        fits, fit_positions, scale_unit=71.0
    )
    print(
        f"overlapping duplicate pair: kept={kept}, "
        f"absorbed={absorbed_flags}, groups={duplicate_groups}"
    )
    assert kept == [(0, 552.3)]
    assert absorbed_flags == {1: ["duplicate_of:0"]}
    assert duplicate_groups == {0: [1]}


def test_disjoint_split_fragment_pair_collapses_to_one():
    # Modeled on the real left/right split-fragment pair (stave 1, y~834-840)
    # from the same reference page: one physical line interrupted by a
    # page-layout gap, detected as two disjoint-x boxes.
    fits = [
        make_fit_at_y(
            834.6, x_start=1288, x_end=2304, n_pixels_used=8376, residual_mean=2.350
        ),
        make_fit_at_y(
            840.7, x_start=2695, x_end=3420, n_pixels_used=6321, residual_mean=2.491
        ),
    ]
    fit_positions = [(0, 834.6), (1, 840.7)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        fits, fit_positions, scale_unit=71.0
    )
    print(
        f"disjoint split-fragment pair: kept={kept}, "
        f"absorbed={absorbed_flags}, groups={duplicate_groups}"
    )
    assert kept == [(0, 834.6)]
    assert absorbed_flags == {1: ["companion_of:0"]}
    assert duplicate_groups == {0: [1]}


def test_compound_four_box_cluster_collapses_to_one():
    # Modeled on the real 4-box group (stave 1, y~931-937): two duplicate
    # left-half boxes and two duplicate right-half boxes, all fragments of
    # one real line. fit0/fit1 tie on both n_pixels_used and residual_mean,
    # so the lowest fit_index (0) must win the primary tiebreak.
    fits = [
        make_fit_at_y(
            931.4, x_start=1325, x_end=2312, n_pixels_used=6996, residual_mean=1.912
        ),  # LEFT
        make_fit_at_y(
            931.4, x_start=1317, x_end=2290, n_pixels_used=6996, residual_mean=1.912
        ),  # LEFT dup, tied with fit0
        make_fit_at_y(
            937.2, x_start=2714, x_end=3446, n_pixels_used=4951, residual_mean=2.052
        ),  # RIGHT
        make_fit_at_y(
            937.2, x_start=2719, x_end=3417, n_pixels_used=4911, residual_mean=2.046
        ),  # RIGHT dup
    ]
    fit_positions = [(0, 931.4), (1, 931.4), (2, 937.2), (3, 937.2)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        fits, fit_positions, scale_unit=71.0
    )
    print(
        f"compound 4-box cluster: kept={kept}, "
        f"absorbed={absorbed_flags}, groups={duplicate_groups}"
    )
    assert kept == [(0, 931.4)]
    assert duplicate_groups == {0: [1, 2, 3]}
    assert absorbed_flags[1] == ["duplicate_of:0"]  # same-side (LEFT), overlapping x
    assert absorbed_flags[2] == ["companion_of:0"]  # disjoint x (RIGHT vs primary LEFT)
    assert absorbed_flags[3] == ["companion_of:0"]


def test_primary_tiebreak_by_residual_mean():
    # Equal n_pixels_used; the lower residual_mean should win.
    fits = [
        make_fit_at_y(500.0, x_start=100, x_end=900, n_pixels_used=5000, residual_mean=3.0),
        make_fit_at_y(501.0, x_start=100, x_end=900, n_pixels_used=5000, residual_mean=1.5),
    ]
    fit_positions = [(0, 500.0), (1, 501.0)]
    kept, absorbed_flags, _ = _reconcile_duplicate_fits(fits, fit_positions, scale_unit=71.0)
    print(f"tiebreak by residual_mean: kept={kept}, absorbed={absorbed_flags}")
    assert kept == [(1, 501.0)]
    assert absorbed_flags == {0: ["duplicate_of:1"]}


def test_primary_tiebreak_by_fit_index():
    # Equal n_pixels_used AND equal residual_mean; the lower fit_index wins.
    fits = [
        make_fit_at_y(500.0, x_start=100, x_end=900, n_pixels_used=5000, residual_mean=2.0),
        make_fit_at_y(501.0, x_start=100, x_end=900, n_pixels_used=5000, residual_mean=2.0),
    ]
    fit_positions = [(0, 500.0), (1, 501.0)]
    kept, absorbed_flags, _ = _reconcile_duplicate_fits(fits, fit_positions, scale_unit=71.0)
    print(f"tiebreak by fit_index: kept={kept}, absorbed={absorbed_flags}")
    assert kept == [(0, 500.0)]
    assert absorbed_flags == {1: ["duplicate_of:0"]}


def test_close_but_genuinely_distinct_lines_are_not_merged():
    # Modeled on the real fit56/fit51 boundary case (stave 2, reference
    # page): 18.9px y-gap at scale_unit=71 (y_threshold=10.65px) -- both are
    # real, distinct lines and must remain separately indexed within the
    # stave, not collapsed by the new reconciliation step.
    fit56 = make_fit_at_y(
        1185.5, x_start=1373, x_end=3412, n_pixels_used=15700, residual_mean=5.286
    )
    fit51 = make_fit_at_y(
        1204.4, x_start=1282, x_end=3395, n_pixels_used=30061, residual_mean=23.815
    )
    result = group_staves([fit56, fit51], scale_unit=71.0)
    stave_ids = [a.stave_id for a in result.assignments]
    within = [a.within_stave_index for a in result.assignments]
    print(
        f"must-not-merge boundary case: stave_ids={stave_ids}, within={within}, "
        f"duplicate_reconciliation={result.duplicate_reconciliation}"
    )
    assert stave_ids == [0, 0]
    assert within == [0, 1]
    assert result.duplicate_reconciliation == {}


def test_reconciliation_invisible_to_stage6_counts():
    # One stave of 5 real lines, one of which also has a duplicate box. The
    # reported line_count_distribution/mode must reflect 5, not 6, and every
    # original fit (real or absorbed) must still get a StaveAssignment.
    ys = [100, 130, 160, 190, 220]
    fits = [make_fit_at_y(y, n_pixels_used=1000, residual_mean=1.0) for y in ys]
    # Duplicate of the y=160 line: heavily overlapping x (same default
    # x_start/x_end), near-identical y, fewer pixels so it loses the primary
    # tiebreak.
    fits.insert(3, make_fit_at_y(161.0, n_pixels_used=500, residual_mean=1.0))
    result = group_staves(fits, scale_unit=10.0)
    print(
        f"reconciliation invisible to counts: "
        f"distribution={result.line_count_distribution}, "
        f"mode={result.mode_lines_per_stave}, flags={result.flags}, "
        f"n_assignments={len(result.assignments)}, n_fits={len(fits)}"
    )
    assert len(result.assignments) == len(fits) == 6
    assert result.line_count_distribution == {5: 1}
    assert result.mode_lines_per_stave == 5
    assert any(f.startswith("reconciled_duplicate_fits:") for f in result.flags)
    absorbed = [a for a in result.assignments if a.within_stave_index is None]
    assert len(absorbed) == 1
    assert absorbed[0].flags[0].startswith("duplicate_of:")


def test_disjoint_pair_with_slope_confirmed_by_curve_agreement():
    # Two fragments of one real sloped line: y=800+0.008*(x-2500), page-wide.
    # Own-center samples (each fragment's own arbitrary midpoint) disagree by
    # 10.0px -- just inside the old y_threshold=10.65 gate -- purely because
    # they're sampled at different x's on a sloped line. Evaluated at the
    # shared gap midpoint (x=2500) instead, both curves agree exactly
    # (0.0px), so this should merge under the new, tighter
    # curve-agreement tolerance (7.1px), not just the old coarse gate.
    left = make_sloped_fit(
        [0.0, 0.008, 790.4], x_start=0, x_end=1000, x_page_offset=1300.0, n_pixels_used=8000
    )
    right = make_sloped_fit(
        [0.0, 0.008, 801.6], x_start=0, x_end=700, x_page_offset=2700.0, n_pixels_used=6000
    )
    fit_positions = [(0, 794.4), (1, 804.4)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        [left, right], fit_positions, scale_unit=71.0
    )
    print(f"sloped pair confirmed by curve agreement: kept={kept}, absorbed={absorbed_flags}")
    assert kept == [(0, 794.4)]
    assert absorbed_flags == {1: ["companion_of:0"]}
    assert duplicate_groups == {0: [1]}


def test_disjoint_pair_coarse_match_rejected_by_curve_disagreement():
    # The key new-capability test: two oppositely-sloped fragments engineered
    # so their own-center samples agree EXACTLY (0.0px apart -- the old,
    # pre-refinement check would have merged these unconditionally, since
    # 0.0 <= y_threshold trivially). But their curves evaluated at the shared
    # gap midpoint (x=2500) diverge to 37.5px apart -- comfortably beyond the
    # curve-agreement tolerance (21.3px at this scale_unit, with real margin
    # so this isn't a razor's-edge case -- see the real disjoint pair found
    # during validation with 16.72px of *genuine-match* disagreement at a
    # similar ~200px extrapolation distance, which is what this tolerance is
    # calibrated against) -- revealing these are NOT consistent with being
    # one continuous line. Must NOT merge.
    left = make_sloped_fit(
        [0.0, 0.25, 750.0], x_start=0, x_end=1000, x_page_offset=1300.0, n_pixels_used=8000
    )
    right = make_sloped_fit(
        [0.0, -0.25, 877.5], x_start=0, x_end=700, x_page_offset=2700.0, n_pixels_used=6000
    )
    fit_positions = [(0, 875.0), (1, 875.0)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        [left, right], fit_positions, scale_unit=71.0
    )
    print(
        f"sloped pair rejected by curve disagreement: kept={kept}, "
        f"absorbed={absorbed_flags}"
    )
    assert sorted(kept) == [(0, 875.0), (1, 875.0)]
    assert absorbed_flags == {}
    assert duplicate_groups == {}


def test_disjoint_pair_with_opposite_curvature_still_merges():
    # Regression test for a real pair found during validation (reference
    # page layer_1_3801 x 5013, fits 21/0): two independently-fit quadratics
    # of the SAME real line, with OPPOSITE-signed curvature (concave-down vs.
    # concave-up) -- consistent with each fragment only capturing its own
    # local piece of a wavy/arching real line (this codebase already
    # documents "S-shaped waves" as a real phenomenon on these
    # manuscripts). Own-centers agree to 5.8px (comfortably inside
    # y_threshold), but evaluated at the shared gap midpoint the curves
    # disagree by ~16.76px -- more than the ORIGINAL, tighter
    # curve-agreement tolerance (7.1px) rejected, which is exactly the
    # regression this test guards against. Must merge under the
    # recalibrated tolerance (21.3px).
    left = make_sloped_fit(
        [-2.664909018764964e-05, 0.034700448331980764, 44.680095941384224],
        x_start=27, x_end=990, x_page_offset=1299.0, y_page_offset=875.96, n_pixels_used=6996,
    )
    right = make_sloped_fit(
        [1.9448343513398842e-05, -0.008710123178918078, 22.86596280637449],
        x_start=0, x_end=700, x_page_offset=2714.0, y_page_offset=915.0, n_pixels_used=4951,
    )
    fit_positions = [(0, 931.4), (1, 937.2)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        [left, right], fit_positions, scale_unit=71.0
    )
    print(f"opposite-curvature real pair: kept={kept}, absorbed={absorbed_flags}")
    assert kept == [(0, 931.4)]
    assert absorbed_flags == {1: ["companion_of:0"]}
    assert duplicate_groups == {0: [1]}


def test_disjoint_pair_guard_falls_back_when_narrow_fragment_would_blow_up():
    # A 50px-wide fragment's own guard cap (25px = 0.5*50) is far below the
    # 200px extrapolation the shared gap midpoint would require. Naively
    # trusting the curve there would claim a wildly wrong value (this
    # fragment's mild quadratic evaluates to 1025 at that distance, vs. the
    # other fragment's flat 903 -- a 122px "disagreement" that would
    # wrongly block a real match). The guard must force a fallback to the
    # coarse own-center comparison instead (1.75px apart, well inside
    # y_threshold=10.65), so the pair still merges via the safe path.
    narrow_left = make_sloped_fit(
        [0.002, 0.0, 900.0], x_start=0, x_end=50, x_page_offset=2000.0, n_pixels_used=1200
    )
    right = make_sloped_fit(
        [0.0, 0.0, 903.0], x_start=0, x_end=750, x_page_offset=2450.0, n_pixels_used=9000
    )
    fit_positions = [(0, 901.25), (1, 903.0)]
    kept, absorbed_flags, duplicate_groups = _reconcile_duplicate_fits(
        [narrow_left, right], fit_positions, scale_unit=71.0
    )
    print(
        f"narrow fragment guard fallback: kept={kept}, absorbed={absorbed_flags}"
    )
    assert kept == [(1, 903.0)]
    assert absorbed_flags == {0: ["companion_of:1"]}
    assert duplicate_groups == {1: [0]}


def test_diagnostic_renders():
    ys = [100, 130, 160, 190, 310, 340, 370, 400]
    fits = [make_fit_at_y(y) for y in ys]
    save_path = Path("/tmp/group_staves_diag.png")
    if save_path.exists():
        save_path.unlink()
    _ = group_staves(fits, scale_unit=10.0, save_path=save_path, page_size=(1000, 500))
    assert save_path.exists()
    print(f"diagnostic saved: {save_path}, size={save_path.stat().st_size} bytes")


if __name__ == "__main__":
    test_compute_gap_distribution()
    test_cut_threshold_floor()
    test_cut_threshold_bimodal()
    test_assign_staves_split()
    test_full_grouping_two_staves_of_four()
    test_grouping_with_missing_line_no_interpolation()
    test_grouping_with_missing_line_and_interpolation()
    test_failed_fits_excluded_from_grouping()
    test_no_fits_available()
    test_overlapping_duplicate_pair_collapses_to_one()
    test_disjoint_split_fragment_pair_collapses_to_one()
    test_compound_four_box_cluster_collapses_to_one()
    test_primary_tiebreak_by_residual_mean()
    test_primary_tiebreak_by_fit_index()
    test_close_but_genuinely_distinct_lines_are_not_merged()
    test_reconciliation_invisible_to_stage6_counts()
    test_disjoint_pair_with_slope_confirmed_by_curve_agreement()
    test_disjoint_pair_coarse_match_rejected_by_curve_disagreement()
    test_disjoint_pair_with_opposite_curvature_still_merges()
    test_disjoint_pair_guard_falls_back_when_narrow_fragment_would_blow_up()
    test_diagnostic_renders()
    print("\nAll group_staves sanity checks passed.")
