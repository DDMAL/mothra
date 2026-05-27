"""Sanity check for group_staves on synthetic FitResult sequences."""
import sys
from pathlib import Path

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
    ys = [100, 130, 160, 190,    # stave 0
          310, 340, 370, 400]    # stave 1
    fits = [make_fit_at_y(y) for y in ys]
    result = group_staves(fits, scale_unit=10.0)
    stave_ids = [a.stave_id for a in result.assignments]
    print(f"two staves of four: stave_ids={stave_ids}, "
          f"mode={result.mode_lines_per_stave}, "
          f"distribution={result.line_count_distribution}, "
          f"cut={result.cut_threshold_px:.1f}, flags={result.flags}")
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
    ys = [100, 130, 160, 190,    # stave 0: 4 lines
          310, 340, 400]         # stave 1: 3 lines (one missing in middle)
    fits = [make_fit_at_y(y) for y in ys]
    result = group_staves(fits, scale_unit=10.0)
    stave_ids = [a.stave_id for a in result.assignments]
    print(f"missing-line (no interp): stave_ids={stave_ids}, "
          f"mode={result.mode_lines_per_stave}, "
          f"distribution={result.line_count_distribution}, "
          f"flags={result.flags}")
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
    # Interpolation is currently stubbed (returns empty list; MVP deferred).
    # This test verifies that grouping still runs correctly with
    # interpolate_missing=True and that the stave grouping is sound.
    # The missing-line case uses a high scale_unit so the 50-px gap within
    # stave 1 stays below the cut threshold and both staves stay intact.
    ys_stave_0 = [100, 130, 160, 190]
    ys_stave_1 = [400, 430, 480]  # widest internal gap is 50
    ys = ys_stave_0 + ys_stave_1
    fits = [make_fit_at_y(y) for y in ys]
    result = group_staves(fits, scale_unit=55.0, interpolate_missing=True)
    print(f"missing-line (interp ON, stub): "
          f"stave_ids={[a.stave_id for a in result.assignments]}, "
          f"mode={result.mode_lines_per_stave}, "
          f"n_interpolated={len(result.interpolated_lines)}")
    stave_ids = [a.stave_id for a in result.assignments]
    assert stave_ids[:4] == [0, 0, 0, 0]
    assert stave_ids[4:] == [1, 1, 1]
    assert result.mode_lines_per_stave == 4
    # Interpolation is stubbed; interpolated_lines is empty until implemented.
    assert result.interpolated_lines == []


def test_failed_fits_excluded_from_grouping():
    # Mix of usable and failed fits.
    ys = [100, 130, 160, 190]
    fits = [make_fit_at_y(y) for y in ys]
    # Insert a failed fit (no y_values).
    fits.insert(2, FitResult(flags=["fit_did_not_converge"]))
    result = group_staves(fits, scale_unit=10.0)
    print(f"with one failed fit: "
          f"stave_ids={[a.stave_id for a in result.assignments]}, "
          f"flags on failed={result.assignments[2].flags}")
    # Failed fit has no stave_id.
    assert result.assignments[2].stave_id is None
    assert "no_y_position_available" in result.assignments[2].flags
    # The four usable fits still grouped as one stave.
    usable_assignments = [a for a in result.assignments if a.stave_id is not None]
    assert all(a.stave_id == 0 for a in usable_assignments)


def test_no_fits_available():
    # All fits are failed (no y_values); distinct from an empty list.
    # An empty list → 'no_fits_available'; non-empty all-failed → 'no_fits_with_y_positions'.
    fits = [FitResult(flags=["fit_did_not_converge"]),
            FitResult(flags=["no_fit_attempted"])]
    result = group_staves(fits, scale_unit=10.0)
    print(f"no usable fits: flags={result.flags}")
    assert "no_fits_with_y_positions" in result.flags


def test_diagnostic_renders():
    ys = [100, 130, 160, 190, 310, 340, 370, 400]
    fits = [make_fit_at_y(y) for y in ys]
    save_path = Path("/tmp/group_staves_diag.png")
    if save_path.exists():
        save_path.unlink()
    _ = group_staves(fits, scale_unit=10.0,
                      save_path=save_path,
                      page_size=(1000, 500))
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
    test_diagnostic_renders()
    print("\nAll group_staves sanity checks passed.")