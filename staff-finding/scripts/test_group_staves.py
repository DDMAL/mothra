#!/usr/bin/env python3
"""Quick test of group_staves implementation."""

import sys
from fit_centerline import FitResult
from group_staves import group_staves

def test_basic_grouping():
    """Test basic stave grouping with synthetic data."""
    
    # Create synthetic fits with y-positions corresponding to 2 staves
    # Stave 1: fits at y = 100, 110, 120, 130 (gaps = 10 each, intra-stave)
    # Stave 2: fits at y = 200, 210, 220, 230 (gaps = 10 each, intra-stave)
    # Gap between staves: 70 pixels (inter-stave)
    
    fits = []
    y_positions = [100, 110, 120, 130, 200, 210, 220, 230]
    
    for i, y in enumerate(y_positions):
        fit = FitResult(
            x_start=0,
            x_end=100,
            y_values=[y] * 101,  # Constant y across x range
            coefficients=[0.0, 0.0, y],  # Flat line at y
            residual_mean=0.0,
            residual_max=0.0,
            n_pixels_used=101,
            flags=[]
        )
        fits.append(fit)
    
    # Run grouping
    result = group_staves(
        fits=fits,
        scale_unit=5.0,  # h = 5 pixels
        interpolate_missing=False,
    )
    
    print("Test: Basic stave grouping")
    print(f"  Mode lines per stave: {result.mode_lines_per_stave}")
    print(f"  Line count distribution: {result.line_count_distribution}")
    print(f"  Cut threshold: {result.cut_threshold_px:.2f} px")
    print(f"  Gap distribution: {result.gap_distribution}")
    print(f"  Flags: {result.flags}")
    print()
    
    # Verify assignments
    print("  Assignments:")
    for assignment in result.assignments:
        print(f"    Fit {assignment.fit_index}: stave_id={assignment.stave_id}, "
              f"within_index={assignment.within_stave_index}, "
              f"y_at_center={assignment.y_at_center}")
    
    # Expected: 2 staves with 4 lines each
    expected_stave_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    actual_stave_ids = [a.stave_id for a in result.assignments]
    
    print()
    if actual_stave_ids == expected_stave_ids:
        print("  PASS: Stave assignments correct")
        return True
    else:
        print(f"  FAIL: Expected stave_ids {expected_stave_ids}, got {actual_stave_ids}")
        return False


def test_empty_fits():
    """Test handling of empty fit list."""
    
    result = group_staves(
        fits=[],
        scale_unit=5.0,
    )
    
    print("Test: Empty fits list")
    print(f"  Flags: {result.flags}")
    
    if "no_fits_available" in result.flags:
        print("  PASS: Correctly flagged empty input")
        return True
    else:
        print("  FAIL: Should have flagged empty input")
        return False


def test_no_y_values():
    """Test handling of fits with no y_values."""
    
    # Create a fit with no y_values (failed fit)
    fit = FitResult(
        x_start=0,
        x_end=0,
        y_values=[],  # No y values
        coefficients=[],
        flags=["fit_did_not_converge"]
    )
    
    result = group_staves(
        fits=[fit],
        scale_unit=5.0,
    )
    
    print("Test: Fit with no y_values")
    print(f"  Flags: {result.flags}")
    print(f"  Assignments: {len(result.assignments)}")
    if result.assignments:
        print(f"    Assignment 0: stave_id={result.assignments[0].stave_id}, "
              f"flags={result.assignments[0].flags}")
    
    if ("no_fits_with_y_positions" in result.flags and 
        len(result.assignments) == 1 and
        result.assignments[0].stave_id is None):
        print("  PASS: Correctly handled fit with no y_values")
        return True
    else:
        print("  FAIL: Should have recorded unassigned fit")
        return False


if __name__ == "__main__":
    try:
        test1 = test_basic_grouping()
        print()
        test2 = test_empty_fits()
        print()
        test3 = test_no_y_values()
        print()
        
        if test1 and test2 and test3:
            print("All tests PASSED")
            sys.exit(0)
        else:
            print("Some tests FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
