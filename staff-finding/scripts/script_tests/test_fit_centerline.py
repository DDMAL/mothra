"""Sanity check for fit_centerline.py on synthetic inputs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude")
from component_filter import filter_components, ComponentFilterResult
from fit_centerline import fit_centerline


def make_synthetic_crop(width=400, height=80, line_y=40, line_thickness=4,
                        add_neume=False, empty=False):
    crop = np.full((height, width, 3), 255, dtype=np.uint8)
    if empty:
        return crop
    half = line_thickness // 2
    crop[line_y - half:line_y + half + 1, :] = 30
    if add_neume:
        cx = width // 2
        crop[line_y - 10:line_y + 10, cx - 8:cx + 8] = 30
    return crop


def test_clean_line_fit():
    crop = make_synthetic_crop()
    filter_result = filter_components(crop, scale_unit=4.0)
    fit_result = fit_centerline(filter_result, scale_unit=4.0,
                                crop=crop, save_path=Path("/tmp/fit_clean.png"))
    # Line y is at 40; fit should land near 40 for all sample positions.
    median_y = float(np.median(fit_result.y_values))
    print(f"clean line: median fitted y={median_y:.2f}, "
          f"residual mean={fit_result.residual_mean:.3f}, "
          f"x range=[{fit_result.x_start}, {fit_result.x_end}]")
    assert 39 <= median_y <= 41, f"expected fit near y=40, got {median_y}"
    # For a 4-pixel-thick line, a perfectly centered fit has residuals up to
    # 2 pixels at the line's edges, so mean residuals around 1 px are correct.
    assert fit_result.residual_mean < 2.0
    assert fit_result.flags == []


def test_line_with_neume_fit():
    crop = make_synthetic_crop(add_neume=True)
    filter_result = filter_components(crop, scale_unit=4.0)
    fit_result = fit_centerline(filter_result, scale_unit=4.0,
                                crop=crop, save_path=Path("/tmp/fit_with_neume.png"))
    median_y = float(np.median(fit_result.y_values))
    # The neume blob is centered on the line, so contributes symmetric pixels;
    # robust fit should still land near 40.
    print(f"line+neume: median fitted y={median_y:.2f}, "
          f"residual mean={fit_result.residual_mean:.3f}")
    assert 38 <= median_y <= 42, \
        f"expected fit near y=40 with neume tolerated, got {median_y}"


def test_empty_filter_result_fit():
    # Synthesize an empty ComponentFilterResult directly.
    empty_result = ComponentFilterResult()
    fit_result = fit_centerline(empty_result, scale_unit=4.0)
    print(f"empty input: y_values={fit_result.y_values}, flags={fit_result.flags}")
    assert fit_result.y_values == []
    assert "no_fit_attempted" in fit_result.flags


def test_sloped_line_fit():
    # A linear-slope line, easy case for quadratic fit (which subsumes linear).
    crop = np.full((80, 400, 3), 255, dtype=np.uint8)
    for x in range(400):
        y = int(20 + (x / 400) * 30)  # y goes from 20 to 50 across the box
        crop[y - 1:y + 2, x] = 30
    filter_result = filter_components(crop, scale_unit=4.0)
    fit_result = fit_centerline(filter_result, scale_unit=4.0,
                                crop=crop, save_path=Path("/tmp/fit_sloped.png"))
    # Check endpoints land near where the line actually is.
    y_at_start = fit_result.y_values[0]
    y_at_end = fit_result.y_values[-1]
    print(f"sloped line: y at x_start={y_at_start:.2f} (expect ~20), "
          f"y at x_end={y_at_end:.2f} (expect ~50)")
    assert 18 <= y_at_start <= 23
    assert 47 <= y_at_end <= 52


if __name__ == "__main__":
    test_clean_line_fit()
    test_line_with_neume_fit()
    test_empty_filter_result_fit()
    test_sloped_line_fit()
    print("\nAll fit_centerline sanity checks passed.")