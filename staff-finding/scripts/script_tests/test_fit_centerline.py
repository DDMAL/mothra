"""Sanity check for fit_centerline.py on synthetic inputs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def test_two_line_box_line_following():
    """Line-following isolates the nearer staffline when the kept-pixel set
    contains two lines (e.g. because connected-component merging joined them).

    The component filter is bypassed here — we construct a ComponentFilterResult
    directly with pixels from both lines so that the test exercises only the
    line-following logic inside fit_centerline.

    Geometry (scale_unit=15, realistic for square-notation manuscripts):
      - OLS puts initial guess at y~50 (midpoint of 30 and 70)
      - Both lines are ~20 px from that midpoint, both in the linear Huber
        regime (f_scale = 0.5*15 = 7.5 px)
      - With equal pixel counts both sides cancel -> Huber stays near y=50,
        residual_mean ~ 20 px >> trigger (1.0*15 = 15 px) -> line-following fires
      - seed = crop vertical center = 40 px
      - band_half = 1.5*15 = 22.5 px -> band [17.5, 62.5]
      - inner line (y~30) IS in band; outer line (y~70) is NOT -> trace locks y~30
    """
    scale_unit = 15.0
    crop = np.full((80, 400, 3), 255, dtype=np.uint8)
    crop[28:32, :] = 30   # inner line at y~30 (10 px from seed=40)
    crop[68:72, :] = 30   # outer line at y~70 (30 px from seed=40, outside band)

    # Build a filter result containing pixels from BOTH lines, replicating the
    # scenario where two lines end up in the same kept-pixel set.
    coords: list[tuple[int, int]] = []
    for x in range(400):
        for y in [28, 29, 30, 31]:  # inner line rows
            coords.append((x, y))
        for y in [68, 69, 70, 71]:  # outer line rows
            coords.append((x, y))
    filter_result = ComponentFilterResult(coords=coords)

    fit_result = fit_centerline(filter_result, scale_unit=scale_unit,
                                crop=crop,
                                save_path=Path("/tmp/fit_two_line.png"))
    median_y = float(np.median(fit_result.y_values))
    print(f"two-line box: median_y={median_y:.2f}, "
          f"residual_mean={fit_result.residual_mean:.3f}, "
          f"flags={fit_result.flags}")
    assert any("line_following_applied" in f for f in fit_result.flags), \
        f"expected line_following_applied in flags; got {fit_result.flags}"
    # Trace should lock onto the inner line (y~30, inside the initial band).
    assert 27 <= median_y <= 33, \
        f"expected fit near y=30 (inner line); got median_y={median_y:.2f}"
    assert fit_result.residual_mean < 3.0, \
        f"expected tight refit (< 3 px); got residual_mean={fit_result.residual_mean:.3f}"


if __name__ == "__main__":
    test_clean_line_fit()
    test_line_with_neume_fit()
    test_empty_filter_result_fit()
    test_sloped_line_fit()
    test_two_line_box_line_following()
    print("\nAll fit_centerline sanity checks passed.")
