"""Sanity check for component_filter.py with synthetic inputs."""

import sys
import numpy as np

sys.path.insert(0, "/home/claude")
from component_filter import filter_components


def make_synthetic_crop(
    width=400,
    height=80,
    line_y=40,
    line_thickness=4,
    add_neume=False,
    add_neighbor_fragment=False,
    add_noise_speck=False,
    empty=False,
):
    """Make a synthetic crop: white background, dark line, optional extras."""
    # White background (255), so ink will be added as low values.
    crop = np.full((height, width, 3), 255, dtype=np.uint8)

    if empty:
        return crop

    # Draw the main staffline as a dark horizontal stripe.
    half = line_thickness // 2
    crop[line_y - half : line_y + half + 1, :] = 30

    if add_neume:
        # A neume blob on the line, centered.
        cx = width // 2
        crop[line_y - 10 : line_y + 10, cx - 8 : cx + 8] = 30

    if add_neighbor_fragment:
        # A short fragment near the top of the box (intruding neighbor line).
        crop[5:8, 50:150] = 30

    if add_noise_speck:
        # A tiny speck of ink — should be filtered as below min size.
        crop[70:72, 200:202] = 30

    return crop


def test_clean_line():
    crop = make_synthetic_crop()
    result = filter_components(crop, scale_unit=4.0)
    assert len(result.coords) > 0, "should have kept pixels"
    assert "no_components_survived" not in result.flags
    assert "multiple_components_kept" not in result.flags
    print(f"clean line: {len(result.coords)} kept pixels, flags={result.flags}")


def test_line_with_neume():
    crop = make_synthetic_crop(add_neume=True)
    result = filter_components(crop, scale_unit=4.0)
    assert len(result.coords) > 0
    # The neume is part of the line's connected component, so should be kept.
    # Confirm by checking that some kept pixels are near the box center.
    ys = [y for _, y in result.coords]
    # Neume blob extends y=30..49 (since [30:50] is exclusive on top end).
    assert (
        min(ys) <= 30 and max(ys) >= 49
    ), f"kept component should include neume vertical extent, got y range {min(ys)}..{max(ys)}"
    print(f"line+neume: {len(result.coords)} kept pixels, flags={result.flags}")


def test_line_with_neighbor_fragment():
    crop = make_synthetic_crop(add_neighbor_fragment=True)
    result = filter_components(crop, scale_unit=4.0)
    assert len(result.coords) > 0
    # The neighbor fragment is short (100 wide) vs. the main line (400 wide).
    # Both should pass filters, but main line should win on horizontal extent.
    # Kept pixels should be near y=40 (the main line), not near y=6 (the fragment).
    ys = [y for _, y in result.coords]
    median_y = sorted(ys)[len(ys) // 2]
    assert 35 < median_y < 45, f"kept median y should be near main line, got {median_y}"
    # The fragment should be in 'discarded' as 'not_top_scoring'.
    discarded_reasons = [d.get("reason") for d in result.discarded]
    print(
        f"line+neighbor: kept median y={median_y}, "
        f"discarded reasons={discarded_reasons}, flags={result.flags}"
    )


def test_noise_speck_filtered():
    crop = make_synthetic_crop(add_noise_speck=True)
    result = filter_components(crop, scale_unit=4.0)
    # The speck is 2x2 = 4 pixels; min_size = 5 * 4.0 = 20. Should be discarded.
    discarded_reasons = [d.get("reason") for d in result.discarded]
    assert (
        "below_min_size" in discarded_reasons
    ), f"speck should be filtered as below_min_size; got reasons={discarded_reasons}"
    print(f"noise speck: discarded reasons={discarded_reasons}")


def test_empty_crop():
    crop = make_synthetic_crop(empty=True)
    result = filter_components(crop, scale_unit=4.0)
    assert (
        "no_components_survived" in result.flags
    ), f"empty crop should set the flag; got flags={result.flags}"
    assert len(result.coords) == 0
    print(f"empty crop: flags={result.flags}, coords empty={len(result.coords) == 0}")


def test_visualization_saves():
    from pathlib import Path

    crop = make_synthetic_crop(add_neume=True, add_neighbor_fragment=True)
    save_path = Path("/tmp/component_filter_diag.png")
    if save_path.exists():
        save_path.unlink()
    _ = filter_components(crop, scale_unit=4.0, save_path=save_path)
    assert save_path.exists(), "diagnostic PNG should have been saved"
    print(f"visualization saved to {save_path}, size={save_path.stat().st_size} bytes")


if __name__ == "__main__":
    test_clean_line()
    test_line_with_neume()
    test_line_with_neighbor_fragment()
    test_noise_speck_filtered()
    test_empty_crop()
    test_visualization_saves()
    print("\nAll sanity checks passed.")
