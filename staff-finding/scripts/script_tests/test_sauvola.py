"""Sanity check for Sauvola binarization in the component filter."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude")
from component_filter import filter_components


def make_faint_line_crop(
    width=400,
    height=80,
    line_y=40,
    thickness=4,
    line_intensity=170,
    background_noise=False,
):
    """Light gray line on near-white background. Otsu can fail on such low
    contrast; Sauvola should handle it.
    """
    crop = np.full((height, width, 3), 250, dtype=np.uint8)
    if background_noise:
        # Add gentle noise across the background.
        noise = np.random.default_rng(0).integers(
            -10, 11, size=crop.shape, dtype=np.int16
        )
        crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    half = thickness // 2
    crop[line_y - half : line_y + half + 1, :] = line_intensity
    return crop


def test_sauvola_recovers_faint_line():
    crop = make_faint_line_crop(line_intensity=200)  # very faint
    save_otsu = Path("/tmp/sauvola_test_otsu.png")
    save_sauvola = Path("/tmp/sauvola_test_sauvola.png")
    for p in (save_otsu, save_sauvola):
        if p.exists():
            p.unlink()

    res_otsu = filter_components(
        crop, scale_unit=4.0, save_path=save_otsu, binarization="otsu"
    )
    res_sauvola = filter_components(
        crop, scale_unit=4.0, save_path=save_sauvola, binarization="sauvola"
    )

    n_otsu = len(res_otsu.coords)
    n_sauvola = len(res_sauvola.coords)
    print(f"faint line: Otsu kept {n_otsu} pixels, Sauvola kept {n_sauvola}")
    # We don't strictly require Otsu to fail (Otsu can handle some faint cases
    # cleanly), but Sauvola should at least produce a non-empty result.
    assert n_sauvola > 0, "Sauvola should recover some line pixels"


def test_sauvola_runs_on_normal_line():
    # Normal-contrast line; Sauvola should still work, not just on faint ones.
    crop = make_faint_line_crop(line_intensity=30)
    res = filter_components(crop, scale_unit=4.0, binarization="sauvola")
    n_kept = len(res.coords)
    print(f"normal-contrast line + sauvola: kept {n_kept} pixels")
    assert n_kept > 100, "Sauvola should keep a substantial fraction of a clear line"
    assert "no_components_survived" not in res.flags


def test_invalid_method_raises():
    crop = make_faint_line_crop()
    try:
        filter_components(crop, scale_unit=4.0, binarization="bogus")
    except ValueError as e:
        print(f"invalid method correctly raised: {e}")
        return
    raise AssertionError("expected ValueError for unknown binarization method")


def test_default_is_sauvola():
    """Confirm the default binarization method is now Sauvola (post-promotion)."""
    crop = make_faint_line_crop(line_intensity=200)
    res_default = filter_components(crop, scale_unit=4.0)
    res_sauvola_explicit = filter_components(
        crop, scale_unit=4.0, binarization="sauvola"
    )
    # Same input + same method = identical kept-pixel counts.
    assert len(res_default.coords) == len(
        res_sauvola_explicit.coords
    ), "default should match explicit sauvola"
    print(
        f"default binarization is Sauvola: "
        f"{len(res_default.coords)} kept pixels both ways"
    )


if __name__ == "__main__":
    test_sauvola_recovers_faint_line()
    test_sauvola_runs_on_normal_line()
    test_invalid_method_raises()
    test_default_is_sauvola()
    print("\nSauvola sanity checks passed.")
