#!/usr/bin/env python3
"""
Implicit-neural-representation staffline runner.

Drops into the pipeline after YOLO detection.  For each staffline box, fits a
tiny MLP f(x) → y by gradient descent directly on the grayscale page image —
no component filtering, no curve fitting on ink coordinates.  The network is
optimised at test time to predict y positions that sample dark (ink) pixels,
guided by a smoothness regulariser and a band-clamping constraint.

How it differs from the GP approach
------------------------------------
The GP fitter (gp_centerlines/) operates on a pre-extracted set of filtered
ink-pixel (x, y) coordinates: it regresses a smooth Matérn curve through those
coordinates and returns posterior uncertainty as a free by-product.  The
implicit neural approach bypasses coordinate extraction entirely — the MLP is
differentiably connected to the grayscale tensor via bilinear grid_sample, so
gradients from pixel brightness flow directly into the network weights.  The
trade-off is no uncertainty estimate, but the approach is independent of the
component-filter quality and can adapt to faint or broken stafflines.

What the positional encoding buys
-----------------------------------
A plain MLP with a scalar x input would need many hidden units to represent
warp that varies on the scale of tens of pixels.  Sinusoidal positional
encoding maps x to a fixed bank of sin/cos features at geometrically-spaced
frequencies before the first linear layer, making high-frequency spatial
variation explicit in the input.  A 32-unit hidden layer is then sufficient to
capture sharp local deviations without overfitting.

Usage:
    python run_implicit_neural_page.py \\
        --page  staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \\
        --yolo  staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \\
        --staffline-class 0 \\
        --output staff-finding/e2e_tests/implicit_neural
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Locate shared utilities relative to this file.
_EXP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_EXP_DIR))
from shared_utils import (
    ExperimentFitResult,
    compute_scale_unit,
    filter_to_class,
    load_page_gray,
    parse_yolo_txt,
    run_grouping_and_save,
)
from implicit_neural_fitter import (
    implicit_neural_fit,
    N_FREQS,
    HIDDEN,
    LR,
    N_STEPS,
    LAMBDA_SMOOTH,
    BAND_HALF_MULTIPLIER,
)

# ---------------------------------------------------------------------------
# Page driver
# ---------------------------------------------------------------------------


def run_implicit_neural_page(
    page_path: Path,
    yolo_path: Path,
    output_dir: Path,
    staffline_class: int = 0,
    n_freqs: int = N_FREQS,
    hidden: int = HIDDEN,
    lr: float = LR,
    n_steps: int = N_STEPS,
    lambda_smooth: float = LAMBDA_SMOOTH,
    band_half_multiplier: float = BAND_HALF_MULTIPLIER,
    extend_to_page: bool = False,
    use_valley_threshold: bool = False,
    channel: str = "gray",
) -> None:
    """Run implicit-neural fitting on every staffline detection of a page.

    Args:
        channel: Which image channel to use for the brightness loss.
                 "gray"  — standard luminance grayscale (default, BGR→gray).
                 "green" — green channel only; red ink appears dark because red
                           has low green content, potentially improving detection
                           in regions where red stafflines compete with black text.
                 "blue"  — blue channel only; red ink appears very dark.
                 "red"   — red channel only; red ink appears bright (worst for ink).
    """
    page_name = page_path.stem
    out = output_dir / f"{page_name}_implicit_neural"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading page: {page_path}")
    page_bgr, gray_standard, w, h = load_page_gray(page_path)
    print(f"  Page size: {w} x {h}")

    # Select working grayscale based on channel argument.
    channel = channel.lower()
    if channel == "gray":
        gray = gray_standard
    elif channel == "green":
        gray = page_bgr[:, :, 1]  # OpenCV stores as BGR; index 1 = green
    elif channel == "blue":
        gray = page_bgr[:, :, 0]  # index 0 = blue
    elif channel == "red":
        gray = page_bgr[:, :, 2]  # index 2 = red
    else:
        raise ValueError(f"Unknown channel '{channel}'; choose gray/green/blue/red.")

    if channel != "gray":
        print(f"  Using channel: {channel}")

    print(f"Parsing YOLO detections: {yolo_path}")
    all_detections = parse_yolo_txt(yolo_path)
    stafflines = filter_to_class(all_detections, staffline_class)
    print(f"  Stafflines (class {staffline_class}): {len(stafflines)}")
    if not stafflines:
        print("  No stafflines; nothing to do.")
        return

    scale_unit = compute_scale_unit(stafflines, w, h)
    print(f"  Scale unit h: {scale_unit:.1f} px")

    fit_results: list[ExperimentFitResult] = []
    boxes: list[tuple[int, int, int, int]] = []

    print(
        f"Fitting {len(stafflines)} stafflines (n_steps={n_steps}, hidden={hidden}, n_freqs={n_freqs})..."
    )
    for idx, det in enumerate(stafflines):
        ulx, uly, lrx, lry = det.to_pixel_box(w, h)
        y_hint = (uly + lry) / 2.0

        # Default: trace only within the YOLO box x-range.  The MLP can
        # overfit to image texture when given a very long span with no ink,
        # so per-box is safer than extending to the full page width.
        x_start = 0 if extend_to_page else ulx
        x_end = w - 1 if extend_to_page else lrx

        y_values, meta = implicit_neural_fit(
            gray=gray,
            y_hint=y_hint,
            x_start=x_start,
            x_end=x_end,
            scale_unit=scale_unit,
            n_freqs=n_freqs,
            hidden=hidden,
            lr=lr,
            n_steps=n_steps,
            lambda_smooth=lambda_smooth,
            band_half_multiplier=band_half_multiplier,
        )

        print(
            f"  [{idx+1:>3}/{len(stafflines)}]  y_hint={y_hint:.1f}  "
            f"loss={meta['final_loss']:.4f}  data={meta['final_data_loss']:.4f}"
        )

        # y_values are page-absolute; store with y_page_offset=0 (already absolute).
        fit = ExperimentFitResult(
            x_start=x_start,
            x_end=x_end,
            y_values=y_values,
            x_page_offset=0.0,
            y_page_offset=0.0,
            method="implicit_neural_mlp",
            meta=meta,
        )
        fit_results.append(fit)
        boxes.append((ulx, uly, lrx, lry))

    run_grouping_and_save(
        page_name=page_name,
        fit_results=fit_results,
        boxes=boxes,
        scale_unit=scale_unit,
        page_bgr=page_bgr,
        output_dir=out,
        use_valley_threshold=use_valley_threshold,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staffline detection via implicit neural representation (MLP test-time optimisation)."
    )
    parser.add_argument(
        "--page", type=Path, required=True, help="Path to page image (jpg/png)."
    )
    parser.add_argument(
        "--yolo", type=Path, required=True, help="Path to YOLO detections .txt file."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory root."
    )
    parser.add_argument(
        "--staffline-class",
        type=int,
        default=0,
        help="YOLO class index for stafflines (default %(default)s).",
    )
    parser.add_argument(
        "--n-freqs",
        type=int,
        default=N_FREQS,
        help="Positional encoding frequency bands (default %(default)s).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=HIDDEN,
        help="MLP hidden layer width (default %(default)s).",
    )
    parser.add_argument(
        "--lr", type=float, default=LR, help="Adam learning rate (default %(default)s)."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=N_STEPS,
        help="Gradient steps per staffline (default %(default)s).",
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=LAMBDA_SMOOTH,
        help="Smoothness regulariser weight (default %(default)s).",
    )
    parser.add_argument(
        "--band-half-multiplier",
        type=float,
        default=BAND_HALF_MULTIPLIER,
        help="Clamp band = ± multiplier × scale_unit (default %(default)s).",
    )
    parser.add_argument(
        "--extend-to-page",
        action="store_true",
        help="Trace full page width instead of per-box x-range.",
    )
    parser.add_argument(
        "--valley-threshold",
        action="store_true",
        help="Use valley-finding gap detection for stave grouping.",
    )
    parser.add_argument(
        "--channel",
        choices=["gray", "green", "blue", "red"],
        default="gray",
        help=(
            "Image channel used for the brightness loss.  "
            "'gray' (default): standard luminance grayscale.  "
            "'green': green channel — red ink appears dark, useful when red "
            "stafflines compete with black text/initials.  "
            "'blue': blue channel — red ink very dark, may amplify parchment noise.  "
            "'red': red channel — red ink bright (not recommended for ink detection)."
        ),
    )
    args = parser.parse_args()

    run_implicit_neural_page(
        page_path=args.page,
        yolo_path=args.yolo,
        output_dir=args.output,
        staffline_class=args.staffline_class,
        n_freqs=args.n_freqs,
        hidden=args.hidden,
        lr=args.lr,
        n_steps=args.n_steps,
        lambda_smooth=args.lambda_smooth,
        band_half_multiplier=args.band_half_multiplier,
        extend_to_page=args.extend_to_page,
        use_valley_threshold=args.valley_threshold,
        channel=args.channel,
    )


if __name__ == "__main__":
    main()
