#!/usr/bin/env python3
"""
DP-tracing staffline runner.

Drops into the pipeline after YOLO detection.  For each staffline box, runs
dp_trace() on the raw grayscale page image — no binarization, no component
filter, no curve fitting.  Outputs the same JSOMR JSON format as run_page.py
so eval_page.py / eval_batch.py can compare it directly.

Usage:
    python run_dp_page.py \\
        --page  staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \\
        --yolo  staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \\
        --staffline-class 0 \\
        --output staff-finding/e2e_tests/29may/Gent15_17_right/run_page/dp_tracing
"""

import argparse
import sys
from pathlib import Path

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
from dp_tracer import dp_trace, BAND_HALF_MULTIPLIER, MAX_STEP_PX, BLUR_SIGMA_MULTIPLIER


# ---------------------------------------------------------------------------
# Page driver
# ---------------------------------------------------------------------------

def run_dp_page(
    page_path: Path,
    yolo_path: Path,
    output_dir: Path,
    staffline_class: int = 0,
    band_half_multiplier: float = BAND_HALF_MULTIPLIER,
    max_step_px: int = MAX_STEP_PX,
    blur_sigma_multiplier: float = BLUR_SIGMA_MULTIPLIER,
    extend_to_page: bool = True,
) -> None:
    """Run DP tracing on every staffline detection of a page."""
    page_name = page_path.stem
    out = output_dir / f"{page_name}_dp"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading page: {page_path}")
    page_bgr, gray, w, h = load_page_gray(page_path)
    print(f"  Page size: {w} x {h}")

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

    print(f"Tracing {len(stafflines)} stafflines...")
    for idx, det in enumerate(stafflines):
        ulx, uly, lrx, lry = det.to_pixel_box(w, h)
        y_hint = (uly + lry) / 2.0

        # Optionally extend the trace to the full page width so we recover the
        # staffline even where the YOLO box was clipped or narrow.
        x_start = 0 if extend_to_page else ulx
        x_end   = w - 1 if extend_to_page else lrx

        xs, ys = dp_trace(
            gray=gray,
            y_hint=y_hint,
            x_start=x_start,
            x_end=x_end,
            scale_unit=scale_unit,
            band_half_multiplier=band_half_multiplier,
            max_step_px=max_step_px,
            blur_sigma_multiplier=blur_sigma_multiplier,
        )

        # y_values are page-absolute; store with y_page_offset=0 (already absolute)
        fit = ExperimentFitResult(
            x_start=int(xs[0]),
            x_end=int(xs[-1]),
            y_values=list(ys),
            x_page_offset=0.0,
            y_page_offset=0.0,
            method="dp_trace",
            meta={
                "band_half_multiplier": band_half_multiplier,
                "max_step_px": max_step_px,
                "blur_sigma_multiplier": blur_sigma_multiplier,
            },
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
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staffline detection via DP shortest-path tracing."
    )
    parser.add_argument("--page",  type=Path, required=True)
    parser.add_argument("--yolo",  type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staffline-class", type=int, default=0)
    parser.add_argument("--band-half-multiplier", type=float,
                        default=BAND_HALF_MULTIPLIER,
                        help="Search band = ± multiplier × h  (default %(default)s)")
    parser.add_argument("--max-step-px", type=int, default=MAX_STEP_PX,
                        help="Max y-shift per column (default %(default)s)")
    parser.add_argument("--blur-sigma-multiplier", type=float,
                        default=BLUR_SIGMA_MULTIPLIER,
                        help="Gaussian pre-blur sigma = multiplier × h (default %(default)s)")
    parser.add_argument("--no-extend", action="store_true",
                        help="Trace only within the YOLO box x-range (default: extend to full page)")
    args = parser.parse_args()

    run_dp_page(
        page_path=args.page,
        yolo_path=args.yolo,
        output_dir=args.output,
        staffline_class=args.staffline_class,
        band_half_multiplier=args.band_half_multiplier,
        max_step_px=args.max_step_px,
        blur_sigma_multiplier=args.blur_sigma_multiplier,
        extend_to_page=not args.no_extend,
    )


if __name__ == "__main__":
    main()
