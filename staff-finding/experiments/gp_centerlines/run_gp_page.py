#!/usr/bin/env python3
"""
GP-centerline staffline runner.

Runs the existing pipeline up through the component filter (Sauvola binarization
+ connected-component analysis), then replaces the quadratic Huber fit with a
Gaussian Process regression.  The GP output is a smooth curve that tracks
parchment warp with per-column uncertainty estimates included in the JSOMR JSON.

WHY VALLEY-THRESHOLD GROUPING IS USED HERE
-------------------------------------------
The default stave grouping in group_staves.py sets the inter-stave cut threshold
as: max(median_gap × 1.5, scale_unit_h).  This works well for the existing
pipeline (Sauvola + quadratic Huber) because the fitted y-values closely track
the YOLO box centres, making intra-stave gaps reliably smaller than h.

GP-derived centerlines behave differently in two ways:
  1. The GP fits to filtered ink pixel positions in crop-local coordinates, not
     directly to the YOLO box centre.  The resulting page-absolute y-values can
     differ from the YOLO centre by several pixels per line.
  2. Because this manuscript's stafflines are spaced approximately h apart within
     a stave, intra-stave gaps in the sorted y-distribution cluster around h —
     the same value as the threshold floor.  The median-based threshold then
     lands inside the intra-stave cluster and incorrectly splits staves, producing
     many small "staves" of 2–3 lines rather than the true count.

The valley-finding threshold (_find_valley_threshold in group_staves.py) instead
locates the largest jump between consecutive distinct gap values — the natural
boundary of the bimodal distribution — and places the cut there.  On this page
the intra-stave gaps cluster at 0–17 px and inter-stave gaps at 25–44 px, so the
valley sits at ~21 px, well above the ambiguous 15 px floor.

This flag is off by default in group_staves.group_staves() and shared_utils so
that the existing pipeline and other experiment runners are unaffected.  It is
enabled here because GP centerlines reliably exhibit the above distribution
pattern, and it can always be disabled via --no-valley-threshold for comparison.

Usage:
    python run_gp_page.py \\
        --page  staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \\
        --yolo  staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \\
        --staffline-class 0 \\
        --output staff-finding/e2e_tests/29may/Gent15_17_right/run_page/gp_centerlines
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

_EXP_DIR = Path(__file__).parent.parent
_SCRIPTS_DIR = _EXP_DIR.parent / "scripts"
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR))

from shared_utils import (
    ExperimentFitResult,
    compute_scale_unit,
    filter_to_class,
    load_page_gray,
    parse_yolo_txt,
    run_grouping_and_save,
)
from gp_fitter import gp_fit, LENGTH_SCALE_INIT, LENGTH_SCALE_BOUNDS, N_RESTARTS

# Reuse existing component filter from the main pipeline.
from component_filter import filter_components

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


DEFAULT_CROP_PADDING = 2
DEFAULT_BINARIZATION = "sauvola"


def _crop(
    image: np.ndarray,
    ulx: int,
    uly: int,
    lrx: int,
    lry: int,
    padding: int,
    w: int,
    h: int,
) -> tuple[np.ndarray, tuple]:
    ulx_p = max(0, ulx - padding)
    uly_p = max(0, uly - padding)
    lrx_p = min(w, lrx + padding)
    lry_p = min(h, lry + padding)
    return image[uly_p:lry_p, ulx_p:lrx_p], (ulx_p, uly_p, lrx_p, lry_p)


def run_gp_page(
    page_path: Path,
    yolo_path: Path,
    output_dir: Path,
    staffline_class: int = 0,
    crop_padding: int = DEFAULT_CROP_PADDING,
    binarization: str = DEFAULT_BINARIZATION,
    length_scale_init: float = LENGTH_SCALE_INIT,
    n_restarts: int = N_RESTARTS,
    use_valley_threshold: bool = True,
) -> None:
    page_name = page_path.stem
    out = output_dir / f"{page_name}_gp"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading page: {page_path}")
    page_bgr, _, w, h = load_page_gray(page_path)
    page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
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

    print(f"Processing {len(stafflines)} stafflines (component filter + GP fit)...")
    for idx, det in enumerate(stafflines):
        ulx, uly, lrx, lry = det.to_pixel_box(w, h)
        crop, actual_box = _crop(page_rgb, ulx, uly, lrx, lry, crop_padding, w, h)
        ulx_a, uly_a, lrx_a, lry_a = actual_box

        if crop.size == 0:
            fit_results.append(ExperimentFitResult(flags=["empty_crop"]))
            boxes.append((ulx, uly, lrx, lry))
            continue

        # Component filter (same as existing pipeline).
        filter_result = filter_components(
            crop=crop,
            scale_unit=scale_unit,
            save_path=None,
            merge_components=True,
            binarization=binarization,
        )

        if not filter_result.coords:
            fit_results.append(
                ExperimentFitResult(
                    flags=["no_kept_pixels"],
                    x_page_offset=float(ulx_a),
                    y_page_offset=float(uly_a),
                )
            )
            boxes.append(actual_box)
            continue

        coords = filter_result.coords
        xs_arr = np.array([c[0] for c in coords])
        x_start = int(xs_arr.min())
        x_end = int(xs_arr.max())

        y_pred, y_std, meta = gp_fit(
            coords=coords,
            x_query_start=x_start,
            x_query_end=x_end,
            length_scale_init=length_scale_init,
            n_restarts=n_restarts,
        )

        if meta.get("error"):
            # GP fit failed — do not fabricate a centerline; exclude from grouping.
            fit_results.append(
                ExperimentFitResult(
                    x_page_offset=float(ulx_a),
                    y_page_offset=float(uly_a),
                    method="gp_matern25",
                    meta=meta,
                    flags=[f"gp_fit_failed:{meta['error']}"],
                )
            )
            boxes.append(actual_box)
            continue

        fit = ExperimentFitResult(
            x_start=x_start,
            x_end=x_end,
            y_values=y_pred,
            x_page_offset=float(ulx_a),
            y_page_offset=float(uly_a),
            method="gp_matern25",
            meta=meta,
        )
        # Store uncertainty in flags as a summary statistic (mean std across x).
        mean_std = float(np.mean(y_std)) if y_std else 0.0
        fit.flags = [f"mean_uncertainty_px:{mean_std:.2f}"]

        fit_results.append(fit)
        boxes.append(actual_box)

        if (idx + 1) % 20 == 0:
            print(f"  ...{idx + 1}/{len(stafflines)} done")

    run_grouping_and_save(
        page_name=page_name,
        fit_results=fit_results,
        boxes=boxes,
        scale_unit=scale_unit,
        page_bgr=page_bgr,
        output_dir=out,
        use_valley_threshold=use_valley_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staffline detection via Gaussian Process centerline fitting."
    )
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staffline-class", type=int, default=0)
    parser.add_argument("--crop-padding", type=int, default=DEFAULT_CROP_PADDING)
    parser.add_argument(
        "--binarization", default=DEFAULT_BINARIZATION, choices=["sauvola", "otsu"]
    )
    parser.add_argument(
        "--length-scale-init",
        type=float,
        default=LENGTH_SCALE_INIT,
        help="Initial GP length scale in pixels (default %(default)s)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=N_RESTARTS,
        help="Kernel hyperparameter optimisation restarts (default %(default)s)",
    )
    parser.add_argument(
        "--no-valley-threshold",
        action="store_true",
        help="Use the default median-based grouping threshold instead of "
        "valley-finding.  Useful for comparing grouping methods; see "
        "module docstring for a full explanation of why valley-finding "
        "is the default here.",
    )
    args = parser.parse_args()

    run_gp_page(
        page_path=args.page,
        yolo_path=args.yolo,
        output_dir=args.output,
        staffline_class=args.staffline_class,
        crop_padding=args.crop_padding,
        binarization=args.binarization,
        length_scale_init=args.length_scale_init,
        n_restarts=args.n_restarts,
        use_valley_threshold=not args.no_valley_threshold,
    )


if __name__ == "__main__":
    main()
