#!/usr/bin/env python3
"""
Periodicity-comb DP staffline runner.

Drops into the pipeline after YOLO detection.  For each staffline box this
runner:

  1. Crops a vertical strip ±2·scale_unit around the YOLO y-centre (wider than
     the box itself, so the autocorrelation can see adjacent stafflines).
  2. Calls estimate_period() on that crop to measure the inter-staffline period h
     from the vertical intensity profile.
  3. Calls periodicity_trace() on the full page with that h, which runs DP using
     a comb-filter cost: at each candidate row y the cost includes darkness at
     y ± h, y ± 2h, … (one additional tooth on each side per extra pair).

Why the comb filter matters
---------------------------
Plain DP rewards dark pixels at the hypothesised row.  In noisy or warped
manuscripts the staffline signal alone may be weak.  The comb adds a
self-consistency prior: if the hypothesis is "this is a staffline at row y",
then we expect stafflines at y ± h too.  Rows between stafflines see bright
parchment at the tooth positions, increasing their cost.  The result is a
detector that is more selective about staying on staff positions rather than
wandering into the gaps.

n_teeth=1 degenerates to plain DP (no comb effect).
n_teeth=3 (default) uses y-h, y, y+h.

Outputs the same JSOMR JSON format as run_page.py so eval_page.py /
eval_batch.py can compare it directly against the other tracers.

Usage:
    python run_periodicity_page.py \\
        --page  staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \\
        --yolo  staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \\
        --staffline-class 0 \\
        --output staff-finding/e2e_tests/29may/Gent15_17_right/run_page/periodicity
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
from periodicity_detector import (
    compute_dark_field,
    estimate_period,
    periodicity_trace,
    BAND_HALF_MULTIPLIER,
    BLUR_SIGMA_MULTIPLIER,
    MAX_STEP_PX,
    N_TEETH,
    TEETH_WEIGHT,
)

# ---------------------------------------------------------------------------
# Page driver
# ---------------------------------------------------------------------------


def run_periodicity_page(
    page_path: Path,
    yolo_path: Path,
    output_dir: Path,
    staffline_class: int = 0,
    band_half_multiplier: float = BAND_HALF_MULTIPLIER,
    max_step_px: int = MAX_STEP_PX,
    blur_sigma_multiplier: float = BLUR_SIGMA_MULTIPLIER,
    n_teeth: int = N_TEETH,
    teeth_weight: float = TEETH_WEIGHT,
    extend_to_page: bool = True,
    use_valley_threshold: bool = False,
) -> None:
    """Run periodicity-comb DP tracing on every staffline detection of a page."""
    page_name = page_path.stem
    out = output_dir / f"{page_name}_periodicity"
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

    # Computed once per page, not once per staffline: smoothed/dark depend
    # only on gray and scale_unit, both invariant across the whole page.
    dark_field = compute_dark_field(gray, scale_unit, blur_sigma_multiplier)

    print(
        f"Tracing {len(stafflines)} stafflines (n_teeth={n_teeth}, "
        f"teeth_weight={teeth_weight})..."
    )

    for idx, det in enumerate(stafflines):
        ulx, uly, lrx, lry = det.to_pixel_box(w, h)
        y_hint = (uly + lry) / 2.0

        # Crop a strip ±4·scale_unit around the YOLO y-centre, restricted to
        # the YOLO box x-range.  Two changes from the initial run:
        #   - Vertical margin increased from ±2h to ±4h so the crop spans
        #     2–3 adjacent stafflines, making the inter-line period the
        #     dominant autocorrelation peak rather than sub-line ink texture.
        #   - X-range clamped to [ulx, lrx] to exclude text and neume columns
        #     outside the stave region, which swamp the vertical mean profile
        #     and produce false short-period peaks (~6 px in the first run).
        crop_margin = int(round(4.0 * scale_unit))
        crop_y_lo = max(0, int(np.floor(y_hint)) - crop_margin)
        crop_y_hi = min(h - 1, int(np.ceil(y_hint)) + crop_margin)
        period_crop = gray[crop_y_lo : crop_y_hi + 1, ulx : lrx + 1]

        h_est, autocorr_conf = estimate_period(period_crop, scale_unit)

        # Period range gate + confidence floor.
        # Two failure modes observed in early runs:
        #   - h_est ≈ 6 px  (sub-scale): autocorrelation locks onto the ink
        #     stroke width rather than the inter-line spacing; confidence can
        #     be high (≥ 0.3), so a confidence floor alone doesn't catch it.
        #   - h_est ≈ 37–38 px (supra-scale): autocorrelation picks up the
        #     inter-stave period; comb teeth then reach neighbouring staves and
        #     pull the DP trace off course, worsening mode.
        # Fix: accept h_est only if it falls in [0.7·h, 1.5·h].  Outside that
        # band, or if confidence is below MIN_AUTOCORR_CONF, fall back to h.
        MIN_AUTOCORR_CONF = 0.3
        H_RATIO_LO, H_RATIO_HI = 0.7, 1.5
        if (
            autocorr_conf < MIN_AUTOCORR_CONF
            or h_est < H_RATIO_LO * scale_unit
            or h_est > H_RATIO_HI * scale_unit
        ):
            h_est = scale_unit

        # Optionally extend the trace to the full page width.
        x_start = 0 if extend_to_page else ulx
        x_end = w - 1 if extend_to_page else lrx

        xs, ys = periodicity_trace(
            gray=gray,
            y_hint=y_hint,
            x_start=x_start,
            x_end=x_end,
            scale_unit=scale_unit,
            h_est=h_est,
            band_half_multiplier=band_half_multiplier,
            max_step_px=max_step_px,
            blur_sigma_multiplier=blur_sigma_multiplier,
            dark=dark_field,
            n_teeth=n_teeth,
            teeth_weight=teeth_weight,
        )

        # y_values are page-absolute; store with y_page_offset=0 (already absolute).
        fit = ExperimentFitResult(
            x_start=int(xs[0]),
            x_end=int(xs[-1]),
            y_values=list(ys),
            x_page_offset=0.0,
            y_page_offset=0.0,
            method="periodicity_comb_dp",
            meta={
                "h_est_px": round(h_est, 1),
                "autocorr_confidence": round(autocorr_conf, 3),
                "n_teeth": n_teeth,
                "teeth_weight": teeth_weight,
            },
        )
        fit_results.append(fit)
        boxes.append((ulx, uly, lrx, lry))

        print(
            f"  [{idx + 1:3d}/{len(stafflines)}] y={y_hint:.0f}  "
            f"h_est={h_est:.1f}px  conf={autocorr_conf:.2f}"
        )

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
        description="Staffline detection via periodicity-comb DP tracing."
    )
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staffline-class", type=int, default=0)
    parser.add_argument(
        "--band-half-multiplier",
        type=float,
        default=BAND_HALF_MULTIPLIER,
        help="Search band = ± multiplier × h  (default %(default)s)",
    )
    parser.add_argument(
        "--max-step-px",
        type=int,
        default=MAX_STEP_PX,
        help="Max y-shift per column (default %(default)s)",
    )
    parser.add_argument(
        "--blur-sigma-multiplier",
        type=float,
        default=BLUR_SIGMA_MULTIPLIER,
        help="Gaussian pre-blur sigma = multiplier × h (default %(default)s)",
    )
    parser.add_argument(
        "--n-teeth",
        type=int,
        default=N_TEETH,
        help="Comb teeth count (odd; 1 = plain DP) (default %(default)s)",
    )
    parser.add_argument(
        "--teeth-weight",
        type=float,
        default=TEETH_WEIGHT,
        help="Weight per off-centre tooth (0 = plain DP) (default %(default)s)",
    )
    parser.add_argument(
        "--no-extend",
        action="store_true",
        help="Trace only within the YOLO box x-range (default: extend to full page)",
    )
    parser.add_argument(
        "--valley-threshold",
        action="store_true",
        help="Use valley-finding gap detection in stave grouping (default: off)",
    )
    args = parser.parse_args()

    run_periodicity_page(
        page_path=args.page,
        yolo_path=args.yolo,
        output_dir=args.output,
        staffline_class=args.staffline_class,
        band_half_multiplier=args.band_half_multiplier,
        max_step_px=args.max_step_px,
        blur_sigma_multiplier=args.blur_sigma_multiplier,
        n_teeth=args.n_teeth,
        teeth_weight=args.teeth_weight,
        extend_to_page=not args.no_extend,
        use_valley_threshold=args.valley_threshold,
    )


if __name__ == "__main__":
    main()
