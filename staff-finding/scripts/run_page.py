"""
Page-level driver for the Stage 1 component filter.

Given a page image, its YOLO detection output, and (optionally) the BGR
(background removal) model checkpoint, this script:

  1. Loads the page (RGB).
  2. Optionally runs BGR inference once on the full page, producing an
     ink-on-white image. With --no-bgr, the original page is used directly.
  3. Parses the YOLO .txt detections and filters to the staffline class.
  4. For each staffline detection: crops the (BGR-processed or raw) page, runs
     the component filter, saves a diagnostic PNG.
  5. Writes a per-page summary CSV.

Usage:
    python run_page.py \\
        --page /path/to/page.jpg \\
        --yolo /path/to/page.txt \\
        --bgr-model /path/to/best_model.pth \\
        --output /path/to/output_dir \\
        [--staffline-class 2] \\
        [--crop-padding 2] \\
        [--device cpu] \\
        [--no-bgr]

When --no-bgr is set, --bgr-model is not required and BGR is skipped entirely.
Output goes to <output>/<page_name>_no_bgr/ so it doesn't clobber a BGR run.

See ADR-001 for component-filter design decisions.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from component_filter import filter_components
from fit_centerline import fit_centerline
from group_staves import group_staves
from yolo_io import parse_yolo_txt, filter_to_class, YoloDetection
from bgr_adapter import (
    load_bgr_model,
    run_bgr_inference,
    DEFAULT_BGR_WINDOW_SIZE,
    DEFAULT_BGR_STRIDE,
    DEFAULT_BGR_CONFIDENCE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STAFFLINE_CLASS = 2
DEFAULT_CROP_PADDING_PX = 2  # small margin around YOLO box; see driver plan


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def crop_with_padding(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to box plus padding, clamped to image bounds.

    Returns the cropped array and the actual pixel box used (after clamping).
    """
    h, w = image.shape[:2]
    ulx, uly, lrx, lry = box
    ulx_p = max(0, ulx - padding)
    uly_p = max(0, uly - padding)
    lrx_p = min(w, lrx + padding)
    lry_p = min(h, lry + padding)
    crop = image[uly_p:lry_p, ulx_p:lrx_p]
    return crop, (ulx_p, uly_p, lrx_p, lry_p)


# ---------------------------------------------------------------------------
# Scale unit derivation
# ---------------------------------------------------------------------------


def compute_page_scale_unit(
    detections: list[YoloDetection],
    image_width: int,
    image_height: int,
) -> float:
    """Page-level scale unit h: median pixel height of staffline boxes.

    A tight detector box's height is roughly the line thickness plus a small
    slack. Median across the page is more robust than mean for the occasional
    skewed/tall box.
    """
    heights = []
    for d in detections:
        _, uly, _, lry = d.to_pixel_box(image_width, image_height)
        heights.append(lry - uly)
    if not heights:
        return 0.0
    return float(np.median(heights))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def process_page(
    page_path: Path,
    yolo_path: Path,
    bgr_model_path: Optional[Path],
    output_dir: Path,
    staffline_class: int = DEFAULT_STAFFLINE_CLASS,
    crop_padding: int = DEFAULT_CROP_PADDING_PX,
    device: str = "cpu",
    bgr_window_size: int = DEFAULT_BGR_WINDOW_SIZE,
    bgr_stride: int = DEFAULT_BGR_STRIDE,
    bgr_confidence: float = DEFAULT_BGR_CONFIDENCE,
    use_bgr: bool = True,
    merge_components: bool = True,
    binarization: str = "sauvola",
) -> None:
    """Run the full page pipeline. See module docstring for sequence."""
    page_name = page_path.stem
    # Tag output dir with the BGR/no-BGR, merge/no-merge, and binarization
    # variants so multiple runs with different settings coexist. Defaults
    # (BGR on, merge on, Sauvola) produce an untagged directory.
    variant_parts = []
    if not use_bgr:
        variant_parts.append("no_bgr")
    if not merge_components:
        variant_parts.append("no_merge")
    if binarization != "sauvola":
        variant_parts.append(binarization)
    variant_suffix = ("_" + "_".join(variant_parts)) if variant_parts else ""
    page_output_dir = output_dir / f"{page_name}{variant_suffix}"
    page_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load page ---
    print(f"Loading page: {page_path}")
    bgr_loaded = cv2.imread(str(page_path))
    if bgr_loaded is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")
    page_rgb = cv2.cvtColor(bgr_loaded, cv2.COLOR_BGR2RGB)
    h, w = page_rgb.shape[:2]
    print(f"  Page size: {w} x {h}")

    # --- Parse YOLO ---
    print(f"Parsing YOLO detections: {yolo_path}")
    all_detections = parse_yolo_txt(yolo_path)
    print(f"  Total detections: {len(all_detections)}")
    stafflines = filter_to_class(all_detections, staffline_class)
    print(f"  Stafflines (class {staffline_class}): {len(stafflines)}")
    if not stafflines:
        print("  No stafflines on this page; nothing to do.")
        return

    # --- Compute scale unit ---
    h_scale = compute_page_scale_unit(stafflines, w, h)
    print(f"  Page scale unit h (median staffline box height): {h_scale:.1f} px")

    # --- BGR or skip ---
    if use_bgr:
        if bgr_model_path is None:
            raise ValueError("BGR is enabled but no --bgr-model was provided.")
        print(f"Loading BGR model: {bgr_model_path}")
        bgr_model = load_bgr_model(str(bgr_model_path), device)
        print("Running BGR inference (this can take a minute on CPU)...")
        page_for_crops = run_bgr_inference(
            bgr_model,
            page_rgb,
            window_size=bgr_window_size,
            stride=bgr_stride,
            confidence=bgr_confidence,
            device=device,
        )
        # Save the BGR-processed page for inspection.
        processed_save_path = page_output_dir / f"{page_name}_bgr.png"
        cv2.imwrite(
            str(processed_save_path), cv2.cvtColor(page_for_crops, cv2.COLOR_RGB2BGR)
        )
        print(f"  Saved BGR-processed page to {processed_save_path}")
    else:
        print("BGR disabled (--no-bgr): cropping directly from original page.")
        page_for_crops = page_rgb

    # --- Per-box processing ---
    summary_rows = []
    fit_results = []  # collected for stage 2 grouping after the per-box loop!
    print(f"Processing {len(stafflines)} staffline boxes...")
    for idx, detection in enumerate(stafflines):
        box = detection.to_pixel_box(w, h)
        crop, actual_box = crop_with_padding(page_for_crops, box, crop_padding)

        if crop.size == 0:
            print(f"  Box {idx}: degenerate crop (empty), skipping.")
            continue

        diag_path = page_output_dir / f"box_{idx:04d}.png"
        result = filter_components(
            crop=crop,
            scale_unit=h_scale,
            save_path=diag_path,
            merge_components=merge_components,
            binarization=binarization,
        )

        # Fit a centerline to the kept pixels. Diagnostic goes to its own PNG.
        fit_diag_path = page_output_dir / f"box_{idx:04d}_fit.png"
        fit_result = fit_centerline(
            filter_result=result,
            scale_unit=h_scale,
            crop=crop,
            save_path=fit_diag_path,
        )
        # Record the crop's page-absolute origin so downstream grouping and
        # visualization can convert crop-local coords to page-absolute.
        fit_result.x_page_offset = float(actual_box[0])  # ulx
        fit_result.y_page_offset = float(actual_box[1])  # uly
        fit_results.append(fit_result)
        summary_rows.append(
            {
                "box_index": idx,
                "ulx": actual_box[0],
                "uly": actual_box[1],
                "lrx": actual_box[2],
                "lry": actual_box[3],
                "n_kept_pixels": len(result.coords),
                "flags": ";".join(result.flags),
                "n_discarded": len(result.discarded),
                "top_score": _top_score_of(result),
                "diagnostic_path": str(diag_path.relative_to(output_dir)),
                "fit_x_start": fit_result.x_start,
                "fit_x_end": fit_result.x_end,
                "fit_residual_mean": round(fit_result.residual_mean, 3),
                "fit_residual_max": round(fit_result.residual_max, 3),
                "fit_flags": ";".join(fit_result.flags),
                "fit_diagnostic_path": str(fit_diag_path.relative_to(output_dir)),
                "stave_id": None,  # to be filled in by stage 2
                "within_stave_index": None,  # to be filled in by stage 2
                "grouping_flags": None,  # to be filled in by stage 2
            }
        )
    # --- Stage 2: stave grouping ---
    grouping_diag_path = page_output_dir / f"{page_name}_stave_grouping.png"
    grouping_result = group_staves(
        fits=fit_results,
        scale_unit=h_scale,
        save_path=grouping_diag_path,
        page_size=(w, h),
        page_image=bgr_loaded,  # Pass the full page image (BGR format for cv2) for visualization
    )
    print(
        f"  Stave grouping: mode={grouping_result.mode_lines_per_stave} "
        f"lines/stave, distribution={grouping_result.line_count_distribution}, "
        f"flags={grouping_result.flags}"
    )

    # Print detailed stave assignments with evidence
    if grouping_result.assignments:
        print(
            f"  Cut threshold: {grouping_result.cut_threshold_px:.1f} px "
            f"(gaps >= this → inter-stave boundary)"
        )
        if grouping_result.gap_distribution:
            print(
                f"  Gaps between consecutive fits: {[f'{g:.0f}' for g in grouping_result.gap_distribution]}"
            )

        print("  Stave assignments:")
        by_stave = {}
        for asg in grouping_result.assignments:
            if asg.stave_id is not None:
                if asg.stave_id not in by_stave:
                    by_stave[asg.stave_id] = []
                by_stave[asg.stave_id].append(asg)

        # Print grouped staves with y-positions
        for stave_id in sorted(by_stave.keys()):
            assignments_in_stave = sorted(
                by_stave[stave_id], key=lambda a: a.within_stave_index
            )
            fit_indices = [str(a.fit_index) for a in assignments_in_stave]
            y_positions = [
                f"{a.y_at_center:.0f}px" if a.y_at_center is not None else "?"
                for a in assignments_in_stave
            ]
            print(f"    Stave {stave_id}: fits [{', '.join(fit_indices)}]")
            print(f"               y-positions: [{', '.join(y_positions)}]")

        # Print unassigned fits if any
        unassigned = [a for a in grouping_result.assignments if a.stave_id is None]
        if unassigned:
            print(
                f"    Unassigned: fits [{', '.join(str(a.fit_index) for a in unassigned)}]"
            )
            for a in unassigned:
                flags_str = f" ({', '.join(a.flags)})" if a.flags else ""
                print(f"               fit {a.fit_index}{flags_str}")

    # Fold stave assignments back into the summary rows.
    for asg in grouping_result.assignments:
        if asg.fit_index < len(summary_rows):
            summary_rows[asg.fit_index]["stave_id"] = asg.stave_id
            summary_rows[asg.fit_index]["within_stave_index"] = asg.within_stave_index
            summary_rows[asg.fit_index]["grouping_flags"] = ";".join(asg.flags)

    # --- Write detailed grouping report ---
    grouping_report_path = page_output_dir / "stave_grouping_report.txt"
    with open(grouping_report_path, "w") as f:
        f.write("STAVE GROUPING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Page: {page_name}\n")
        f.write(f"Mode lines per stave: {grouping_result.mode_lines_per_stave}\n")
        f.write(f"Line count distribution: {grouping_result.line_count_distribution}\n")
        f.write(f"Cut threshold (px): {grouping_result.cut_threshold_px:.1f}\n")
        f.write(
            f"Flags: {', '.join(grouping_result.flags) if grouping_result.flags else 'none'}\n\n"
        )

        f.write("GAPS BETWEEN CONSECUTIVE FITS:\n")
        f.write("-" * 60 + "\n")
        if grouping_result.gap_distribution:
            for i, gap in enumerate(grouping_result.gap_distribution):
                marker = (
                    ">>> INTER-STAVE <<<"
                    if gap >= grouping_result.cut_threshold_px
                    else ""
                )
                f.write(f"  Gap {i}: {gap:.1f} px  {marker}\n")
        else:
            f.write("  (No gaps; 0 or 1 fit)\n")

        f.write("\nSTAVE ASSIGNMENTS:\n")
        f.write("-" * 60 + "\n")

        by_stave = {}
        for asg in grouping_result.assignments:
            if asg.stave_id is not None:
                if asg.stave_id not in by_stave:
                    by_stave[asg.stave_id] = []
                by_stave[asg.stave_id].append(asg)

        for stave_id in sorted(by_stave.keys()):
            assignments_in_stave = sorted(
                by_stave[stave_id], key=lambda a: a.within_stave_index
            )
            f.write(f"\nStave {stave_id}:\n")
            for asg in assignments_in_stave:
                y_str = (
                    f"{asg.y_at_center:.1f}"
                    if asg.y_at_center is not None
                    else "unknown"
                )
                flags_str = f" [{', '.join(asg.flags)}]" if asg.flags else ""
                f.write(
                    f"  Fit {asg.fit_index}: line {asg.within_stave_index}, y={y_str}px{flags_str}\n"
                )

        unassigned = [a for a in grouping_result.assignments if a.stave_id is None]
        if unassigned:
            f.write(f"\nUnassigned fits:\n")
            for asg in unassigned:
                flags_str = f" [{', '.join(asg.flags)}]" if asg.flags else ""
                f.write(f"  Fit {asg.fit_index}{flags_str}\n")

    # --- Write JSOMR JSON (per-line schema, design doc §5.8) ---
    jsomr_path = page_output_dir / f"{page_name}_stafflines.json"
    _write_jsomr_json(
        page_name=page_name,
        stafflines=stafflines,
        summary_rows=summary_rows,
        fit_results=fit_results,
        grouping_result=grouping_result,
        scale_unit=h_scale,
        image_width=w,
        image_height=h,
        save_path=jsomr_path,
    )
    print(f"Wrote JSOMR stafflines: {jsomr_path}")

    # --- Write summary CSV ---
    summary_path = page_output_dir / "summary.csv"
    fieldnames = [
        "box_index",
        "ulx",
        "uly",
        "lrx",
        "lry",
        "n_kept_pixels",
        "flags",
        "n_discarded",
        "top_score",
        "diagnostic_path",
        "fit_x_start",
        "fit_x_end",
        "fit_residual_mean",
        "fit_residual_max",
        "fit_flags",
        "fit_diagnostic_path",
        "stave_id",
        "within_stave_index",
        "grouping_flags",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote per-page summary: {summary_path}")
    print(f"Outputs under: {page_output_dir}")


def _write_jsomr_json(
    page_name: str,
    stafflines: list,
    summary_rows: list[dict],
    fit_results: list,
    grouping_result,
    scale_unit: float,
    image_width: int,
    image_height: int,
    save_path: Path,
) -> None:
    """Write per-line JSOMR records matching the design doc §5.8 schema.

    Each record has: id, source, bounding_box, centerline, fit, quality,
    scale_unit, column_id, stave_id.  Stave assignments are folded in from
    the grouping result.  Interpolated lines are not yet emitted (deferred
    per design doc §6.3).
    """
    # Build a stave-assignment lookup keyed by fit index.
    asg_by_fit: dict[int, object] = {
        a.fit_index: a for a in grouping_result.assignments
    }

    records = []
    for row_idx, row in enumerate(summary_rows):
        fit = fit_results[row_idx]
        asg = asg_by_fit.get(row_idx)

        stave_id = asg.stave_id if asg else None
        grouping_flags = asg.flags if asg else []

        all_flags = (
            [f for f in row["flags"].split(";") if f]
            + [f for f in row["fit_flags"].split(";") if f]
            + grouping_flags
        )

        record = {
            "id": f"{page_name}_line{row_idx:04d}",
            "source": "detected",
            "bounding_box": {
                "ulx": row["ulx"],
                "uly": row["uly"],
                "lrx": row["lrx"],
                "lry": row["lry"],
            },
            "centerline": {
                "x_start": fit.x_start,
                "x_end": fit.x_end,
                "y_values": fit.y_values,
            },
            "centerline_page": {
                "x_start": int(fit.x_start + fit.x_page_offset),
                "x_end":   int(fit.x_end   + fit.x_page_offset),
                "y_values": [round(y + fit.y_page_offset, 1) for y in fit.y_values],
            },
            "fit": {
                "method": "quadratic_huber",
                "coefficients": fit.coefficients,
                "residual_mean": round(fit.residual_mean, 3),
                "residual_max": round(fit.residual_max, 3),
                "n_pixels_used": fit.n_pixels_used,
                "n_pixels_total": fit.n_pixels_total,
            },
            "quality": {
                "confidence": None,  # not yet computed; placeholder per §5.8
                "flags": all_flags,
            },
            "scale_unit": scale_unit,
            "column_id": None,
            "stave_id": stave_id,
            "within_stave_index": asg.within_stave_index if asg else None,
        }
        records.append(record)

    with save_path.open("w") as f:
        json.dump(records, f, indent=2)


def _top_score_of(result) -> Optional[float]:
    """Pull the kept component's total score from a ComponentFilterResult."""
    for entry in result.score_breakdown.values():
        if entry.get("kept"):
            return float(entry["total"])
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 component filter on every staffline detection of a page."
    )
    parser.add_argument("--page", type=Path, required=True, help="Page image path")
    parser.add_argument("--yolo", type=Path, required=True, help="YOLO .txt path")
    parser.add_argument(
        "--bgr-model",
        type=Path,
        required=False,
        help="BGR model checkpoint (required unless --no-bgr is set)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory root"
    )
    parser.add_argument(
        "--staffline-class",
        type=int,
        default=DEFAULT_STAFFLINE_CLASS,
        help=f"YOLO class id for stafflines (default: {DEFAULT_STAFFLINE_CLASS})",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=DEFAULT_CROP_PADDING_PX,
        help=f"Pixels of padding around YOLO box (default: {DEFAULT_CROP_PADDING_PX})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for BGR inference (cuda/cpu)",
    )
    parser.add_argument("--bgr-window-size", type=int, default=DEFAULT_BGR_WINDOW_SIZE)
    parser.add_argument("--bgr-stride", type=int, default=DEFAULT_BGR_STRIDE)
    parser.add_argument("--bgr-confidence", type=float, default=DEFAULT_BGR_CONFIDENCE)
    parser.add_argument(
        "--no-bgr",
        action="store_true",
        help="Skip BGR preprocessing entirely; crop directly from original page.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable component merging; fit to the single highest-scoring "
        "connected component instead of the merged cluster.",
    )
    parser.add_argument(
        "--otsu",
        action="store_true",
        help="Use Otsu global thresholding instead of the default Sauvola. "
        "Retained for comparison; Sauvola is preferred on most manuscripts.",
    )
    args = parser.parse_args()

    use_bgr = not args.no_bgr
    merge_components = not args.no_merge
    binarization = "otsu" if args.otsu else "sauvola"
    if use_bgr and args.bgr_model is None:
        parser.error("--bgr-model is required unless --no-bgr is set.")

    process_page(
        page_path=args.page,
        yolo_path=args.yolo,
        bgr_model_path=args.bgr_model,
        output_dir=args.output,
        staffline_class=args.staffline_class,
        crop_padding=args.crop_padding,
        device=args.device,
        bgr_window_size=args.bgr_window_size,
        bgr_stride=args.bgr_stride,
        bgr_confidence=args.bgr_confidence,
        use_bgr=use_bgr,
        merge_components=merge_components,
        binarization=binarization,
    )


if __name__ == "__main__":
    main()
