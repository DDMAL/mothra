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
        [--staffline-class 0] \\
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
from ultralytics import YOLO

from component_filter import filter_components
from fit_centerline import fit_centerline
from group_staves import group_staves, page_absolute_x_range
from yolo_io import parse_yolo_txt, filter_to_class, YoloDetection
from fallback_redetect import (
    FallbackCandidate,
    identify_probe_regions,
    validate_and_select_candidates,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SF-9 fix: the bundled single-class stave-detector checkpoint (produced by
# train_staffline_detector.py) remaps the merged-project's class 2 down to
# raw class 0 -- its own docstring says explicitly to pass --staffline-class 0
# when piping its output into this pipeline. `2` was the wrong default: a run
# without an explicit override silently matched nothing (every real box is
# class 0), producing an empty stafflines set with no warning.
DEFAULT_STAFFLINE_CLASS = 0
DEFAULT_CROP_PADDING_PX = 2  # small margin around YOLO box; see driver plan

# Mirrors bgr_adapter.py's own defaults, duplicated here rather than imported:
# bgr_adapter is only imported lazily, inside process_page()'s use_bgr branch,
# so --no-bgr (and --help) don't need the external inference_simple dependency
# to be present just to resolve these function/argparse default values.
DEFAULT_BGR_WINDOW_SIZE = 512
DEFAULT_BGR_STRIDE = 256
DEFAULT_BGR_CONFIDENCE = 0.5

# --- Fallback missed-detection re-probe (opt-in via --fallback-redetect) ---
DEFAULT_FALLBACK_CONF = 0.15  # deliberate midpoint between the proven page-wide
# default (0.25) and a page-wide value already shown noisy/fragmentary (0.05) --
# see fallback_redetect.py module docstring and the plan for the full derivation.
DEFAULT_FALLBACK_IOU = 0.7  # matches detect_stafflines.py's own default; a no-op
# only if --fallback-weights points at an end2end (NMS-free) checkpoint.
FALLBACK_IMGSZ = 1280  # matches train_staffline_detector.py's own training imgsz
# ("staff lines are thin horizontal features that need spatial detail") -- the
# probe crop is small enough that this doesn't downsample as aggressively as
# a full page would at the same imgsz.


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
# Fallback missed-detection re-probe (opt-in via --fallback-redetect)
# ---------------------------------------------------------------------------


def _sibling_widths_for_stave(
    stave_id: int,
    grouping_result,
    fit_results: list,
) -> list[float]:
    """Page-absolute x-widths of a stave's own already-detected fits, for
    fallback_redetect.is_plausible_width's relative-width comparison."""
    widths = []
    for asg in grouping_result.assignments:
        if asg.stave_id != stave_id:
            continue
        rng = page_absolute_x_range(fit_results[asg.fit_index])
        if rng is not None:
            widths.append(rng[1] - rng[0])
    return widths


def run_fallback_redetect(
    model: "YOLO",
    bgr_loaded: np.ndarray,
    page_for_crops: np.ndarray,
    fit_results: list,
    summary_rows: list[dict],
    grouping_result,
    scale_unit: float,
    staffline_class: int,
    crop_padding: int,
    merge_components: bool,
    binarization: str,
    page_output_dir: Path,
    output_dir: Path,
    fallback_conf: float,
    fallback_iou: float,
    page_width: int,
) -> tuple[int, list[str]]:
    """Probe under-populated staves for detections the primary pass missed.

    Mutates fit_results/summary_rows in place, appending one entry per
    accepted candidate (matching the schema the main per-box loop in
    process_page() already produces, so the caller's second group_staves()
    call and _write_jsomr_json need no special-casing for these entries).

    Returns (n_added, report_lines) -- report_lines is a per-stave evidence
    trail (territory bounds, h_est, candidates found/passed/accepted,
    cap-exceeded) for fallback_redetect_report.txt.
    """
    regions = identify_probe_regions(
        fits=fit_results,
        assignments=grouping_result.assignments,
        rhythm_anomalies=grouping_result.rhythm_anomalies,
        gap_distribution=grouping_result.gap_distribution,
        scale_unit=scale_unit,
        min_threshold=grouping_result.min_threshold_px,
        cut_threshold=grouping_result.cut_threshold_px,
        mode_n=grouping_result.mode_lines_per_stave,
        page_width=page_width,
    )

    report_lines: list[str] = []
    n_added = 0

    if not regions:
        report_lines.append("No under-populated staves found; nothing to probe.\n")
        return n_added, report_lines

    for region in regions:
        report_lines.append(f"\nStave {region.stave_id}:\n")
        report_lines.append(
            f"  territory: y=[{region.y_start:.1f}, {region.y_end:.1f}], "
            f"x=[{region.x_start:.1f}, {region.x_end:.1f}]\n"
        )
        report_lines.append(
            f"  h_est={region.h_est:.1f}px, lines_observed={region.lines_observed}, "
            f"mode_n={region.mode_n}, max_new_lines={region.max_new_lines}\n"
        )

        y0, y1 = int(region.y_start), int(region.y_end)
        x0, x1 = int(region.x_start), int(region.x_end)
        probe_crop = bgr_loaded[y0:y1, x0:x1]
        if probe_crop.size == 0:
            report_lines.append("  degenerate probe crop (empty); skipping.\n")
            continue

        result = model.predict(
            source=probe_crop,
            conf=fallback_conf,
            iou=fallback_iou,
            imgsz=FALLBACK_IMGSZ,
            save=False,
            verbose=False,
        )[0]

        candidates: list[FallbackCandidate] = []
        candidate_boxes: list[tuple[int, int, int, int]] = []
        # Keyed by id(candidate): box_index (used above, len(fit_results) at
        # probe time) collided every candidate in a region onto the same
        # box_XXXX_fallback.png, each overwriting the last, and the
        # acceptance loop below recomputed a different box_index anyway, so
        # summary_rows ended up pointing at a path nothing had written.
        # Per-region, per-candidate paths are unique from the moment each
        # candidate's diagnostics are written, so the accepted subset can
        # carry its own actual path forward instead of guessing a new one.
        candidate_paths: dict[int, tuple[Path, Path]] = {}
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) != staffline_class:
                    continue
                bx0, by0, bx1, by1 = box.xyxy[0].tolist()
                # Crop-local (within the probe) -> page-absolute.
                page_box = (
                    int(round(bx0)) + x0,
                    int(round(by0)) + y0,
                    int(round(bx1)) + x0,
                    int(round(by1)) + y0,
                )
                crop, actual_box = crop_with_padding(page_for_crops, page_box, crop_padding)
                if crop.size == 0:
                    continue
                cand_index = len(candidates)
                diag_path = page_output_dir / f"fallback_s{region.stave_id:02d}_c{cand_index:02d}.png"
                filter_result = filter_components(
                    crop=crop,
                    scale_unit=scale_unit,
                    save_path=diag_path,
                    merge_components=merge_components,
                    binarization=binarization,
                )
                fit_diag_path = page_output_dir / f"fallback_s{region.stave_id:02d}_c{cand_index:02d}_fit.png"
                fit_result = fit_centerline(
                    filter_result=filter_result,
                    scale_unit=scale_unit,
                    crop=crop,
                    save_path=fit_diag_path,
                )
                fit_result.x_page_offset = float(actual_box[0])
                fit_result.y_page_offset = float(actual_box[1])
                stage1_score = _top_score_of(filter_result) or 0.0
                candidate = FallbackCandidate(
                    fit=fit_result,
                    yolo_confidence=float(box.conf[0]),
                    stage1_score=stage1_score,
                )
                candidates.append(candidate)
                candidate_boxes.append(actual_box)
                candidate_paths[id(candidate)] = (diag_path, fit_diag_path)

        sibling_widths = _sibling_widths_for_stave(region.stave_id, grouping_result, fit_results)
        accepted, cap_exceeded = validate_and_select_candidates(region, candidates, sibling_widths)

        report_lines.append(
            f"  candidates found={len(candidates)}, accepted={len(accepted)}, "
            f"cap_exceeded={cap_exceeded}\n"
        )
        if cap_exceeded:
            report_lines.append(
                f"  FLAG: fallback_found_more_than_expected:{region.stave_id}\n"
            )

        for candidate in accepted:
            fit_result = candidate.fit
            fit_result.flags = list(fit_result.flags) + [
                "fallback_redetected",
                f"fallback_conf:{candidate.yolo_confidence:.3f}",
            ]
            box_index = len(fit_results)
            fit_results.append(fit_result)
            actual_box = (
                int(fit_result.x_page_offset),
                int(fit_result.y_page_offset),
                int(fit_result.x_page_offset + (fit_result.x_end - fit_result.x_start)),
                int(fit_result.y_page_offset + 1),  # box height isn't tracked post-fit; not load-bearing downstream
            )
            cand_diag_path, cand_fit_diag_path = candidate_paths[id(candidate)]
            summary_rows.append(
                {
                    "box_index": box_index,
                    "ulx": actual_box[0],
                    "uly": actual_box[1],
                    "lrx": actual_box[2],
                    "lry": actual_box[3],
                    "n_kept_pixels": None,
                    "flags": "",
                    "n_discarded": None,
                    "top_score": candidate.stage1_score,
                    "diagnostic_path": str(cand_diag_path.relative_to(output_dir)),
                    "fit_x_start": fit_result.x_start,
                    "fit_x_end": fit_result.x_end,
                    "fit_residual_mean": round(fit_result.residual_mean, 3),
                    "fit_residual_max": round(fit_result.residual_max, 3),
                    "fit_flags": ";".join(fit_result.flags),
                    "fit_diagnostic_path": str(cand_fit_diag_path.relative_to(output_dir)),
                    "stave_id": None,
                    "within_stave_index": None,
                    "grouping_flags": None,
                }
            )
            n_added += 1

    return n_added, report_lines


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
    fallback_redetect: bool = False,
    fallback_weights: Optional[Path] = None,
    fallback_conf: float = DEFAULT_FALLBACK_CONF,
    fallback_iou: float = DEFAULT_FALLBACK_IOU,
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
        from bgr_adapter import load_bgr_model, run_bgr_inference

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

    # --- Stage 2.5: fallback missed-detection re-probe (opt-in) ---
    fallback_report_lines: list[str] = []
    if fallback_redetect:
        print("Running fallback missed-detection re-probe...")
        fallback_model = YOLO(str(fallback_weights))
        n_added, fallback_report_lines = run_fallback_redetect(
            model=fallback_model,
            bgr_loaded=bgr_loaded,
            page_for_crops=page_for_crops,
            fit_results=fit_results,
            summary_rows=summary_rows,
            grouping_result=grouping_result,
            scale_unit=h_scale,
            staffline_class=staffline_class,
            crop_padding=crop_padding,
            merge_components=merge_components,
            binarization=binarization,
            page_output_dir=page_output_dir,
            output_dir=output_dir,
            fallback_conf=fallback_conf,
            fallback_iou=fallback_iou,
            page_width=w,
        )
        print(f"  Fallback re-probe: {n_added} line(s) recovered")
        if n_added:
            # Re-run grouping on the augmented fit list. A separate
            # diagnostic path keeps the pre-fallback pass independently
            # inspectable.
            post_fallback_diag_path = page_output_dir / f"{page_name}_stave_grouping_post_fallback.png"
            grouping_result = group_staves(
                fits=fit_results,
                scale_unit=h_scale,
                save_path=post_fallback_diag_path,
                page_size=(w, h),
                page_image=bgr_loaded,
            )
            print(
                f"  Stave grouping (post-fallback): mode={grouping_result.mode_lines_per_stave} "
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

    # --- Write fallback re-probe report (only when the pass actually ran) ---
    if fallback_redetect:
        fallback_report_path = page_output_dir / "fallback_redetect_report.txt"
        with open(fallback_report_path, "w") as f:
            f.write("FALLBACK MISSED-DETECTION RE-PROBE REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"fallback_conf={fallback_conf}, fallback_iou={fallback_iou}\n")
            f.writelines(fallback_report_lines)
        print(f"Wrote fallback re-probe report: {fallback_report_path}")

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
        source = "fallback_redetected" if "fallback_redetected" in all_flags else "detected"

        record = {
            "id": f"{page_name}_line{row_idx:04d}",
            "source": source,
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
                # confidence intentionally left null -- DL-13
                # (ALPHA_TRANSITION_PLAN.md) is still open on whether/how to
                # compute this. Matches landing-page/scripts/staffline_stage.py's
                # identical field (the live landing-app path); this file is the
                # standalone CLI, kept in sync so JSOMR output shape doesn't
                # silently diverge between the two.
                "confidence": None,
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
    parser.add_argument(
        "--fallback-redetect",
        action="store_true",
        help="After grouping, re-probe under-populated staves (a suspiciously "
        "low line count relative to the page mode) for detections the "
        "primary pass may have missed entirely. Off by default. Requires "
        "--fallback-weights.",
    )
    parser.add_argument(
        "--fallback-weights",
        type=Path,
        required=False,
        help="Stave-detector .pt checkpoint for the fallback re-probe "
        "(required when --fallback-redetect is set; typically the same "
        "checkpoint used for the page's primary detection pass).",
    )
    parser.add_argument(
        "--fallback-conf",
        type=float,
        default=DEFAULT_FALLBACK_CONF,
        help=f"Confidence threshold for the fallback re-probe (default: {DEFAULT_FALLBACK_CONF})",
    )
    parser.add_argument(
        "--fallback-iou",
        type=float,
        default=DEFAULT_FALLBACK_IOU,
        help=f"IoU threshold for the fallback re-probe (default: {DEFAULT_FALLBACK_IOU})",
    )
    args = parser.parse_args()

    use_bgr = not args.no_bgr
    merge_components = not args.no_merge
    binarization = "otsu" if args.otsu else "sauvola"
    if use_bgr and args.bgr_model is None:
        parser.error("--bgr-model is required unless --no-bgr is set.")
    if args.fallback_redetect and args.fallback_weights is None:
        parser.error("--fallback-weights is required when --fallback-redetect is set.")

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
        fallback_redetect=args.fallback_redetect,
        fallback_weights=args.fallback_weights,
        fallback_conf=args.fallback_conf,
        fallback_iou=args.fallback_iou,
    )


if __name__ == "__main__":
    main()
