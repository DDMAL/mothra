"""
eval_paco_neume_masking.py — local mock-eval: does masking phase-1 music/text
YOLO boxes out of the paco-classifier's stafflines-only layer approximate
what a real 3-class (background/staff/neume) paco model might do?

## Why this exists

`paco-classifier-service` (see `paco-classifier-service/main.py`) is currently
a 2-class pixel classifier — every pixel is argmax'd between "background"
(`model_0.h5`) and "stafflines" (`model_1.h5`). Neume ink is never isolated
into its own class, so it stays in the stafflines-only layer that
`tasks_predict.py::_run_medieval_inference()` hands to the stave YOLO model.
The `paco-classifier` submodule's own training config
(`paco-classifier/Configs/config.yaml`) supports a genuine 3-way split
(background / staff / neumes — a `model_neumes.h5`), but no such checkpoint
is trained yet. This script is a local, offline mock-up of what removing
neume ink from the stafflines layer might do to stave detection, using the
phase-1 text/music YOLO boxes as a stand-in for what a real 3-class model
would (ideally) segment out on its own.

It never talks to `paco-classifier-service` over HTTP — it calls
`recognition_engine.process_image_msae()` in-process directly, using the same
`[model_0.h5, model_1.h5]` paths and `PATCH_HEIGHT=PATCH_WIDTH=256` as the
service (see `_load_paco_layer` below). This also sidesteps a real bug found
while writing this script: `main.py`'s `/classify` calls
`process_image_msae(..., progress_callback=...)`, a parameter the checked-out
submodule's `process_image_msae()` doesn't accept — filed as
https://github.com/DDMAL/mothra/issues/308 (needs live-request verification,
not fixed here — out of scope for this eval).

## Variants produced, per page

  paco_baseline              — the current, real 2-class stafflines layer,
                                unmodified. Neume ink stays in.
  paco_neume_masked_naive    — every phase-1 music-class YOLO box is painted
                                solid white in the layer before stave
                                detection runs. Rebuilds the "naive" mode
                                from an earlier (now-lost) glyph-masking
                                experiment — see NOTE below.
  paco_neume_masked_band     — same boxes, but a thin horizontal band
                                (height = `--band-multiplier` * scale_unit,
                                centered on the nearest overlapping baseline
                                stave box, falling back to the glyph box's
                                own vertical center) is preserved unmasked
                                inside each box, so real on-line ink isn't
                                deleted along with the neume ink around it.
                                Rebuilds the "band" mode from the same prior
                                experiment.

## NOTE: this rebuilds a lost experiment, on a new base image

`staff-finding/e2e_tests/{gent_right_exp1,ms234_064_exp1}/` (both untracked,
not in git) hold results from an earlier "glyph masking" experiment: masking
YOLO text/music boxes out of a **raw-page** staffline crop, tried as `naive`
vs `band`. `band` matched the `no_bgr` baseline on both pages; `naive`
measurably degraded one page and outright lost a detected line on the more
complex page. The driver script (`scripts/infer_glyph_boxes.py`) and writeup
(`experiments/glyph_masking/NOTES.md`) that produced those results are gone —
not in git history, any stash, or any worktree as of 2026-08-27. The
`naive`/`band` masking logic here is rebuilt from the two READMEs' prose
description, not the original code, and is now applied to the **paco
stafflines layer** as the base image instead of the raw page — that's the
actual novelty being tested here (whether masking still helps/hurts once
paco's own background/stafflines split has already run). Specifically, the
"estimated line" the band mode preserves a strip around is reconstructed as
"the nearest baseline-detected stave box that overlaps this glyph box in x" —
a reasonable reading of the prose, but a reconstruction, not a verified match
to the original algorithm. Treat this script's `_band_center_y()` as the one
piece worth double-checking against intent if the numbers look surprising.

## Noticed issues (local list — not filed, see plan's "Bug tracking" section)

  - paco-classifier-service/main.py's /classify progress_callback mismatch:
    filed as https://github.com/DDMAL/mothra/issues/308 (the one item that
    graduated to a real issue; everything else found stays in this list).
  - (add further items here as they turn up while running this script)

## Usage

    python eval_paco_neume_masking.py \\
        --manifest manifest.csv \\
        --tm-weights ../../landing-page/scripts/assets/models/medieval/text_music_detector_fulldata.pt \\
        --stave-weights ../../landing-page/scripts/assets/models/medieval/stave_detector_fulldata.pt \\
        --paco-models-dir ../../paco-classifier/models_v4 \\
        --output ../e2e_tests/paco_neume_masking_exp1

Manifest CSV columns: page_name,image,gt_txt (see load_manifest below).

No services need to be running — everything executes in-process. Needs
ultralytics/torch (for YOLO) and tensorflow/keras (for the paco classifier)
in the active environment; see CLAUDE.md's "Running locally" section for
which local venvs already have these.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from component_filter import filter_components
from fit_centerline import fit_centerline
from group_staves import group_staves
from yolo_io import YoloDetection
from run_page import (
    compute_page_scale_unit,
    crop_with_padding,
    _write_jsomr_json,
    DEFAULT_CROP_PADDING_PX,
)
from eval_page import CSV_FIELDS, evaluate as eval_page_evaluate, _print_summary
from eval_batch import _print_aggregate_summary

PACO_DIR = Path(__file__).resolve().parent.parent.parent / "paco-classifier"
sys.path.insert(0, str(PACO_DIR))

# ---------------------------------------------------------------------------
# Constants — mirrors paco-classifier-service/main.py and medieval_models.py
# so this stays representative of production behavior.
# ---------------------------------------------------------------------------

PACO_PATCH_HEIGHT = 256
PACO_PATCH_WIDTH = 256
PACO_BACKGROUND_LABEL = 0
PACO_STAFFLINES_LABEL = 1

# medieval_models.py: TEXT_MUSIC_CLASS_MAP = {0: 0, 1: 1} — text_music_detector's
# own raw class ids are already 0=text, 1=music (identity map); running the
# checkpoint directly via ultralytics (not through landing's merged-slot
# logic) means we read box.cls[0] straight off, no remapping needed.
MUSIC_CLASS_ID = 1
# STAVE_CLASS_MAP = {0: 2} — the stave_detector's only raw class (0) maps to
# merged slot 2; run directly, its raw output is always class 0.
STAVE_RAW_CLASS_ID = 0

DEFAULT_TM_CONF = 0.5  # yolo_inference.py's shared text/music default
DEFAULT_STAVE_CONF = 0.25  # yolo_inference.py's DEFAULT_STAVE_CONFIDENCE (SF-1)
DEFAULT_IOU = 0.7

# Band-masking: preserved-strip height, in multiples of the page scale unit
# (median stave-box height). 1.5x gives a bit of margin above/below the
# staff-detector's own box height, in the same spirit as run_page.py's
# DEFAULT_CROP_PADDING_PX margin.
DEFAULT_BAND_MULTIPLIER = 1.5

VARIANTS = ("paco_baseline", "paco_neume_masked_naive", "paco_neume_masked_band")


# ---------------------------------------------------------------------------
# Step 1-2: page load + paco layer (in-process, no HTTP)
# ---------------------------------------------------------------------------


def _load_paco_layer(page_bgr: np.ndarray, models_dir: Path) -> np.ndarray:
    """Run the paco classifier in-process and return the stafflines-only
    layer as an RGB array (white outside the mask), matching
    tasks_predict.py::_decode_paco_layer's channel-order convention exactly.

    Mirrors paco-classifier-service/main.py's classify() + _layer_to_rgba_png
    (main.py:152-173) without the HTTP/SSE/threading wrapper around it, and
    without the broken progress_callback kwarg (see this module's docstring
    and https://github.com/DDMAL/mothra/issues/308).
    """
    from Paco_classifier import recognition_engine

    model_paths = [
        str(models_dir / "model_0.h5"),
        str(models_dir / "model_1.h5"),
    ]
    label_map = recognition_engine.process_image_msae(
        page_bgr, model_paths, PACO_PATCH_HEIGHT, PACO_PATCH_WIDTH, mode="logical",
    )
    if label_map.shape[:2] != page_bgr.shape[:2]:
        raise RuntimeError(
            f"paco label map shape {label_map.shape[:2]} != page shape {page_bgr.shape[:2]}"
        )
    label_range = np.array(PACO_STAFFLINES_LABEL, dtype=np.uint8)
    mask = cv2.inRange(label_map, label_range, label_range)
    layer_bgr = cv2.bitwise_and(page_bgr, page_bgr, mask=mask)
    layer_bgr[mask == 0] = (255, 255, 255)
    return cv2.cvtColor(layer_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Step 3/5: live YOLO detection (mirrors parity_harness.py::detect_live)
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[str, "YOLO"] = {}


def _load_yolo_cached(weights: Path) -> "YOLO":
    """Loading a YOLO checkpoint from disk takes real time; detect_live()
    below is called ~5x per page (music once, stave 3x) across 3 pages, so
    an uncached loader would reload the same two checkpoints 15 times.
    Cached by weights path, module-lifetime -- fine for a short-lived CLI
    run like this one."""
    key = str(weights)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def detect_live(weights: Path, image_rgb: np.ndarray, conf: float, iou: float,
                 device: Optional[str], keep_class_id: int) -> list[YoloDetection]:
    """Run a YOLO checkpoint on `image_rgb`, keeping only `keep_class_id`.

    Ultralytics expects BGR for ndarray sources (see yolo_inference.py's
    to_bgr() comment) -- every array in this script is RGB internally, only
    flipped at this boundary, matching run_page.py/parity_harness.py's own
    convention.
    """
    model = _load_yolo_cached(weights)
    kwargs = dict(source=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                  conf=conf, iou=iou, save=False, verbose=False)
    if device:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    detections: list[YoloDetection] = []
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            if int(box.cls[0]) != keep_class_id:
                continue
            x, y, w, h = box.xywhn[0].tolist()
            detections.append(YoloDetection(keep_class_id, x, y, w, h))
    return detections


def _to_pixel_boxes(detections: list[YoloDetection], w: int, h: int) -> list[tuple[int, int, int, int]]:
    return [d.to_pixel_box(w, h) for d in detections]


# ---------------------------------------------------------------------------
# Step 4: masking
# ---------------------------------------------------------------------------


def _band_center_y(
    box: tuple[int, int, int, int],
    stave_boxes_px: list[tuple[int, int, int, int]],
) -> float:
    """The band-masking "estimated line" position for one glyph box:
    the y-center of the nearest baseline-detected stave box that overlaps
    this glyph box in x, falling back to the glyph box's own y-center when
    no stave box overlaps it at all. See this module's docstring NOTE for
    why this is a reconstruction, not a verified match to the original
    (lost) algorithm.
    """
    ulx, uly, lrx, lry = box
    box_yc = (uly + lry) / 2.0
    best_yc, best_dist = None, None
    for sulx, suly, slrx, slry in stave_boxes_px:
        if slrx <= ulx or sulx >= lrx:
            continue  # no x-overlap
        syc = (suly + slry) / 2.0
        dist = abs(syc - box_yc)
        if best_dist is None or dist < best_dist:
            best_dist, best_yc = dist, syc
    return best_yc if best_yc is not None else box_yc


def mask_boxes_white(
    image_rgb: np.ndarray,
    boxes_px: list[tuple[int, int, int, int]],
    mode: str,
    stave_boxes_px: Optional[list[tuple[int, int, int, int]]] = None,
    scale_unit: Optional[float] = None,
    band_multiplier: float = DEFAULT_BAND_MULTIPLIER,
) -> np.ndarray:
    """Paint music-class glyph boxes white in `image_rgb`.

    mode="naive": the whole box, every time (can delete real on-line ink —
        this is the risky mode the prior experiment found lost a line).
    mode="band": everything in the box EXCEPT a preserved horizontal strip
        of height `band_multiplier * scale_unit` centered on
        `_band_center_y()`'s estimate.
    """
    out = image_rgb.copy()
    h, w = out.shape[:2]
    for box in boxes_px:
        ulx, uly, lrx, lry = box
        ulx, uly = max(0, ulx), max(0, uly)
        lrx, lry = min(w, lrx), min(h, lry)
        if lrx <= ulx or lry <= uly:
            continue
        if mode == "naive":
            out[uly:lry, ulx:lrx] = 255
        elif mode == "band":
            band_yc = _band_center_y(box, stave_boxes_px or [])
            half = (band_multiplier * scale_unit) / 2.0 if scale_unit else (lry - uly) / 2.0
            band_top = int(round(band_yc - half))
            band_bot = int(round(band_yc + half))
            if band_top > uly:
                out[uly:min(band_top, lry), ulx:lrx] = 255
            if band_bot < lry:
                out[max(band_bot, uly):lry, ulx:lrx] = 255
        else:
            raise ValueError(f"unknown mask mode: {mode}")
    return out


# ---------------------------------------------------------------------------
# Step 6: Stage 1/2 (component filter -> centerline fit -> stave grouping)
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    variant: str
    image_rgb: np.ndarray
    stave_boxes_px: list[tuple[int, int, int, int]]
    scale_unit: float
    fit_results: list
    boxes: list[tuple[int, int, int, int]]
    filter_results: list
    grouping_result: object
    jsomr_path: Path


def run_stage_1_2(
    variant: str,
    image_rgb: np.ndarray,
    detections: list[YoloDetection],
    page_name: str,
    page_output_dir: Path,
) -> VariantResult:
    """Component filter -> centerline fit -> stave grouping, one variant.
    Mirrors run_page.py::process_page's per-box loop + group_staves call,
    but writes none of run_page.py's per-box diagnostic PNGs (see this
    module's docstring on the deliberately small diagnostic set)."""
    h, w = image_rgb.shape[:2]
    scale_unit = compute_page_scale_unit(detections, w, h)

    fit_results, boxes, filter_results = [], [], []
    for det in detections:
        box = det.to_pixel_box(w, h)
        crop, actual_box = crop_with_padding(image_rgb, box, DEFAULT_CROP_PADDING_PX)
        if crop.size == 0:
            continue
        filter_result = filter_components(crop, scale_unit=scale_unit)
        fit_result = fit_centerline(filter_result=filter_result, scale_unit=scale_unit, crop=crop)
        fit_result.x_page_offset = float(actual_box[0])
        fit_result.y_page_offset = float(actual_box[1])
        fit_results.append(fit_result)
        boxes.append(actual_box)
        filter_results.append(filter_result)

    grouping_diag_path = page_output_dir / f"{page_name}_{variant}_5_grouping.png"
    grouping_result = group_staves(
        fits=fit_results, scale_unit=scale_unit, save_path=grouping_diag_path,
        page_size=(w, h), page_image=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )

    summary_rows = []
    for idx, (fit, box, fres) in enumerate(zip(fit_results, boxes, filter_results)):
        summary_rows.append({
            "box_index": idx, "ulx": box[0], "uly": box[1], "lrx": box[2], "lry": box[3],
            "n_kept_pixels": len(fres.coords), "flags": ";".join(fres.flags),
            "n_discarded": len(fres.discarded), "top_score": None,
            "diagnostic_path": "", "fit_x_start": fit.x_start, "fit_x_end": fit.x_end,
            "fit_residual_mean": round(fit.residual_mean, 3),
            "fit_residual_max": round(fit.residual_max, 3),
            "fit_flags": ";".join(fit.flags), "fit_diagnostic_path": "",
            "stave_id": None, "within_stave_index": None, "grouping_flags": None,
        })
    for asg in grouping_result.assignments:
        if asg.fit_index < len(summary_rows):
            summary_rows[asg.fit_index]["stave_id"] = asg.stave_id
            summary_rows[asg.fit_index]["within_stave_index"] = asg.within_stave_index
            summary_rows[asg.fit_index]["grouping_flags"] = ";".join(asg.flags)

    jsomr_path = page_output_dir / f"{page_name}_{variant}_stafflines.json"
    _write_jsomr_json(
        page_name=f"{page_name}_{variant}", stafflines=detections, summary_rows=summary_rows,
        fit_results=fit_results, grouping_result=grouping_result, scale_unit=scale_unit,
        image_width=w, image_height=h, save_path=jsomr_path,
    )

    return VariantResult(
        variant=variant, image_rgb=image_rgb,
        stave_boxes_px=[d.to_pixel_box(w, h) for d in detections],
        scale_unit=scale_unit, fit_results=fit_results, boxes=boxes,
        filter_results=filter_results, grouping_result=grouping_result, jsomr_path=jsomr_path,
    )


# ---------------------------------------------------------------------------
# Step 8: diagnostics (small, fixed set -- see docstring)
# ---------------------------------------------------------------------------


def write_diagnostics(vr: VariantResult, music_boxes_px: list, page_name: str, page_output_dir: Path) -> None:
    bgr = cv2.cvtColor(vr.image_rgb, cv2.COLOR_RGB2BGR)

    # 1. Separated layer itself.
    cv2.imwrite(str(page_output_dir / f"{page_name}_{vr.variant}_1_layer.png"), bgr)

    # 2. Stave box placement (+ the music boxes that drove any masking, in blue).
    box_overlay = bgr.copy()
    for ulx, uly, lrx, lry in vr.stave_boxes_px:
        cv2.rectangle(box_overlay, (ulx, uly), (lrx, lry), (0, 200, 0), 2)
    for ulx, uly, lrx, lry in music_boxes_px:
        cv2.rectangle(box_overlay, (ulx, uly), (lrx, lry), (255, 120, 0), 1)
    cv2.imwrite(str(page_output_dir / f"{page_name}_{vr.variant}_2_boxes.png"), box_overlay)

    # 3. Component-filter overview: kept pixels (green) + discarded
    # components' bounding boxes (red), composited at page location -- one
    # file, not one per box. `coords` is the kept (active) component/cluster's
    # pixel list (component_filter.py's ComponentFilterResult.coords);
    # `discarded` entries carry only crop-local bbox stats (x,y,w,h), not a
    # pixel list, so discarded components are drawn as rectangles, not a
    # pixel mask -- see component_filter.py:216-221.
    filter_overlay = bgr.copy()
    for fres, box in zip(vr.filter_results, vr.boxes):
        ulx, uly, _, _ = box
        for x, y in fres.coords:
            fy, fx = uly + int(y), ulx + int(x)
            if 0 <= fy < filter_overlay.shape[0] and 0 <= fx < filter_overlay.shape[1]:
                filter_overlay[fy, fx] = (0, 220, 0)
        for entry in fres.discarded:
            rx0, ry0 = ulx + entry["x"], uly + entry["y"]
            rx1, ry1 = rx0 + entry["w"], ry0 + entry["h"]
            cv2.rectangle(filter_overlay, (rx0, ry0), (rx1, ry1), (0, 0, 220), 1)
    cv2.imwrite(str(page_output_dir / f"{page_name}_{vr.variant}_3_component_filter.png"), filter_overlay)

    # 4. Centerline fits, page-absolute, composited.
    fit_overlay = bgr.copy()
    for fit in vr.fit_results:
        if not fit.y_values:
            continue
        xs = np.arange(fit.x_start, fit.x_start + len(fit.y_values)) + fit.x_page_offset
        ys = np.asarray(fit.y_values) + fit.y_page_offset
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(fit_overlay, [pts], isClosed=False, color=(0, 140, 255), thickness=2)
    cv2.imwrite(str(page_output_dir / f"{page_name}_{vr.variant}_4_centerlines.png"), fit_overlay)

    # 5. Stave grouping -- already written by group_staves(save_path=...) as
    # f"{page_name}_{variant}_5_grouping.png" inside run_stage_1_2().


# ---------------------------------------------------------------------------
# Per-page orchestration
# ---------------------------------------------------------------------------


def process_page(
    page_name: str, image_path: Path, gt_path: Path, output_dir: Path,
    tm_weights: Path, stave_weights: Path, paco_models_dir: Path,
    tm_conf: float, stave_conf: float, iou: float, device: Optional[str],
    band_multiplier: float,
) -> list[dict]:
    print(f"\n=== {page_name} ===")
    page_output_dir = output_dir / page_name
    page_output_dir.mkdir(parents=True, exist_ok=True)

    page_bgr = cv2.imread(str(image_path))
    if page_bgr is None:
        raise FileNotFoundError(f"could not read page image: {image_path}")
    page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    h, w = page_rgb.shape[:2]

    print("  Running paco classifier (in-process)...")
    layer_baseline_rgb = _load_paco_layer(page_bgr, paco_models_dir)

    print("  Detecting music/text boxes on the raw page...")
    music_detections = detect_live(tm_weights, page_rgb, tm_conf, iou, device, MUSIC_CLASS_ID)
    music_boxes_px = _to_pixel_boxes(music_detections, w, h)
    print(f"    {len(music_boxes_px)} music-class box(es)")

    print("  Detecting stave boxes on the unmasked paco layer (baseline)...")
    baseline_stave_dets = detect_live(stave_weights, layer_baseline_rgb, stave_conf, iou, device, STAVE_RAW_CLASS_ID)
    baseline_stave_px = _to_pixel_boxes(baseline_stave_dets, w, h)
    scale_unit = compute_page_scale_unit(baseline_stave_dets, w, h)
    print(f"    {len(baseline_stave_px)} stave box(es), scale_unit={scale_unit:.1f}px")

    layer_naive_rgb = mask_boxes_white(layer_baseline_rgb, music_boxes_px, mode="naive")
    layer_band_rgb = mask_boxes_white(
        layer_baseline_rgb, music_boxes_px, mode="band",
        stave_boxes_px=baseline_stave_px, scale_unit=scale_unit, band_multiplier=band_multiplier,
    )

    variant_images = {
        "paco_baseline": (layer_baseline_rgb, baseline_stave_dets),
        "paco_neume_masked_naive": (layer_naive_rgb, None),
        "paco_neume_masked_band": (layer_band_rgb, None),
    }

    rows = []
    for variant in VARIANTS:
        image_rgb, precomputed_dets = variant_images[variant]
        if precomputed_dets is None:
            print(f"  Detecting stave boxes on {variant}...")
            precomputed_dets = detect_live(stave_weights, image_rgb, stave_conf, iou, device, STAVE_RAW_CLASS_ID)
        print(f"  Running Stage 1/2 for {variant} ({len(precomputed_dets)} stave box(es))...")
        vr = run_stage_1_2(variant, image_rgb, precomputed_dets, page_name, page_output_dir)
        write_diagnostics(vr, music_boxes_px, page_name, page_output_dir)

        metrics = eval_page_evaluate(
            gt_path=gt_path, pred_path=vr.jsomr_path, image_path=image_path,
            staffline_class=0, gt_source="corrected", variant=variant, page_name=page_name,
        )
        _print_summary(metrics)
        rows.append(metrics)

    return rows


# ---------------------------------------------------------------------------
# Manifest + CLI
# ---------------------------------------------------------------------------

REQUIRED_MANIFEST_COLS = {"page_name", "image", "gt_txt"}


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest {manifest_path} is empty.")
    missing = REQUIRED_MANIFEST_COLS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True, help="CSV: page_name,image,gt_txt")
    p.add_argument("--tm-weights", type=Path, required=True, help="text_music_detector .pt")
    p.add_argument("--stave-weights", type=Path, required=True, help="stave_detector .pt")
    p.add_argument("--paco-models-dir", type=Path, required=True, help="dir with model_0.h5/model_1.h5")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tm-conf", type=float, default=DEFAULT_TM_CONF)
    p.add_argument("--stave-conf", type=float, default=DEFAULT_STAVE_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--device", default=None)
    p.add_argument("--band-multiplier", type=float, default=DEFAULT_BAND_MULTIPLIER)
    args = p.parse_args()

    manifest_rows = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for row in manifest_rows:
        page_name = row["page_name"].strip()
        image_path = Path(row["image"].strip())
        gt_path = Path(row["gt_txt"].strip())
        if not image_path.exists() or not gt_path.exists():
            print(f"SKIP {page_name}: missing image or gt_txt ({image_path}, {gt_path})")
            continue
        try:
            all_rows.extend(process_page(
                page_name=page_name, image_path=image_path, gt_path=gt_path,
                output_dir=args.output, tm_weights=args.tm_weights,
                stave_weights=args.stave_weights, paco_models_dir=args.paco_models_dir,
                tm_conf=args.tm_conf, stave_conf=args.stave_conf, iou=args.iou,
                device=args.device, band_multiplier=args.band_multiplier,
            ))
        except Exception as exc:
            print(f"ERROR on {page_name}: {exc}")
            raise

    out_csv = args.output / "eval_batch_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} row(s) to {out_csv}")
    if all_rows:
        _print_aggregate_summary(all_rows)


if __name__ == "__main__":
    main()
