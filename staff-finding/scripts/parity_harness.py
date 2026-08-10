"""
Parity harness: measure landing-vs-standalone staffline-pipeline divergences.

Runs the SAME Stage 1/2 pipeline (component filter -> centerline fit -> stave
grouping) under controlled toggles, each isolating one known divergence
between mothra-landing's invocation (landing-page/scripts/staffline_stage.py)
and the standalone driver (run_page.py / test_model.sh "basic settings"):

  toggle              divergence (ALPHA_TRANSITION_PLAN.md findings register)
  ------------------  ----------------------------------------------------
  --no-pass-crop      SF-5 (D1): fit_centerline called without `crop`, which
                      changes the line-following seed_y
  --channel-order bgr SF-4 (D5c): landing hands component_filter a BGR array;
                      _binarize hardcodes COLOR_RGB2GRAY
  --image working-copy SF-2 (D5a): landing runs on the client-side resized,
                      re-encoded JPEG working copy (imageResize.ts semantics)
  --image paco-layer  SF-3 (D5b): landing crops the paco-classifier
                      stafflines-only layer, not the raw page
  --conf 0.5          SF-1 (D3): landing's default stave confidence is 0.5;
                      the validated standalone runs used 0.25

Baseline semantics (all toggles at standalone settings) reproduce
run_page.py --no-bgr exactly: image loaded BGR->RGB, sauvola binarization,
merge_components on, crop padding 2, fit_centerline WITH crop, group_staves
with median cut threshold and no interpolation.

Usage (one variant):
    python parity_harness.py --page page.jpg --yolo page.txt \
        --staffline-class 0 --output out/ --label baseline

    python parity_harness.py --page page.jpg --yolo page.txt \
        --staffline-class 0 --output out/ --label no_crop --no-pass-crop \
        --baseline out/baseline.json

Usage (full attribution sweep -- one flip at a time, plus a landing-exact
combined run; writes <output>/report.md):
    python parity_harness.py --page page.jpg --yolo page.txt \
        --staffline-class 0 --output out/ --sweep \
        [--weights models/stave_detector_fulldata.pt]   # enables conf variants
        [--paco-url http://localhost:8003]              # enables paco variants

Notes:
  - Detections come from --yolo (fixed box set; conf toggles are then
    unavailable, since YOLO txt carries no confidences) or live from
    --weights at --conf.
  - The paco variants need paco-classifier-service running (default
    http://localhost:8003, override with --paco-url); they are skipped with
    a note if the service is unreachable.
  - Working-copy simulation mirrors landing-page/src/utils/imageResize.ts:
    files <= 5 MB are passed through untouched; larger files are scaled by
    sqrt(2MB/size) and JPEG-re-encoded starting at quality 90, stepping down
    exactly as the frontend does.
  - When a variant changes image dimensions (working copy), its page-absolute
    coordinates are rescaled into the baseline frame before comparison.

This harness intentionally does NOT import run_page.py (its module-level
bgr_adapter import chain requires an unvendored external repo) nor anything
from landing-page/ (separate package). It re-states both invocations from
first principles against the shared package modules, with file:line
references to what it mirrors.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from component_filter import filter_components
from fit_centerline import fit_centerline
from group_staves import group_staves
from yolo_io import parse_yolo_txt, filter_to_class, YoloDetection

CROP_PADDING_PX = 2  # run_page.py DEFAULT_CROP_PADDING_PX == staffline_stage.CROP_PADDING_PX
MATCH_THRESHOLD_SCALE = 0.5  # line-match tolerance, in units of baseline scale_unit
                             # (same convention as eval_page.py)

# imageResize.ts constants (landing-page/src/utils/imageResize.ts:1-2)
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
TARGET_RESIZE_BYTES = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Image preparation variants
# ---------------------------------------------------------------------------


def _load_original_rgb(page_path: Path) -> np.ndarray:
    """Exactly run_page.py:377-380 -- cv2.imread then BGR->RGB."""
    bgr = cv2.imread(str(page_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _simulate_working_copy(page_path: Path) -> tuple[bytes, str]:
    """Mirror imageResize.ts resizeImageFile(): returns (jpeg_bytes, note).

    Files at or under MAX_IMAGE_SIZE_BYTES are returned untouched (the
    frontend only offers the resize modal above that threshold)."""
    raw = page_path.read_bytes()
    if len(raw) <= MAX_IMAGE_SIZE_BYTES:
        return raw, f"untouched ({len(raw)} B <= 5 MB threshold)"

    bgr = cv2.imread(str(page_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")

    scale = min(1.0, (TARGET_RESIZE_BYTES / len(raw)) ** 0.5)
    quality = 0.9
    encoded: Optional[bytes] = None
    for _attempt in range(6):
        width = max(1, round(bgr.shape[1] * scale))
        height = max(1, round(bgr.shape[0] * scale))
        resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(round(quality * 100))])
        if not ok:
            raise RuntimeError("cv2.imencode failed while simulating working copy")
        encoded = buf.tobytes()
        if len(encoded) <= TARGET_RESIZE_BYTES:
            break
        if quality > 0.5:
            quality -= 0.15
        else:
            scale *= 0.8
    note = (
        f"resized {bgr.shape[1]}x{bgr.shape[0]} -> scale {scale:.3f}, "
        f"final quality {quality:.2f}, {len(encoded)} B"
    )
    return encoded, note


def _fetch_paco_layer(image_bytes: bytes, mime_type: str, paco_url: str) -> bytes:
    """POST one page to paco-classifier-service /classify; return the
    stafflines-layer PNG bytes. Mirrors landing-page/scripts/paco_api.py's
    request shape (multipart field name 'image')."""
    boundary = uuid.uuid4().hex
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="image"; filename="page.png"\r\n'
    body += f"Content-Type: {mime_type}\r\n\r\n".encode()
    body += image_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        paco_url.rstrip("/") + "/classify",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode())
    return base64.b64decode(payload["stafflines_png_base64"])


@dataclass
class PreparedImage:
    array_rgb: np.ndarray          # always true RGB channel order
    notes: list[str] = field(default_factory=list)


def prepare_image(page_path: Path, image_variant: str, paco_base: str,
                  paco_url: str) -> PreparedImage:
    notes: list[str] = []
    if image_variant == "original":
        return PreparedImage(_load_original_rgb(page_path), ["original file"])

    if image_variant == "working-copy":
        jpeg_bytes, note = _simulate_working_copy(page_path)
        notes.append(f"working-copy: {note}")
        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        return PreparedImage(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), notes)

    if image_variant == "paco-layer":
        if paco_base == "working-copy":
            base_bytes, note = _simulate_working_copy(page_path)
            mime = "image/jpeg"
            notes.append(f"paco base = working-copy ({note})")
        else:
            base_bytes = page_path.read_bytes()
            mime = "image/jpeg" if page_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            notes.append("paco base = original file")
        layer_png = _fetch_paco_layer(base_bytes, mime, paco_url)
        # Landing decodes with IMREAD_COLOR (BGR, alpha dropped) --
        # tasks_predict.py:64. We decode the same way, then convert to true
        # RGB so --channel-order alone controls what the pipeline sees.
        bgr = cv2.imdecode(np.frombuffer(layer_png, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("could not decode stafflines PNG from paco-classifier-service")
        notes.append("paco stafflines layer (alpha dropped, as landing does)")
        return PreparedImage(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), notes)

    raise ValueError(f"unknown image variant: {image_variant}")


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


def detect_live(weights: Path, image_rgb: np.ndarray, conf: float, iou: float,
                device: Optional[str]) -> list[YoloDetection]:
    """Run the stave detector on the prepared image at the given confidence.

    Ultralytics expects BGR for ndarray sources, so we hand it BGR -- this
    matches both detect_stafflines.py (file path input, decoded BGR
    internally) and landing's paco-layer call (already-BGR array)."""
    from ultralytics import YOLO  # deferred: only needed for conf variants

    model = YOLO(str(weights))
    kwargs = dict(source=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                  conf=conf, iou=iou, save=False, verbose=False)
    if device:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    detections: list[YoloDetection] = []
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            x, y, w, h = box.xywhn[0].tolist()
            detections.append(YoloDetection(int(box.cls[0]), x, y, w, h))
    return detections


# ---------------------------------------------------------------------------
# The shared pipeline core, with the two invocation styles
# ---------------------------------------------------------------------------


def compute_page_scale_unit(detections: list[YoloDetection], w: int, h: int) -> float:
    """Verbatim run_page.py compute_page_scale_unit / staffline_stage
    _compute_page_scale_unit (identical on both sides)."""
    heights = []
    for d in detections:
        _, uly, _, lry = d.to_pixel_box(w, h)
        heights.append(lry - uly)
    return float(np.median(heights)) if heights else 0.0


def crop_with_padding(image: np.ndarray, box, padding: int):
    h, w = image.shape[:2]
    ulx, uly, lrx, lry = box
    ulx_p, uly_p = max(0, ulx - padding), max(0, uly - padding)
    lrx_p, lry_p = min(w, lrx + padding), min(h, lry + padding)
    return image[uly_p:lry_p, ulx_p:lrx_p], (ulx_p, uly_p, lrx_p, lry_p)


def run_variant(image_arr: np.ndarray, detections: list[YoloDetection],
                pass_crop: bool) -> dict:
    """Run Stage 1/2 once. `image_arr` is handed to the pipeline exactly as
    given (the caller controls channel order); `pass_crop` toggles SF-5.

    With pass_crop=True this is run_page.py:430-453's per-box loop; with
    pass_crop=False it is staffline_stage.py:210-222's."""
    h, w = image_arr.shape[:2]
    scale_unit = compute_page_scale_unit(detections, w, h)

    fit_results, boxes = [], []
    for det in detections:
        box = det.to_pixel_box(w, h)
        crop, actual_box = crop_with_padding(image_arr, box, CROP_PADDING_PX)
        if crop.size == 0:
            continue
        filter_result = filter_components(crop, scale_unit=scale_unit)
        if pass_crop:
            fit = fit_centerline(filter_result=filter_result, scale_unit=scale_unit, crop=crop)
        else:
            fit = fit_centerline(filter_result, scale_unit=scale_unit)
        fit.x_page_offset = float(actual_box[0])
        fit.y_page_offset = float(actual_box[1])
        fit_results.append(fit)
        boxes.append(actual_box)

    if not fit_results:
        return {"image_size": [w, h], "scale_unit": scale_unit, "n_boxes": len(detections),
                "n_fits": 0, "lines": [], "stave_count": 0, "mode_lines_per_stave": None,
                "line_count_distribution": {}, "cut_threshold_px": None, "grouping_flags": []}

    grouping = group_staves(fit_results, scale_unit=scale_unit)
    asg_by_fit = {a.fit_index: a for a in grouping.assignments}

    lines = []
    for idx, (fit, box) in enumerate(zip(fit_results, boxes)):
        asg = asg_by_fit.get(idx)
        ys = np.asarray(fit.y_values, dtype=float) + fit.y_page_offset
        lines.append({
            "fit_index": idx,
            "bounding_box": list(box),
            "x_start_page": float(fit.x_start + fit.x_page_offset),
            "x_end_page": float(fit.x_end + fit.x_page_offset),
            "y_mid_page": float(ys[len(ys) // 2]) if len(ys) else None,
            "y_mean_page": float(ys.mean()) if len(ys) else None,
            "stave_id": asg.stave_id if asg else None,
            "within_stave_index": asg.within_stave_index if asg else None,
            "fit_flags": list(fit.flags),
            "grouping_flags": list(asg.flags) if asg else [],
            "residual_mean": round(float(fit.residual_mean), 3),
        })

    stave_ids = {ln["stave_id"] for ln in lines if ln["stave_id"] is not None}
    return {
        "image_size": [w, h],
        "scale_unit": scale_unit,
        "n_boxes": len(detections),
        "n_fits": len(fit_results),
        "n_assigned": sum(1 for ln in lines if ln["stave_id"] is not None),
        "stave_count": len(stave_ids),
        "mode_lines_per_stave": grouping.mode_lines_per_stave,
        "line_count_distribution": {str(k): v for k, v in grouping.line_count_distribution.items()},
        "cut_threshold_px": round(float(grouping.cut_threshold_px), 1),
        "grouping_flags": list(grouping.flags),
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(baseline: dict, variant: dict) -> dict:
    """Match assigned lines by page-absolute y (variant rescaled into the
    baseline frame when dimensions differ), greedy nearest within
    MATCH_THRESHOLD_SCALE x baseline scale_unit; report y-MAE + unmatched."""
    h_base = baseline["image_size"][1]
    h_var = variant["image_size"][1]
    y_scale = h_base / h_var if h_var else 1.0

    base_ys = sorted(
        ln["y_mid_page"] for ln in baseline["lines"]
        if ln["stave_id"] is not None and ln["y_mid_page"] is not None
    )
    var_ys = sorted(
        ln["y_mid_page"] * y_scale for ln in variant["lines"]
        if ln["stave_id"] is not None and ln["y_mid_page"] is not None
    )
    threshold = MATCH_THRESHOLD_SCALE * baseline["scale_unit"]

    used = [False] * len(var_ys)
    abs_errors, unmatched_base = [], 0
    for by in base_ys:
        best_j, best_d = None, None
        for j, vy in enumerate(var_ys):
            if used[j]:
                continue
            d = abs(vy - by)
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j is not None and best_d is not None and best_d <= threshold:
            used[best_j] = True
            abs_errors.append(best_d)
        else:
            unmatched_base += 1
    unmatched_variant = used.count(False)

    return {
        "matched": len(abs_errors),
        "unmatched_baseline": unmatched_base,
        "unmatched_variant": unmatched_variant,
        "y_mae_px": round(float(np.mean(abs_errors)), 2) if abs_errors else None,
        "y_max_px": round(float(np.max(abs_errors)), 2) if abs_errors else None,
        "match_threshold_px": round(threshold, 1),
        "y_rescale_factor": round(y_scale, 4),
        "stave_count_delta": variant["stave_count"] - baseline["stave_count"],
        "mode_delta": (
            (variant["mode_lines_per_stave"] or 0) - (baseline["mode_lines_per_stave"] or 0)
        ),
    }


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

SWEEP_ORDER = [
    # label, description, overrides {image, channel, pass_crop, conf}
    ("baseline", "standalone settings: original image, RGB, crop seed", {}),
    ("no_crop_seed", "SF-5/D1: fit_centerline without crop", {"pass_crop": False}),
    ("bgr_channel", "SF-4/D5c: BGR array into RGB2GRAY binarize", {"channel": "bgr"}),
    ("working_copy", "SF-2/D5a: client-resize simulated working copy", {"image": "working-copy"}),
    ("paco_layer", "SF-3/D5b: paco stafflines layer (RGB, crop seed)", {"image": "paco-layer"}),
    ("conf_landing", "SF-1/D3: detection at landing default conf 0.5", {"conf": 0.5}),
    ("landing_exact", "all landing settings combined",
     {"image": "paco-layer", "paco_base": "working-copy", "channel": "bgr",
      "pass_crop": False, "conf": 0.5}),
]


def _report_row(label: str, desc: str, v: dict, cmp_: Optional[dict]) -> str:
    dist = ",".join(f"{k}:{n}" for k, n in sorted(v["line_count_distribution"].items()))
    if cmp_ is None:
        tail = "— (baseline)"
    else:
        tail = (f"Δstaves {cmp_['stave_count_delta']:+d} · Δmode {cmp_['mode_delta']:+d} · "
                f"matched {cmp_['matched']} (yMAE {cmp_['y_mae_px']} px, max {cmp_['y_max_px']}) · "
                f"lost {cmp_['unmatched_baseline']} / new {cmp_['unmatched_variant']}")
    return (f"| {label} | {v['n_fits']}/{v['n_boxes']} | {v['stave_count']} | "
            f"{v['mode_lines_per_stave']} | {dist} | {v['cut_threshold_px']} | {tail} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--page", type=Path, required=True, help="Page image file.")
    p.add_argument("--yolo", type=Path, help="YOLO .txt detections (fixed box set).")
    p.add_argument("--weights", type=Path, help="Stave-detector .pt for live detection.")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence (with --weights).")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU (with --weights).")
    p.add_argument("--device", default=None, help="Detection device (with --weights).")
    p.add_argument("--staffline-class", type=int, default=0,
                   help="Class id to keep from --yolo (0 for detect_stafflines.py output, "
                        "2 for landing-produced merged annotations).")
    p.add_argument("--image", choices=["original", "working-copy", "paco-layer"],
                   default="original")
    p.add_argument("--paco-base", choices=["original", "working-copy"], default="original",
                   help="Which bytes to send to the paco classifier for --image paco-layer.")
    p.add_argument("--paco-url", default="http://localhost:8003")
    p.add_argument("--channel-order", choices=["rgb", "bgr"], default="rgb",
                   help="Channel order of the array handed to the pipeline "
                        "(rgb = standalone-correct; bgr = landing's actual behavior).")
    p.add_argument("--no-pass-crop", action="store_true",
                   help="Call fit_centerline without crop (landing behavior, SF-5).")
    p.add_argument("--label", default="variant", help="Name for this run's output JSON.")
    p.add_argument("--baseline", type=Path, help="A previous run's JSON to diff against.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sweep", action="store_true",
                   help="Run the standard attribution matrix and write report.md.")
    args = p.parse_args()

    if not args.yolo and not args.weights:
        p.error("need --yolo (fixed boxes) or --weights (live detection)")
    args.output.mkdir(parents=True, exist_ok=True)

    fixed_detections = None
    if args.yolo:
        fixed_detections = filter_to_class(parse_yolo_txt(args.yolo), args.staffline_class)
        print(f"Fixed detections: {len(fixed_detections)} (class {args.staffline_class})")

    def build_and_run(image_variant: str, paco_base: str, channel: str,
                      pass_crop: bool, conf: Optional[float]) -> dict:
        prep = prepare_image(args.page, image_variant, paco_base, args.paco_url)
        if conf is not None:
            if not args.weights:
                raise RuntimeError("conf variant requested but no --weights given")
            dets = detect_live(args.weights, prep.array_rgb, conf, args.iou, args.device)
            det_note = f"live detection conf={conf} iou={args.iou}: {len(dets)} boxes"
        else:
            assert fixed_detections is not None, "fixed-box variant needs --yolo"
            dets = fixed_detections
            det_note = f"fixed boxes from {args.yolo.name}: {len(dets)}"
        arr = prep.array_rgb if channel == "rgb" else prep.array_rgb[:, :, ::-1].copy()
        result = run_variant(arr, dets, pass_crop)
        result["settings"] = {
            "image": image_variant, "paco_base": paco_base, "channel_order": channel,
            "pass_crop": pass_crop, "conf": conf,
            "notes": prep.notes + [det_note],
        }
        return result

    if not args.sweep:
        result = build_and_run(
            args.image, args.paco_base, args.channel_order,
            pass_crop=not args.no_pass_crop,
            conf=None if args.yolo else args.conf,
        )
        if args.baseline:
            base = json.loads(args.baseline.read_text())
            result["comparison_vs_baseline"] = compare(base, result)
        out = args.output / f"{args.label}.json"
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "lines"}, indent=2))
        print(f"Wrote {out}")
        return

    # --- sweep ---
    results: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    for label, desc, ov in SWEEP_ORDER:
        image_variant = ov.get("image", "original")
        conf = ov.get("conf")
        if conf is not None and not args.weights:
            skipped.append((label, "needs --weights"))
            continue
        if conf is None and fixed_detections is None:
            skipped.append((label, "needs --yolo"))
            continue
        print(f"\n=== {label}: {desc}")
        try:
            results[label] = build_and_run(
                image_variant, ov.get("paco_base", args.paco_base),
                ov.get("channel", "rgb"), ov.get("pass_crop", True), conf,
            )
        except Exception as exc:  # paco down, weights missing, etc. -- keep sweeping
            print(f"    SKIPPED: {exc}")
            skipped.append((label, str(exc)))
            continue
        (args.output / f"{label}.json").write_text(json.dumps(results[label], indent=2))

    if "baseline" not in results:
        print("Baseline variant failed; no report possible.", file=sys.stderr)
        sys.exit(1)
    base = results["baseline"]

    lines = [
        f"# Parity sweep: {args.page.name}",
        "",
        f"- page: `{args.page}`",
        f"- detections: " + (f"`{args.yolo}` (class {args.staffline_class})" if args.yolo else "live"),
        f"- baseline scale_unit: {base['scale_unit']:.1f} px · image {base['image_size'][0]}x{base['image_size'][1]}",
        "",
        "| variant | fits/boxes | staves | mode | distribution | cut px | vs baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, desc, _ov in SWEEP_ORDER:
        if label not in results:
            continue
        v = results[label]
        cmp_ = None if label == "baseline" else compare(base, v)
        if cmp_ is not None:
            v["comparison_vs_baseline"] = cmp_
            (args.output / f"{label}.json").write_text(json.dumps(v, indent=2))
        lines.append(_report_row(label, desc, v, cmp_))
    lines.append("")
    for label, desc, _ov in SWEEP_ORDER:
        if label in results:
            lines.append(f"- **{label}** — {desc}")
    if skipped:
        lines.append("")
        lines.append("Skipped: " + "; ".join(f"{l} ({r})" for l, r in skipped))
    report = "\n".join(lines) + "\n"
    (args.output / "report.md").write_text(report)
    print("\n" + report)
    print(f"Wrote {args.output}/report.md")


if __name__ == "__main__":
    main()
