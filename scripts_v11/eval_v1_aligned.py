#!/usr/bin/env python3
"""
eval_v1_aligned.py

Two modes that can be run independently or together:

  eval   -- Evaluate a YOLOv11 checkpoint against a labelled data split and
            report per-class mAP@50 (the same style as the v1 YOLOv8 baseline:
            "mAP text=0.77, music=0.67").  Also reports mAP@50-95, P, and R so
            the two baselines can be compared side-by-side.  A CSV summary is
            saved to <out_dir>/eval_summary.csv.

  infer  -- Run batch inference on a directory of images and write, for every
            image, a per-image sub-directory containing:
              <stem>_predicted.jpg   bbox overlay (same style as inference-outputs)
              <stem>.json            mothra-annotator-compatible JSON
            and an aggregated all_predictions.json at the top of <out_dir>.

Use --mode eval, --mode infer, or --mode both (default).

Examples
--------
# Evaluate only (on the test split from a training run)
python scripts_v11/eval_v1_aligned.py \\
    --mode eval \\
    --weights outputs/yolo11/runs/detect/weights/best.pt \\
    --data outputs/yolo11/datasets/data.yaml \\
    --split test \\
    --out-dir models/eval_v1_aligned

# Infer only (on holdout images)
python scripts_v11/eval_v1_aligned.py \\
    --mode infer \\
    --weights outputs/yolo11/runs/detect/weights/best.pt \\
    --source data/holdout \\
    --out-dir models/predictions_v11_aligned

# Both at once
python scripts_v11/eval_v1_aligned.py \\
    --mode both \\
    --weights outputs/yolo11/runs/detect/weights/best.pt \\
    --data outputs/yolo11/datasets/data.yaml \\
    --source data/holdout \\
    --out-dir models/eval_v1_aligned
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

# ── constants ──────────────────────────────────────────────────────────────────

# These match mothra_base11.yaml evaluation defaults and v1's *inference* settings.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
# Ultralytics' internal default for model.val(). Using a higher conf in val
# truncates the PR curve and under-reports AP. Keep this independent of DEFAULT_CONF,
# which is the inference-time threshold used for visualisation.
VAL_CONF = 0.001

# YOLO class index (0-based) → mothra-annotator classId (1-based)
YOLO_TO_ANNOTATOR_CLASS = {0: 1, 1: 2, 2: 3}
CLASS_NAMES = ["text", "music", "staves"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ── helpers ────────────────────────────────────────────────────────────────────

def _collect_images(images_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.rglob(f"*{ext}"))
        images.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def _yolo_xywhn_to_annotator_bbox(
    box_xywhn: list[float], img_w: int, img_h: int
) -> list[int]:
    """Normalised YOLO centre-xywh → pixel [x, y, w, h] with top-left origin."""
    cx, cy, w, h = box_xywhn
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    return [round(x), round(y), round(w * img_w), round(h * img_h)]


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── eval mode ──────────────────────────────────────────────────────────────────

def run_eval(
    weights: Path,
    data_yaml: Path,
    split: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    out_dir: Path,
    class_names: list[str],
) -> dict:
    """
    Evaluate checkpoint and report per-class mAP@50 matching v1 metric style.
    Returns a dict of results for optional downstream use.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION  (split={split}, conf={conf}, iou={iou})")
    print(f"{'='*60}")
    print(f"  Weights : {weights}")
    print(f"  Data    : {data_yaml}")
    print(f"  Device  : {device}\n")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        conf=VAL_CONF,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save_json=False,
        verbose=False,
    )

    map50     = metrics.box.map50
    map50_95  = metrics.box.map
    precision = metrics.box.mp
    recall    = metrics.box.mr
    # per-class AP@50. metrics.box.maps is per-class mAP@50-95 (NOT what v1 reports).
    # ap50 is ordered by ap_class_index; classes with 0 GT are absent.
    ap50_arr = list(metrics.box.ap50)
    class_idx = list(metrics.box.ap_class_index)
    per_class_ap50 = [float("nan")] * len(class_names)
    for slot, cls_i in enumerate(class_idx):
        if 0 <= int(cls_i) < len(class_names):
            per_class_ap50[int(cls_i)] = float(ap50_arr[slot])

    print(f"\n{'─'*40}")
    print(f"  (val conf={VAL_CONF}; iou={iou})")
    print(f"  Overall mAP@50    : {map50:.4f}")
    print(f"  Overall mAP@50-95 : {map50_95:.4f}  (stricter; not reported by v1)")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"{'─'*40}")
    print("  Per-class mAP@50 (v1 style):")
    for i, name in enumerate(class_names):
        ap = per_class_ap50[i] if i < len(per_class_ap50) else float("nan")
        print(f"    {name:<10s}: {ap:.4f}")
    print(f"{'─'*40}\n")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval_summary.csv"
    rows = [
        ["metric", "value"],
        ["split", split],
        ["weights", str(weights)],
        ["conf_val", VAL_CONF],
        ["conf_infer", conf],
        ["iou", iou],
        ["imgsz", imgsz],
        ["map50_overall", f"{map50:.4f}"],
        ["map50_95_overall", f"{map50_95:.4f}"],
        ["precision", f"{precision:.4f}"],
        ["recall", f"{recall:.4f}"],
    ]
    for i, name in enumerate(class_names):
        ap = per_class_ap50[i] if i < len(per_class_ap50) else float("nan")
        rows.append([f"map50_{name}", f"{ap:.4f}"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"  Saved CSV: {csv_path}")

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "per_class_ap50": {class_names[i]: per_class_ap50[i] for i in range(len(class_names))},
    }


# ── infer mode ─────────────────────────────────────────────────────────────────

def run_infer(
    weights: Path,
    source: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    out_dir: Path,
    line_width: int = 2,
    font_size: float = 14.0,
) -> None:
    """
    Batch inference that mirrors the inference-outputs branch format exactly:
      <out_dir>/<stem>/<stem>_predicted.jpg
      <out_dir>/<stem>/<stem>.json          (mothra-annotator schema)
      <out_dir>/all_predictions.json
    """
    print(f"\n{'='*60}")
    print(f"  INFERENCE  (conf={conf}, iou={iou})")
    print(f"{'='*60}")
    print(f"  Weights : {weights}")
    print(f"  Source  : {source}")
    print(f"  Output  : {out_dir}\n")

    image_paths = _collect_images(source)
    if not image_paths:
        print(f"  No images found in {source}")
        return

    print(f"  Found {len(image_paths)} image(s). Running inference...")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    timestamp = datetime.now(timezone.utc).isoformat()
    all_sessions: list[dict] = []

    for image_path in image_paths:
        stem = image_path.stem
        image_out_dir = out_dir / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=False,
            save_txt=False,
            verbose=False,
        )
        result = results[0]
        img_h, img_w = result.orig_shape

        # Visualisation — fixed font/line scale avoids Ultralytics auto-scale
        # producing enormous labels on large manuscript pages.
        plotted = result.plot(
            conf=True,
            line_width=line_width,
            font_size=font_size,
            labels=True,
            boxes=True,
        )
        jpg_path = image_out_dir / f"{stem}_predicted.jpg"
        cv2.imwrite(str(jpg_path), plotted)

        # mothra-annotator JSON
        annotations: list[dict] = []
        for box in result.boxes:
            yolo_cls = int(box.cls.item())
            annotator_cls = YOLO_TO_ANNOTATOR_CLASS.get(yolo_cls, yolo_cls + 1)
            bbox = _yolo_xywhn_to_annotator_bbox(box.xywhn[0].tolist(), img_w, img_h)
            annotations.append(
                {
                    "id": str(uuid.uuid4()),
                    "classId": annotator_cls,
                    "bbox": bbox,
                    "confidence": round(float(box.conf.item()), 4),
                    "timestamp": timestamp,
                }
            )

        session = {
            "imageName": image_path.name,
            "imageWidth": img_w,
            "imageHeight": img_h,
            "annotations": annotations,
        }
        json_path = image_out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(session, indent=2))
        all_sessions.append(session)

        n = len(annotations)
        print(f"  {image_path.name}: {n} detection{'s' if n != 1 else ''}")

    agg_path = out_dir / "all_predictions.json"
    agg_path.write_text(json.dumps(all_sessions, indent=2))
    print(f"\n  Done. Outputs under: {out_dir}/")
    print(f"  Aggregated JSON  : {agg_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv11 eval + inference aligned with v1 YOLOv8 baseline metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "infer", "both"],
        default="both",
        help="Which mode(s) to run (default: both)",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional path to a mothra YAML config. "
            "Used to read imgsz and device defaults; "
            "command-line flags take precedence."
        ),
    )

    # eval-specific
    eval_grp = parser.add_argument_group("eval options")
    eval_grp.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml (required for eval mode)",
    )
    eval_grp.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)",
    )

    # infer-specific
    infer_grp = parser.add_argument_group("infer options")
    infer_grp.add_argument(
        "--source",
        default=None,
        help="Directory of images for inference (required for infer mode)",
    )
    infer_grp.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bbox line width for overlay images (default: 2)",
    )
    infer_grp.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Label font size for overlay images (default: 14.0)",
    )

    # shared
    parser.add_argument(
        "--out-dir",
        default="models/eval_v1_aligned",
        help="Output directory (default: models/eval_v1_aligned)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"Confidence threshold (default: {DEFAULT_CONF}, matching v1)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help=f"IoU threshold (default: {DEFAULT_IOU}, matching v1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default: from --config or 640)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. 0, cpu (default: from --config or 0)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ── resolve paths ──────────────────────────────────────────────────────────
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        sys.exit(f"ERROR: weights not found: {weights}")

    out_dir = Path(args.out_dir).resolve()

    # ── read optional config for imgsz / device defaults ──────────────────────
    cfg_imgsz  = 640
    cfg_device = "0"
    if args.config:
        cfg = _load_config(Path(args.config))
        cfg_imgsz  = int(cfg.get("training", {}).get("image_size", 640))
        cfg_device = str(cfg.get("training", {}).get("device", "0"))
        cfg_conf = float(cfg.get("evaluation", {}).get("confidence_threshold", DEFAULT_CONF))
        cfg_iou  = float(cfg.get("evaluation", {}).get("iou_threshold", DEFAULT_IOU))
    else:
        cfg_conf = DEFAULT_CONF
        cfg_iou  = DEFAULT_IOU

    conf   = args.conf   if args.conf   is not None else cfg_conf
    iou    = args.iou    if args.iou    is not None else cfg_iou
    imgsz  = args.imgsz  if args.imgsz  is not None else cfg_imgsz
    device = args.device if args.device is not None else cfg_device

    # ── eval ───────────────────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        if not args.data:
            sys.exit("ERROR: --data <data.yaml> is required for eval mode")
        data_yaml = Path(args.data).resolve()
        if not data_yaml.is_file():
            sys.exit(f"ERROR: data.yaml not found: {data_yaml}")

        run_eval(
            weights=weights,
            data_yaml=data_yaml,
            split=args.split,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            out_dir=out_dir,
            class_names=CLASS_NAMES,
        )

    # ── infer ──────────────────────────────────────────────────────────────────
    if args.mode in ("infer", "both"):
        if not args.source:
            sys.exit("ERROR: --source <images_dir> is required for infer mode")
        source = Path(args.source).resolve()
        if not source.exists():
            sys.exit(f"ERROR: source directory not found: {source}")

        run_infer(
            weights=weights,
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            out_dir=out_dir,
            line_width=args.line_width,
            font_size=args.font_size,
        )


if __name__ == "__main__":
    main()
