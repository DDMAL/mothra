#!/usr/bin/env python3
"""
Batch inference on holdout manuscript images (YOLOv11).

Writes visualizations and YOLO txt sidecars similarly to a dedicated predictions folder
(e.g. ``models/predictions_v11``), reading images from ``<data_root>/holdout`` by default.

Usage (from repo root):

    python scripts_v11/predict_holdout.py \\
        --config configs/mothra_base11.yaml \\
        --weights outputs/yolo11/runs/detect/weights/best.pt

    # Custom IO / smaller box labels (large manuscript pages default to huge auto font)
    python scripts_v11/predict_holdout.py \\
        --config configs/mothra_base11.yaml \\
        --source data/holdout \\
        --out-dir models/predictions_v11 \\
        --font-size 12 \\
        --line-width 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402


def _resolve_cfg_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _holdout_image_dir(data_root: Path) -> Path:
    """Prefer ``holdout/images`` if it exists and has files, else ``holdout``."""
    holdout = data_root / "holdout"
    images = holdout / "images"
    if images.is_dir() and any(images.iterdir()):
        return images
    return holdout


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv11 holdout predictions + visualizations")
    parser.add_argument("--config", type=str, required=True, help="Training YAML (paths, device, imgsz, thresholds)")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pt checkpoint; default: newest best.pt under config output_root",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image directory or file; default: <data_root>/holdout (or holdout/images if present)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/predictions_v11",
        help="Output directory: images at top level, labels/ for YOLO txt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (default: evaluation.confidence_threshold in YAML)",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Label font size on saved images (Ultralytics default scales with image size and is often huge)",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Bounding box line width for saved visualizations",
    )
    args = parser.parse_args()

    config_path = _resolve_cfg_path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()

    if args.source:
        source = _resolve_cfg_path(args.source)
    else:
        source = _holdout_image_dir(data_root)

    if not source.exists():
        raise FileNotFoundError(
            f"Holdout source not found: {source}\n"
            "Create data/holdout (or data/holdout/images) or pass --source explicitly."
        )

    trainer = MothraTrainer(str(config_path))
    if args.weights:
        weights = _resolve_cfg_path(args.weights)
    else:
        weights = trainer._find_best_weights()

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir = _resolve_cfg_path(args.out_dir)

    eval_cfg = config.get("evaluation", {}) or {}
    conf = args.conf if args.conf is not None else float(eval_cfg.get("confidence_threshold", 0.25))
    iou = float(eval_cfg.get("iou_threshold", 0.7))
    imgsz = int(config["training"]["image_size"])
    device = config["training"]["device"]

    print(f"📂 Source: {source}")
    print(f"📦 Weights: {weights}")
    print(f"💾 Output: {out_dir}")
    print(
        f"🎚  conf={conf}, iou={iou}, imgsz={imgsz}, device={device}, "
        f"font_size={args.font_size}, line_width={args.line_width}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    # Ultralytics ``predict(save=True)`` does not pass ``font_size`` into ``Results.plot``; large pages get enormous
    # labels. Plot and save ourselves with fixed font/box scale.
    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=False,
        save_txt=False,
        stream=False,
    )

    for r in results:
        path = Path(r.path)
        stem = path.stem
        out_name = path.name
        # OpenCV 4.13+ requires integer rectangle thickness; argparse gives 2.0 for ``--line-width 2``.
        plotted = r.plot(
            conf=True,
            line_width=int(round(args.line_width)),
            font_size=args.font_size,
            labels=True,
            boxes=True,
        )
        cv2.imwrite(str(out_dir / out_name), plotted)
        txt_path = labels_dir / f"{stem}.txt"
        txt_path.unlink(missing_ok=True)
        r.save_txt(txt_path, save_conf=True)

    print(f"\n✅ Done. Visualizations and labels under: {out_dir}")


if __name__ == "__main__":
    main()
