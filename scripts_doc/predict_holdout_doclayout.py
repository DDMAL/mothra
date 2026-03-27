#!/usr/bin/env python3
"""
Batch inference on holdout images for DocLayout-YOLO checkpoints.

Usage:
    python scripts_doc/predict_holdout_doclayout.py \
      --config configs/mothra_doclayout_base.yaml \
      --weights outputs/doclayout_yolo/runs/<run_name>/weights/best.pt \
      --out-dir models/predictions_doclayout \
      --font-size 12 \
      --line-width 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _holdout_image_dir(data_root: Path) -> Path:
    holdout = data_root / "holdout"
    images = holdout / "images"
    if images.is_dir() and any(images.iterdir()):
        return images
    return holdout


def _patch_doclayout_ckpt_compat() -> None:
    """
    Make YOLOv10DetectionModel discoverable from ultralytics.nn.tasks
    when loading DocLayout-YOLO serialized checkpoints.
    """
    import sys

    doclayout_root = (Path.cwd() / "DocLayout-YOLO").resolve()
    if doclayout_root.is_dir() and str(doclayout_root) not in sys.path:
        sys.path.insert(0, str(doclayout_root))

    try:
        import ultralytics.nn.tasks as utasks
        from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
    except Exception:
        return

    if not hasattr(utasks, "YOLOv10DetectionModel"):
        utasks.YOLOv10DetectionModel = YOLOv10DetectionModel


def _load_doclayout_model(weights: Path):
    import sys

    doclayout_root = (Path.cwd() / "DocLayout-YOLO").resolve()
    if not doclayout_root.is_dir():
        raise FileNotFoundError(f"DocLayout-YOLO folder not found: {doclayout_root}")
    if str(doclayout_root) not in sys.path:
        sys.path.insert(0, str(doclayout_root))

    # Patch before loading ckpt to support serialized class paths.
    _patch_doclayout_ckpt_compat()
    from doclayout_yolo import YOLOv10

    return YOLOv10(str(weights))


def main() -> None:
    parser = argparse.ArgumentParser(description="DocLayout-YOLO holdout predictions + visualizations")
    parser.add_argument("--config", type=str, required=True, help="DocLayout training YAML")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image directory or file; default: <data_root>/holdout (or holdout/images if present)",
    )
    parser.add_argument("--out-dir", type=str, default="models/predictions_doclayout", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--font-size", type=float, default=12.0, help="Label font size")
    parser.add_argument("--line-width", type=float, default=2.0, help="Bounding box line width")
    args = parser.parse_args()

    with open(_resolve(args.config), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = _resolve(config["paths"]["data_root"])
    source = _resolve(args.source) if args.source else _holdout_image_dir(data_root)
    weights = _resolve(args.weights)
    out_dir = _resolve(args.out_dir)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = _load_doclayout_model(weights)

    imgsz = int(config["training"]["image_size"])
    device = config["training"]["device"]

    print(f"📂 Source: {source}")
    print(f"📦 Weights: {weights}")
    print(f"💾 Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=False,
        save_txt=False,
        stream=False,
    )

    for r in results:
        path = Path(r.path)
        plotted = r.plot(
            conf=True,
            line_width=int(round(args.line_width)),
            font_size=args.font_size,
            labels=True,
            boxes=True,
        )
        cv2.imwrite(str(out_dir / path.name), plotted)
        txt_path = labels_dir / f"{path.stem}.txt"
        txt_path.unlink(missing_ok=True)
        r.save_txt(txt_path, save_conf=True)

    print(f"\n✅ Done. Visualizations and labels under: {out_dir}")


if __name__ == "__main__":
    main()

