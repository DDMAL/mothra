#!/usr/bin/env python3
"""
Mothra Training Script - YOLOv11 for Medieval Manuscript Layout Detection.

Implementation lives in ``mothra_trainer.py`` (no imports from ``scripts/``).

Usage:
    python scripts_v11/train_mothra.py --config configs/mothra_base11.yaml
    python scripts_v11/train_mothra.py --config configs/mothra_base11.yaml --split-type random
    python scripts_v11/train_mothra.py --config configs/mothra_base11.yaml --resume
    python scripts_v11/train_mothra.py --config configs/mothra_base11.yaml --predict path/to/image.png
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Mothra YOLOv11 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--predict", type=str, help="Run prediction on single image")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help='Images subdirectory name (default: images). Use "greyscale" for greyscale images.',
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="manuscript",
        choices=["manuscript", "random"],
        help='Split strategy: "manuscript" (default) or "random" (better for small datasets)',
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    trainer = MothraTrainer(args.config)

    if args.predict:
        print(f"🔮 Running prediction on: {args.predict}")
        trainer.predict_sample(args.predict)
        return

    print("=" * 60)
    print("🦋 MOTHRA - YOLOv11 Training for Medieval Manuscripts")
    print("=" * 60)

    if "/" in args.images_dir or args.images_dir.startswith("images"):
        images_dir = trainer.data_root / args.images_dir
    else:
        images_dir = trainer.data_root / "images" / args.images_dir

    labels_dir = trainer.data_root / "yolo_labels"

    print(f"📂 Images directory: {images_dir}")
    print(f"📂 Labels directory: {labels_dir}")
    print()

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    if args.split_type == "random":
        splits, split_log = trainer.random_page_split(images_dir, labels_dir)
    else:
        splits, split_log = trainer.manuscript_aware_split(images_dir, labels_dir)

    dataset_dir = trainer.output_root / "datasets" / f"mothra_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_yaml_path = trainer.organize_yolo_dataset(splits, dataset_dir, split_log)

    if args.eval_only:
        trainer.evaluate(data_yaml_path)
    else:
        trainer.train(data_yaml_path, resume=args.resume)
        trainer.evaluate(data_yaml_path)

    print("\n✅ Done! Check results in:", trainer.output_root / "runs")
    print("📊 Dataset organized in:", dataset_dir)


if __name__ == "__main__":
    main()
