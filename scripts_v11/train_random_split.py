#!/usr/bin/env python3
"""
Mothra Training - PAGE-LEVEL Random Split (NOT manuscript-aware), YOLOv11.

Implementation lives in ``simple_trainer.py`` (no imports from ``scripts/``).

Usage:
    python scripts_v11/train_random_split.py --config configs/mothra_base11.yaml
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

from simple_trainer import SimpleTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Mothra YOLOv11 Training - Random Page Split")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    trainer = SimpleTrainer(args.config)

    print("=" * 60)
    print("🦋 MOTHRA - Random Page-Level Split Training (YOLOv11)")
    print("=" * 60)
    print("⚠️  WARNING: Pages from same manuscript may be split across train/val")
    print("=" * 60)

    images_dir = trainer.data_root / "images"
    labels_dir = trainer.data_root / "yolo_labels"

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    splits, split_log = trainer.random_page_split(images_dir, labels_dir)

    dataset_dir = trainer.output_root / "datasets" / f"random_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_yaml_path = trainer.organize_yolo_dataset(splits, dataset_dir, split_log)

    trainer.train(data_yaml_path, resume=args.resume)

    print("\n✅ Done! Check results in:", trainer.output_root / "runs")
    print("📊 Dataset organized in:", dataset_dir)


if __name__ == "__main__":
    main()
