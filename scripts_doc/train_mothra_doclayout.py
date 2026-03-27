#!/usr/bin/env python3
"""
Mothra training script using the vendored DocLayout-YOLO implementation.

Important:
- Does NOT modify or import from scripts/ or scripts_v11/.
- Keeps DocLayout-YOLO-specific integration under scripts_doc/.

Usage:
  python scripts_doc/train_mothra_doclayout.py --config configs/mothra_doclayout_base.yaml
  python scripts_doc/train_mothra_doclayout.py --config configs/mothra_doclayout_base.yaml --split-type random
  python scripts_doc/train_mothra_doclayout.py --config configs/mothra_doclayout_base.yaml --resume
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime

import numpy as np

from doclayout_trainer import DocLayoutTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Mothra DocLayout-YOLO Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume training")
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
        help='Split strategy: "manuscript" (default) or "random"',
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    trainer = DocLayoutTrainer(args.config)

    if "/" in args.images_dir or args.images_dir.startswith("images"):
        images_dir = trainer.data_root / args.images_dir
    else:
        images_dir = trainer.data_root / "images" / args.images_dir

    labels_dir = trainer.data_root / "yolo_labels"

    print("=" * 60)
    print("🦋 MOTHRA - DocLayout-YOLO Training")
    print("=" * 60)
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

    trainer.train(data_yaml_path, resume=args.resume)

    print("\n✅ Done! Check results in:", (trainer.output_root / "runs").resolve())
    print("📊 Dataset organized in:", dataset_dir.resolve())


if __name__ == "__main__":
    main()

