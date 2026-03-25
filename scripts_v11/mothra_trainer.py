#!/usr/bin/env python3
"""
Mothra YOLOv11 trainer core (self-contained under scripts_v11/).

Dataset splitting, YOLO layout, training, eval, and predict — duplicated from
``scripts/train_mothra.py`` so ``scripts_v11`` does not import ``scripts/``.
Training uses ``yolo11{size}.pt`` checkpoints.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO


class MothraTrainer:
    """
    YOLO training for medieval manuscripts with manuscript-aware or random splits.
    Works with YOLO labels exported from the Mothra annotator.
    """

    def __init__(self, config_path: str | Path):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path(self.config["paths"]["project_root"])
        self.data_root = Path(self.config["paths"]["data_root"])
        self.output_root = Path(self.config["paths"]["output_root"])

        self.setup_directories()

    def _yolo_runs_project(self) -> Path:
        """Absolute Ultralytics ``project`` directory."""
        return (self.output_root / "runs").resolve()

    def _yolo_weights_dir(self) -> Path:
        return self._yolo_runs_project() / "detect" / "weights"

    def _find_best_weights(self) -> Path:
        preferred = self._yolo_weights_dir() / "best.pt"
        if preferred.exists():
            return preferred
        candidates: list[Path] = []
        if self.output_root.exists():
            candidates.extend(self.output_root.rglob("weights/best.pt"))
        misroot = Path.cwd() / "runs" / "detect"
        if misroot.is_dir():
            candidates.extend(misroot.rglob("weights/best.pt"))
        tag = self.output_root.resolve().name
        tagged = [p for p in candidates if tag in p.parts]
        pool = tagged or candidates
        if not pool:
            raise FileNotFoundError(
                f"No best.pt found. Expected {preferred} "
                f"(use absolute project path for training; see _yolo_runs_project)."
            )
        return max(pool, key=lambda p: p.stat().st_mtime)

    def setup_directories(self) -> None:
        for dir_path in [self.output_root, self.output_root / "runs", self.output_root / "datasets"]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def manuscript_aware_split(self, images_dir, labels_dir, split_ratios=None):
        print("📚 Creating manuscript-aware data splits...")

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        manuscript_groups = defaultdict(list)

        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                continue

            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"  ⚠️  No label file for {img_file.name}, skipping")
                continue

            manuscript_id = self.extract_manuscript_id(img_file.name)
            manuscript_groups[manuscript_id].append({"image": img_file, "label": label_file})

        if not manuscript_groups:
            raise ValueError(f"No image-label pairs found in {images_dir} and {labels_dir}")

        n_manuscripts = len(manuscript_groups)

        if split_ratios is None:
            if n_manuscripts <= 3:
                split_ratios = (0.5, 0.5, 0.0) if n_manuscripts == 2 else (0.34, 0.33, 0.33)
                print(f"  ⚠️  Tiny dataset ({n_manuscripts} manuscripts) - adjusted splits")
            elif n_manuscripts <= 6:
                split_ratios = (0.5, 0.25, 0.25)
                print(f"  ⚠️  Small dataset ({n_manuscripts} manuscripts) - adjusted splits")
            else:
                split_ratios = (0.7, 0.15, 0.15)

        manuscript_ids = list(manuscript_groups.keys())
        random.shuffle(manuscript_ids)

        n_total = len(manuscript_ids)
        n_train = max(1, int(n_total * split_ratios[0]))
        n_val = max(1, int(n_total * split_ratios[1])) if n_total > 2 else 0

        if n_train + n_val > n_total:
            n_val = max(0, n_total - n_train)

        train_mss = manuscript_ids[:n_train]
        val_mss = manuscript_ids[n_train : n_train + n_val]
        test_mss = manuscript_ids[n_train + n_val :]

        print(f"  Split: {len(train_mss)} train, {len(val_mss)} val, {len(test_mss)} test manuscripts")

        splits = {
            "train": [pair for ms in train_mss for pair in manuscript_groups[ms]],
            "val": [pair for ms in val_mss for pair in manuscript_groups[ms]],
            "test": [pair for ms in test_mss for pair in manuscript_groups[ms]],
        }

        print(f"  Images: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

        split_log = {
            "train_manuscripts": train_mss,
            "val_manuscripts": val_mss,
            "test_manuscripts": test_mss,
            "split_ratios": split_ratios,
            "timestamp": datetime.now().isoformat(),
        }

        return splits, split_log

    def random_page_split(self, images_dir, labels_dir, split_ratios=(0.7, 0.15, 0.15)):
        print("🎲 Creating random page-level splits (NOT manuscript-aware)...")
        print("⚠️  WARNING: Pages from same manuscript may appear in train/val")

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        pairs = []

        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                continue

            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"  ⚠️  No label file for {img_file.name}, skipping")
                continue

            pairs.append({"image": img_file, "label": label_file})

        if not pairs:
            raise ValueError(f"No image-label pairs found in {images_dir} and {labels_dir}")

        print(f"  Found {len(pairs)} image-label pairs")

        random.shuffle(pairs)

        n_total = len(pairs)
        n_train = max(1, int(n_total * split_ratios[0]))
        n_val = max(1, int(n_total * split_ratios[1])) if n_total > 2 else 0

        if n_train + n_val > n_total:
            n_val = max(0, n_total - n_train)

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val :]

        print(f"  Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pages")

        splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        split_log = {
            "split_type": "random_page_level",
            "warning": "Pages from same manuscript may appear in different splits - DATA LEAKAGE POSSIBLE",
            "train_images": [p["image"].name for p in train_pairs],
            "val_images": [p["image"].name for p in val_pairs],
            "test_images": [p["image"].name for p in test_pairs],
            "split_ratios": split_ratios,
            "timestamp": datetime.now().isoformat(),
        }

        return splits, split_log

    def extract_manuscript_id(self, filename):
        stem = Path(filename).stem
        stem_normalized = stem.replace("_", " ")
        parts = stem_normalized.split()

        manuscript_parts = []
        for part in parts:
            if part.lower() == "copy":
                break
            if part.startswith("p.") and any(c.isdigit() for c in part):
                break
            if len(part) >= 3 and any(c.isdigit() for c in part):
                if part[-1] in ["r", "v"] or part[-2:] in ["rr", "vv"]:
                    break
            manuscript_parts.append(part)

        if manuscript_parts:
            return " ".join(manuscript_parts)

        if len(parts) >= 3:
            return " ".join(parts[:3])
        if len(parts) >= 2:
            return " ".join(parts[:2])
        return stem

    def organize_yolo_dataset(self, splits, output_dir, split_log):
        print("📂 Organizing YOLO dataset...")

        output_dir = Path(output_dir)

        for split_name, pairs in splits.items():
            split_dir = output_dir / split_name
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for pair in pairs:
                shutil.copy(pair["image"], images_dir / pair["image"].name)
                shutil.copy(pair["label"], labels_dir / pair["label"].name)

        data_yaml = {
            "path": str(output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": self.config["classes"],
            "nc": len(self.config["classes"]),
        }

        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        log_path = output_dir / "split_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(split_log, f, indent=2)

        print(f"  ✅ YOLO dataset organized at: {output_dir}")
        print(f"  ✅ data.yaml created at: {yaml_path}")
        print(f"  ✅ Split log saved at: {log_path}")

        return yaml_path

    def train(self, data_yaml_path, resume=False):
        print("🎯 Starting training...")

        model_size = self.config["model"]["size"]
        last_ckpt = self._yolo_weights_dir() / "last.pt"
        if resume and last_ckpt.exists():
            model = YOLO(str(last_ckpt))
            print("  📂 Resuming from checkpoint")
        else:
            model = YOLO(f"yolo11{model_size}.pt")
            print(f"  📦 Loading pretrained YOLOv11{model_size}")

        train_args = {
            "data": str(data_yaml_path),
            "epochs": self.config["training"]["epochs"],
            "imgsz": self.config["training"]["image_size"],
            "batch": self.config["training"]["batch_size"],
            "lr0": self.config["training"]["learning_rate"],
            "patience": self.config["training"]["patience"],
            "save": True,
            "save_period": self.config["training"]["save_period"],
            "project": str(self._yolo_runs_project()),
            "name": "detect",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "verbose": True,
            "device": self.config["training"]["device"],
            "workers": self.config["training"]["workers"],
            "augment": True,
        }

        if "augmentation" in self.config:
            aug = self.config["augmentation"]
            train_args.update(
                {
                    "hsv_h": aug.get("hsv_h", 0.015),
                    "hsv_s": aug.get("hsv_s", 0.7),
                    "hsv_v": aug.get("hsv_v", 0.4),
                    "degrees": aug.get("degrees", 10),
                    "translate": aug.get("translate", 0.1),
                    "scale": aug.get("scale", 0.5),
                    "shear": aug.get("shear", 2.0),
                    "perspective": aug.get("perspective", 0.0),
                    "flipud": aug.get("flipud", 0.0),
                    "fliplr": aug.get("fliplr", 0.5),
                    "mosaic": aug.get("mosaic", 1.0),
                    "mixup": aug.get("mixup", 0.0),
                }
            )

        results = model.train(**train_args)
        print("  ✅ Training complete!")
        return results

    def evaluate(self, data_yaml_path, weights_path=None):
        print("📊 Evaluating model...")

        if weights_path is None:
            weights_path = self._find_best_weights()

        model = YOLO(str(weights_path))

        metrics = model.val(
            data=str(data_yaml_path),
            split="val",
            save_json=True,
            save_hybrid=True,
        )

        print(f"\n{'=' * 60}")
        print(f"  mAP@50: {metrics.box.map50:.4f}")
        print(f"  mAP@50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        print(f"{'=' * 60}\n")

        print("  Per-class performance:")
        for i, class_name in enumerate(self.config["classes"]):
            print(f"    {class_name:10s}: AP@50 = {metrics.box.maps[i]:.4f}")

        return metrics

    def predict_sample(self, image_path, weights_path=None, save=True):
        if weights_path is None:
            weights_path = self._find_best_weights()

        model = YOLO(str(weights_path))

        results = model.predict(
            source=str(image_path),
            save=save,
            save_txt=True,
            save_conf=True,
            project=str((self.output_root / "predictions").resolve()),
            name="sample",
            exist_ok=True,
        )

        return results
