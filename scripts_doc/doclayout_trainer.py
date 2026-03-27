#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


class DocLayoutTrainer:
    """
    DocLayout-YOLO training wrapper for Mothra-style YOLO labels.

    - Keeps all modifications self-contained under scripts_doc/.
    - Reuses the dataset split + YOLO folder layout approach from existing scripts,
      but calls DocLayout-YOLO's training API (YOLOv10).
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path(self.config["paths"]["project_root"]).resolve()
        self.data_root = Path(self.config["paths"]["data_root"]).resolve()
        self.output_root = Path(self.config["paths"]["output_root"]).resolve()

        self._ensure_doclayout_on_path()
        self.setup_directories()

    def _ensure_doclayout_on_path(self) -> None:
        """
        Allow importing raw DocLayout-YOLO (vendored under repo root).
        """
        doclayout_root = self.project_root / "DocLayout-YOLO"
        if not doclayout_root.is_dir():
            raise FileNotFoundError(f"DocLayout-YOLO folder not found at: {doclayout_root}")
        if str(doclayout_root) not in sys.path:
            sys.path.insert(0, str(doclayout_root))

    def setup_directories(self) -> None:
        for dir_path in [self.output_root, self.output_root / "runs", self.output_root / "datasets"]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def extract_manuscript_id(self, filename: str) -> str:
        stem = Path(filename).stem
        stem_normalized = stem.replace("_", " ")
        parts = stem_normalized.split()

        manuscript_parts: list[str] = []
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

    def manuscript_aware_split(self, images_dir: str | Path, labels_dir: str | Path, split_ratios=None):
        print("📚 Creating manuscript-aware data splits...")

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        manuscript_groups: dict[str, list[dict[str, Path]]] = defaultdict(list)

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
            "split_type": "manuscript",
            "train_manuscripts": train_mss,
            "val_manuscripts": val_mss,
            "test_manuscripts": test_mss,
            "split_ratios": split_ratios,
            "timestamp": datetime.now().isoformat(),
        }
        return splits, split_log

    def random_page_split(self, images_dir: str | Path, labels_dir: str | Path, split_ratios=(0.7, 0.15, 0.15)):
        print("🎲 Creating random page-level splits (NOT manuscript-aware)...")
        print("⚠️  WARNING: Pages from same manuscript may appear in train/val")

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        pairs: list[dict[str, Path]] = []
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
            "split_ratios": split_ratios,
            "timestamp": datetime.now().isoformat(),
        }
        return splits, split_log

    def organize_yolo_dataset(self, splits, output_dir: str | Path, split_log: dict) -> Path:
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
            json.dump(split_log, f, indent=2, ensure_ascii=False)

        print(f"  ✅ YOLO dataset organized at: {output_dir}")
        print(f"  ✅ data.yaml created at: {yaml_path}")
        print(f"  ✅ Split log saved at: {log_path}")
        return yaml_path

    def _build_model_spec(self) -> tuple[str, str]:
        """
        Returns (model_spec, pretrain_name) used for naming/logging.
        """
        model_cfg = self.config.get("doclayout", {}).get("model")
        if model_cfg is None:
            raise KeyError('Missing config key: doclayout.model (e.g. "m" or "m-doclayout")')

        pretrain = self.config.get("doclayout", {}).get("pretrain")
        if pretrain is None:
            # Allow YAML arch selection: "yolov10m-doclayout.yaml" etc.
            if str(model_cfg).endswith(".yaml"):
                return str(model_cfg), "None"
            return f"yolov10{model_cfg}.yaml", "None"

        if pretrain == "coco":
            return f"yolov10{model_cfg}.pt", "coco"

        pretrain_path = str(pretrain)
        if pretrain_path.endswith(".pt"):
            return pretrain_path, Path(pretrain_path).stem

        raise ValueError('doclayout.pretrain must be null, "coco", or a .pt path')

    def _patch_ultralytics_yolov10_compat(self) -> None:
        """
        Compatibility shim for loading YOLOv10 *.pt in environments where the
        installed `ultralytics` package does not expose YOLOv10DetectionModel.
        """
        try:
            import ultralytics.nn.tasks as utasks
            from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
        except Exception:
            return

        if not hasattr(utasks, "YOLOv10DetectionModel"):
            utasks.YOLOv10DetectionModel = YOLOv10DetectionModel

    def train(self, data_yaml_path: str | Path, resume: bool = False):
        from doclayout_yolo import YOLOv10  # imported after sys.path fix

        tcfg = self.config["training"]
        dcfg = self.config.get("doclayout", {})
        aug = self.config.get("augmentation", {})

        model_spec, pretrain_name = self._build_model_spec()
        if str(model_spec).endswith(".pt"):
            self._patch_ultralytics_yolov10_compat()
        model = YOLOv10(model_spec)

        project_dir = (self.output_root / "runs").resolve()

        name = (
            f"{Path(model_spec).stem}"
            f"_mothra_epoch{tcfg['epochs']}"
            f"_imgsz{tcfg['image_size']}"
            f"_bs{tcfg['batch_size']}"
            f"_pretrain_{pretrain_name}"
        )

        print("🎯 Starting DocLayout-YOLO training...")
        print(f"  📦 Model: {model_spec}")
        print(f"  📂 Data:  {data_yaml_path}")
        print(f"  🏷️  Run:   {project_dir} / {name}")

        results = model.train(
            data=str(data_yaml_path),
            epochs=int(tcfg["epochs"]),
            warmup_epochs=float(dcfg.get("warmup_epochs", 3.0)),
            lr0=float(tcfg.get("learning_rate", dcfg.get("lr0", 0.02))),
            optimizer=str(dcfg.get("optimizer", "auto")),
            momentum=float(dcfg.get("momentum", 0.9)),
            imgsz=int(tcfg["image_size"]),
            mosaic=float(aug.get("mosaic", dcfg.get("mosaic", 1.0))),
            batch=int(tcfg["batch_size"]),
            device=str(tcfg.get("device", dcfg.get("device", "0"))),
            workers=int(tcfg.get("workers", dcfg.get("workers", 4))),
            plots=bool(dcfg.get("plots", False)),
            exist_ok=bool(dcfg.get("exist_ok", True)),
            val=bool(dcfg.get("val", True)),
            val_period=int(dcfg.get("val_period", 1)),
            resume=bool(resume),
            save_period=int(tcfg.get("save_period", 10)),
            patience=int(tcfg.get("patience", 100)),
            project=str(project_dir),
            name=name,
        )
        print("  ✅ Training complete!")
        return results

