#!/usr/bin/env python3
"""
Optuna search for YOLOv11 augmentation/training on mothra (v2 — expanded space).

Why a v2 (vs. optuna_search_v11_aug.py)
=======================================

The original script searched a small fixed set of training/aug knobs and
covered some regions that the prior 121-trial run already showed are dead
ends (e.g. image_size=640, degrees>=7, mosaic=1.0). It also fixed a number
of Ultralytics knobs that are known to matter for OMR / manuscript layout:
the LR schedule, weight decay, warmup, close_mosaic, copy_paste, etc.

This v2 makes three changes:

1. Tighter, evidence-based search space
   - image_size in {960, 1280, 1536}  (640 dropped — always lost; 1536 added)
   - degrees in {0, 2, 5}, shear in {0, 1, 2}  (no rotations >5° or shear=3:
     winners cluster at 0)
   - hsv_s in {0.5, 0.65, 0.8}  (0.8 dominated previous winners)
   - mosaic in {0, 0.5, 1.0} — both 0 and 0.5 produced top-5 trials
   - mixup in {0, 0.05, 0.1}

2. New axes that previous runs never explored
   - optimizer in {AdamW, SGD} with optimizer-conditional LR ranges
   - cos_lr + lrf (final-LR fraction)  — cosine schedule is usually a free win
   - weight_decay, warmup_epochs, label_smoothing
   - close_mosaic — disable mosaic in last N epochs (the standard "polish"
     trick that often unlocks the final 1-2 mAP)
   - copy_paste, erasing, hsv_h, perspective
   - batch_size is memory-aware (when image_size≥1280, bs≤8)

3. Conditional sampling + warm start
   - close_mosaic only sampled when mosaic > 0
   - LR / batch ranges depend on optimizer and image_size
   - Top configurations from the prior 121-trial study are enqueue_trial'd
     so TPE does not have to rediscover them from scratch.

Operationally:
   - A `TunableTrainer` subclass surfaces the new Ultralytics knobs that the
     stock MothraTrainer.train() hard-codes away.
   - Defaults to a fresh storage / study / out-dir to avoid mixing stats
     with the v1 search.

Recommended first use:
   python scripts_v11/optuna_search_v11_aug_v2.py --n-trials 50
   # OOM? --max-image-size 1280  (drops 1536 from the search)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Search-space constants
# ---------------------------------------------------------------------------

_IMAGE_SIZES = [960, 1280, 1536]
_BATCH_LARGE_IMG = [4, 6, 8]    # image_size >= 1280
_BATCH_SMALL_IMG = [8, 12, 16]  # image_size  < 1280

_LR_ADAMW = (2e-4, 3e-3)        # log-uniform
_LR_SGD = (1e-3, 2e-2)          # log-uniform


# Best 2 configurations from the prior 121-trial v1 study
# (models/optuna_v11_omr_search1/results.csv). Used as TPE warm start
# so the new study does not have to rediscover them.
_KNOWN_GOOD_CONFIGS: list[dict[str, Any]] = [
    {
        # trial5 of v1 — best: mAP50-95 = 0.5105
        "image_size": 1280,
        "batch_size_lg": 8,            # conditional name when image_size>=1280
        "optimizer": "AdamW",
        "lr_adamw": 5e-4,              # conditional name when optimizer=AdamW
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 2.0,
        "hsv_s": 0.8,
        "hsv_v": 0.1,
        "mosaic": 0.5,
        "close_mosaic": 20,            # conditional: mosaic>0
        "mixup": 0.05,
    },
    {
        # cluster at mAP50-95 ~ 0.499 (trial 92-95 in v1)
        "image_size": 960,
        "batch_size_md": 16,
        "optimizer": "AdamW",
        "lr_adamw": 2e-3,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.2,
        "shear": 0.0,
        "hsv_s": 0.8,
        "hsv_v": 0.25,
        "mosaic": 0.0,                 # close_mosaic not used here
        "mixup": 0.05,
    },
]


# CSV result columns (single source of truth)
_RESULT_FIELDS = [
    "trial_number", "tag", "status", "error",
    "epochs", "best_map50_95", "best_epoch",
    "best_pt", "run_output_root",
    # training schedule
    "image_size", "batch_size", "optimizer", "learning_rate",
    "cos_lr", "lrf", "weight_decay", "warmup_epochs", "label_smoothing",
    # geometric aug
    "degrees", "translate", "scale", "shear", "perspective",
    # color aug
    "hsv_h", "hsv_s", "hsv_v",
    # mosaic family
    "mosaic", "close_mosaic", "mixup", "copy_paste", "erasing",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _fmt_float_tag(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def _tag_from_params(params: dict, trial_number: int) -> str:
    # Short tag — the full param set is persisted in the CSV anyway.
    return "_".join([
        f"trial{trial_number}",
        f"img{params['image_size']}",
        f"bs{params['batch_size']}",
        params["optimizer"].lower(),
        f"lr{_fmt_float_tag(params['learning_rate'])}",
        f"mo{_fmt_float_tag(params['mosaic'])}",
    ])


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def _max_map50_95_from_results_csv(results_csv: Path) -> tuple[float, int | None]:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    best_map = -1.0
    best_epoch: int | None = None

    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise KeyError(f"No CSV header found in {results_csv}")

        col_exact = "metrics/mAP50-95(B)"
        if col_exact in reader.fieldnames:
            col = col_exact
        else:
            candidates = [c for c in reader.fieldnames if "mAP50-95" in c]
            b_candidates = [c for c in candidates if "B" in c]
            if len(b_candidates) == 1:
                col = b_candidates[0]
            elif len(candidates) == 1:
                col = candidates[0]
            else:
                raise KeyError(
                    f"Could not find an mAP50-95 column in {results_csv}. "
                    f"Found columns: {reader.fieldnames}"
                )

        for row in reader:
            epoch = row.get("epoch", "")
            try:
                m = float(row[col])
            except Exception:
                continue
            if m > best_map:
                best_map = m
                best_epoch = int(epoch) if str(epoch).isdigit() else None

    return best_map, best_epoch


def _delete_checkpoints_except_best(weights_dir: Path, best_pt: Path) -> None:
    if not weights_dir.exists():
        return
    for pt in weights_dir.glob("*.pt"):
        if pt.name == best_pt.name:
            continue
        try:
            pt.unlink()
        except Exception:
            pass


def _ensure_results_header(results_csv_path: Path) -> None:
    if results_csv_path.exists():
        return
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(_RESULT_FIELDS)


def _append_result_row(
    results_csv_path: Path,
    *,
    trial_number: int,
    tag: str,
    params: dict,
    epochs: int,
    best_map: float,
    best_epoch: int | None,
    best_pt_dest: Path,
    run_root: Path,
    status: str,
    err_msg: str,
) -> None:
    row_map = {
        "trial_number": trial_number,
        "tag": tag,
        "status": status,
        "error": err_msg,
        "epochs": epochs,
        "best_map50_95": best_map,
        "best_epoch": best_epoch if best_epoch is not None else "",
        "best_pt": str(best_pt_dest) if best_pt_dest.exists() else "",
        "run_output_root": str(run_root),
    }
    for k in _RESULT_FIELDS:
        if k not in row_map:
            row_map[k] = params.get(k, "")
    with open(results_csv_path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([row_map[k] for k in _RESULT_FIELDS])
        f.flush()


# ---------------------------------------------------------------------------
# Tunable trainer — exposes Ultralytics knobs MothraTrainer.train() hides
# ---------------------------------------------------------------------------


class TunableTrainer(MothraTrainer):
    """
    Thin override of MothraTrainer.train() that pipes additional Ultralytics
    fields straight from the config into model.train(...). Original behavior
    (model loading, checkpoint paths, dataset wiring) is unchanged.

    Recognized extra fields under config["training"]:
        optimizer, cos_lr, lrf, weight_decay, momentum,
        warmup_epochs, warmup_momentum, warmup_bias_lr,
        label_smoothing, dropout, box, cls, dfl

    Recognized extra fields under config["augmentation"]:
        hsv_h, perspective, fliplr, flipud,
        close_mosaic, copy_paste, erasing
    """

    _EXTRA_TRAIN_KEYS = (
        "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        "label_smoothing", "dropout",
        "box", "cls", "dfl",
    )
    _EXTRA_AUG_KEYS = ("close_mosaic", "copy_paste", "erasing")

    def train(self, data_yaml_path, resume: bool = False):
        from ultralytics import YOLO  # imported lazily, mirrors parent

        print("Starting tunable training...")

        tcfg = self.config["training"]
        model_size = self.config["model"]["size"]
        last_ckpt = self._yolo_weights_dir() / "last.pt"
        if resume and last_ckpt.exists():
            model = YOLO(str(last_ckpt))
            print("  Resuming from checkpoint")
        else:
            model = YOLO(f"yolo11{model_size}.pt")
            print(f"  Loading pretrained YOLOv11{model_size}")

        train_args: dict[str, Any] = {
            "data": str(data_yaml_path),
            "epochs": int(tcfg["epochs"]),
            "imgsz": int(tcfg["image_size"]),
            "batch": int(tcfg["batch_size"]),
            "lr0": float(tcfg["learning_rate"]),
            "patience": int(tcfg["patience"]),
            "save": True,
            "save_period": int(tcfg["save_period"]),
            "project": str(self._yolo_runs_project()),
            "name": "detect",
            "exist_ok": True,
            "pretrained": True,
            "verbose": True,
            "device": tcfg["device"],
            "workers": int(tcfg["workers"]),
            "augment": True,
            "optimizer": str(tcfg.get("optimizer", "AdamW")),
            "cos_lr": bool(tcfg.get("cos_lr", False)),
        }
        for k in self._EXTRA_TRAIN_KEYS:
            if k in tcfg and tcfg[k] is not None:
                train_args[k] = tcfg[k]

        if "augmentation" in self.config:
            aug = self.config["augmentation"]
            for k, default in [
                ("hsv_h", 0.015),
                ("hsv_s", 0.7),
                ("hsv_v", 0.4),
                ("degrees", 10.0),
                ("translate", 0.1),
                ("scale", 0.5),
                ("shear", 2.0),
                ("perspective", 0.0),
                ("flipud", 0.0),
                ("fliplr", 0.5),
                ("mosaic", 1.0),
                ("mixup", 0.0),
            ]:
                train_args[k] = aug.get(k, default)
            for k in self._EXTRA_AUG_KEYS:
                if k in aug and aug[k] is not None:
                    train_args[k] = aug[k]

            # Ultralytics requires close_mosaic < epochs.
            if "close_mosaic" in train_args:
                cm = int(train_args["close_mosaic"])
                ep = int(train_args["epochs"])
                train_args["close_mosaic"] = max(0, min(cm, max(1, ep - 1)))

        # Honor external CUDA_VISIBLE_DEVICES isolation: ultralytics'
        # select_device() does `os.environ["CUDA_VISIBLE_DEVICES"] = device`
        # (utils/torch_utils.py:221), which would pin every worker to physical
        # GPU 0 if we pass device='0'. Setting device=None makes select_device
        # take the empty-string branch and pick cuda:0 = the only GPU CVD
        # already isolated us to.
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            train_args["device"] = None

        results = model.train(**train_args)
        print("  Training complete.")
        return results


# ---------------------------------------------------------------------------
# Dataset preparation (unchanged from v1)
# ---------------------------------------------------------------------------


def _prepare_fixed_dataset(base_cfg: dict, base_cfg_path: Path, args) -> tuple[Path, Path, Path]:
    dataset_yaml_marker_path = _resolve_path(args.out_dir) / "data_yaml_path.txt"

    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = base_cfg_path.parent.parent.resolve()

    for k in ("project_root", "data_root"):
        if k in base_cfg.get("paths", {}):
            v = Path(base_cfg["paths"][k])
            if not v.is_absolute():
                base_cfg["paths"][k] = str((repo_root / v).resolve())

    search_root = (repo_root / "outputs" / "yolo11_optuna_search_v2").resolve()
    search_root.mkdir(parents=True, exist_ok=True)

    dataset_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = search_root / "datasets" / dataset_id

    if dataset_yaml_marker_path.exists() and not args.force_regenerate_dataset:
        data_yaml_path = Path(dataset_yaml_marker_path.read_text(encoding="utf-8").strip()).resolve()
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Marker exists but data.yaml not found: {data_yaml_path}")
        print(f"Reuse dataset split: {data_yaml_path}")
        return repo_root, search_root, data_yaml_path

    dataset_cfg = dict(base_cfg)
    dataset_cfg["paths"] = dict(base_cfg["paths"])
    dataset_cfg["training"] = dict(base_cfg["training"])
    dataset_cfg["paths"]["output_root"] = str(dataset_root)

    tmp_dataset_cfg = dataset_root / "dataset_gen_config.yaml"
    _write_yaml(tmp_dataset_cfg, dataset_cfg)

    trainer_for_split = MothraTrainer(str(tmp_dataset_cfg))

    images_dir = trainer_for_split.data_root / args.images_dir
    if not images_dir.exists():
        alt_images_dir = trainer_for_split.data_root / "images" / args.images_dir
        if alt_images_dir.exists():
            images_dir = alt_images_dir

    labels_dir = trainer_for_split.data_root / "yolo_labels"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images dir not found: {images_dir} (or data_root/images/{args.images_dir})"
        )
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    if args.split_type == "random":
        splits, split_log = trainer_for_split.random_page_split(images_dir, labels_dir)
    else:
        splits, split_log = trainer_for_split.manuscript_aware_split(images_dir, labels_dir)

    data_yaml_dir = dataset_root / "dataset"
    data_yaml_path = trainer_for_split.organize_yolo_dataset(splits, data_yaml_dir, split_log)
    dataset_yaml_marker_path.write_text(str(data_yaml_path), encoding="utf-8")

    print(f"Generated dataset split: {data_yaml_path}")
    return repo_root, search_root, data_yaml_path


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial, *, max_image_size: int = 1536) -> dict:
    """
    See module docstring for the rationale behind every range. Keep this
    function pure (only reads from `trial`); _KNOWN_GOOD_CONFIGS depends on
    the exact optuna parameter names defined here.
    """
    p: dict[str, Any] = {}

    # --- image / batch (memory-aware) ---
    img_choices = [s for s in _IMAGE_SIZES if s <= max_image_size]
    p["image_size"] = trial.suggest_categorical("image_size", img_choices)

    if p["image_size"] >= 1280:
        p["batch_size"] = trial.suggest_categorical("batch_size_lg", _BATCH_LARGE_IMG)
    else:
        p["batch_size"] = trial.suggest_categorical("batch_size_md", _BATCH_SMALL_IMG)

    # --- optimizer + LR (LR range is optimizer-conditional) ---
    p["optimizer"] = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    if p["optimizer"] == "AdamW":
        p["learning_rate"] = trial.suggest_float("lr_adamw", *_LR_ADAMW, log=True)
    else:
        p["learning_rate"] = trial.suggest_float("lr_sgd", *_LR_SGD, log=True)

    # --- LR schedule + regularization ---
    p["cos_lr"] = trial.suggest_categorical("cos_lr", [True, False])
    p["lrf"] = trial.suggest_float("lrf", 0.01, 0.2, log=True)
    p["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    p["warmup_epochs"] = trial.suggest_int("warmup_epochs", 1, 5)
    p["label_smoothing"] = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])

    # --- geometric augmentation (manuscripts are mostly upright/flat) ---
    p["degrees"] = trial.suggest_categorical("degrees", [0.0, 2.0, 5.0])
    p["translate"] = trial.suggest_categorical("translate", [0.05, 0.1, 0.15])
    p["scale"] = trial.suggest_categorical("scale", [0.1, 0.2, 0.3])
    p["shear"] = trial.suggest_categorical("shear", [0.0, 1.0, 2.0])
    p["perspective"] = trial.suggest_categorical("perspective", [0.0, 1e-4, 5e-4])

    # --- color augmentation (parchment / ink / lighting variation) ---
    p["hsv_h"] = trial.suggest_categorical("hsv_h", [0.0, 0.005, 0.015])
    p["hsv_s"] = trial.suggest_categorical("hsv_s", [0.5, 0.65, 0.8])
    p["hsv_v"] = trial.suggest_categorical("hsv_v", [0.1, 0.25, 0.4])

    # --- mosaic family ---
    p["mosaic"] = trial.suggest_categorical("mosaic", [0.0, 0.5, 1.0])
    if p["mosaic"] > 0.0:
        p["close_mosaic"] = trial.suggest_categorical("close_mosaic", [10, 20, 50])
    else:
        p["close_mosaic"] = 0  # not sampled — irrelevant when mosaic is off

    p["mixup"] = trial.suggest_categorical("mixup", [0.0, 0.05, 0.1])
    p["copy_paste"] = trial.suggest_categorical("copy_paste", [0.0, 0.1, 0.3])
    p["erasing"] = trial.suggest_categorical("erasing", [0.0, 0.2, 0.4])

    return p


def _enqueue_warm_starts(study: optuna.Study, max_image_size: int) -> int:
    """Seed the study with the best v1 configs that fit the image-size cap."""
    enqueued = 0
    for cfg in _KNOWN_GOOD_CONFIGS:
        if cfg.get("image_size", 0) > max_image_size:
            continue
        try:
            study.enqueue_trial(cfg, skip_if_exists=True)
            enqueued += 1
        except Exception as e:
            print(f"  warm-start skipped: {e}")
    return enqueued


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search v2 for YOLOv11 on mothra OMR/layout task")
    parser.add_argument("--base-config", type=str, default="configs/mothra_base11.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default=None, help="Override training.device, e.g. 0 or [0,1].")
    parser.add_argument("--split-type", type=str, default="manuscript", choices=["manuscript", "random"])
    parser.add_argument("--images-dir", type=str, default="images", help="Images dir under data_root.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/optuna_v11_omr_search_v2",
        help="Where to write best_<tag>.pt and results.csv",
    )
    parser.add_argument(
        "--force-regenerate-dataset",
        action="store_true",
        help="Regenerate dataset split even if a previous data.yaml exists in --out-dir.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1536,
        help="Cap image_size search (drop to 1280 if 1536 OOMs).",
    )

    # Optuna
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials this worker will attempt (upper bound).")
    parser.add_argument(
        "--max-total-trials",
        type=int,
        default=None,
        help="Stop this worker once the study has N globally-completed trials. "
             "Use this to cap total trials in multi-worker setups; otherwise each "
             "worker independently runs --n-trials.",
    )
    parser.add_argument("--study-name", type=str, default="yolo11_omr_optuna_v2")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_yolo11_omr_v2.db",
        help="Optuna storage URL. SQLite is easiest for resume.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler type.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip enqueueing the prior best v1 configs.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    base_cfg_path = _resolve_path(args.base_config)
    base_cfg = _load_yaml(base_cfg_path)

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = out_dir / "results.csv"
    _ensure_results_header(results_csv_path)

    repo_root, search_root, data_yaml_path = _prepare_fixed_dataset(base_cfg, base_cfg_path, args)

    if args.sampler == "random":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        # multivariate=True helps when several knobs interact (e.g. lr + bs).
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    if not args.no_warm_start:
        n = _enqueue_warm_starts(study, args.max_image_size)
        if n:
            print(f"Warm-started study with {n} known-good configs.")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, max_image_size=args.max_image_size)
        tag = _tag_from_params(params, trial.number)
        run_root = (search_root / "runs" / tag).resolve()

        print(f"\n[Trial {trial.number}] {tag}")
        for k, v in params.items():
            print(f"    {k}: {v}")

        run_cfg = dict(base_cfg)
        run_cfg["paths"] = dict(base_cfg["paths"])
        run_cfg["model"] = dict(base_cfg.get("model", {}))
        run_cfg["training"] = dict(base_cfg["training"])
        run_cfg["augmentation"] = dict(base_cfg.get("augmentation", {}))
        run_cfg["evaluation"] = dict(base_cfg.get("evaluation", {}))

        # ---- training overrides ----
        run_cfg["training"]["epochs"] = int(args.epochs)
        run_cfg["training"]["save_period"] = -1
        run_cfg["training"]["learning_rate"] = float(params["learning_rate"])
        run_cfg["training"]["batch_size"] = int(params["batch_size"])
        run_cfg["training"]["image_size"] = int(params["image_size"])
        run_cfg["training"]["optimizer"] = str(params["optimizer"])
        run_cfg["training"]["cos_lr"] = bool(params["cos_lr"])
        run_cfg["training"]["lrf"] = float(params["lrf"])
        run_cfg["training"]["weight_decay"] = float(params["weight_decay"])
        run_cfg["training"]["warmup_epochs"] = float(params["warmup_epochs"])
        run_cfg["training"]["label_smoothing"] = float(params["label_smoothing"])
        if args.device is not None:
            run_cfg["training"]["device"] = args.device

        # ---- augmentation overrides ----
        run_cfg["augmentation"]["hsv_h"] = float(params["hsv_h"])
        run_cfg["augmentation"]["hsv_s"] = float(params["hsv_s"])
        run_cfg["augmentation"]["hsv_v"] = float(params["hsv_v"])
        run_cfg["augmentation"]["degrees"] = float(params["degrees"])
        run_cfg["augmentation"]["translate"] = float(params["translate"])
        run_cfg["augmentation"]["scale"] = float(params["scale"])
        run_cfg["augmentation"]["shear"] = float(params["shear"])
        run_cfg["augmentation"]["perspective"] = float(params["perspective"])
        run_cfg["augmentation"]["mosaic"] = float(params["mosaic"])
        run_cfg["augmentation"]["close_mosaic"] = int(params["close_mosaic"])
        run_cfg["augmentation"]["mixup"] = float(params["mixup"])
        run_cfg["augmentation"]["copy_paste"] = float(params["copy_paste"])
        run_cfg["augmentation"]["erasing"] = float(params["erasing"])
        # Forced — manuscripts must not be flipped.
        run_cfg["augmentation"]["fliplr"] = 0.0
        run_cfg["augmentation"]["flipud"] = 0.0

        run_cfg["paths"]["output_root"] = str(run_root)

        tmp_run_cfg = run_root / "config.yaml"
        _write_yaml(tmp_run_cfg, run_cfg)

        best_pt_dest = out_dir / f"best_{tag}.pt"
        status = "ok"
        err_msg = ""
        best_map = -1.0
        best_epoch: int | None = None

        try:
            trainer = TunableTrainer(str(tmp_run_cfg))
            trainer.train(data_yaml_path, resume=False)

            best_pt = trainer._yolo_weights_dir() / "best.pt"
            if not best_pt.exists():
                found = list(run_root.rglob("weights/best.pt"))
                if found:
                    best_pt = max(found, key=lambda p: p.stat().st_mtime)
                else:
                    raise FileNotFoundError(f"best.pt not found under: {run_root}")

            best_pt_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, best_pt_dest)

            _delete_checkpoints_except_best(trainer._yolo_weights_dir(), best_pt)

            results_csv = run_root / "runs" / "detect" / "results.csv"
            best_map, best_epoch = _max_map50_95_from_results_csv(results_csv)

            trial.set_user_attr("tag", tag)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_pt", str(best_pt_dest))
            trial.set_user_attr("run_root", str(run_root))

        except Exception as e:
            status = "failed"
            err_msg = str(e)
            _append_result_row(
                results_csv_path,
                trial_number=trial.number,
                tag=tag,
                params=params,
                epochs=args.epochs,
                best_map=best_map,
                best_epoch=best_epoch,
                best_pt_dest=best_pt_dest,
                run_root=run_root,
                status=status,
                err_msg=err_msg,
            )
            print(f"  ERROR: {err_msg}")
            raise

        _append_result_row(
            results_csv_path,
            trial_number=trial.number,
            tag=tag,
            params=params,
            epochs=args.epochs,
            best_map=best_map,
            best_epoch=best_epoch,
            best_pt_dest=best_pt_dest,
            run_root=run_root,
            status=status,
            err_msg=err_msg,
        )

        print(
            f"  -> status={status} best_map50_95={best_map} "
            f"best_epoch={best_epoch} best_pt={best_pt_dest}"
        )
        return best_map

    callbacks: list = []
    if args.max_total_trials is not None:
        target = args.max_total_trials

        def _stop_when_global_done(study_: optuna.Study, trial_: optuna.trial.FrozenTrial) -> None:
            n_done = sum(1 for t in study_.trials if t.state == optuna.trial.TrialState.COMPLETE)
            if n_done >= target:
                print(f"  global complete trials = {n_done} >= {target}, stopping worker.")
                study_.stop()

        callbacks.append(_stop_when_global_done)

    study.optimize(objective, n_trials=args.n_trials, callbacks=callbacks)

    print("\n===== BEST TRIAL =====")
    print("Best value:", study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print("Best attrs:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
