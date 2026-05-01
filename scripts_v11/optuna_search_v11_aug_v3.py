#!/usr/bin/env python3
"""
Optuna search v3 for YOLOv11 on mothra (large-budget run, ~600 trials).

Goals over v2
=============

v2 (57 trials) found a strong prior: optimizer=AdamW, cos_lr=False,
label_smoothing=0.1, hsv_h=0, copy_paste∈[0.1,0.3], mosaic=0 winning often,
hsv_s∈[0.65,0.8], geometric augs minimal. v3 builds on that:

1. **Drop dead regions** (v2 evidence): SGD, cos_lr=True, image_size=1536,
   erasing>0, perspective>1e-4, hsv_h>0, label_smoothing<0.1.

2. **Fix proven priors** to save budget: optimizer=AdamW, cos_lr=False,
   label_smoothing=0.1, hsv_h=0, erasing=0, perspective=0, flipud=0, fliplr=0.

3. **Add 8 new axes** that v2 never explored:
     model_size (n/s/m)              — m may be overkill on 17 images
     box/cls/dfl loss weights        — text/music/staves are very different scales
     dropout                         — small-data regularization
     multi_scale                     — image-size jitter at train time
     warmup_momentum, warmup_bias_lr — fine-tuning warmup behavior
     patience                        — early-stop tuning

4. **Refine grids**: image_size now {768,896,960,1024,1152,1280},
   wider lr/lrf/wd ranges, finer geometric/aug levels.

5. **No warm-start race**: only a single warm-start config (v2 best, mAP=0.4953).
   Stage 1 worker is expected to complete that single trial before fan-out, so
   fan-out workers find study.trials non-empty and the SQLite ask() race that
   killed 3 v2 workers cannot happen.

6. **Better val set**: random_page_split with ratios (0.65, 0.25, 0.10) →
   ~12 train / 4 val / 2 test. Page-level leakage exists but every trial sees
   the same split, so trial *ranking* is preserved (only absolute mAP biased).

Run:
    # Single launch — stage-1 finishes warm-start before fan-out:
    python scripts_v11/optuna_search_v11_aug_v3.py \
        --n-trials 100 --epochs 200 --max-total-trials 600

    # If 1280 OOMs:
    --max-image-size 1152
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

_MODEL_SIZES = ["n", "s", "m"]
_IMAGE_SIZES = [768, 896, 960, 1024, 1152, 1280]
# batch_size lanes (constraint: large img or large model -> small batch)
_BATCH_LARGE = [4, 6, 8]   # image_size>=1024 OR model_size=='m'
_BATCH_SMALL = [8, 12, 16]

# Split ratios for random_page_split — gives ~12/4/2 with 18 raw images.
_SPLIT_RATIOS = (0.65, 0.25, 0.10)

# Single warm-start: v2 trial #20 (mAP=0.4953). Keys must match optuna param
# names in _suggest_params (including conditional names like batch_size_lg).
_KNOWN_GOOD_CONFIGS: list[dict[str, Any]] = [
    {
        # v2 trial #20: mAP=0.4953 — the best config from the previous run.
        # Defaults filled in for axes that v2 didn't search (dropout, patience,
        # box/cls/dfl, multi_scale, warmup_momentum, warmup_bias_lr) come from
        # ultralytics defaults that were implicitly used in v2.
        "model_size": "m",
        "image_size": 960,
        "batch_size_lg": 8,        # 960 + model=m → large lane
        "lr": 9.4e-4,
        "lrf": 0.063,
        "weight_decay": 2.0e-4,
        "warmup_epochs": 2,
        "warmup_momentum": 0.8,    # ultralytics default
        "warmup_bias_lr": 0.1,     # ultralytics default
        "dropout": 0.0,
        "patience": 50,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "degrees": 2.0,
        "translate": 0.15,
        "scale": 0.2,
        "shear": 0.0,
        "multi_scale": False,
        "hsv_s": 0.8,
        "hsv_v": 0.1,
        "mosaic": 0.0,             # close_mosaic intentionally omitted
        "mixup": 0.05,
        "copy_paste": 0.3,
    },
]


# CSV columns
_RESULT_FIELDS = [
    "trial_number", "tag", "status", "error",
    "epochs", "best_map50_95", "best_epoch",
    "best_pt", "run_output_root",
    # model
    "model_size",
    # training schedule (optimizer fixed to AdamW, cos_lr fixed to False)
    "image_size", "batch_size", "learning_rate", "lrf",
    "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "dropout", "patience",
    # loss weights
    "box", "cls", "dfl",
    # geometric aug (perspective fixed to 0; fliplr/flipud forced 0)
    "degrees", "translate", "scale", "shear", "multi_scale",
    # color aug (hsv_h fixed to 0)
    "hsv_s", "hsv_v",
    # mosaic family (erasing fixed to 0)
    "mosaic", "close_mosaic", "mixup", "copy_paste",
]


# ---------------------------------------------------------------------------
# Utilities (unchanged from v2 unless noted)
# ---------------------------------------------------------------------------


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _fmt_float_tag(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def _tag_from_params(params: dict, trial_number: int) -> str:
    return "_".join([
        f"trial{trial_number}",
        f"yolo11{params['model_size']}",
        f"img{params['image_size']}",
        f"bs{params['batch_size']}",
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
# Tunable trainer — extends v2's pass-through with multi_scale + model_size
# ---------------------------------------------------------------------------


class TunableTrainer(MothraTrainer):
    """
    Same idea as v2's TunableTrainer but adds two more knobs:
        - multi_scale (training section)
        - model_size still read via self.config["model"]["size"] (already in
          MothraTrainer); we just allow runtime override via the search.
    """

    _EXTRA_TRAIN_KEYS = (
        "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        "label_smoothing", "dropout",
        "box", "cls", "dfl",
        "multi_scale",
    )
    _EXTRA_AUG_KEYS = ("close_mosaic", "copy_paste", "erasing")

    def train(self, data_yaml_path, resume: bool = False):
        from ultralytics import YOLO

        print("Starting tunable v3 training...")

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

            if "close_mosaic" in train_args:
                cm = int(train_args["close_mosaic"])
                ep = int(train_args["epochs"])
                train_args["close_mosaic"] = max(0, min(cm, max(1, ep - 1)))

        # Honor external CUDA_VISIBLE_DEVICES (see v2 commentary).
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            train_args["device"] = None

        results = model.train(**train_args)
        print("  Training complete.")
        return results


# ---------------------------------------------------------------------------
# Dataset preparation — uses random_page_split with custom ratios
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

    search_root = (repo_root / "outputs" / "yolo11_optuna_search_v3").resolve()
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

    if args.split_type == "manuscript":
        splits, split_log = trainer_for_split.manuscript_aware_split(images_dir, labels_dir)
    else:
        splits, split_log = trainer_for_split.random_page_split(
            images_dir, labels_dir, split_ratios=_SPLIT_RATIOS
        )

    data_yaml_dir = dataset_root / "dataset"
    data_yaml_path = trainer_for_split.organize_yolo_dataset(splits, data_yaml_dir, split_log)
    dataset_yaml_marker_path.write_text(str(data_yaml_path), encoding="utf-8")

    print(f"Generated dataset split (random_page, ratios={_SPLIT_RATIOS}): {data_yaml_path}")
    return repo_root, search_root, data_yaml_path


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial, *, max_image_size: int = 1280) -> dict:
    """
    See module docstring for rationale. Notes:
      - optimizer fixed to AdamW; cos_lr fixed to False; label_smoothing
        fixed to 0.1; hsv_h, erasing, perspective, flipud, fliplr fixed to 0.
      - batch_size lane is conditional on image_size and model_size to fit
        memory (model=m + img=1280 doesn't fit at batch=16).
      - close_mosaic only sampled when mosaic > 0.
    """
    p: dict[str, Any] = {}

    # ===== model & training scale =====
    p["model_size"] = trial.suggest_categorical("model_size", _MODEL_SIZES)

    img_choices = [s for s in _IMAGE_SIZES if s <= max_image_size]
    p["image_size"] = trial.suggest_categorical("image_size", img_choices)

    if p["image_size"] >= 1024 or p["model_size"] == "m":
        p["batch_size"] = trial.suggest_categorical("batch_size_lg", _BATCH_LARGE)
    else:
        p["batch_size"] = trial.suggest_categorical("batch_size_md", _BATCH_SMALL)

    # ===== LR + regularization =====
    p["learning_rate"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    p["lrf"] = trial.suggest_float("lrf", 0.005, 0.3, log=True)
    p["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # ===== warmup =====
    p["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 8)
    p["warmup_momentum"] = trial.suggest_float("warmup_momentum", 0.5, 0.95)
    p["warmup_bias_lr"] = trial.suggest_float("warmup_bias_lr", 0.01, 0.5, log=True)

    # ===== regularization =====
    p["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
    p["patience"] = trial.suggest_categorical("patience", [30, 50, 100])

    # ===== loss weights (defaults 7.5/0.5/1.5) =====
    p["box"] = trial.suggest_categorical("box", [5.0, 7.5, 10.0])
    p["cls"] = trial.suggest_categorical("cls", [0.3, 0.5, 0.7, 1.0])
    p["dfl"] = trial.suggest_categorical("dfl", [1.0, 1.5, 2.0])

    # ===== geometric aug =====
    p["degrees"] = trial.suggest_categorical("degrees", [0.0, 1.0, 2.0, 3.0, 5.0])
    p["translate"] = trial.suggest_categorical("translate", [0.0, 0.05, 0.1, 0.15, 0.2])
    p["scale"] = trial.suggest_categorical("scale", [0.05, 0.1, 0.15, 0.2, 0.3])
    p["shear"] = trial.suggest_categorical("shear", [0.0, 1.0, 2.0, 3.0])
    # multi_scale=True triggers ZeroDivisionError in torch._decomp._compute_scale
    # for some (imgsz, scale) combinations — see v3 fail trials 23/24/25/27/28/31.
    # Disabled until ultralytics/torch fix it.
    p["multi_scale"] = False

    # ===== color aug =====
    p["hsv_s"] = trial.suggest_categorical("hsv_s", [0.5, 0.65, 0.75, 0.8, 0.9])
    p["hsv_v"] = trial.suggest_categorical("hsv_v", [0.05, 0.1, 0.2, 0.3])

    # ===== mosaic family =====
    p["mosaic"] = trial.suggest_categorical("mosaic", [0.0, 0.25, 0.5, 0.75, 1.0])
    if p["mosaic"] > 0.0:
        p["close_mosaic"] = trial.suggest_categorical("close_mosaic", [10, 20, 30, 50, 100])
    else:
        p["close_mosaic"] = 0

    p["mixup"] = trial.suggest_categorical("mixup", [0.0, 0.05, 0.1, 0.15])
    p["copy_paste"] = trial.suggest_categorical("copy_paste", [0.0, 0.1, 0.2, 0.3, 0.5])

    return p


def _enqueue_warm_starts(study: optuna.Study, max_image_size: int) -> int:
    """Seed with v2 best (single config), only if study is empty."""
    if len(study.trials) > 0:
        return 0  # not the first worker; another worker already seeded
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
    parser = argparse.ArgumentParser(description="Optuna search v3 for YOLOv11 on mothra")
    parser.add_argument("--base-config", type=str, default="configs/mothra_base11.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default=None, help="Override training.device.")
    parser.add_argument(
        "--split-type",
        type=str,
        default="random",
        choices=["manuscript", "random"],
        help="Default 'random' (random_page_split) for v3, since manuscript-aware "
             "yields val=1 image with 18 raw pages.",
    )
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/optuna_v11_omr_search_v3",
        help="Where to write best_<tag>.pt and results.csv",
    )
    parser.add_argument("--force-regenerate-dataset", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=1280)

    parser.add_argument("--n-trials", type=int, default=100,
                        help="Per-worker upper bound (use --max-total-trials for global cap).")
    parser.add_argument("--max-total-trials", type=int, default=None,
                        help="Stop this worker once the study has N globally-completed trials.")
    parser.add_argument("--study-name", type=str, default="yolo11_omr_optuna_v3")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_yolo11_omr_v3.db",
        help="Optuna storage URL.",
    )
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    parser.add_argument("--no-warm-start", action="store_true")
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
            print(f"Warm-started study with {n} known-good config(s).")

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

        # ---- model ----
        run_cfg["model"]["size"] = str(params["model_size"])

        # ---- training overrides ----
        run_cfg["training"]["epochs"] = int(args.epochs)
        run_cfg["training"]["save_period"] = -1
        run_cfg["training"]["learning_rate"] = float(params["learning_rate"])
        run_cfg["training"]["batch_size"] = int(params["batch_size"])
        run_cfg["training"]["image_size"] = int(params["image_size"])
        run_cfg["training"]["optimizer"] = "AdamW"           # FIXED
        run_cfg["training"]["cos_lr"] = False                 # FIXED
        run_cfg["training"]["lrf"] = float(params["lrf"])
        run_cfg["training"]["weight_decay"] = float(params["weight_decay"])
        run_cfg["training"]["warmup_epochs"] = float(params["warmup_epochs"])
        run_cfg["training"]["warmup_momentum"] = float(params["warmup_momentum"])
        run_cfg["training"]["warmup_bias_lr"] = float(params["warmup_bias_lr"])
        run_cfg["training"]["label_smoothing"] = 0.1          # FIXED
        run_cfg["training"]["dropout"] = float(params["dropout"])
        run_cfg["training"]["patience"] = int(params["patience"])
        run_cfg["training"]["box"] = float(params["box"])
        run_cfg["training"]["cls"] = float(params["cls"])
        run_cfg["training"]["dfl"] = float(params["dfl"])
        run_cfg["training"]["multi_scale"] = bool(params["multi_scale"])
        if args.device is not None:
            run_cfg["training"]["device"] = args.device

        # ---- augmentation overrides ----
        run_cfg["augmentation"]["hsv_h"] = 0.0                # FIXED
        run_cfg["augmentation"]["hsv_s"] = float(params["hsv_s"])
        run_cfg["augmentation"]["hsv_v"] = float(params["hsv_v"])
        run_cfg["augmentation"]["degrees"] = float(params["degrees"])
        run_cfg["augmentation"]["translate"] = float(params["translate"])
        run_cfg["augmentation"]["scale"] = float(params["scale"])
        run_cfg["augmentation"]["shear"] = float(params["shear"])
        run_cfg["augmentation"]["perspective"] = 0.0          # FIXED
        run_cfg["augmentation"]["mosaic"] = float(params["mosaic"])
        run_cfg["augmentation"]["close_mosaic"] = int(params["close_mosaic"])
        run_cfg["augmentation"]["mixup"] = float(params["mixup"])
        run_cfg["augmentation"]["copy_paste"] = float(params["copy_paste"])
        run_cfg["augmentation"]["erasing"] = 0.0              # FIXED
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
