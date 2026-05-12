#!/usr/bin/env python3
"""
Optuna search for YOLOv11 augmentation/training on mothra.

Key behavior:
1) Fix dataset split once (manuscript-aware by default) and reuse the same data.yaml.
2) For each Optuna trial:
   - sample a hyperparameter set
   - train for N epochs
   - disable periodic epoch checkpoints (save_period=-1)
   - copy only best.pt to `out_dir/best_<tag>.pt`
   - delete extra checkpoints to reduce clutter
3) Append summary to a single CSV results file.
4) Persist Optuna study to SQLite so runs can be resumed.

Recommended first use:
- Start with 20-40 trials
- Keep search space discrete / conservative
- Reuse the same dataset split for fair comparison
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _fmt_float_tag(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def _tag_from_params(params: dict, trial_number: int) -> str:
    parts = [
        f"trial{trial_number}",
        f"lr{_fmt_float_tag(params['learning_rate'])}",
        f"bs{params['batch_size']}",
        f"img{params['image_size']}",
        f"deg{_fmt_float_tag(params['degrees'])}",
        f"tr{_fmt_float_tag(params['translate'])}",
        f"sc{_fmt_float_tag(params['scale'])}",
        f"sh{_fmt_float_tag(params['shear'])}",
        f"mo{_fmt_float_tag(params['mosaic'])}",
        f"mu{_fmt_float_tag(params['mixup'])}",
        f"hsvs{_fmt_float_tag(params['hsv_s'])}",
        f"hsvv{_fmt_float_tag(params['hsv_v'])}",
    ]
    return "_".join(parts)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def _max_map50_95_from_results_csv(results_csv: Path) -> tuple[float, int | None]:
    """
    Return (best_map50_95, best_epoch).
    Expects Ultralytics results.csv header contains `metrics/mAP50-95(B)`,
    but we tolerate version differences.
    """
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
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial_number",
                "tag",
                "learning_rate",
                "batch_size",
                "image_size",
                "degrees",
                "translate",
                "scale",
                "shear",
                "mosaic",
                "mixup",
                "hsv_s",
                "hsv_v",
                "epochs",
                "best_map50_95",
                "best_epoch",
                "best_pt",
                "run_output_root",
                "status",
                "error",
            ]
        )


def _append_result_row(
    results_csv_path: Path,
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
    with open(results_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                trial_number,
                tag,
                params["learning_rate"],
                params["batch_size"],
                params["image_size"],
                params["degrees"],
                params["translate"],
                params["scale"],
                params["shear"],
                params["mosaic"],
                params["mixup"],
                params["hsv_s"],
                params["hsv_v"],
                epochs,
                best_map,
                best_epoch if best_epoch is not None else "",
                str(best_pt_dest) if best_pt_dest.exists() else "",
                str(run_root),
                status,
                err_msg,
            ]
        )
        f.flush()


def _prepare_fixed_dataset(base_cfg: dict, base_cfg_path: Path, args) -> tuple[Path, Path, Path]:
    """
    Returns:
        repo_root, search_root, data_yaml_path
    """
    dataset_yaml_marker_path = _resolve_path(args.out_dir) / "data_yaml_path.txt"

    # Reproducibility for deterministic split generation
    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = base_cfg_path.parent.parent.resolve()

    # Resolve config paths relative to repo root
    for k in ("project_root", "data_root"):
        if k in base_cfg.get("paths", {}):
            v = Path(base_cfg["paths"][k])
            if not v.is_absolute():
                base_cfg["paths"][k] = str((repo_root / v).resolve())

    search_root = (repo_root / "outputs" / "yolo11_optuna_search").resolve()
    search_root.mkdir(parents=True, exist_ok=True)

    dataset_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = search_root / "datasets" / dataset_id

    if dataset_yaml_marker_path.exists() and not args.force_regenerate_dataset:
        data_yaml_path = Path(dataset_yaml_marker_path.read_text(encoding="utf-8").strip()).resolve()
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Marker exists but data.yaml not found: {data_yaml_path}")
        print(f"✅ Reuse dataset split: {data_yaml_path}")
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

    print(f"✅ Generated dataset split: {data_yaml_path}")
    return repo_root, search_root, data_yaml_path


def _suggest_params(trial: optuna.Trial) -> dict:
    """
    Conservative search space for OMR / manuscript layout detection.

    Why these:
    - image_size: thin staff lines and small details benefit from larger sizes
    - lr / batch_size: training stability and final quality
    - geometric aug: moderate only, because page layout should remain realistic
    - mosaic/mixup: may help or hurt document layout; worth testing in a limited range
    - hsv_s / hsv_v: parchment / lighting / ink variation
    """
    params = {
        # training
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 5e-4, 1e-3, 2e-3]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16]),
        "image_size": trial.suggest_categorical("image_size", [640, 960, 1280]),

        # augmentation
        "degrees": trial.suggest_categorical("degrees", [0.0, 3.0, 5.0, 7.0, 10.0]),
        "translate": trial.suggest_categorical("translate", [0.0, 0.05, 0.1, 0.15]),
        "scale": trial.suggest_categorical("scale", [0.1, 0.2, 0.3, 0.4]),
        "shear": trial.suggest_categorical("shear", [0.0, 1.0, 2.0, 3.0]),
        "mosaic": trial.suggest_categorical("mosaic", [0.0, 0.5, 1.0]),
        "mixup": trial.suggest_categorical("mixup", [0.0, 0.05, 0.1]),
        "hsv_s": trial.suggest_categorical("hsv_s", [0.2, 0.35, 0.5, 0.65, 0.8]),
        "hsv_v": trial.suggest_categorical("hsv_v", [0.1, 0.25, 0.4, 0.55]),
    }

    # Optional gentle constraints:
    # If no mosaic, mixup usually matters less; but we don't hard-code that.
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search for YOLOv11 on mothra OMR/layout task")
    parser.add_argument("--base-config", type=str, default="configs/mothra_base11.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default=None, help="Override training.device, e.g. 0 or [0,1].")
    parser.add_argument("--split-type", type=str, default="manuscript", choices=["manuscript", "random"])
    parser.add_argument("--images-dir", type=str, default="images", help="Images dir under data_root.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/optuna_v11_omr_search",
        help="Where to write best_<tag>.pt and results.csv",
    )
    parser.add_argument(
        "--force-regenerate-dataset",
        action="store_true",
        help="Regenerate dataset split even if a previous data.yaml exists in --out-dir.",
    )

    # Optuna
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--study-name", type=str, default="yolo11_omr_optuna")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_yolo11_omr.db",
        help="Optuna storage URL. SQLite is easiest for resume.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler type.",
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
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        tag = _tag_from_params(params, trial.number)
        run_root = (search_root / "runs" / tag).resolve()

        print(f"\n[Trial {trial.number}] {tag}")

        run_cfg = dict(base_cfg)
        run_cfg["paths"] = dict(base_cfg["paths"])
        run_cfg["model"] = dict(base_cfg.get("model", {}))
        run_cfg["training"] = dict(base_cfg["training"])
        run_cfg["augmentation"] = dict(base_cfg["augmentation"])
        run_cfg["evaluation"] = dict(base_cfg.get("evaluation", {}))

        # ---- overrides ----
        run_cfg["training"]["epochs"] = int(args.epochs)
        run_cfg["training"]["save_period"] = -1  # disable epoch*.pt
        run_cfg["training"]["learning_rate"] = float(params["learning_rate"])
        run_cfg["training"]["batch_size"] = int(params["batch_size"])
        run_cfg["training"]["image_size"] = int(params["image_size"])
        if args.device is not None:
            run_cfg["training"]["device"] = args.device

        run_cfg["augmentation"]["degrees"] = float(params["degrees"])
        run_cfg["augmentation"]["translate"] = float(params["translate"])
        run_cfg["augmentation"]["scale"] = float(params["scale"])
        run_cfg["augmentation"]["shear"] = float(params["shear"])
        run_cfg["augmentation"]["mosaic"] = float(params["mosaic"])
        run_cfg["augmentation"]["mixup"] = float(params["mixup"])
        run_cfg["augmentation"]["hsv_s"] = float(params["hsv_s"])
        run_cfg["augmentation"]["hsv_v"] = float(params["hsv_v"])

        run_cfg["paths"]["output_root"] = str(run_root)

        tmp_run_cfg = run_root / "config.yaml"
        _write_yaml(tmp_run_cfg, run_cfg)

        best_pt_dest = out_dir / f"best_{tag}.pt"
        status = "ok"
        err_msg = ""
        best_map = -1.0
        best_epoch: int | None = None

        try:
            trainer = MothraTrainer(str(tmp_run_cfg))
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
                results_csv_path=results_csv_path,
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
            results_csv_path=results_csv_path,
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

    study.optimize(objective, n_trials=args.n_trials)

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