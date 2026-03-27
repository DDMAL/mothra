#!/usr/bin/env python3
"""
Grid search for YOLOv11 augmentation on mothra.

Key behavior:
1) Fix dataset split once (manuscript-aware by default) and reuse the same data.yaml.
2) For each augmentation combo:
   - train for N epochs (default 200)
   - disable periodic epoch checkpoints (save_period=-1)
   - copy only best.pt to `models/grid_search_v11_aug/best_<tag>.pt`
   - delete last.pt and epoch*.pt to reduce checkpoint clutter
3) Append summary to a single CSV results file so you can compare runs.

Default grid (81 runs):
    degrees: [0, 5, 10]
    scale:   [0.2, 0.3, 0.4]
    mosaic:  [0.0, 0.5, 1.0]
    hsv_s:   [0.3, 0.5, 0.7]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _fmt_float_tag(x: float) -> str:
    # Avoid filenames like 0.3000000004; keep stable formatting.
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _tag_from_combo(degrees: float, scale: float, mosaic: float, hsv_s: float) -> str:
    return f"deg{_fmt_float_tag(degrees)}_sc{_fmt_float_tag(scale)}_mo{_fmt_float_tag(mosaic)}_hsvs{_fmt_float_tag(hsv_s)}"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False)


def _max_map50_95_from_results_csv(results_csv: Path) -> tuple[float, int | None]:
    """
    Return (best_map50_95, best_epoch).
    Expects Ultralytics results.csv header contains `metrics/mAP50-95(B)`.
    """
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    best_map = -1.0
    best_epoch: int | None = None

    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise KeyError(f"No CSV header found in {results_csv}")

        # Be tolerant to Ultralytics column naming differences across versions.
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
                    f"Expected `{col_exact}` or a header containing `mAP50-95`. "
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

    if best_epoch is None:
        return best_map, None
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search augmentation for YOLOv11 (mothra)")
    parser.add_argument("--base-config", type=str, default="configs/mothra_base11.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default=None, help="Override training.device, e.g. 0 or [0,1].")
    parser.add_argument("--split-type", type=str, default="manuscript", choices=["manuscript", "random"])
    parser.add_argument("--images-dir", type=str, default="images", help='Data images dir under data_root, default `images`.')
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/grid_search_v11_aug",
        help="Where to write best_<tag>.pt and results.csv",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit combos for quick tests.")
    parser.add_argument("--start-idx", type=int, default=0, help="Skip first N combos (resume support).")
    parser.add_argument(
        "--force-regenerate-dataset",
        action="store_true",
        help="Regenerate dataset split even if a previous data.yaml exists in --out-dir.",
    )
    args = parser.parse_args()

    base_cfg_path = _resolve_path(args.base_config)
    base_cfg = _load_yaml(base_cfg_path)

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = out_dir / "results.csv"
    dataset_yaml_marker_path = out_dir / "data_yaml_path.txt"

    # Reproducibility for deterministic splits.
    random.seed(42)
    np.random.seed(42)

    repo_root = base_cfg_path.parent.parent.resolve()

    # Resolve config paths relative to repo root (MothraTrainer uses paths relative to CWD).
    for k in ("project_root", "data_root"):
        if k in base_cfg.get("paths", {}):
            v = Path(base_cfg["paths"][k])
            if not v.is_absolute():
                base_cfg["paths"][k] = str((repo_root / v).resolve())

    # Prepare fixed dataset once.
    grid_root = (repo_root / "outputs" / "yolo11_grid_search").resolve()
    grid_root.mkdir(parents=True, exist_ok=True)
    grid_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = grid_root / "datasets" / grid_id

    if dataset_yaml_marker_path.exists() and not args.force_regenerate_dataset:
        data_yaml_path = Path(dataset_yaml_marker_path.read_text(encoding="utf-8").strip()).resolve()
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Marker exists but data.yaml not found: {data_yaml_path}")
        print(f"✅ Reuse dataset split: {data_yaml_path}")
    else:
        dataset_cfg = dict(base_cfg)
        dataset_cfg["paths"] = dict(base_cfg["paths"])
        dataset_cfg["paths"]["output_root"] = str(dataset_root)  # only for bookkeeping
        dataset_cfg["training"] = dict(base_cfg["training"])

        # Save temporary config for dataset generation.
        tmp_dataset_cfg = dataset_root / "dataset_gen_config.yaml"
        _write_yaml(tmp_dataset_cfg, dataset_cfg)

        trainer_for_split = MothraTrainer(str(tmp_dataset_cfg))

        images_dir = trainer_for_split.data_root / args.images_dir
        if not images_dir.exists():
            # if user passes a subdir name like greyscale, original scripts expect data_root/images/<subdir>
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

    # Default grid (81 runs).
    degrees_list = [0, 5, 10]
    scale_list = [0.2, 0.3, 0.4]
    mosaic_list = [0.0, 0.5, 1.0]
    hsv_s_list = [0.3, 0.5, 0.7]

    combos = list(itertools.product(degrees_list, scale_list, mosaic_list, hsv_s_list))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]
    combos = combos[args.start_idx :]

    # Resume support: only skip tags with status=ok and a present best checkpoint.
    done_tags: set[str] = set()
    if results_csv_path.exists():
        with open(results_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "tag" in reader.fieldnames:
                for row in reader:
                    tag = row.get("tag", "")
                    status = row.get("status", "")
                    if not tag:
                        continue
                    best_pt_dest = out_dir / f"best_{tag}.pt"
                    if status == "ok" and best_pt_dest.exists():
                        done_tags.add(tag)

    # Ensure results header.
    if not results_csv_path.exists():
        with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "tag",
                    "degrees",
                    "scale",
                    "mosaic",
                    "hsv_s",
                    "epochs",
                    "best_map50_95",
                    "best_epoch",
                    "best_pt",
                    "run_output_root",
                    "status",
                    "error",
                ]
            )

    # Main loop
    for idx, (degrees, scale, mosaic, hsv_s) in enumerate(combos, start=1):
        tag = _tag_from_combo(degrees, scale, mosaic, hsv_s)
        if tag in done_tags:
            print(f"[{idx}/{len(combos)}] Skip already done: {tag}")
            continue

        print(f"[{idx}/{len(combos)}] Run {tag}")

        run_root = (grid_root / "runs" / tag).resolve()
        run_cfg = dict(base_cfg)  # shallow copy
        run_cfg["paths"] = dict(base_cfg["paths"])
        run_cfg["training"] = dict(base_cfg["training"])
        run_cfg["augmentation"] = dict(base_cfg["augmentation"])

        # Override augmentation params for this combo
        run_cfg["augmentation"]["degrees"] = float(degrees)
        run_cfg["augmentation"]["scale"] = float(scale)
        run_cfg["augmentation"]["mosaic"] = float(mosaic)
        run_cfg["augmentation"]["hsv_s"] = float(hsv_s)

        # Override training params
        run_cfg["training"]["epochs"] = int(args.epochs)
        run_cfg["training"]["save_period"] = -1  # disable epoch*.pt
        if args.device is not None:
            run_cfg["training"]["device"] = args.device

        run_cfg["paths"]["output_root"] = str(run_root)

        tmp_run_cfg = run_root / "config.yaml"
        _write_yaml(tmp_run_cfg, run_cfg)

        # Train
        best_pt_dest = out_dir / f"best_{tag}.pt"
        status = "ok"
        err_msg = ""
        best_map = -1.0
        best_epoch: int | None = None

        try:
            trainer = MothraTrainer(str(tmp_run_cfg))
            trainer.train(data_yaml_path, resume=False)

            # Canonical ultralytics paths (given our v11 trainer uses absolute project).
            best_pt = trainer._yolo_weights_dir() / "best.pt"
            if not best_pt.exists():
                # Fallback: search.
                found = list(run_root.rglob("weights/best.pt"))
                if found:
                    best_pt = max(found, key=lambda p: p.stat().st_mtime)
                else:
                    raise FileNotFoundError(f"best.pt not found under: {run_root}")

            # Copy only best checkpoint.
            best_pt_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, best_pt_dest)

            # Reduce checkpoint clutter.
            _delete_checkpoints_except_best(trainer._yolo_weights_dir(), best_pt)

            # Read best metric from results.csv
            results_csv = run_root / "runs" / "detect" / "results.csv"
            best_map, best_epoch = _max_map50_95_from_results_csv(results_csv)

        except Exception as e:
            status = "failed"
            err_msg = str(e)
            print(f"  ERROR: {err_msg}")

        # Append to results.csv
        with open(results_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    tag,
                    degrees,
                    scale,
                    mosaic,
                    hsv_s,
                    args.epochs,
                    best_map,
                    best_epoch if best_epoch is not None else "",
                    str(best_pt_dest) if best_pt_dest.exists() else "",
                    str(run_root),
                    status,
                    err_msg,
                ]
            )
            f.flush()

        print(f"  -> status={status} best_map50_95={best_map} best_epoch={best_epoch} dest={best_pt_dest}")


if __name__ == "__main__":
    main()

