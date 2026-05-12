#!/usr/bin/env python3
"""
grid_search_v11_size_res.py — v2 grid search varying (model_size, image_size).

Successor to ``grid_search_v11_aug.py``. Augmentation params are held at the
winners found in v1 (``degrees=0, scale=0.3, mosaic=0, hsv_s=0.5``); this grid
explores model capacity vs input resolution because

  - the v1 winner already saturates text/music mAP@50 (≈ v1 baseline) but
    collapses to ~0 on ``staves``;
  - staves bboxes have mean aspect ratio w/h≈36 and mean height ≈0.9 % of
    image height — at ``imgsz=640`` they project to ~6 pixels tall, near
    YOLO's smallest feature stride, which makes higher input resolution the
    most plausible fix.

Plan rationale lives in ``next_grid_search_plan.md``.

Defaults: 8 runs (2 model sizes × 4 image sizes). The script aborts the
single run if it OOMs after auto-halving batch_size; it continues with the
rest of the grid.

Each run produces:
  - ``outputs/yolo11_grid_search_v2/runs/<tag>/`` — ultralytics training dir
  - ``models/grid_search_v11_size_res/best_<tag>.pt`` — copy of best.pt
  - one row in ``models/grid_search_v11_size_res/results.csv``
  - if ``--post-eval`` (default on), ``models/eval_v1_aligned_v2/<tag>/``
    with the v1-aligned per-class mAP@50 and holdout viz.

Example::

    python scripts_v11/grid_search_v11_size_res.py \\
        --data-yaml outputs/yolo11_grid_search_v2/datasets/<ts>/dataset/data.yaml \\
        --epochs 300 \\
        --device 5

    # Resume after interruption — finished tags are skipped
    python scripts_v11/grid_search_v11_size_res.py --data-yaml ... --device 5

Build the data.yaml first with ``scripts_v11/build_split_v2.py``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent

# Fixed augmentation profile = v1 winner
FIXED_AUG = {
    "degrees": 0.0,
    "scale": 0.3,
    "mosaic": 0.0,
    "hsv_s": 0.5,
}

# Default grid axes
MODEL_SIZES = ["s", "m"]
IMAGE_SIZES = [640, 1024, 1280, 1600]

# (model_size, imgsz) → starting batch size (will halve on OOM)
DEFAULT_BATCH = {
    ("s", 640): 32,
    ("s", 1024): 16,
    ("s", 1280): 12,
    ("s", 1600): 8,
    ("m", 640): 16,
    ("m", 1024): 12,
    ("m", 1280): 8,
    ("m", 1600): 4,
}


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _tag(model_size: str, imgsz: int) -> str:
    return f"yolo11{model_size}_img{imgsz}"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _max_map50_95_from_results_csv(results_csv: Path) -> tuple[float, int | None]:
    """Lift the best mAP50-95 row from Ultralytics' per-epoch results.csv."""
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    best = -1.0
    best_epoch: int | None = None
    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95(B)", "metrics/mAP50-95"):
                if key in row and row[key]:
                    val = float(row[key])
                    if val > best:
                        best = val
                        best_epoch = int(float(row.get("epoch", -1))) + 1
                    break
    return best, best_epoch


def _delete_checkpoints_except_best(weights_dir: Path, best_pt: Path) -> None:
    if not weights_dir.exists():
        return
    for pt in weights_dir.glob("*.pt"):
        if pt.resolve() == best_pt.resolve():
            continue
        try:
            pt.unlink()
        except OSError:
            pass


def _run_post_eval(
    best_pt: Path,
    data_yaml: Path,
    source: Path,
    out_dir: Path,
    imgsz: int,
    device: str,
) -> int:
    """Call eval_v1_aligned.py to attach per-class mAP@50 + holdout viz."""
    eval_script = SCRIPT_DIR / "eval_v1_aligned.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--mode",
        "both",
        "--weights",
        str(best_pt),
        "--data",
        str(data_yaml),
        "--split",
        "test",
        "--source",
        str(source),
        "--out-dir",
        str(out_dir),
        "--imgsz",
        str(imgsz),
        "--device",
        device,
    ]
    print("  >> post-eval:", " ".join(cmd))
    return subprocess.call(cmd)


def _train_one(
    base_cfg: dict,
    data_yaml: Path,
    model_size: str,
    imgsz: int,
    epochs: int,
    device: str,
    run_root: Path,
) -> Path:
    """Train one (model_size, imgsz) run; return path to best.pt."""
    cfg = {
        "paths": dict(base_cfg["paths"]),
        "training": dict(base_cfg["training"]),
        "augmentation": dict(base_cfg.get("augmentation", {})),
        "classes": base_cfg["classes"],
        "model": dict(base_cfg["model"]),
    }
    cfg["paths"]["output_root"] = str(run_root)
    cfg["model"]["size"] = model_size
    cfg["training"]["epochs"] = epochs
    cfg["training"]["image_size"] = imgsz
    cfg["training"]["batch_size"] = DEFAULT_BATCH.get((model_size, imgsz), 8)
    cfg["training"]["save_period"] = -1
    cfg["training"]["device"] = device
    cfg["augmentation"].update(FIXED_AUG)

    run_cfg_path = run_root / "config.yaml"
    _write_yaml(run_cfg_path, cfg)

    trainer = MothraTrainer(str(run_cfg_path))
    trainer.train(data_yaml, resume=False)

    best_pt = trainer._yolo_weights_dir() / "best.pt"
    if not best_pt.exists():
        found = list(run_root.rglob("weights/best.pt"))
        if not found:
            raise FileNotFoundError(f"best.pt missing under {run_root}")
        best_pt = max(found, key=lambda p: p.stat().st_mtime)
    return best_pt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--base-config", default="configs/mothra_base11.yaml")
    parser.add_argument(
        "--data-yaml",
        required=True,
        help="data.yaml from build_split_v2.py (frozen split).",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'.")
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        default=MODEL_SIZES,
        choices=["n", "s", "m", "l", "x"],
    )
    parser.add_argument("--image-sizes", nargs="+", type=int, default=IMAGE_SIZES)
    parser.add_argument(
        "--out-dir",
        default="models/grid_search_v11_size_res",
        help="Where best_<tag>.pt and results.csv live.",
    )
    parser.add_argument(
        "--grid-output-root",
        default="outputs/yolo11_grid_search_v2/runs",
        help="Where per-run training dirs go.",
    )
    parser.add_argument(
        "--no-post-eval",
        action="store_true",
        help="Skip the v1-aligned eval+infer pass after each run.",
    )
    parser.add_argument(
        "--holdout-source",
        default="data/holdout",
        help="Directory of holdout images for post-eval infer mode.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit total runs (testing).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Skip first N combos (resume).",
    )
    args = parser.parse_args(argv)

    base_cfg_path = _resolve(args.base_config)
    base_cfg = _load_yaml(base_cfg_path)
    data_yaml = _resolve(args.data_yaml)
    if not data_yaml.is_file():
        sys.exit(f"ERROR: data.yaml not found: {data_yaml}")

    # Resolve repo-relative paths inside the config.
    for k in ("project_root", "data_root"):
        if k in base_cfg.get("paths", {}):
            v = Path(base_cfg["paths"][k])
            if not v.is_absolute():
                base_cfg["paths"][k] = str((REPO_ROOT / v).resolve())

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = out_dir / "results.csv"

    grid_root = _resolve(args.grid_output_root)
    grid_root.mkdir(parents=True, exist_ok=True)

    holdout_source = _resolve(args.holdout_source)

    eval_out_root = _resolve("models/eval_v1_aligned_v2") if not args.no_post_eval else None

    combos = list(itertools.product(args.model_sizes, args.image_sizes))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]
    combos = combos[args.start_idx :]

    # Resume: skip rows already marked ok with checkpoint present.
    done_tags: set[str] = set()
    if results_csv_path.exists():
        with results_csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok" and (out_dir / f"best_{row['tag']}.pt").exists():
                    done_tags.add(row["tag"])
    else:
        with results_csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "tag",
                    "model_size",
                    "image_size",
                    "epochs",
                    "best_map50_95",
                    "best_epoch",
                    "best_pt",
                    "run_output_root",
                    "post_eval_dir",
                    "status",
                    "error",
                    "timestamp",
                ]
            )

    for idx, (model_size, imgsz) in enumerate(combos, start=1):
        tag = _tag(model_size, imgsz)
        if tag in done_tags:
            print(f"[{idx}/{len(combos)}] skip done: {tag}")
            continue

        print(f"\n[{idx}/{len(combos)}] >>> {tag}")
        run_root = (grid_root / tag).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        status = "ok"
        err = ""
        best_pt_dest = out_dir / f"best_{tag}.pt"
        best_map = -1.0
        best_epoch: int | None = None
        post_eval_dir = ""

        try:
            best_pt = _train_one(
                base_cfg,
                data_yaml,
                model_size,
                imgsz,
                args.epochs,
                args.device,
                run_root,
            )
            shutil.copy2(best_pt, best_pt_dest)
            _delete_checkpoints_except_best(best_pt.parent, best_pt)

            results_csv = run_root / "runs" / "detect" / "results.csv"
            best_map, best_epoch = _max_map50_95_from_results_csv(results_csv)

            if not args.no_post_eval:
                eval_dir = eval_out_root / tag  # type: ignore[union-attr]
                rc = _run_post_eval(
                    best_pt_dest,
                    data_yaml,
                    holdout_source,
                    eval_dir,
                    imgsz,
                    args.device,
                )
                if rc != 0:
                    print(f"  WARN: post-eval exit code {rc} for {tag}")
                post_eval_dir = str(eval_dir)

        except Exception as e:  # broad on purpose — one bad combo shouldn't kill grid
            status = "failed"
            err = repr(e)
            print(f"  ERROR: {err}")

        with results_csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    tag,
                    model_size,
                    imgsz,
                    args.epochs,
                    f"{best_map:.5f}" if best_map >= 0 else "",
                    best_epoch if best_epoch is not None else "",
                    str(best_pt_dest) if status == "ok" else "",
                    str(run_root),
                    post_eval_dir,
                    status,
                    err,
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )

        print(f"[{idx}/{len(combos)}] <<< {tag}  status={status}  mAP50-95={best_map:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
