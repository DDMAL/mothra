#!/usr/bin/env python3
"""
eval_grid_v1_aligned.py

Re-evaluates grid search checkpoints and reports per-class mAP@50,
aligning results with the v1 YOLOv8 baseline metric style
(e.g. "mAP text=0.77, music=0.67").

The existing results CSV records overall mAP@50 / mAP@50-95 from training
curves, but not per-class breakdown.  This script fills that gap by
running model.val() on every checkpoint whose status is "ok" and whose
.pt file actually exists.

Output
------
  <out-dir>/grid_eval_v1_aligned.csv   per-run per-class mAP@50 summary
  <out-dir>/top<N>/<stem>/             _predicted.jpg + .json for the top N
                                        checkpoints ranked by overall mAP@50

Usage
-----
# Evaluate all ok checkpoints and visualise top 3
python scripts_v11/eval_grid_v1_aligned.py \\
    --results-csv models/results_multi_metrics_best_per_metric.csv \\
    --data-yaml   models/grid_search_v11_aug/data_yaml_path.txt \\
    --source      data/holdout \\
    --out-dir     models/grid_eval_v1_aligned \\
    --top-n       3

# Evaluate only (skip inference visualisation)
python scripts_v11/eval_grid_v1_aligned.py \\
    --results-csv models/results_multi_metrics_best_per_metric.csv \\
    --data-yaml   models/grid_search_v11_aug/data_yaml_path.txt \\
    --out-dir     models/grid_eval_v1_aligned \\
    --top-n       0

# Dry run: print what would be evaluated without touching the GPU
python scripts_v11/eval_grid_v1_aligned.py \\
    --results-csv models/results_multi_metrics_best_per_metric.csv \\
    --data-yaml   models/grid_search_v11_aug/data_yaml_path.txt \\
    --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_CONF   = 0.25   # matches v1 and mothra_base11.yaml
DEFAULT_IOU    = 0.7    # matches v1 and mothra_base11.yaml
CLASS_NAMES    = ["text", "music", "staves"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

YOLO_TO_ANNOTATOR_CLASS = {0: 1, 1: 2, 2: 3}

# ── helpers ────────────────────────────────────────────────────────────────────

def _collect_images(images_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.rglob(f"*{ext}"))
        images.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def _yolo_xywhn_to_annotator_bbox(
    box_xywhn: list[float], img_w: int, img_h: int
) -> list[int]:
    cx, cy, w, h = box_xywhn
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    return [round(x), round(y), round(w * img_w), round(h * img_h)]


def _resolve(p: str | Path) -> Path:
    return Path(p).resolve()


def _read_data_yaml_arg(data_yaml_arg: str) -> Path:
    """
    Accept either:
      - a direct path to data.yaml
      - a path to a data_yaml_path.txt marker (written by grid_search_v11_aug.py)
    """
    p = _resolve(data_yaml_arg)
    if not p.exists():
        sys.exit(f"ERROR: --data-yaml path not found: {p}")
    if p.suffix == ".txt":
        content = p.read_text(encoding="utf-8").strip()
        resolved = _resolve(content)
        if not resolved.exists():
            sys.exit(f"ERROR: data.yaml listed in marker file not found: {resolved}")
        return resolved
    return p


def _load_results_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── per-run eval ───────────────────────────────────────────────────────────────

def eval_checkpoint(
    tag: str,
    weights: Path,
    data_yaml: Path,
    split: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> dict | None:
    """
    Run model.val() and return a dict of per-class and overall metrics.
    Returns None on failure.
    """
    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save_json=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"    ERROR evaluating {tag}: {exc}")
        return None

    per_class = list(metrics.box.maps)  # AP@50 per class

    result = {
        "tag": tag,
        "map50_overall": metrics.box.map50,
        "map50_95_overall": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }
    for i, name in enumerate(CLASS_NAMES):
        result[f"map50_{name}"] = per_class[i] if i < len(per_class) else float("nan")

    return result


# ── inference + JSON for top-N ─────────────────────────────────────────────────

def infer_one(
    tag: str,
    weights: Path,
    source: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    out_dir: Path,
    line_width: int = 2,
    font_size: float = 14.0,
) -> None:
    """Write _predicted.jpg + mothra-annotator JSON for every image in source."""
    image_paths = _collect_images(source)
    if not image_paths:
        print(f"    No images in {source}")
        return

    model = YOLO(str(weights))
    timestamp = datetime.now(timezone.utc).isoformat()
    all_sessions: list[dict] = []

    for image_path in image_paths:
        stem = image_path.stem
        image_out_dir = out_dir / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=False,
            save_txt=False,
            verbose=False,
        )
        result = results[0]
        img_h, img_w = result.orig_shape

        plotted = result.plot(
            conf=True,
            line_width=line_width,
            font_size=font_size,
            labels=True,
            boxes=True,
        )
        cv2.imwrite(str(image_out_dir / f"{stem}_predicted.jpg"), plotted)

        annotations: list[dict] = []
        for box in result.boxes:
            yolo_cls = int(box.cls.item())
            ann_cls  = YOLO_TO_ANNOTATOR_CLASS.get(yolo_cls, yolo_cls + 1)
            bbox     = _yolo_xywhn_to_annotator_bbox(box.xywhn[0].tolist(), img_w, img_h)
            annotations.append({
                "id":         str(uuid.uuid4()),
                "classId":    ann_cls,
                "bbox":       bbox,
                "confidence": round(float(box.conf.item()), 4),
                "timestamp":  timestamp,
            })

        session = {
            "imageName":  image_path.name,
            "imageWidth":  img_w,
            "imageHeight": img_h,
            "annotations": annotations,
        }
        (image_out_dir / f"{stem}.json").write_text(json.dumps(session, indent=2))
        all_sessions.append(session)

    agg = out_dir / "all_predictions.json"
    agg.write_text(json.dumps(all_sessions, indent=2))
    n_det = sum(len(s["annotations"]) for s in all_sessions)
    print(f"    {len(all_sessions)} image(s), {n_det} detection(s) → {out_dir}/")


# ── main ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluate grid search checkpoints with per-class mAP@50.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-csv",
        default="models/results_multi_metrics_best_per_metric.csv",
        help="Grid search summary CSV (default: models/results_multi_metrics_best_per_metric.csv)",
    )
    p.add_argument(
        "--data-yaml",
        required=True,
        help=(
            "Path to data.yaml OR to the data_yaml_path.txt marker written by "
            "grid_search_v11_aug.py (e.g. models/grid_search_v11_aug/data_yaml_path.txt)"
        ),
    )
    p.add_argument(
        "--source",
        default=None,
        help="Image directory for top-N inference (required when --top-n > 0)",
    )
    p.add_argument(
        "--out-dir",
        default="models/grid_eval_v1_aligned",
        help="Output directory (default: models/grid_eval_v1_aligned)",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=3,
        help=(
            "Number of top-ranked checkpoints (by mAP@50) to also run inference "
            "and save visualisations for.  Set to 0 to skip inference entirely. "
            "(default: 3)"
        ),
    )
    p.add_argument("--conf",   type=float, default=DEFAULT_CONF, help=f"Confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--iou",    type=float, default=DEFAULT_IOU,  help=f"IoU threshold (default: {DEFAULT_IOU})")
    p.add_argument("--imgsz",  type=int,   default=640,          help="Image size for eval/infer (default: 640)")
    p.add_argument("--device", type=str,   default="0",          help="Device, e.g. 0 or cpu (default: 0)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be evaluated without running anything",
    )
    p.add_argument(
        "--line-width", type=int,   default=2,    help="Bbox line width for overlay images (default: 2)")
    p.add_argument(
        "--font-size",  type=float, default=14.0, help="Label font size for overlay images (default: 14.0)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    results_csv = _resolve(args.results_csv)
    if not results_csv.exists():
        sys.exit(f"ERROR: results CSV not found: {results_csv}")

    data_yaml = _read_data_yaml_arg(args.data_yaml)
    out_dir   = _resolve(args.out_dir)

    rows = _load_results_csv(results_csv)

    # ── filter to ok runs with a checkpoint path ───────────────────────────────
    candidates: list[tuple[str, Path, dict]] = []  # (tag, weights_path, row)
    skipped_no_pt = 0
    skipped_failed = 0

    for row in rows:
        if row.get("status", "").strip() != "ok":
            skipped_failed += 1
            continue
        raw_pt = row.get("best_pt", "").strip()
        if not raw_pt:
            skipped_no_pt += 1
            continue
        pt = _resolve(raw_pt)
        if not pt.exists():
            skipped_no_pt += 1
            continue
        candidates.append((row["tag"].strip(), pt, row))

    print(f"\nGrid search results: {len(rows)} total rows")
    print(f"  status=ok with checkpoint present : {len(candidates)}")
    print(f"  status!=ok (failed/skipped)        : {skipped_failed}")
    print(f"  ok but checkpoint file missing     : {skipped_no_pt}")

    if not candidates:
        sys.exit(
            "\nNo checkpoints found on disk.\n"
            "Grid search .pt files are expected at the paths listed in "
            f"{results_csv} (best_pt column).\n"
            "Make sure the grid search has been run and checkpoints are accessible."
        )

    if args.dry_run:
        print("\n-- DRY RUN -- would evaluate:")
        for tag, pt, _ in candidates:
            print(f"  {tag:50s}  {pt}")
        if args.top_n > 0:
            print(f"\nWould also run inference for top {args.top_n} checkpoints.")
        return

    # ── evaluate all candidates ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_results: list[dict] = []

    print(f"\nEvaluating {len(candidates)} checkpoint(s) on split='{args.split}'...\n")
    for i, (tag, pt, _) in enumerate(candidates, 1):
        print(f"  [{i}/{len(candidates)}] {tag}")
        r = eval_checkpoint(
            tag=tag,
            weights=pt,
            data_yaml=data_yaml,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
        if r is None:
            continue
        eval_results.append(r)
        print(
            f"    mAP@50  overall={r['map50_overall']:.4f}  "
            + "  ".join(f"{n}={r[f'map50_{n}']:.4f}" for n in CLASS_NAMES)
        )

    # ── write summary CSV ──────────────────────────────────────────────────────
    if eval_results:
        csv_path = out_dir / "grid_eval_v1_aligned.csv"
        fieldnames = [
            "tag", "map50_overall", "map50_95_overall",
            "precision", "recall",
        ] + [f"map50_{n}" for n in CLASS_NAMES]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in sorted(eval_results, key=lambda x: x["map50_overall"], reverse=True):
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})

        print(f"\nSaved evaluation summary: {csv_path}")

        # ── print top-5 leaderboard ────────────────────────────────────────────
        ranked = sorted(eval_results, key=lambda x: x["map50_overall"], reverse=True)
        print(f"\n{'─'*70}")
        print(f"  Top-{min(5, len(ranked))} by mAP@50 (v1-aligned, split={args.split})")
        print(f"  {'tag':<45}  {'overall':>8}  " + "  ".join(f"{n:>8}" for n in CLASS_NAMES))
        print(f"{'─'*70}")
        for r in ranked[:5]:
            per = "  ".join(f"{r[f'map50_{n}']:>8.4f}" for n in CLASS_NAMES)
            print(f"  {r['tag']:<45}  {r['map50_overall']:>8.4f}  {per}")
        print(f"{'─'*70}\n")
    else:
        print("\nNo successful evaluations; CSV not written.")
        ranked = []

    # ── inference for top N ────────────────────────────────────────────────────
    if args.top_n > 0 and ranked:
        if not args.source:
            print(
                f"WARNING: --top-n={args.top_n} but --source not provided; "
                "skipping inference visualisation."
            )
        else:
            source = _resolve(args.source)
            if not source.exists():
                print(f"WARNING: --source {source} not found; skipping inference.")
            else:
                top = ranked[:args.top_n]
                print(f"Running inference for top {len(top)} checkpoint(s)...")
                # map tag → weights path
                tag_to_pt = {tag: pt for tag, pt, _ in candidates}
                for rank, r in enumerate(top, 1):
                    tag = r["tag"]
                    pt  = tag_to_pt.get(tag)
                    if pt is None:
                        print(f"  [{rank}] {tag}: checkpoint path not found, skipping")
                        continue
                    infer_dir = out_dir / f"top{args.top_n}" / f"{rank:02d}_{tag}"
                    print(f"  [{rank}] {tag}  (mAP@50={r['map50_overall']:.4f})")
                    infer_one(
                        tag=tag,
                        weights=pt,
                        source=source,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=args.device,
                        out_dir=infer_dir,
                        line_width=args.line_width,
                        font_size=args.font_size,
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
