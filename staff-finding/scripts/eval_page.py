#!/usr/bin/env python3
"""
eval_page.py — per-page evaluation of staffline detection pipeline output.

Compares pipeline JSOMR output (fitted centerlines) against ground-truth YOLO
annotations to produce line-level precision/recall, split ratio, and centerline
y-MAE for matched pairs.

A GT staffline box and a predicted fit are considered a match when the
prediction's mean page-absolute y-center is within MATCH_THRESHOLD_MULTIPLIER *
scale_unit pixels of the GT box's y-center.  Matching is 1:1 (greedy by
ascending distance).

Usage:
    python eval_page.py \\
        --gt   staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \\
        --pred staff-finding/e2e_tests/.../GentAnt1475_0017_AC_rightcrop_stafflines.json \\
        --image staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \\
        --gt-source corrected_kyrie \\
        --variant sauvola_no_bgr \\
        --output metrics.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from yolo_io import filter_to_class, parse_yolo_txt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# SF-9 fix: the bundled single-class stave-detector checkpoint remaps the
# merged-project's class 2 down to raw class 0 (see
# train_staffline_detector.py's SOURCE_CLASS=2/TARGET_CLASS=0 remap) -- the
# previous comment here cited a run_inference.py/CLASS_NAMES that doesn't
# exist anywhere in this repo. `2` silently matched nothing against this
# model's real (class-0) output.
STAFFLINE_CLASS_DEFAULT = 0
MATCH_THRESHOLD_MULTIPLIER = 0.5  # fraction of scale_unit → max y-distance for a match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_dims(image_path: Path) -> tuple[int, int]:
    """Return (width, height) by reading the image header."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def _pred_page_y(item: dict) -> np.ndarray:
    """Page-absolute y-values for a JSOMR staffline item.

    Prefers the `centerline_page` block (already page-absolute) when present;
    falls back to `centerline` + `bounding_box` for legacy crop-local records
    that predate `centerline_page`. That fallback assumes `bounding_box` is
    present -- true for detected lines, never true for interpolated ones (no
    crop of their own) -- so an interpolated record without `centerline_page`
    (a stale fixture from before the writer emitted it for those too) returns
    empty rather than crashing on bounding_box=None.
    """
    centerline_page = item.get("centerline_page")
    if centerline_page is not None:
        return np.array(centerline_page["y_values"], dtype=float)
    if item.get("bounding_box") is None:
        return np.array([], dtype=float)
    return (
        np.array(item["centerline"]["y_values"], dtype=float)
        + item["bounding_box"]["uly"]
    )


def _pred_page_x(item: dict) -> np.ndarray:
    """Page-absolute x positions corresponding to each y_value.

    See _pred_page_y for the bounding_box=None fallback case; kept in sync
    with it since callers zip the two together.
    """
    centerline_page = item.get("centerline_page")
    if centerline_page is not None:
        x_start = centerline_page["x_start"]
        n = len(centerline_page["y_values"])
        return np.arange(x_start, x_start + n)
    if item.get("bounding_box") is None:
        return np.array([], dtype=int)
    x_start = item["centerline"]["x_start"]
    n = len(item["centerline"]["y_values"])
    return np.arange(x_start, x_start + n) + item["bounding_box"]["ulx"]


def _mean_page_y(item: dict) -> float:
    return float(np.mean(_pred_page_y(item)))


def _y_mae(gt_cy: float, pred_item: dict, gt_ulx: int, gt_lrx: int) -> float | None:
    """Mean absolute y-error between GT box y-center and matched prediction.

    Measured over the x-overlap of the GT box and the prediction's x-range.
    Returns None when there is no x-overlap.
    """
    xs = _pred_page_x(pred_item)
    ys = _pred_page_y(pred_item)
    mask = (xs >= gt_ulx) & (xs <= gt_lrx)
    if not mask.any():
        return None
    return float(np.mean(np.abs(ys[mask] - gt_cy)))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    gt_path: Path,
    pred_path: Path,
    image_path: Path,
    staffline_class: int = STAFFLINE_CLASS_DEFAULT,
    gt_source: str = "unknown",
    variant: str = "unknown",
    page_name: str | None = None,
) -> dict:
    """Run evaluation for a single (GT, prediction) pair.

    Returns a flat dict suitable for writing as one CSV row.
    """
    image_width, image_height = _image_dims(image_path)

    # --- Ground truth ---
    gt_detections = filter_to_class(parse_yolo_txt(gt_path), staffline_class)
    gt_boxes = []
    for d in gt_detections:
        ulx, uly, lrx, lry = d.to_pixel_box(image_width, image_height)
        gt_boxes.append(
            {
                "ulx": ulx,
                "uly": uly,
                "lrx": lrx,
                "lry": lry,
                "cy": (uly + lry) / 2.0,
            }
        )

    # --- Predictions ---
    with pred_path.open() as f:
        pred_items = json.load(f)
    # Skip items without a fitted centerline (no_y_position_available)
    pred_items = [p for p in pred_items if p["centerline"]["y_values"]]

    if not gt_boxes:
        print(f"  WARNING: no GT stafflines in {gt_path} (class {staffline_class})")
    if not pred_items:
        print(f"  WARNING: no valid predictions in {pred_path}")

    scale_unit = pred_items[0]["scale_unit"] if pred_items else 15.0
    threshold = MATCH_THRESHOLD_MULTIPLIER * scale_unit

    # --- Greedy 1:1 matching by ascending y-distance ---
    pred_mean_ys = [_mean_page_y(p) for p in pred_items]

    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, py in enumerate(pred_mean_ys):
            dist = abs(gt["cy"] - py)
            if dist <= threshold:
                candidates.append((dist, gi, pi))
    candidates.sort()

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    mae_values: list[float] = []

    for dist, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        mae = _y_mae(
            gt_boxes[gi]["cy"], pred_items[pi], gt_boxes[gi]["ulx"], gt_boxes[gi]["lrx"]
        )
        if mae is not None:
            mae_values.append(mae)

    n_gt = len(gt_boxes)
    n_pred = len(pred_items)
    n_matched = len(matched_gt)

    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    split_ratio = n_pred / n_gt if n_gt > 0 else float("nan")
    mean_y_mae = float(np.mean(mae_values)) if mae_values else float("nan")

    return {
        "page": page_name or gt_path.stem,
        "variant": variant,
        "gt_source": gt_source,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "n_unmatched_pred": n_pred - n_matched,
        "n_unmatched_gt": n_gt - n_matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "split_ratio": round(split_ratio, 4),
        "mean_y_mae_px": round(mean_y_mae, 2),
        "match_threshold_px": round(threshold, 2),
        "scale_unit_px": round(scale_unit, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "page",
    "variant",
    "gt_source",
    "timestamp",
    "n_gt",
    "n_pred",
    "n_matched",
    "n_unmatched_pred",
    "n_unmatched_gt",
    "precision",
    "recall",
    "f1",
    "split_ratio",
    "mean_y_mae_px",
    "match_threshold_px",
    "scale_unit_px",
]


def _print_summary(metrics: dict) -> None:
    print(f"\n  Page:        {metrics['page']}")
    print(f"  Variant:     {metrics['variant']}")
    print(f"  GT source:   {metrics['gt_source']}")
    print(f"  GT lines:    {metrics['n_gt']}")
    print(
        f"  Pred lines:  {metrics['n_pred']}  (split ratio {metrics['split_ratio']:.3f})"
    )
    print(f"  Matched:     {metrics['n_matched']}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  Mean y-MAE:  {metrics['mean_y_mae_px']:.2f} px")
    print(
        f"  Threshold:   {metrics['match_threshold_px']:.1f} px  "
        f"(= 0.5 × {metrics['scale_unit_px']:.1f} px scale unit)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate staffline detection: JSOMR predictions vs. YOLO GT."
    )
    parser.add_argument(
        "--gt", required=True, type=Path, help="Ground-truth YOLO .txt file."
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="Pipeline JSOMR *_stafflines.json output.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Page image (used to convert YOLO normalised coords).",
    )
    parser.add_argument(
        "--staffline-class",
        type=int,
        default=STAFFLINE_CLASS_DEFAULT,
        help=f"YOLO class id for stafflines (default {STAFFLINE_CLASS_DEFAULT}).",
    )
    parser.add_argument(
        "--gt-source",
        default="unknown",
        help="Label for the GT file — annotator name, 'corrected', etc.",
    )
    parser.add_argument(
        "--variant",
        default="unknown",
        help="Pipeline variant label, e.g. 'sauvola_no_bgr'.",
    )
    parser.add_argument(
        "--page-name",
        default=None,
        help="Override page identifier in output (defaults to GT filename stem).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Append one CSV row to this file (created with header if new).",
    )
    args = parser.parse_args()

    metrics = evaluate(
        gt_path=args.gt,
        pred_path=args.pred,
        image_path=args.image,
        staffline_class=args.staffline_class,
        gt_source=args.gt_source,
        variant=args.variant,
        page_name=args.page_name,
    )

    _print_summary(metrics)

    if args.output:
        write_header = not args.output.exists()
        with args.output.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"  Appended to {args.output}")


if __name__ == "__main__":
    main()
