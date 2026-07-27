#!/usr/bin/env python3
"""
eval_batch.py — batch evaluation across pages and pipeline variants.

Reads a manifest CSV that lists (GT, prediction, image, metadata) triples and
runs eval_page.evaluate() on each row, collecting all results into a single
output CSV.

Manifest CSV columns (order does not matter, extra columns are ignored):
    page_name    — human-readable page id, used in output
    image        — path to the page image file
    gt_txt       — path to the ground-truth YOLO .txt file
    gt_source    — annotator label or 'corrected', 'raw', etc.
    pred_json    — path to the pipeline *_stafflines.json output
    variant      — pipeline variant label, e.g. 'sauvola_no_bgr'

Optional manifest columns:
    staffline_class   — YOLO class id for stafflines (default 0)

Usage:
    python eval_batch.py --manifest eval_manifest.csv --output eval_results.csv

To produce a quick summary after running:
    python eval_batch.py --manifest eval_manifest.csv --output eval_results.csv --summarize
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_page import CSV_FIELDS, evaluate, _print_summary

# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

REQUIRED_MANIFEST_COLS = {
    "page_name",
    "image",
    "gt_txt",
    "gt_source",
    "pred_json",
    "variant",
}


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest {manifest_path} is empty.")

    missing = REQUIRED_MANIFEST_COLS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    return rows


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_aggregate_summary(results: list[dict]) -> None:
    if not results:
        return

    variants = sorted({r["variant"] for r in results})
    metrics_to_agg = ["precision", "recall", "f1", "split_ratio", "mean_y_mae_px"]

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (mean ± std across pages)")
    print("=" * 70)

    for variant in variants:
        rows = [r for r in results if r["variant"] == variant]
        print(f"\n  Variant: {variant}  ({len(rows)} page(s))")
        for m in metrics_to_agg:
            vals = [float(r[m]) for r in rows if r[m] not in ("", "nan")]
            if not vals:
                continue
            a = np.array(vals)
            print(
                f"    {m:<20s}  {a.mean():.4f} ± {a.std():.4f}"
                f"  [min {a.min():.4f}  max {a.max():.4f}]"
            )

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch staffline evaluation from a manifest CSV."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="CSV manifest file (see module docstring for columns).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path — results are written here (overwritten each run).",
    )
    parser.add_argument(
        "--staffline-class",
        type=int,
        default=0,
        help="YOLO class id for stafflines (overrides manifest column if set).",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print an aggregate summary table after processing.",
    )
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest)
    print(f"Loaded manifest: {len(manifest_rows)} row(s) from {args.manifest}")

    all_results = []
    n_ok = 0
    n_err = 0

    for i, row in enumerate(manifest_rows, start=1):
        page = row["page_name"].strip()
        image = Path(row["image"].strip())
        gt_txt = Path(row["gt_txt"].strip())
        gt_source = row["gt_source"].strip()
        pred_json = Path(row["pred_json"].strip())
        variant = row["variant"].strip()
        staffline_class = int(
            row.get("staffline_class", args.staffline_class) or args.staffline_class
        )

        print(f"\n[{i}/{len(manifest_rows)}] {page}  |  {variant}")

        missing_files = [p for p in (image, gt_txt, pred_json) if not p.exists()]
        if missing_files:
            print(f"  ERROR: missing file(s): {[str(p) for p in missing_files]}")
            n_err += 1
            continue

        try:
            metrics = evaluate(
                gt_path=gt_txt,
                pred_path=pred_json,
                image_path=image,
                staffline_class=staffline_class,
                gt_source=gt_source,
                variant=variant,
                page_name=page,
            )
            _print_summary(metrics)
            all_results.append(metrics)
            n_ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_err += 1

    # --- Write output CSV ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nWrote {len(all_results)} result(s) to {args.output}")
    if n_err:
        print(f"  ({n_err} row(s) skipped due to errors)")

    if args.summarize and all_results:
        _print_aggregate_summary(all_results)


if __name__ == "__main__":
    main()
