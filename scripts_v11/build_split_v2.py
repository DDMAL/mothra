#!/usr/bin/env python3
"""
build_split_v2.py — create a frozen manuscript-aware split for v2 grid search.

Improvements over the 2026-03-26 split used by `grid_search_v11_aug.py`:

1. Picks up *all* labelled images, including new ones unzipped from
   `inference-outputs/First round of corrections - zips/*.zip`.
2. Re-samples the manuscript shuffle until every split contains at least
   `--min-staves` instances of the staves class (class 2). The previous
   split happened to land 194 staves boxes in test, but the constraint
   was implicit; making it explicit prevents the next dataset edit from
   silently breaking the staves metric.
3. Writes a self-describing `split_log.json` (no DATA LEAKAGE warnings —
   manuscript-level only) and a ready-to-train `data.yaml`.

Usage::

    # Preview only — no files written
    python scripts_v11/build_split_v2.py --dry-run

    # Build a frozen split using corrected annotations from the zips
    python scripts_v11/build_split_v2.py \\
        --images-dir data/images \\
        --extra-labels-zips 'inference-outputs/First round of corrections - zips/*.zip' \\
        --output-root outputs/yolo11_grid_search_v2/datasets \\
        --seed 42 \\
        --min-staves 50

After this runs, point the grid script at
`<output-root>/<timestamp>/dataset/data.yaml`.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mothra_trainer import MothraTrainer  # noqa: E402

CLASS_NAMES = ["text", "music", "staves"]
STAVES_CLASS_ID = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _extract_manuscript_id(filename: str) -> str:
    """Reuse the trainer's manuscript-ID heuristic without instantiating it."""
    return MothraTrainer.extract_manuscript_id(None, filename)  # type: ignore[arg-type]


def _read_labels(label_path: Path) -> list[int]:
    """Return the list of class IDs found in a YOLO label file."""
    try:
        with label_path.open() as f:
            return [int(line.split()[0]) for line in f if line.strip()]
    except (ValueError, OSError):
        return []


def _collect_pairs(images_dir: Path, label_lookup: dict[str, Path]) -> list[dict]:
    """Pair every image in `images_dir` with its label by stem."""
    pairs: list[dict] = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_lookup.get(img.stem)
        if label is None:
            print(f"  skip (no label): {img.name}")
            continue
        pairs.append(
            {
                "image": img,
                "label": label,
                "manuscript": _extract_manuscript_id(img.name),
                "classes": _read_labels(label),
            }
        )
    return pairs


def _stage_labels(
    primary_labels_dir: Path,
    extra_zips: list[str],
    work_dir: Path,
) -> dict[str, Path]:
    """
    Build a {stem → label_path} mapping.

    Labels from `extra_zips` (treated as corrected ground truth) win over
    labels in `primary_labels_dir`.
    """
    lookup: dict[str, Path] = {}

    if primary_labels_dir.exists():
        for txt in primary_labels_dir.glob("*.txt"):
            lookup[txt.stem] = txt
        print(f"  primary labels: {len(lookup)} from {primary_labels_dir}")

    for pattern in extra_zips:
        for zip_path in glob.glob(pattern):
            stage = work_dir / Path(zip_path).stem
            stage.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(stage)
            for txt in stage.rglob("*.txt"):
                lookup[txt.stem] = txt
        print(f"  after merging {pattern}: {len(lookup)} labels")

    return lookup


def _try_split(
    pairs: list[dict],
    split_ratios: tuple[float, float, float],
    min_staves_each: int,
    rng: random.Random,
    attempts: int,
) -> tuple[dict, dict] | None:
    """Shuffle manuscript IDs until every split has enough staves."""
    by_ms: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_ms[p["manuscript"]].append(p)

    ms_ids = list(by_ms.keys())
    n = len(ms_ids)
    n_train = max(1, int(n * split_ratios[0]))
    n_val = max(1, int(n * split_ratios[1])) if n > 2 else 0
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
        n_train = n - n_val - 1

    for attempt in range(attempts):
        rng.shuffle(ms_ids)
        train_ms = ms_ids[:n_train]
        val_ms = ms_ids[n_train : n_train + n_val]
        test_ms = ms_ids[n_train + n_val :]

        splits = {
            "train": [p for ms in train_ms for p in by_ms[ms]],
            "val": [p for ms in val_ms for p in by_ms[ms]],
            "test": [p for ms in test_ms for p in by_ms[ms]],
        }

        ok = all(
            sum(c == STAVES_CLASS_ID for p in splits[s] for c in p["classes"])
            >= min_staves_each
            for s in ("train", "val", "test")
        )
        if ok:
            return splits, {
                "train_manuscripts": train_ms,
                "val_manuscripts": val_ms,
                "test_manuscripts": test_ms,
                "attempt": attempt,
            }

    return None


def _write_dataset(
    splits: dict,
    out_root: Path,
    split_log_extra: dict,
    split_ratios: tuple[float, float, float],
    min_staves_each: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = out_root / timestamp / "dataset"
    dataset.mkdir(parents=True, exist_ok=False)

    for name, pairs in splits.items():
        (dataset / name / "images").mkdir(parents=True)
        (dataset / name / "labels").mkdir(parents=True)
        for p in pairs:
            shutil.copy(p["image"], dataset / name / "images" / p["image"].name)
            shutil.copy(p["label"], dataset / name / "labels" / p["label"].name)

    data_yaml = {
        "path": str(dataset),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    (dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    log = {
        "split_type": "manuscript_aware_v2",
        "split_ratios": list(split_ratios),
        "min_staves_each_split": min_staves_each,
        "class_counts": {
            name: dict(_per_class_counts(pairs)) for name, pairs in splits.items()
        },
        "image_counts": {name: len(pairs) for name, pairs in splits.items()},
        "timestamp": datetime.now().isoformat(),
        **split_log_extra,
    }
    (dataset / "split_log.json").write_text(json.dumps(log, indent=2))

    return dataset / "data.yaml"


def _per_class_counts(pairs: list[dict]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for p in pairs:
        for c in p["classes"]:
            if 0 <= c < len(CLASS_NAMES):
                counter[CLASS_NAMES[c]] += 1
    return counter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument(
        "--primary-labels-dir",
        default="data/yolo_labels",
        help="Existing YOLO .txt directory; overridden by zips when stems collide.",
    )
    parser.add_argument(
        "--extra-labels-zips",
        action="append",
        default=[],
        help="Glob(s) to zip archives carrying corrected labels. Repeatable.",
    )
    parser.add_argument("--output-root", default="outputs/yolo11_grid_search_v2/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument(
        "--min-staves",
        type=int,
        default=50,
        help="Required staves bbox count in each of train/val/test.",
    )
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute split candidates and print stats; do not copy files.",
    )
    args = parser.parse_args(argv)

    images_dir = Path(args.images_dir).resolve()
    primary_labels_dir = Path(args.primary_labels_dir).resolve()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        print(">> staging labels")
        label_lookup = _stage_labels(primary_labels_dir, args.extra_labels_zips, tmp)

        print(">> pairing images")
        pairs = _collect_pairs(images_dir, label_lookup)
        if not pairs:
            sys.exit(f"ERROR: no labelled images found under {images_dir}")
        manuscripts = sorted({p["manuscript"] for p in pairs})
        print(f"  total pairs: {len(pairs)} across {len(manuscripts)} manuscripts")
        for ms in manuscripts:
            n = sum(1 for p in pairs if p["manuscript"] == ms)
            print(f"    - {ms}: {n} pages")

        print(">> searching for a staves-balanced split")
        rng = random.Random(args.seed)
        attempt = _try_split(
            pairs,
            tuple(args.split_ratios),
            args.min_staves,
            rng,
            args.max_attempts,
        )
        if attempt is None:
            sys.exit(
                f"ERROR: no split satisfies min_staves={args.min_staves} per split "
                f"after {args.max_attempts} attempts. Lower --min-staves or annotate "
                f"more staves-bearing manuscripts."
            )
        splits, split_extra = attempt

        for name in ("train", "val", "test"):
            counts = _per_class_counts(splits[name])
            print(
                f"  {name:5s}: imgs={len(splits[name]):3d}  "
                + "  ".join(f"{c}={counts[c]}" for c in CLASS_NAMES)
            )

        if args.dry_run:
            print(">> dry-run; nothing written")
            return 0

        out_root = Path(args.output_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        data_yaml_path = _write_dataset(
            splits,
            out_root,
            split_extra,
            tuple(args.split_ratios),
            args.min_staves,
        )

    print(f">> data.yaml: {data_yaml_path}")
    print(f">> split_log: {data_yaml_path.parent / 'split_log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
