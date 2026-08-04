"""
Train a YOLO26 staff-line detector on the addtl-gt dataset.

Model: YOLO26 (https://docs.ultralytics.com/models/yolo26)
  - NMS-free end-to-end architecture (yolo26n.pt / yolo26s.pt / ...)
  - MuSGD optimizer (applied automatically by ultralytics)

Class mapping
  Source TXT files use class 2 (staffline) from the multi-class project model.
  This script remaps 2 → 0 so the trained model is a clean single-class detector.
  If you run inference with this model and pipe results into the existing
  pipeline, pass --staffline-class 0 instead of the default 2.

Augmentation strategy
  Staff lines appear in both red and black inks on aged parchment.
  We apply heavy hue/saturation shifts so the model generalises across ink
  colours (hsv_s=0.7 collapses saturation toward zero, covering greyscale scans).
  Minor brightness jitter covers the range of photographic conditions found
  in the dataset.

Usage
  python scripts/train_staffline_detector.py [options]

  --source-dir   path to addtl-gt (default: staff-finding/addtl-gt)
  --dataset-dir  where to build the YOLO dataset tree (default: staff-finding/addtl-gt/stafflines)
  --model        base weights, e.g. yolo26n.pt  (default: yolo26n.pt)
  --epochs       training epochs (default: 300)
  --imgsz        input resolution (default: 1280)
  --batch        batch size, -1 = auto (default: -1)
  --device       cpu / 0 / 0,1 (default: auto)
  --project      run output directory (default: staff-finding/runs/stafflines)
  --name         run name (default: yolo26_stafflines)
  --resume       path to a previous run's last.pt to resume training
  --val-split    fraction of images held out for validation (default: 0.2)
  --seed         random seed for the train/val split (default: 42)
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent  # staff-finding/

    p = argparse.ArgumentParser(description="Train YOLO26 staff-line detector")
    p.add_argument("--source-dir", type=Path, default=here / "addtl-gt")
    p.add_argument("--dataset-dir", type=Path, default=here / "addtl-gt" / "stafflines")
    p.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Base YOLO26 weights (yolo26n/s/m/l/x.pt). "
        "A nano model is recommended with only 26 training images.",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input resolution. 1280 is recommended: staff lines are "
        "thin horizontal features that need spatial detail.",
    )
    p.add_argument("--batch", type=int, default=-1, help="-1 = ultralytics auto-batch")
    p.add_argument(
        "--device",
        default="",
        help="Training device: '' = auto, '0' = first GPU, 'cpu'",
    )
    p.add_argument("--project", type=Path, default=here / "runs" / "stafflines")
    p.add_argument("--name", default="yolo26_stafflines")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from last.pt of a previous run",
    )
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

SOURCE_CLASS = 2  # class ID used in the source .txt files
TARGET_CLASS = 0  # single-class model: staffline = 0


def remap_label_file(src: Path, dst: Path) -> None:
    """Copy a YOLO .txt label, keeping only SOURCE_CLASS boxes remapped to TARGET_CLASS."""
    lines: list[str] = []
    with src.open() as fh:
        for raw in fh:
            parts = raw.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls != SOURCE_CLASS:
                continue  # drop non-staffline annotations
            lines.append(" ".join([str(TARGET_CLASS), *parts[1:]]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def build_dataset(
    source_dir: Path, dataset_dir: Path, val_split: float, seed: int
) -> Path:
    """
    Build the YOLO folder tree under dataset_dir and return the path to
    the generated dataset YAML file.

    Layout produced:
      dataset_dir/
        images/train/   images/val/
        labels/train/   labels/val/
      dataset_dir/stafflines.yaml
    """
    images = sorted(source_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg files found in {source_dir}")

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_split))
    val_set = set(p.stem for p in shuffled[:n_val])

    print(f"\nDataset split ({len(images)} images total):")
    print(f"  train : {len(images) - n_val}")
    print(f"  val   : {n_val}  {sorted(val_set)}\n")

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_path in images:
        split = "val" if img_path.stem in val_set else "train"

        # image
        dst_img = dataset_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # label (remap class)
        src_lbl = img_path.with_suffix(".txt")
        if not src_lbl.exists():
            print(f"  WARNING: no label for {img_path.name}, skipping")
            continue
        dst_lbl = dataset_dir / "labels" / split / src_lbl.name
        remap_label_file(src_lbl, dst_lbl)

    yaml_path = dataset_dir / "stafflines.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: staffline\n"
    )
    print(f"Dataset YAML written to {yaml_path}\n")
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace, yaml_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run:\n" "  pip install 'ultralytics>=8.4'"
        ) from exc

    if args.resume:
        model = YOLO(str(args.resume))
    else:
        model = YOLO(args.model)

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=str(args.project),
        name=args.name,
        resume=bool(args.resume),
        # ── Augmentation ────────────────────────────────────────────────────
        # Hue shift: wide range lets the model see red-ink staves become
        # blue/green/yellow/black during training → colour-agnostic features.
        hsv_h=0.5,
        # Saturation: 0→saturated colour, high→washed out / near-greyscale.
        # Staff lines in old photographs are often desaturated.
        hsv_s=0.7,
        # Brightness/value jitter for varying photographic exposure.
        hsv_v=0.4,
        # Minor rotation: staff lines are nearly horizontal but scans can be
        # slightly skewed.
        degrees=2.0,
        # Horizontal flip: left↔right symmetry is fine for staff lines.
        fliplr=0.5,
        # No vertical flip — upside-down manuscripts would confuse notation.
        flipud=0.0,
        # Scale jitter.
        scale=0.5,
        # Mosaic: paste four training images together — very useful for the
        # small dataset (26 images) because it creates many novel compositions.
        mosaic=1.0,
        # MixUp: gentle blending, helps with overlapping annotations.
        mixup=0.1,
        # Copy-paste: disabled (not useful for full-page layout images).
        copy_paste=0.0,
        # ── Optimiser / scheduler ────────────────────────────────────────────
        # YOLO26 uses MuSGD by default (no need to set optimizer explicitly).
        lr0=0.01,
        lrf=0.01,  # final LR = lr0 * lrf
        warmup_epochs=5,
        patience=50,  # early-stop if no mAP gain for 50 epochs
        # ── Misc ─────────────────────────────────────────────────────────────
        # Save top-3 checkpoints so you can recover if last.pt is the worst.
        save_period=50,
        exist_ok=True,
        verbose=True,
    )

    print("\nTraining complete.")
    best = args.project / args.name / "weights" / "best.pt"
    if best.exists():
        print(f"Best weights: {best}")
    else:
        print("best.pt not found — check the run output directory.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("YOLO26 Staff-Line Detector — Training Setup")
    print("=" * 60)
    print(f"  Source GT dir : {args.source_dir}")
    print(f"  Dataset dir   : {args.dataset_dir}")
    print(f"  Base model    : {args.model}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Input size    : {args.imgsz}")
    print(f"  Val split     : {args.val_split}")
    print(f"  Seed          : {args.seed}")
    print()

    yaml_path = build_dataset(
        source_dir=args.source_dir,
        dataset_dir=args.dataset_dir,
        val_split=args.val_split,
        seed=args.seed,
    )

    train(args, yaml_path)


if __name__ == "__main__":
    main()
