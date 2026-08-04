#!/usr/bin/env python3
"""
detect_stafflines.py — standalone YOLO inference for the stave detector.

Loads a single-class stave-detection YOLO checkpoint once, runs it over one
image or a directory of images, and writes one plain YOLO-format .txt label
file per image (class cx cy w h, normalized — always written, even when a
page has zero detections) plus a manifest.txt of every processed image's
absolute path, so a calling shell script never has to special-case a
missing label file or re-implement recursive image discovery.

Detections keep the model's own native class id (no remapping) — the
bundled stave detector is single-class, so every line is written as class 0.

Usage:
    python detect_stafflines.py --weights staff-finding/models/stave_detector_fulldata.pt \
        --images-dir path/to/images --output path/to/yolo_txt [--conf 0.25] [--device cpu]
    python detect_stafflines.py --weights ... --image path/to/one.jpg --output path/to/yolo_txt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_images(images_dir: Path) -> list[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.rglob(f"*{ext}"))
        images.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(p for p in set(images) if not p.name.startswith("._"))


def detect_and_write(
    weights: Path,
    images: list[Path],
    output_dir: Path,
    conf: float,
    iou: float,
    device: str | None,
) -> None:
    model = YOLO(str(weights))  # load once for the whole batch
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    for img_path in images:
        kwargs = dict(source=str(img_path), conf=conf, iou=iou, save=False, verbose=False)
        if device:
            kwargs["device"] = device
        result = model.predict(**kwargs)[0]

        lines = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        (output_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        manifest_lines.append(str(img_path.resolve()))
        n = len(lines)
        print(f"  {img_path.name}: {n} detection{'s' if n != 1 else ''}")

    (output_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")
    print(f"\nWrote {len(images)} label file(s) to {output_dir}/")
    print(f"Manifest: {output_dir / 'manifest.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the stave-detector YOLO model, writing plain YOLO-format .txt labels."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="A single image file.")
    source.add_argument(
        "--images-dir", type=Path, help="Directory to search recursively for images."
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the stave-detector .pt checkpoint."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory to write <stem>.txt + manifest.txt into."
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default 0.7 = Ultralytics' own implicit "
        "default, so omitting this flag reproduces prior behavior exactly). "
        "Lower values suppress more overlapping duplicate staffline "
        "detections at inference time -- see staff-finding/dox/ for the "
        "layer_1_3801 x 5013 reference-page analysis this default is based "
        "on. Has no effect on end2end=True checkpoints (e.g. an NMS-free "
        "YOLO26 export) -- only affects standard NMS-based checkpoints such "
        "as the current default stave_detector_fulldata.pt.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:N (default: let ultralytics choose).")
    args = parser.parse_args()

    images = [args.image] if args.image else collect_images(args.images_dir)
    if not images:
        parser.error(f"no images found in {args.images_dir}")

    detect_and_write(args.weights, images, args.output, args.conf, args.iou, args.device)


if __name__ == "__main__":
    main()
