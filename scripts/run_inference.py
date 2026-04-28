"""
Batch inference script for Mothra.

Runs a trained YOLOv8 model on a directory of images and writes:
  - <output_dir>/<image_stem>/<image_stem>_predicted.jpg  (bbox overlay)
  - <output_dir>/<image_stem>/<image_stem>.json           (mothra-annotator format)

The JSON format matches https://github.com/DDMAL/mothra-annotator so outputs
can be loaded directly into the annotator for correction.
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

# YOLO class index → mothra-annotator classId (1-indexed)
YOLO_TO_ANNOTATOR_CLASS = {0: 1, 1: 2, 2: 3}
CLASS_NAMES = {0: "text", 1: "music", 2: "staves"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_images(images_dir: Path) -> list[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.rglob(f"*{ext}"))
        images.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def yolo_box_to_annotator_bbox(box_xywhn, img_w: int, img_h: int) -> list[int]:
    """Convert normalized YOLO center xywh → pixel [x, y, w, h] (top-left origin)."""
    cx, cy, w, h = box_xywhn
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    return [round(x), round(y), round(w * img_w), round(h * img_h)]


def run_inference(
    images_dir: Path,
    weights_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
):
    model = YOLO(str(weights_path))
    image_paths = collect_images(images_dir)

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} images. Running inference...")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    all_predictions = []

    for image_path in image_paths:
        stem = image_path.stem
        image_out_dir = output_dir / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False,
        )

        result = results[0]
        img_h, img_w = result.orig_shape

        # Save annotated image directly (avoids YOLO's project/runs/detect layout)
        overlay = result.plot()  # BGR numpy array with boxes drawn
        overlay_path = image_out_dir / f"{stem}_predicted.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        annotations = []
        for box in result.boxes:
            yolo_class_id = int(box.cls.item())
            annotator_class_id = YOLO_TO_ANNOTATOR_CLASS.get(yolo_class_id, yolo_class_id + 1)
            bbox = yolo_box_to_annotator_bbox(box.xywhn[0].tolist(), img_w, img_h)
            annotations.append(
                {
                    "id": str(uuid.uuid4()),
                    "classId": annotator_class_id,
                    "bbox": bbox,
                    "confidence": round(float(box.conf.item()), 4),
                    "timestamp": timestamp,
                }
            )

        session = {
            "imageName": image_path.name,
            "imageWidth": img_w,
            "imageHeight": img_h,
            "annotations": annotations,
        }

        json_path = image_out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(session, indent=2))

        n = len(annotations)
        print(f"  {image_path.name}: {n} detection{'s' if n != 1 else ''}")
        all_predictions.append(session)

    all_json_path = output_dir / "all_predictions.json"
    all_json_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"\nDone. Outputs written to {output_dir}/")
    print(f"Aggregated results: {all_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Mothra batch inference with annotator-compatible JSON output")
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory to search recursively for images",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/v1/yolov8m_17images_7568boxes.pt",
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference-outputs",
        help="Directory to write outputs into",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    args = parser.parse_args()

    run_inference(
        images_dir=Path(args.images_dir),
        weights_path=Path(args.weights),
        output_dir=Path(args.output_dir),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )


if __name__ == "__main__":
    main()
