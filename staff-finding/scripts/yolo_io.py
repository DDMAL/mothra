"""
YOLO detection I/O.

Parsing of YOLO .txt files (the Ultralytics one-detection-per-line format) and
the data structure for one detection. Class-filtering helper included.

Separated from the page driver so it can be reused by other scripts (batch
runners, evaluation harnesses, etc.) without dragging in BGR or component-filter
dependencies.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoloDetection:
    """One detection parsed from a YOLO-format .txt line."""
    class_id: int
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float

    def to_pixel_box(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel (ulx, uly, lrx, lry)."""
        cx = self.x_center_norm * image_width
        cy = self.y_center_norm * image_height
        w = self.width_norm * image_width
        h = self.height_norm * image_height
        ulx = int(round(cx - w / 2))
        uly = int(round(cy - h / 2))
        lrx = int(round(cx + w / 2))
        lry = int(round(cy + h / 2))
        return ulx, uly, lrx, lry


def parse_yolo_txt(yolo_path: Path) -> list[YoloDetection]:
    """Parse a YOLO .txt file. One detection per line: class cx cy w h.

    Malformed or unparseable lines are reported to stdout and skipped; the
    function does not raise on parse failures.
    """
    detections = []
    with yolo_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"  Skipping malformed line {line_num} in {yolo_path}: {line!r}")
                continue
            try:
                detections.append(YoloDetection(
                    class_id=int(parts[0]),
                    x_center_norm=float(parts[1]),
                    y_center_norm=float(parts[2]),
                    width_norm=float(parts[3]),
                    height_norm=float(parts[4]),
                ))
            except ValueError:
                print(f"  Skipping unparseable line {line_num} in {yolo_path}: {line!r}")
    return detections


def filter_to_class(
    detections: list[YoloDetection],
    class_id: int,
) -> list[YoloDetection]:
    """Return only detections with the given class id."""
    return [d for d in detections if d.class_id == class_id]