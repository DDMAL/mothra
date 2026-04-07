#!/usr/bin/env python3
"""
pixel2yolo.py — Convert Pixel.js layer masks to YOLO bounding box format

Issue: https://github.com/DDMAL/mothra/issues/13
Parent: https://github.com/DDMAL/mothra/issues/8

Takes Pixel.js layer PNGs (neumes, staves, text extracted from manuscripts)
and converts them to YOLO-format .txt annotation files for training.

Pixel layers are RGB PNGs on black background:
  - Layer 1 (L1) = neumes  → YOLO class 1 (music)
  - Layer 2 (L2) = staves  → YOLO class 2 (staves)
  - Layer 3 (L3) = text    → YOLO class 0 (text)

Usage:
    # Single folio (provide all 3 layers + original image):
    python pixel2yolo.py \
        --image folio.jpg \
        --l1 PIXEL_L1.png \
        --l2 PIXEL_L2.png \
        --l3 PIXEL_L3.png \
        --output labels/
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


# YOLO class mapping (0-indexed, matching json2yolo.py)
CLASS_MAP = {
    "text": 0,     # Pixel Layer 3
    "music": 1,    # Pixel Layer 1 (neumes)
    "staves": 2,   # Pixel Layer 2
}


def load_mask(path, threshold=15):
    """
    Load a Pixel.js layer PNG and convert to binary mask.
    
    Pixel layers come in two formats:
      - RGB: content on black background (older exports)
      - RGBA: content with alpha, background is transparent (newer exports)
    We composite RGBA onto black, then take max across RGB and threshold.
    """
    img = np.array(Image.open(path))
    
    if img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA: composite onto black background.
            # Background pixels have alpha=0, so RGB*0 = black.
            # Content pixels have alpha=255, so RGB preserved.
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        gray = np.max(img, axis=2)
    else:
        gray = img
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def mask_to_bboxes_staves(binary, min_system_height=20, gap_threshold=15,
                          two_column=False, col_gap_ratio=0.08):
    """
    Convert staves mask to bounding boxes.
    
    Staff systems are groups of 4 horizontal lines close together.
    We use row-projection to find groups of active rows, then compute
    the horizontal extent for each group.
    
    Args:
        binary: Binary mask (255 = content, 0 = background)
        min_system_height: Minimum pixel height for a valid staff system
        gap_threshold: Minimum gap (in rows) between separate systems
        two_column: If True, split staves that span both columns.
                    Uses gap detection first; if no gap is found, falls
                    back to center-split for staves wider than 60% of
                    image width.
        col_gap_ratio: Minimum gap width (as fraction of image width) to
                       consider a column split (default: 0.08 = 8%)
    
    Returns:
        List of (x, y, w, h) bounding boxes in pixel coordinates
    """
    h, w = binary.shape
    
    # Find rows with significant content
    row_sums = binary.sum(axis=1) // 255
    active_rows = np.where(row_sums > 10)[0]
    
    if len(active_rows) == 0:
        return []
    
    # Group active rows into systems based on gaps
    gaps = np.diff(active_rows)
    gap_indices = np.where(gaps > gap_threshold)[0]
    
    systems = []
    prev = 0
    for gi in gap_indices:
        start = active_rows[prev]
        end = active_rows[gi]
        systems.append((start, end))
        prev = gi + 1
    systems.append((active_rows[prev], active_rows[-1]))
    
    # For center-split fallback: a stave spanning >60% of image width
    # is likely a two-column stave that should be split.
    wide_threshold = w * 0.60
    center_x = w // 2
    
    # Convert each system to a bounding box
    bboxes = []
    min_col_gap = int(w * col_gap_ratio)
    
    for y_start, y_end in systems:
        system_height = y_end - y_start
        if system_height < min_system_height:
            continue  # Skip noise
        
        # Find horizontal extent for this system
        row_slice = binary[y_start:y_end + 1, :]
        col_sums = row_slice.sum(axis=0)
        active_cols = np.where(col_sums > 0)[0]
        
        if len(active_cols) == 0:
            continue
        
        x_start = active_cols[0]
        x_end = active_cols[-1]
        box_width = x_end - x_start
        
        if two_column:
            # Strategy 1: gap detection — look for a large horizontal gap
            col_gaps = np.diff(active_cols)
            large_gaps = np.where(col_gaps > min_col_gap)[0]
            
            if len(large_gaps) > 0:
                biggest = large_gaps[np.argmax(col_gaps[large_gaps])]
                left_end = active_cols[biggest]
                right_start = active_cols[biggest + 1]
                
                bboxes.append((x_start, y_start,
                               left_end - x_start, y_end - y_start))
                bboxes.append((right_start, y_start,
                               x_end - right_start, y_end - y_start))
                continue
            
            # Strategy 2: center-split fallback — force split at image
            # center when no gap was found but stave is very wide.
            if box_width > wide_threshold:
                left_w = center_x - x_start
                if left_w > 0:
                    bboxes.append((x_start, y_start, left_w, y_end - y_start))
                right_w = x_end - center_x
                if right_w > 0:
                    bboxes.append((center_x, y_start, right_w, y_end - y_start))
                continue
        
        bboxes.append((x_start, y_start, box_width, y_end - y_start))
    
    return bboxes


def mask_to_bboxes_components(binary, dilation_kernel, min_area=20):
    """
    Convert a mask to bounding boxes using connected components with dilation.
    
    Dilation merges nearby pixels into coherent objects (e.g., parts of a
    neume or characters in a word).
    
    Args:
        binary: Binary mask (255 = content, 0 = background)
        dilation_kernel: (width, height) tuple for the dilation kernel
        min_area: Minimum bounding box area to keep (filters noise)
    
    Returns:
        List of (x, y, w, h) bounding boxes in pixel coordinates
    """
    kw, kh = dilation_kernel
    kernel = np.ones((kh, kw), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
    
    bboxes = []
    for i in range(1, n_labels):  # Skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
        
        bboxes.append((x, y, w, h))
    
    return bboxes


def bboxes_to_yolo(bboxes, class_id, img_width, img_height):
    """
    Convert pixel bounding boxes to YOLO format (normalized center + size).
    
    Args:
        bboxes: List of (x, y, w, h) in pixel coordinates
        class_id: YOLO class ID (0-indexed)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of YOLO format strings: "class_id x_center y_center width height"
    """
    lines = []
    for (x, y, w, h) in bboxes:
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Validate
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 <= w_norm <= 1 and 0 <= h_norm <= 1):
            continue
        
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    return lines


def process_folio(image_path, l1_path, l2_path, l3_path, output_dir,
                  neume_kernel=(12, 12), text_kernel=(9, 7),
                  min_stave_height=20, min_area=20, threshold=15,
                  stave_gap=15, two_column=False):
    """
    Process one folio: convert all Pixel layers to a single YOLO .txt file.
    
    Args:
        image_path: Path to the original manuscript image (for dimensions)
        l1_path: Path to Layer 1 PNG (neumes)
        l2_path: Path to Layer 2 PNG (staves)
        l3_path: Path to Layer 3 PNG (text)
        output_dir: Directory to write the YOLO .txt file
        neume_kernel: Dilation kernel (w, h) for neume grouping
        text_kernel: Dilation kernel (w, h) for text grouping
        min_stave_height: Minimum height for a staff system
        min_area: Minimum component area to keep
        threshold: Pixel intensity threshold for mask binarization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image dimensions from the first available mask
    first_mask = l2_path or l1_path or l3_path
    mask_sample = Image.open(first_mask)
    img_w, img_h = mask_sample.size
    
    # Auto-scale parameters based on image resolution.
    # Kernels were tuned at 1500px width. Scale proportionally for
    # higher-res images so grouping behavior stays consistent.
    REFERENCE_WIDTH = 1500
    scale = img_w / REFERENCE_WIDTH
    
    scaled_neume_kernel = (max(3, round(neume_kernel[0] * scale)),
                          max(3, round(neume_kernel[1] * scale)))
    scaled_text_kernel = (max(3, round(text_kernel[0] * scale)),
                          max(3, round(text_kernel[1] * scale)))
    scaled_stave_gap = max(5, round(stave_gap * scale))
    scaled_min_height = max(10, round(min_stave_height * scale))
    scaled_min_area = max(10, round(min_area * scale * scale))
    
    print(f"  Mask resolution: {img_w}x{img_h}")
    if abs(scale - 1.0) > 0.05:
        print(f"  Auto-scaled (x{scale:.2f}): neume_kernel={scaled_neume_kernel}, text_kernel={scaled_text_kernel}, stave_gap={scaled_stave_gap}")
    if two_column:
        print(f"  Two-column mode: ON (gap ≥ {int(img_w * 0.08)}px, center-split fallback for staves > {int(img_w * 0.60)}px)")
    
    all_lines = []
    
    # --- Layer 2: Staves (class 2) ---
    if l2_path and Path(l2_path).exists():
        mask_staves = load_mask(l2_path, threshold)
        bboxes_staves = mask_to_bboxes_staves(mask_staves, scaled_min_height,
                                              scaled_stave_gap, two_column)
        lines_staves = bboxes_to_yolo(bboxes_staves, CLASS_MAP["staves"], img_w, img_h)
        all_lines.extend(lines_staves)
        print(f"  Staves (class 2): {len(lines_staves)} boxes")
    
    # --- Layer 1: Neumes / Music (class 1) ---
    if l1_path and Path(l1_path).exists():
        mask_neumes = load_mask(l1_path, threshold)
        bboxes_neumes = mask_to_bboxes_components(mask_neumes, scaled_neume_kernel, scaled_min_area)
        lines_neumes = bboxes_to_yolo(bboxes_neumes, CLASS_MAP["music"], img_w, img_h)
        all_lines.extend(lines_neumes)
        print(f"  Music  (class 1): {len(lines_neumes)} boxes")
    
    # --- Layer 3: Text (class 0) ---
    if l3_path and Path(l3_path).exists():
        mask_text = load_mask(l3_path, threshold)
        bboxes_text = mask_to_bboxes_components(mask_text, scaled_text_kernel, scaled_min_area)
        lines_text = bboxes_to_yolo(bboxes_text, CLASS_MAP["text"], img_w, img_h)
        all_lines.extend(lines_text)
        print(f"  Text   (class 0): {len(lines_text)} boxes")
    
    # Write output
    image_stem = Path(image_path).stem
    output_path = output_dir / f"{image_stem}.txt"
    output_path.write_text('\n'.join(all_lines))
    
    print(f"  Total: {len(all_lines)} annotations → {output_path}")
    
    # Auto-detect MOTHRA reference file for comparison
    _compare_with_reference(all_lines, l1_path, l2_path, l3_path)
    
    return output_path


def _compare_with_reference(our_lines, l1_path, l2_path, l3_path):
    """
    Look for a MOTHRA_*_YOLO.txt in the same folder as the Pixel masks.
    If found, print a comparison table.
    """
    from collections import Counter
    
    # Find the folder containing the Pixel masks
    mask_dir = None
    for p in [l2_path, l1_path, l3_path]:
        if p and Path(p).exists():
            mask_dir = Path(p).parent
            break
    
    if mask_dir is None:
        return
    
    # Look for MOTHRA_*_YOLO.txt or MOTHRA_*.txt in that folder
    ref_files = list(mask_dir.glob("MOTHRA_*_YOLO.txt")) + list(mask_dir.glob("MOTHRA_*.txt"))
    # Filter out JSON and non-YOLO files
    ref_files = [f for f in ref_files if f.suffix == '.txt']
    
    if not ref_files:
        return
    
    ref_path = ref_files[0]
    
    with open(ref_path) as f:
        ref_lines = [l.strip() for l in f if l.strip()]
    
    if not ref_lines:
        return
    
    our_classes = Counter(l.split()[0] for l in our_lines)
    ref_classes = Counter(l.split()[0] for l in ref_lines)
    
    class_names = {
        "0": "Text",
        "1": "Music",
        "2": "Staves",
    }
    
    print(f"\n  Comparison with {ref_path.name}:")
    print(f"  {'':>10}  {'Ours':>6}  {'Ref':>6}  {'Diff':>6}")
    for cid in ["0", "1", "2"]:
        name = class_names.get(cid, f"Class {cid}")
        ours = our_classes.get(cid, 0)
        ref = ref_classes.get(cid, 0)
        diff = ours - ref
        print(f"  {name:>10}  {ours:>6}  {ref:>6}  {diff:>+6}")
    
    total_ours = len(our_lines)
    total_ref = len(ref_lines)
    print(f"  {'Total':>10}  {total_ours:>6}  {total_ref:>6}  {total_ours - total_ref:>+6}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pixel.js layer masks to YOLO bounding box format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layer mapping:
  L1 (neumes)  → YOLO class 1 (music)
  L2 (staves)  → YOLO class 2 (staves)  
  L3 (text)    → YOLO class 0 (text)

Examples:
  python pixel2yolo.py --image folio.jpg --l1 L1.png --l2 L2.png --l3 L3.png -o labels/
  python pixel2yolo.py --image folio.jpg --l2 L2.png -o labels/   # staves only
        """
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to original manuscript image')
    parser.add_argument('--l1', type=str, default=None,
                        help='Path to Layer 1 PNG (neumes/music)')
    parser.add_argument('--l2', type=str, default=None,
                        help='Path to Layer 2 PNG (staves)')
    parser.add_argument('--l3', type=str, default=None,
                        help='Path to Layer 3 PNG (text)')
    parser.add_argument('-o', '--output', type=str, default='./yolo_labels',
                        help='Output directory for YOLO .txt files')
    parser.add_argument('--neume-kernel', type=int, nargs=2, default=[12, 12],
                        metavar=('W', 'H'),
                        help='Dilation kernel for neume grouping (default: 12 12)')
    parser.add_argument('--text-kernel', type=int, nargs=2, default=[9, 7],
                        metavar=('W', 'H'),
                        help='Dilation kernel for text grouping (default: 9 7)')
    parser.add_argument('--threshold', type=int, default=15,
                        help='Pixel intensity threshold for mask binarization (default: 15)')
    parser.add_argument('--min-area', type=int, default=20,
                        help='Minimum component area to keep (default: 20)')
    parser.add_argument('--stave-gap', type=int, default=15,
                        help='Max gap (pixels) between staff lines in the same system. '
                             'Increase for manuscripts with widely-spaced lines (default: 15)')
    parser.add_argument('--two-column', action='store_true',
                        help='Enable two-column layout detection. Tries gap '
                             'detection first; falls back to center-split for '
                             'staves wider than 60%% of image width.')
    
    args = parser.parse_args()
    
    if not any([args.l1, args.l2, args.l3]):
        parser.error("At least one layer (--l1, --l2, or --l3) must be provided")
    
    print("=" * 60)
    print("pixel2yolo — Pixel.js masks → YOLO bounding boxes")
    print("=" * 60)
    print(f"  Image: {args.image}")
    print(f"  L1 (neumes): {args.l1 or 'not provided'}")
    print(f"  L2 (staves): {args.l2 or 'not provided'}")
    print(f"  L3 (text):   {args.l3 or 'not provided'}")
    if args.two_column:
        print(f"  Two-column:  enabled")
    print()
    
    process_folio(
        image_path=args.image,
        l1_path=args.l1,
        l2_path=args.l2,
        l3_path=args.l3,
        output_dir=args.output,
        neume_kernel=tuple(args.neume_kernel),
        text_kernel=tuple(args.text_kernel),
        min_area=args.min_area,
        threshold=args.threshold,
        stave_gap=args.stave_gap,
        two_column=args.two_column,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()