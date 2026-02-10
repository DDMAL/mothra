#!/usr/bin/env python3
"""
Convert web annotator JSON files to YOLO format

Usage:
    python convert_to_yolo.py --input-dir collected_jsons/ --output-dir yolo_dataset/
    
This script processes all *_annotations.json files from the web annotator
and converts them to YOLO format .txt files.
"""

import json
import argparse
from pathlib import Path


CLASS_NAMES_MAPPING = {
    "text": 0,
    "music": 1,
    "staves": 2,
}


def convert_json_to_yolo(json_path, output_dir):
    """
    Convert a single web annotator JSON to YOLO format
    
    Args:
        json_path: Path to the JSON annotation file
        output_dir: Directory to save YOLO format .txt file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_w = data['image_width']
    image_h = data['image_height']
    image_path = data['image_path']
    
    if not data['annotations']:
        print(f"⚠️  {json_path.name}: No annotations, skipping")
        return
    
    yolo_lines = []
    for ann in data['annotations']:
        x1, y1, x2, y2 = ann['bbox']
        
        # Get class ID (web tool uses 1-indexed, YOLO uses 0-indexed)
        class_name = ann['class_name']
        class_id = CLASS_NAMES_MAPPING.get(class_name, ann['class_id'] - 1)
        
        # Convert to YOLO format (normalized center + width/height)
        x_center = (x1 + x2) / (2 * image_w)
        y_center = (y1 + y2) / (2 * image_h)
        width = (x2 - x1) / image_w
        height = (y2 - y1) / image_h
        
        # Validate normalized coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                0 <= width <= 1 and 0 <= height <= 1):
            print(f"⚠️  Invalid coordinates in {json_path.name}, annotation #{ann.get('id', '?')}")
            continue
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if not yolo_lines:
        print(f"⚠️  {json_path.name}: No valid annotations after conversion")
        return
    
    # Write YOLO format file
    # Extract original image name from the JSON filename or data
    image_name = Path(image_path).stem if image_path else json_path.stem.replace('_annotations', '')
    output_path = Path(output_dir) / f"{image_name}.txt"
    
    output_path.write_text('\n'.join(yolo_lines))
    
    print(f"✓ {json_path.name} -> {output_path.name} ({len(yolo_lines)} annotations)")


def batch_convert(input_dir, output_dir):
    """
    Convert all JSON files in input directory to YOLO format
    
    Args:
        input_dir: Directory containing JSON annotation files
        output_dir: Directory to save YOLO format files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.glob('*_annotations.json'))
    
    if not json_files:
        print(f"No annotation files found in {input_dir}")
        print("Looking for files matching pattern: *_annotations.json")
        return
    
    print(f"\nFound {len(json_files)} annotation files")
    print(f"Converting to YOLO format...\n")
    
    # Convert each file
    success_count = 0
    for json_file in sorted(json_files):
        try:
            convert_json_to_yolo(json_file, output_path)
            success_count += 1
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
    
    # Write classes.txt
    classes_file = output_path / "classes.txt"
    classes_file.write_text('\n'.join(sorted(CLASS_NAMES_MAPPING.keys(), 
                                            key=lambda x: CLASS_NAMES_MAPPING[x])))
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successfully converted: {success_count}/{len(json_files)} files")
    print(f"  Output directory: {output_path}")
    print(f"  Classes file: {classes_file}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert web annotator JSON files to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_yolo.py --input-dir annotations/ --output-dir yolo_labels/
  python convert_to_yolo.py -i collected_jsons/ -o dataset/labels/train/

The script looks for files matching *_annotations.json in the input directory.
Output will be .txt files in YOLO format plus a classes.txt file.
        """
    )
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                       help='Directory containing JSON annotation files from web tool')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                       help='Directory to save YOLO format .txt files')
    
    args = parser.parse_args()
    
    batch_convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()