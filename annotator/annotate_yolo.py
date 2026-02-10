#!/usr/bin/env python3
"""
Multi-Class Bounding Box Annotation Tool for YOLO Training
Adapted from the Musical Scribal Attribution (muscrat) annotation tool by Kyrie Bouressa

Usage:
    python annotate_yolo.py --image path/to/manuscript.png
    
Controls:
    - Click and drag to select bounding box
    - Number keys 1-9: Select class for next annotation
    - 's' key: Save current annotation
    - 'u' key or Ctrl+Z: Undo last annotation
    - 'd' key: Delete annotation (hover over box)
    - 'e' key: Export annotations to YOLO format
    - 'q' key: Quit and save session
    
Class Selection:
    Press number key before drawing box:
    1 = text
    2 = music  
    3 = staves
    (Add more classes as needed in CLASS_NAMES)
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
import sys


# Define your classes here - modify as needed
CLASS_NAMES = {
    1: "text",
    2: "music", 
    3: "staves",
    # Add more classes as needed:
    # 4: "clef",
    # 5: "neume",
    # etc.
}

# Colors for each class (BGR format)
CLASS_COLORS = {
    1: (255, 0, 0),      # Blue for text
    2: (0, 255, 0),      # Green for music
    3: (0, 0, 255),      # Red for staves
    4: (255, 255, 0),    # Cyan
    5: (255, 0, 255),    # Magenta
    6: (0, 255, 255),    # Yellow
    7: (128, 0, 128),    # Purple
    8: (255, 128, 0),    # Orange
    9: (0, 128, 255),    # Light Blue
}


class YOLOAnnotator:
    def __init__(self, image_path, output_dir="annotations"):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get image dimensions
        self.img_height, self.img_width = self.image.shape[:2]
        
        # State
        self.annotations = []
        self.undo_stack = []  # Stack for undo functionality
        self.current_class = 1  # Default to first class
        self.drawing = False
        self.selection_start = None
        self.selection_end = None
        self.hover_annotation_idx = None
        
        # Window setup
        self.window_name = "YOLO Annotator"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Session file
        self.session_file = self.output_dir / f"{self.image_path.stem}_session.json"
        self.load_session()
        
        # Display initial state
        self.update_display()
    
    def load_session(self):
        """Load previous annotation session if exists"""
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                self.annotations = data.get('annotations', [])
                print(f"Loaded {len(self.annotations)} previous annotations")
    
    def save_session(self):
        """Save annotation session to JSON"""
        session_data = {
            'image_path': str(self.image_path),
            'image_width': self.img_width,
            'image_height': self.img_height,
            'timestamp': datetime.now().isoformat(),
            'class_names': CLASS_NAMES,
            'annotations': self.annotations
        }
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"Session saved: {len(self.annotations)} annotations")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bounding box selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.selection_end = None
            self.drawing = True
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.selection_end = (x, y)
                self.update_display()
            else:
                # Check if hovering over existing annotation (for delete)
                self.hover_annotation_idx = self.find_annotation_at_point(x, y)
                self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selection_start is not None:
                self.selection_end = (x, y)
                self.save_annotation()
                self.selection_start = None
                self.selection_end = None
                self.drawing = False
                self.update_display()
    
    def find_annotation_at_point(self, x, y):
        """Find annotation index at given point (for hover/delete)"""
        for idx, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None
    
    def save_annotation(self):
        """Save current bounding box as annotation"""
        if self.selection_start is None or self.selection_end is None:
            return
        
        x1 = min(self.selection_start[0], self.selection_end[0])
        y1 = min(self.selection_start[1], self.selection_end[1])
        x2 = max(self.selection_start[0], self.selection_end[0])
        y2 = max(self.selection_start[1], self.selection_end[1])
        
        # Validate minimum size
        if x2 - x1 < 10 or y2 - y1 < 10:
            print("Box too small, ignoring")
            return
        
        # Clamp to image bounds
        x1 = max(0, min(x1, self.img_width))
        y1 = max(0, min(y1, self.img_height))
        x2 = max(0, min(x2, self.img_width))
        y2 = max(0, min(y2, self.img_height))
        
        annotation = {
            'class_id': self.current_class,
            'class_name': CLASS_NAMES.get(self.current_class, f"class_{self.current_class}"),
            'bbox': [x1, y1, x2, y2],
            'timestamp': datetime.now().isoformat()
        }
        
        self.annotations.append(annotation)
        self.undo_stack.append(('add', annotation))
        
        print(f"Added annotation #{len(self.annotations)}: {annotation['class_name']} at ({x1},{y1})-({x2},{y2})")
        self.save_session()
    
    def undo_last_action(self):
        """Undo the last annotation action"""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        
        action_type, data = self.undo_stack.pop()
        
        if action_type == 'add':
            # Remove the last annotation
            if self.annotations and self.annotations[-1] == data:
                removed = self.annotations.pop()
                print(f"Undid: {removed['class_name']} annotation")
            else:
                # Find and remove the specific annotation
                try:
                    self.annotations.remove(data)
                    print(f"Undid: {data['class_name']} annotation")
                except ValueError:
                    print("Could not find annotation to undo")
        
        elif action_type == 'delete':
            # Restore deleted annotation
            self.annotations.append(data)
            print(f"Restored: {data['class_name']} annotation")
        
        self.save_session()
        self.update_display()
    
    def delete_annotation(self, idx):
        """Delete annotation at given index"""
        if 0 <= idx < len(self.annotations):
            deleted = self.annotations.pop(idx)
            self.undo_stack.append(('delete', deleted))
            print(f"Deleted annotation: {deleted['class_name']}")
            self.save_session()
            self.update_display()
    
    def update_display(self):
        """Update the display with all annotations and current selection"""
        display = self.original_image.copy()
        
        # Draw existing annotations
        for idx, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class_id']
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Highlight if hovering
            thickness = 3 if idx == self.hover_annotation_idx else 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{ann['class_name']} #{idx}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current selection
        if self.selection_start and self.selection_end:
            color = CLASS_COLORS.get(self.current_class, (255, 255, 255))
            cv2.rectangle(display, self.selection_start, self.selection_end, color, 2)
        
        # Draw UI overlay
        self.draw_ui_overlay(display)
        
        cv2.imshow(self.window_name, display)
    
    def draw_ui_overlay(self, img):
        """Draw UI information overlay"""
        overlay_height = 120
        overlay = np.zeros((overlay_height, img.shape[1], 3), dtype=np.uint8)
        
        # Current class indicator
        y_pos = 20
        current_class_name = CLASS_NAMES.get(self.current_class, f"class_{self.current_class}")
        current_color = CLASS_COLORS.get(self.current_class, (255, 255, 255))
        
        cv2.putText(overlay, f"Current Class: {self.current_class} - {current_class_name}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)
        
        # Class list
        y_pos += 25
        class_text = "Classes: "
        for class_id, class_name in sorted(CLASS_NAMES.items()):
            class_text += f"[{class_id}]{class_name}  "
        cv2.putText(overlay, class_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        y_pos += 25
        cv2.putText(overlay, "Controls: [1-9] Select class | Draw box with mouse | [S] Save | [U/Ctrl+Z] Undo | [D] Delete (hover) | [E] Export | [Q] Quit", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Stats
        y_pos += 25
        stats_text = f"Total annotations: {len(self.annotations)} | Image: {self.img_width}x{self.img_height}"
        cv2.putText(overlay, stats_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine overlay with image
        img[0:overlay_height] = cv2.addWeighted(img[0:overlay_height], 0.7, overlay, 0.3, 0)
    
    def export_yolo_format(self):
        """Export annotations to YOLO format"""
        if not self.annotations:
            print("No annotations to export")
            return
        
        # Create YOLO format directory
        yolo_dir = self.output_dir / "yolo_format"
        yolo_dir.mkdir(exist_ok=True)
        
        # Write YOLO annotation file
        yolo_file = yolo_dir / f"{self.image_path.stem}.txt"
        
        with open(yolo_file, 'w') as f:
            for ann in self.annotations:
                x1, y1, x2, y2 = ann['bbox']
                class_id = ann['class_id'] - 1  # YOLO uses 0-indexed classes
                
                # Convert to YOLO format (normalized center coordinates + width/height)
                x_center = (x1 + x2) / (2 * self.img_width)
                y_center = (y1 + y2) / (2 * self.img_height)
                width = (x2 - x1) / self.img_width
                height = (y2 - y1) / self.img_height
                
                # Write: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Write class names file
        classes_file = yolo_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_id in sorted(CLASS_NAMES.keys()):
                f.write(f"{CLASS_NAMES[class_id]}\n")
        
        # Copy image to YOLO directory
        import shutil
        shutil.copy(str(self.image_path), str(yolo_dir / self.image_path.name))
        
        print(f"\nExported to YOLO format:")
        print(f"  Annotations: {yolo_file}")
        print(f"  Classes: {classes_file}")
        print(f"  Image: {yolo_dir / self.image_path.name}")
        print(f"  Total annotations: {len(self.annotations)}")
    
    def run(self):
        """Main annotation loop"""
        print("\n" + "="*80)
        print("YOLO BOUNDING BOX ANNOTATOR")
        print("="*80)
        print("\nInstructions:")
        print("  1. Press number key (1-9) to select class")
        print("  2. Click and drag to draw bounding box")
        print("  3. Press 's' to save current box")
        print("  4. Press 'u' or Ctrl+Z to undo last action")
        print("  5. Hover over box and press 'd' to delete")
        print("  6. Press 'e' to export to YOLO format")
        print("  7. Press 'q' to quit and save")
        print(f"\nClasses defined:")
        for class_id, class_name in sorted(CLASS_NAMES.items()):
            print(f"  [{class_id}] {class_name}")
        print(f"\nCurrent annotations: {len(self.annotations)}")
        print("="*80 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            # Save annotation (automatic after drawing, but can force with 's')
            elif key == ord('s'):
                if self.selection_start and self.selection_end:
                    self.save_annotation()
                    self.selection_start = None
                    self.selection_end = None
                    self.update_display()
            
            # Undo
            elif key == ord('u') or key == 26:  # 'u' or Ctrl+Z
                self.undo_last_action()
            
            # Delete hovered annotation
            elif key == ord('d'):
                if self.hover_annotation_idx is not None:
                    self.delete_annotation(self.hover_annotation_idx)
                    self.hover_annotation_idx = None
            
            # Export to YOLO format
            elif key == ord('e'):
                self.export_yolo_format()
            
            # Class selection (1-9)
            elif ord('1') <= key <= ord('9'):
                class_num = key - ord('0')
                if class_num in CLASS_NAMES:
                    self.current_class = class_num
                    print(f"Selected class: {class_num} - {CLASS_NAMES[class_num]}")
                    self.update_display()
                else:
                    print(f"Class {class_num} not defined")
            
            # ESC - cancel current selection
            elif key == 27:
                if self.selection_start:
                    print("Cancelled selection")
                    self.selection_start = None
                    self.selection_end = None
                    self.drawing = False
                    self.update_display()
        
        # Save session and export on exit
        self.save_session()
        if self.annotations:
            self.export_yolo_format()
        
        cv2.destroyAllWindows()
        
        print(f"\nFinal statistics:")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Session saved to: {self.session_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate manuscripts with bounding boxes for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python annotate_yolo.py --image manuscript.png
  python annotate_yolo.py --image manuscript.png --output my_annotations
  
Classes can be customized by editing the CLASS_NAMES dictionary in the script.
        """
    )
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to manuscript image")
    parser.add_argument("--output", type=str, default="annotations", 
                       help="Output directory for annotations (default: annotations)")
    
    args = parser.parse_args()
    
    # Validate image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    annotator = YOLOAnnotator(args.image, args.output)
    annotator.run()


if __name__ == "__main__":
    main()