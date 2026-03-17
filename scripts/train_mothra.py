#!/usr/bin/env python3
"""
Mothra Training Script - YOLOv8 for Medieval Manuscript Layout Detection
Phase 1: Detect text, music systems, and staff lines on degraded parchment

The Mothra annotator exports YOLO format directly, so this script:
- Organizes existing YOLO files into train/val/test splits
- Ensures manuscript-aware splitting (no data leakage)
- Trains YOLOv8 on your parchment manuscripts

Usage:
    python train_mothra.py --config configs/mothra_base.yaml
    python train_mothra.py --config configs/mothra_small.yaml --resume
"""

import argparse
import yaml
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class MothraTrainer:
    """
    Simplified YOLO training for medieval manuscripts.
    Works with YOLO files exported directly from Mothra annotator.
    Handles manuscript-aware splitting and training.
    """
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(self.config['paths']['project_root'])
        self.data_root = Path(self.config['paths']['data_root'])
        self.output_root = Path(self.config['paths']['output_root'])
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.output_root, 
                        self.output_root / 'runs',
                        self.output_root / 'datasets']:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def manuscript_aware_split(self, images_dir, labels_dir, split_ratios=(0.7, 0.15, 0.15)):
        """
        Critical: Split by manuscript ID, not randomly by image
        Prevents data leakage where pages from same manuscript appear in train/val
        
        Args:
            images_dir: Directory containing manuscript images
            labels_dir: Directory containing YOLO .txt label files
            split_ratios: (train, val, test) ratios
        """
        print("Creating manuscript-aware data splits...")
        
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        # Group images by manuscript
        manuscript_groups = defaultdict(list)
        
        # Find all images with corresponding labels
        for img_file in images_dir.glob('*'):
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                continue
            
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"  ⚠️  No label file for {img_file.name}, skipping")
                continue
            
            # Extract manuscript ID from filename
            manuscript_id = self.extract_manuscript_id(img_file.name)
            manuscript_groups[manuscript_id].append({
                'image': img_file,
                'label': label_file
            })
        
        if not manuscript_groups:
            raise ValueError(f"No image-label pairs found in {images_dir} and {labels_dir}")
        
        # Split manuscripts (not individual images!)
        manuscript_ids = list(manuscript_groups.keys())
        random.shuffle(manuscript_ids)
        
        n_total = len(manuscript_ids)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        train_mss = manuscript_ids[:n_train]
        val_mss = manuscript_ids[n_train:n_train+n_val]
        test_mss = manuscript_ids[n_train+n_val:]
        
        print(f"  Split: {len(train_mss)} train, {len(val_mss)} val, {len(test_mss)} test manuscripts")
        
        # Create split dictionaries
        splits = {
            'train': [pair for ms in train_mss for pair in manuscript_groups[ms]],
            'val': [pair for ms in val_mss for pair in manuscript_groups[ms]],
            'test': [pair for ms in test_mss for pair in manuscript_groups[ms]]
        }
        
        print(f"  Images: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # Log which manuscripts went where (helpful for debugging)
        split_log = {
            'train_manuscripts': train_mss,
            'val_manuscripts': val_mss,
            'test_manuscripts': test_mss,
            'split_ratios': split_ratios,
            'timestamp': datetime.now().isoformat()
        }
        
        return splits, split_log
    
    def extract_manuscript_id(self, filename):
        """
        Extract manuscript ID from filename
        Handles various naming conventions including spaces
        
        Examples:
          "CH-Fco Ms. 2_006r copy.jpg" -> "CH-Fco Ms. 2"
          "D-KNd 1161 032r.jpg" -> "D-KNd 1161"
          "NZ-Wt MSR-03 013r.png" -> "NZ-Wt MSR-03"
          "AM_194_8vo_01r.png" -> "AM_194_8vo"
        """
        stem = Path(filename).stem
        
        # Replace underscores with spaces for consistent splitting
        stem_normalized = stem.replace('_', ' ')
        parts = stem_normalized.split()
        
        # Strategy: Take parts until we hit a page number
        # Page numbers look like: "006r", "032r", "112v", "p.100"
        manuscript_parts = []
        for part in parts:
            # Stop at page number indicators
            if part.lower() == 'copy':
                break
            if part.startswith('p.') and any(c.isdigit() for c in part):
                break
            # Check for page numbers like "032r", "112v"
            if len(part) >= 3 and any(c.isdigit() for c in part):
                if part[-1] in ['r', 'v'] or part[-2:] in ['rr', 'vv']:
                    break
            manuscript_parts.append(part)
        
        if manuscript_parts:
            return ' '.join(manuscript_parts)
        
        # Fallback: first 2-3 parts
        if len(parts) >= 3:
            return ' '.join(parts[:3])
        elif len(parts) >= 2:
            return ' '.join(parts[:2])
        else:
            return stem
    
    def organize_yolo_dataset(self, splits, output_dir, split_log):
        """
        Organize YOLO files (already in YOLO format) into train/val/test structure
        Copies images and labels to proper directories
        """
        print("Organizing YOLO dataset...")
        
        output_dir = Path(output_dir)
        
        for split_name, pairs in splits.items():
            split_dir = output_dir / split_name
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            for pair in pairs:
                # Copy image
                shutil.copy(pair['image'], images_dir / pair['image'].name)
                
                # Copy label
                shutil.copy(pair['label'], labels_dir / pair['label'].name)
        
        # Create data.yaml for YOLO
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.config['classes'],
            'nc': len(self.config['classes'])
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Save split log for reproducibility
        log_path = output_dir / 'split_log.json'
        with open(log_path, 'w') as f:
            json.dump(split_log, f, indent=2)
        
        print(f"  YOLO dataset organized at: {output_dir}")
        print(f"  data.yaml created at: {yaml_path}")
        print(f"  Split log saved at: {log_path}")
        
        return yaml_path
    
    def train(self, data_yaml_path, resume=False):
        """
        Train YOLOv8 model
        """
        print("Starting training...")
        
        # Load model
        model_size = self.config['model']['size']
        if resume and (self.output_root / 'runs' / 'detect' / 'train' / 'weights' / 'last.pt').exists():
            model = YOLO(str(self.output_root / 'runs' / 'detect' / 'train' / 'weights' / 'last.pt'))
            print(f"  Resuming from checkpoint")
        else:
            model = YOLO(f'yolov8{model_size}.pt')  # n, s, m, l, x
            print(f"  Loading pretrained YOLOv8{model_size}")
        
        # Training arguments
        train_args = {
            'data': str(data_yaml_path),
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['training']['image_size'],
            'batch': self.config['training']['batch_size'],
            'lr0': self.config['training']['learning_rate'],
            'patience': self.config['training']['patience'],
            'save': True,
            'save_period': self.config['training']['save_period'],
            'project': str(self.output_root / 'runs'),
            'name': 'detect',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'device': self.config['training']['device'],
            'workers': self.config['training']['workers'],
            'augment': True,  # Built-in YOLO augmentation
        }
        
        # Add manuscript-specific augmentation config if specified
        if 'augmentation' in self.config:
            aug = self.config['augmentation']
            train_args.update({
                'hsv_h': aug.get('hsv_h', 0.015),
                'hsv_s': aug.get('hsv_s', 0.7),
                'hsv_v': aug.get('hsv_v', 0.4),
                'degrees': aug.get('degrees', 10),
                'translate': aug.get('translate', 0.1),
                'scale': aug.get('scale', 0.5),
                'shear': aug.get('shear', 2.0),
                'perspective': aug.get('perspective', 0.0),
                'flipud': aug.get('flipud', 0.0),
                'fliplr': aug.get('fliplr', 0.5),
                'mosaic': aug.get('mosaic', 1.0),
                'mixup': aug.get('mixup', 0.0),
            })
        
        # Train
        results = model.train(**train_args)
        
        print("  Training complete!")
        return results
    
    def evaluate(self, data_yaml_path, weights_path=None):
        """
        Evaluate trained model
        """
        print("Evaluating model...")
        
        if weights_path is None:
            weights_path = self.output_root / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
        
        model = YOLO(str(weights_path))
        
        # Validate
        metrics = model.val(
            data=str(data_yaml_path),
            split='val',
            save_json=True,
            save_hybrid=True,
        )
        
        print(f"\n{'='*60}")
        print(f"  mAP@50: {metrics.box.map50:.4f}")
        print(f"  mAP@50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        print(f"{'='*60}\n")
        
        # Per-class metrics
        print("  Per-class performance:")
        for i, class_name in enumerate(self.config['classes']):
            print(f"    {class_name:10s}: AP@50 = {metrics.box.maps[i]:.4f}")
        
        return metrics
    
    def predict_sample(self, image_path, weights_path=None, save=True):
        """
        Run inference on a sample image
        """
        if weights_path is None:
            weights_path = self.output_root / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
        
        model = YOLO(str(weights_path))
        
        results = model.predict(
            source=str(image_path),
            save=save,
            save_txt=True,
            save_conf=True,
            project=str(self.output_root / 'predictions'),
            name='sample',
            exist_ok=True,
        )
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Mothra YOLO Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--predict', type=str, help='Run prediction on single image')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize trainer
    trainer = MothraTrainer(args.config)
    
    if args.predict:
        # Prediction mode
        print(f"Running prediction on: {args.predict}")
        trainer.predict_sample(args.predict)
        return
    
    # Prepare data
    print("="*60)
    print("MOTHRA - YOLO Training for Medieval Manuscripts")
    print("="*60)
    
    images_dir = trainer.data_root / 'images'
    labels_dir = trainer.data_root / 'labels'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Create manuscript-aware splits
    splits, split_log = trainer.manuscript_aware_split(images_dir, labels_dir)
    
    # Organize into YOLO dataset structure
    dataset_dir = trainer.output_root / 'datasets' / f"mothra_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_yaml_path = trainer.organize_yolo_dataset(splits, dataset_dir, split_log)
    
    if args.eval_only:
        # Evaluation only
        trainer.evaluate(data_yaml_path)
    else:
        # Train
        trainer.train(data_yaml_path, resume=args.resume)
        
        # Evaluate
        trainer.evaluate(data_yaml_path)
    
    print("\nDone! Check results in:", trainer.output_root / 'runs')
    print("Dataset organized in:", dataset_dir)


if __name__ == '__main__':
    main()