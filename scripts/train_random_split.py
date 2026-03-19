#!/usr/bin/env python3
"""
Mothra Training - PAGE-LEVEL Random Split (NOT manuscript-aware)
For small datasets where manuscript-aware splitting creates empty splits
"""

import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from ultralytics import YOLO

class SimpleTrainer:
    """Simple YOLO training with random page-level splits"""
    
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
    
    def random_page_split(self, images_dir, labels_dir, split_ratios=(0.7, 0.15, 0.15)):
        """
        Random page-level split (NOT manuscript-aware)
        
        Args:
            images_dir: Directory containing manuscript images
            labels_dir: Directory containing YOLO .txt label files
            split_ratios: (train, val, test) ratios
        """
        print("📄 Creating random page-level splits (NOT manuscript-aware)...")
        
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        # Find all image-label pairs
        pairs = []
        
        for img_file in images_dir.glob('*'):
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                continue
            
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"  ⚠️  No label file for {img_file.name}, skipping")
                continue
            
            pairs.append({
                'image': img_file,
                'label': label_file
            })
        
        if not pairs:
            raise ValueError(f"No image-label pairs found in {images_dir} and {labels_dir}")
        
        print(f"  Found {len(pairs)} image-label pairs")
        
        # Random shuffle
        random.shuffle(pairs)
        
        # Split by pages (not manuscripts)
        n_total = len(pairs)
        n_train = max(1, int(n_total * split_ratios[0]))
        n_val = max(1, int(n_total * split_ratios[1])) if n_total > 2 else 0
        
        # Ensure we don't exceed total
        if n_train + n_val > n_total:
            n_val = max(0, n_total - n_train)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train+n_val]
        test_pairs = pairs[n_train+n_val:]
        
        print(f"  Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pages")
        
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        split_log = {
            'split_type': 'random_page_level',
            'warning': 'Pages from same manuscript may appear in different splits',
            'train_images': [p['image'].name for p in train_pairs],
            'val_images': [p['image'].name for p in val_pairs],
            'test_images': [p['image'].name for p in test_pairs],
            'split_ratios': split_ratios,
            'timestamp': datetime.now().isoformat()
        }
        
        return splits, split_log
    
    def organize_yolo_dataset(self, splits, output_dir, split_log):
        """
        Organize YOLO files into train/val/test structure
        """
        print("📂 Organizing YOLO dataset...")
        
        output_dir = Path(output_dir)
        
        for split_name, pairs in splits.items():
            if not pairs:  # Skip empty splits
                continue
                
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
            'val': 'val/images' if splits['val'] else 'test/images',
            'test': 'test/images',
            'names': self.config['classes'],
            'nc': len(self.config['classes'])
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Save split log
        log_path = output_dir / 'split_log.json'
        import json
        with open(log_path, 'w') as f:
            json.dump(split_log, f, indent=2)
        
        print(f"  ✅ YOLO dataset organized at: {output_dir}")
        print(f"  ✅ data.yaml created at: {yaml_path}")
        print(f"  ✅ Split log saved at: {log_path}")
        
        return yaml_path
    
    def train(self, data_yaml_path, resume=False):
        """Train YOLOv8 model"""
        print("🎯 Starting training...")
        
        model_size = self.config['model']['size']
        if resume and (self.output_root / 'runs' / 'detect' / 'train' / 'weights' / 'last.pt').exists():
            model = YOLO(str(self.output_root / 'runs' / 'detect' / 'train' / 'weights' / 'last.pt'))
            print(f"  📂 Resuming from checkpoint")
        else:
            model = YOLO(f'yolov8{model_size}.pt')
            print(f"  📦 Loading pretrained YOLOv8{model_size}")
        
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
            'augment': True,
        }
        
        # Add augmentation config if specified
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
        
        print("  ✅ Training complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description='Mothra YOLO Training - Random Page Split')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Initialize trainer
    trainer = SimpleTrainer(args.config)
    
    print("="*60)
    print("🦋 MOTHRA - Random Page-Level Split Training")
    print("="*60)
    print("⚠️  WARNING: Pages from same manuscript may be in train/val")
    print("="*60)
    
    images_dir = trainer.data_root / 'images'
    labels_dir = trainer.data_root / 'yolo_labels'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Create random page splits
    splits, split_log = trainer.random_page_split(images_dir, labels_dir)
    
    # Organize into YOLO dataset structure
    dataset_dir = trainer.output_root / 'datasets' / f"random_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_yaml_path = trainer.organize_yolo_dataset(splits, dataset_dir, split_log)
    
    # Train
    trainer.train(data_yaml_path, resume=args.resume)
    
    print("\n✅ Done! Check results in:", trainer.output_root / 'runs')
    print("📊 Dataset organized in:", dataset_dir)


if __name__ == '__main__':
    main()