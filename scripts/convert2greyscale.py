#!/usr/bin/env python3
"""
Convert manuscript images to greyscale
Creates greyscale versions in data/images/greyscale/
"""

from PIL import Image
from pathlib import Path

# Paths
images_dir = Path('/Users/ekaterina/Documents/mothra/data/images')
greyscale_dir = images_dir / 'greyscale'

# Create output directory
greyscale_dir.mkdir(exist_ok=True)

print("="*60)
print("Converting Images to Greyscale")
print("="*60)
print()

converted = 0
skipped = 0

for img_file in images_dir.glob('*'):
    # Skip the greyscale directory itself
    if img_file.is_dir():
        continue
    
    # Only process image files
    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        continue
    
    output_file = greyscale_dir / img_file.name
    
    # Skip if already exists
    if output_file.exists():
        print(f"  ⏭️  Skipping {img_file.name} (already exists)")
        skipped += 1
        continue
    
    try:
        # Open image and convert to greyscale
        img = Image.open(img_file)
        grey_img = img.convert('L')  # 'L' mode = greyscale
        
        # Save with same format
        grey_img.save(output_file)
        
        print(f"  ✅ Converted: {img_file.name}")
        converted += 1
        
    except Exception as e:
        print(f"  ❌ Error converting {img_file.name}: {e}")

print()
print("="*60)
print(f"✅ Conversion complete!")
print(f"   Converted: {converted} images")
print(f"   Skipped: {skipped} images")
print(f"   Output: {greyscale_dir}")
print("="*60)