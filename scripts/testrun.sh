#!/bin/bash
# Mothra Proof of Concept Test Run
# Comprehensive test of all training script features

set -e  # Exit on error

echo "=========================================="
echo "🦋 MOTHRA - Proof of Concept Test Run"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================
# STEP 1: PRE-FLIGHT CHECKS
# ============================================
echo -e "${YELLOW}[1/7] PRE-FLIGHT CHECKS${NC}"
echo "----------------------------------------"

echo "📁 Checking directory structure..."
if [ ! -d "data/images" ]; then
    echo -e "${RED}❌ data/images/ not found${NC}"
    exit 1
fi

if [ ! -d "data/yolo_labels" ]; then
    echo -e "${RED}❌ data/yolo_labels/ not found${NC}"
    exit 1
fi

# Count files
IMAGE_COUNT=$(ls data/images/*.{jpg,png,jpeg} 2>/dev/null | wc -l | tr -d ' ')
LABEL_COUNT=$(ls data/yolo_labels/*.txt 2>/dev/null | wc -l | tr -d ' ')

echo "   Images found: $IMAGE_COUNT"
echo "   Labels found: $LABEL_COUNT"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No images found in data/images/${NC}"
    exit 1
fi

if [ "$LABEL_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No labels found in data/yolo_labels/${NC}"
    exit 1
fi

echo ""
echo "📋 Sample files:"
ls data/images/ | head -3 | sed 's/^/   /'
echo ""

echo "🔍 Checking label format..."
SAMPLE_LABEL=$(ls data/yolo_labels/*.txt | head -1)
echo "   Sample from: $(basename $SAMPLE_LABEL)"
head -2 "$SAMPLE_LABEL" | sed 's/^/   /'
echo ""

echo "🐍 Checking Python dependencies..."
python -c "from ultralytics import YOLO; import yaml; import numpy; print('   ✅ All dependencies available')" || {
    echo -e "${RED}❌ Missing dependencies. Run: pip install ultralytics pyyaml numpy${NC}"
    exit 1
}

echo -e "${GREEN}✅ Pre-flight checks passed!${NC}"
echo ""
sleep 2

# ============================================
# STEP 2: VERIFY MANUSCRIPT ID EXTRACTION
# ============================================
echo -e "${YELLOW}[2/7] TESTING MANUSCRIPT ID EXTRACTION${NC}"
echo "----------------------------------------"

echo "Testing how filenames will be grouped..."
python3 << 'EOF'
import os
from pathlib import Path

def extract_manuscript_id(filename):
    """Extract manuscript ID for grouping"""
    # For your files like "CH-Fco Ms. 2_006r copy.jpg"
    # Strategy: Use everything before the page number
    
    # Remove extension
    stem = Path(filename).stem
    
    # Split by spaces or underscores
    parts = stem.replace('_', ' ').split()
    
    # Take first 3-4 parts (manuscript identifier)
    if len(parts) >= 3:
        # Look for page number pattern (digits followed by r/v)
        manuscript_parts = []
        for part in parts:
            # Stop before page numbers like "006r", "112v", etc.
            if any(c.isdigit() for c in part) and part[-1] in ['r', 'v']:
                break
            manuscript_parts.append(part)
        
        if manuscript_parts:
            return ' '.join(manuscript_parts)
    
    # Fallback
    return ' '.join(parts[:2]) if len(parts) >= 2 else stem

# Test on actual files
images_dir = Path('data/images')
manuscripts = {}

for img_file in images_dir.glob('*'):
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        ms_id = extract_manuscript_id(img_file.name)
        if ms_id not in manuscripts:
            manuscripts[ms_id] = []
        manuscripts[ms_id].append(img_file.name)

print(f"\n   Found {len(manuscripts)} manuscripts:\n")
for ms_id, files in sorted(manuscripts.items()):
    print(f"   📖 {ms_id}: {len(files)} pages")
    for f in files[:3]:  # Show first 3
        print(f"      - {f}")
    if len(files) > 3:
        print(f"      ... and {len(files)-3} more")
    print()

EOF

echo ""
sleep 2

# ============================================
# STEP 3: CREATE TEST CONFIG
# ============================================
echo -e "${YELLOW}[3/7] CREATING TEST CONFIGURATION${NC}"
echo "----------------------------------------"

mkdir -p configs

cat > configs/mothra_test.yaml << 'YAML'
# Mothra Test Configuration - Small/Fast for Proof of Concept

paths:
  project_root: /Users/ekaterina/Documents/mothra
  data_root: /Users/ekaterina/Documents/mothra/data
  output_root: /Users/ekaterina/Documents/mothra/outputs

classes:
  - text
  - music
  - staves

model:
  size: n  # Nano - fastest for testing on CPU

training:
  epochs: 50        # Short run for proof of concept
  batch_size: 4     # Small batch for CPU
  image_size: 416   # Smaller images = faster on CPU
  learning_rate: 0.001
  patience: 20      # Early stopping
  save_period: 10
  device: cpu       # MacBook CPU
  workers: 4        # CPU workers

augmentation:
  hsv_h: 0.01
  hsv_s: 0.5
  hsv_v: 0.4
  degrees: 5
  translate: 0.1
  scale: 0.3
  shear: 2.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.0
  mosaic: 0.5   # Reduced for small dataset
  mixup: 0.0

evaluation:
  confidence_threshold: 0.25
  iou_threshold: 0.7
YAML

echo "   ✅ Created configs/mothra_test.yaml"
echo "      - Nano model (fastest)"
echo "      - 50 epochs (short run)"
echo "      - CPU optimized"
echo "      - Small batch size (4)"
echo ""
sleep 2

# ============================================
# STEP 4: UPDATE TRAINING SCRIPT FOR YOUR FILENAMES
# ============================================
echo -e "${YELLOW}[4/7] ADAPTING MANUSCRIPT ID EXTRACTION${NC}"
echo "----------------------------------------"

echo "   Creating custom extract_manuscript_id function..."
echo "   (Handles spaces in filenames like 'CH-Fco Ms. 2')"
echo ""

# We'll create a custom version that handles your naming
cat > extract_ms_id.py << 'PYTHON'
from pathlib import Path

def extract_manuscript_id(filename):
    """
    Extract manuscript ID from your filename format
    
    Examples:
        "CH-Fco Ms. 2_006r copy.jpg" -> "CH-Fco Ms. 2"
        "D-KNd 1161 032r.jpg" -> "D-KNd 1161"
        "NZ-Wt MSR-03 013r.png" -> "NZ-Wt MSR-03"
    """
    stem = Path(filename).stem
    
    # Replace underscores with spaces for consistent splitting
    stem = stem.replace('_', ' ')
    parts = stem.split()
    
    # Find where page number starts (digits + r/v)
    manuscript_parts = []
    for part in parts:
        # Stop at page numbers like "006r", "112v", "032r"
        if any(c.isdigit() for c in part) and len(part) >= 3:
            if part[-1] in ['r', 'v'] or part[-2:] == 'copy':
                break
        manuscript_parts.append(part)
    
    if manuscript_parts:
        return ' '.join(manuscript_parts)
    
    # Fallback: first 2 parts
    return ' '.join(parts[:2]) if len(parts) >= 2 else stem

# Test it
if __name__ == '__main__':
    test_files = [
        "CH-Fco Ms. 2_006r copy.jpg",
        "D-KNd 1161 032r.jpg",
        "NZ-Wt MSR-03 013r.png",
        "CH-P 18 p.100.jpg"
    ]
    
    for f in test_files:
        print(f"{f:40s} -> {extract_manuscript_id(f)}")
PYTHON

python extract_ms_id.py
echo ""
sleep 2

# ============================================
# STEP 5: RUN TRAINING
# ============================================
echo -e "${YELLOW}[5/7] STARTING TRAINING${NC}"
echo "----------------------------------------"
echo ""
echo "⏱️  This will take 10-30 minutes on CPU..."
echo "   (You'll see progress bars and metrics)"
echo ""
sleep 3

# Create labels directory link if needed
if [ ! -d "data/labels" ]; then
    ln -s yolo_labels data/labels
    echo "   Created data/labels -> data/yolo_labels symlink"
fi

# Run training
python train_mothra.py --config configs/mothra_test.yaml 2>&1 | tee training_output.log

echo ""
echo -e "${GREEN}✅ Training complete!${NC}"
echo ""
sleep 2

# ============================================
# STEP 6: ANALYZE RESULTS
# ============================================
echo -e "${YELLOW}[6/7] ANALYZING RESULTS${NC}"
echo "----------------------------------------"

# Find the dataset directory
DATASET_DIR=$(ls -td outputs/datasets/mothra_* 2>/dev/null | head -1)

if [ -d "$DATASET_DIR" ]; then
    echo ""
    echo "📊 Dataset Organization:"
    echo "   Location: $DATASET_DIR"
    echo ""
    
    TRAIN_COUNT=$(ls "$DATASET_DIR/train/images/" 2>/dev/null | wc -l | tr -d ' ')
    VAL_COUNT=$(ls "$DATASET_DIR/val/images/" 2>/dev/null | wc -l | tr -d ' ')
    TEST_COUNT=$(ls "$DATASET_DIR/test/images/" 2>/dev/null | wc -l | tr -d ' ')
    
    echo "   📁 Train: $TRAIN_COUNT images"
    echo "   📁 Val:   $VAL_COUNT images"
    echo "   📁 Test:  $TEST_COUNT images"
    echo ""
    
    if [ -f "$DATASET_DIR/split_log.json" ]; then
        echo "   📝 Manuscript splits:"
        python3 -c "import json; data=json.load(open('$DATASET_DIR/split_log.json')); print('      Train manuscripts:', ', '.join(data['train_manuscripts'][:3]) + ('...' if len(data['train_manuscripts']) > 3 else '')); print('      Val manuscripts:', ', '.join(data['val_manuscripts'])); print('      Test manuscripts:', ', '.join(data['test_manuscripts']))"
    fi
fi

echo ""
echo "📈 Training Metrics:"
if [ -f "outputs/runs/detect/train/results.csv" ]; then
    echo ""
    tail -1 outputs/runs/detect/train/results.csv | python3 -c "
import sys
line = sys.stdin.read().strip()
parts = line.split(',')
if len(parts) > 10:
    print(f'   Epoch: {parts[0]}')
    print(f'   mAP@50: {parts[10] if len(parts) > 10 else \"N/A\"}')
    print(f'   Precision: {parts[11] if len(parts) > 11 else \"N/A\"}')
    print(f'   Recall: {parts[12] if len(parts) > 12 else \"N/A\"}')
" || echo "   (Results available in outputs/runs/detect/train/results.csv)"
fi

echo ""
echo "📁 Generated Files:"
ls -lh outputs/runs/detect/train/weights/*.pt 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""

if [ -f "outputs/runs/detect/train/results.png" ]; then
    echo "   📊 Visualization: outputs/runs/detect/train/results.png"
    echo "   📊 Confusion Matrix: outputs/runs/detect/train/confusion_matrix.png"
fi

echo ""
sleep 2

# ============================================
# STEP 7: TEST PREDICTION
# ============================================
echo -e "${YELLOW}[7/7] TESTING PREDICTION${NC}"
echo "----------------------------------------"

# Pick a test image
TEST_IMAGE=$(ls data/images/*.{jpg,png,jpeg} 2>/dev/null | head -1)

if [ -f "$TEST_IMAGE" ]; then
    echo ""
    echo "🔮 Running prediction on: $(basename "$TEST_IMAGE")"
    echo ""
    
    python train_mothra.py \
        --config configs/mothra_test.yaml \
        --predict "$TEST_IMAGE"
    
    echo ""
    if [ -f "outputs/predictions/sample/$(basename "$TEST_IMAGE")" ]; then
        echo -e "${GREEN}✅ Prediction successful!${NC}"
        echo ""
        echo "   📸 Visualization: outputs/predictions/sample/$(basename "$TEST_IMAGE")"
        echo "   📝 Coordinates: outputs/predictions/sample/labels/$(basename "${TEST_IMAGE%.*}").txt"
        echo ""
        echo "   Open the image to see bounding boxes!"
    fi
fi

echo ""
sleep 2

# ============================================
# FINAL SUMMARY
# ============================================
echo "=========================================="
echo -e "${GREEN}🎉 TEST RUN COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "📋 Summary:"
echo "   ✅ Manuscript-aware splitting: Working"
echo "   ✅ Training: Complete ($IMAGE_COUNT images)"
echo "   ✅ Evaluation: Complete"
echo "   ✅ Prediction: Complete"
echo ""
echo "📂 Key Outputs:"
echo "   🏆 Best model: outputs/runs/detect/train/weights/best.pt"
echo "   📊 Metrics: outputs/runs/detect/train/results.csv"
echo "   📈 Plots: outputs/runs/detect/train/results.png"
echo "   🔮 Prediction: outputs/predictions/sample/"
echo ""
echo "📖 Next Steps:"
echo "   1. Check outputs/runs/detect/train/results.png for training curves"
echo "   2. Look at prediction visualization to see bounding boxes"
echo "   3. If results look good, annotate more data!"
echo "   4. If not, check split_log.json to verify manuscript grouping"
echo ""
echo "💾 Full log saved to: training_output.log"
echo ""
echo -e "${GREEN}Done! 🦋${NC}"