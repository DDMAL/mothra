# Example Annotation Workflow

This guide walks through annotating a single manuscript page from start to finish.

## Before You Begin

Make sure you've completed setup:
```bash
# On Mac/Linux:
chmod +x setup.sh
./setup.sh

# On Windows:
setup.bat
```

## Step-by-Step Example

### 1. Prepare Your Image

Place your manuscript image somewhere accessible, for example:
```
~/Documents/manuscripts/page001.png
```

### 2. Start the Annotator

```bash
# Activate virtual environment first
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate.bat  # Windows

# Run annotator
python annotate_yolo.py --image ~/Documents/manuscripts/page001.png
```

### 3. Annotation Session

**Scenario:** You're annotating a medieval manuscript page with text and musical notation.

**Step 3a: Annotate Text Regions**

1. Press `1` to select "text" class
2. Notice the top bar shows "Current Class: 1 - text" in blue
3. Click and drag around the first text block
4. Box saves automatically when you release mouse
5. Repeat for all text regions on the page

**Step 3b: Annotate Music Regions**

1. Press `2` to select "music" class
2. Notice the top bar now shows "Current Class: 2 - music" in green
3. Draw boxes around musical notation regions
4. If you make a mistake, press `u` to undo

**Step 3c: Annotate Staff Lines**

1. Press `3` to select "staves" class
2. Draw boxes around staff lines (decide on your strategy - one box per system or per line?)
3. Be consistent!

**Step 3d: Review and Fix**

1. Hover over any box to highlight it with a thicker border
2. Press `d` while hovering to delete incorrect annotations
3. Use `u` to undo recent changes
4. Redraw any boxes that need correction

### 4. Export and Save

```bash
# Press 'e' to export to YOLO format
# This creates: annotations/yolo_format/page001.txt

# Press 'q' to quit
# This also exports and saves your session
```

### 5. Check Your Output

```bash
# Check the annotations directory
ls annotations/

# You should see:
# - page001_session.json  (your session file)
# - yolo_format/          (YOLO format export)

# Look at the YOLO format file
cat annotations/yolo_format/page001.txt

# Each line should look like:
# 0 0.523 0.234 0.145 0.089
# 1 0.678 0.456 0.234 0.123
# ...
```

### 6. Continue with More Images

```bash
# Your first session is saved, now do another image
python annotate_yolo.py --image ~/Documents/manuscripts/page002.png

# Repeat the process
# All annotations go to the same annotations/ directory
```

## Example Output Structure

After annotating 3 pages, your directory looks like:

```
.
├── annotate_yolo.py
├── requirements.txt
├── README.md
└── annotations/
    ├── page001_session.json
    ├── page002_session.json
    ├── page003_session.json
    └── yolo_format/
        ├── classes.txt
        ├── page001.txt
        ├── page001.png
        ├── page002.txt
        ├── page002.png
        ├── page003.txt
        └── page003.png
```

## Organizing for YOLO Training

Once you have many annotated images, organize them:

```bash
# Create dataset structure
mkdir -p dataset/images/train dataset/images/val
mkdir -p dataset/labels/train dataset/labels/val

# Split your data (80/20 train/val split example)
# Move first 8 images to train:
cp annotations/yolo_format/page00[1-8].png dataset/images/train/
cp annotations/yolo_format/page00[1-8].txt dataset/labels/train/

# Move last 2 images to val:
cp annotations/yolo_format/page009.png dataset/images/val/
cp annotations/yolo_format/page009.txt dataset/labels/val/
cp annotations/yolo_format/page010.png dataset/images/val/
cp annotations/yolo_format/page010.txt dataset/labels/val/

# Copy classes.txt to dataset root
cp annotations/yolo_format/classes.txt dataset/
```

## Next Steps

1. Annotate more images (aim for at least 50-100 for initial experiments)
2. Organize into train/val splits
3. Follow the YOLO training instructions in the project brief
4. Train your model!

## Tips from Experience

- **Set a goal:** "I'll annotate 10 pages today" is easier than "I'll annotate everything"
- **Take breaks:** Annotation fatigue is real - your accuracy drops after ~30 minutes
- **Be consistent:** Decide on rules early (e.g., "music includes neumes only, not text") and stick to them
- **Use sessions:** Don't try to finish everything in one sitting - sessions save automatically
- **Export often:** Press `e` every 5-10 annotations to ensure your work is backed up
- **Start simple:** Just annotate text vs. music first, add more granular classes later if needed