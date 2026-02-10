# YOLO Manuscript Annotator

Simple bounding box annotation tool for creating YOLO training datasets from manuscript images.

## Quick Start

### 1. Clone this repository
```bash
git clone <your-repo-url>
cd yolo-manuscript-annotator
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the annotator
```bash
python annotate_yolo.py --image path/to/your/manuscript.png
```

## How to Use

### Basic Workflow

1. **Select a class** - Press number key (1, 2, or 3)
   - `1` = text
   - `2` = music
   - `3` = staves

2. **Draw bounding box** - Click and drag with mouse to draw a box around the object

3. **The box saves automatically** when you release the mouse

4. **Repeat** for all objects in the image

5. **Export** - Press `e` to export in YOLO format

6. **Quit** - Press `q` to save and quit

### Keyboard Controls

| Key | Action |
|-----|--------|
| `1-9` | Select class for next annotation |
| Mouse drag | Draw bounding box |
| `u` or `Ctrl+Z` | Undo last annotation |
| `d` | Delete annotation (hover over box first) |
| `e` | Export to YOLO format |
| `s` | Force save current box (usually automatic) |
| `q` | Quit and save session |
| `ESC` | Cancel current box |

### Visual Indicators

- **Colored boxes** show existing annotations
- **Box color** indicates class (Blue=text, Green=music, Red=staves)
- **Thicker outline** when hovering (for deletion)
- **Top bar** shows current class and controls

## Output Files

The tool creates an `annotations/` directory with:

### Session Files
- `{image_name}_session.json` - Your annotation session (resumes if you quit and restart)

### YOLO Format Export
When you press `e` or quit, creates `annotations/yolo_format/`:
- `{image_name}.txt` - YOLO format annotations (one line per box)
- `classes.txt` - List of class names in order
- `{image_name}.png` - Copy of the annotated image

### YOLO Format Details
Each line in the `.txt` file is:
```
class_id x_center y_center width height
```
All coordinates are normalized (0-1 range).

## Customizing Classes

Edit the `CLASS_NAMES` dictionary in `annotate_yolo.py`:

```python
CLASS_NAMES = {
    1: "text",
    2: "music",
    3: "staves",
    4: "your_new_class",  # Add more as needed
    # ...
}
```

You can define up to 9 classes (keys 1-9).

## Tips

- **Work systematically** - Annotate all instances of one class before moving to the next
- **Use undo liberally** - `u` or `Ctrl+Z` if you make a mistake
- **Save often** - Press `e` periodically to export your work
- **Sessions auto-save** - You can quit anytime and resume later (annotations are saved automatically)
- **Be consistent** - Decide on annotation rules (e.g., "include all staff lines in one box" vs "one box per line") and stick to them

## Troubleshooting

**Image won't load:**
- Check the file path is correct
- Ensure the image is a valid format (PNG, JPG, etc.)

**Can't see the image:**
- The window might be too large/small - resize the window
- Try a smaller image if your screen resolution is low

**Annotations seem wrong:**
- Remember YOLO uses normalized coordinates (0-1)
- Check the exported `.txt` file - values should all be between 0 and 1

**Session not resuming:**
- Make sure you're using the same `--output` directory
- Check that `{image_name}_session.json` exists

## Next Steps

Once you have annotated images:

1. Organize your dataset:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. Copy the `.txt` files from `annotations/yolo_format/` to `labels/train/` or `labels/val/`

3. Copy the corresponding images to `images/train/` or `images/val/`

4. Create a `dataset.yaml` file for YOLO training (see YOLO documentation)

## Questions?

See the [project brief](https://github.com/DDMAL/mothra/tree/main?tab=readme-ov-file#mothra-) for more context on the YOLO training pipeline.
