# GitHub Pages Deployment Guide for Mothra

## Deploy as Branch of Mothra Repository

This annotation tool will be deployed as a separate branch (`gh-pages`) in existing Mothra repository.

### Setup Instructions

#### Option 1: Using Command Line (Recommended)

```bash
# Navigate to your Mothra repository
cd /path/to/mothra

# Create and switch to a new orphan branch for GitHub Pages
# (orphan branch = clean slate, no history from main)
git checkout --orphan gh-pages

# Remove all files from the staging area (we want a clean branch)
git rm -rf .

# Copy the annotation tool files
cp /path/to/index.html .
cp /path/to/annotator.js .

# Add and commit
git add index.html annotator.js
git commit -m "Add web-based annotation tool for GitHub Pages"

# Push to GitHub
git push origin gh-pages

# Switch back to your main branch
git checkout main  # or master, depending on your setup
```

#### Option 2: Using GitHub Web Interface

1. **In your Mothra repository on GitHub:**
   - Click the branch dropdown (usually says "main" or "master")
   - Type `gh-pages` in the text box
   - Click "Create branch: gh-pages"

2. **Upload the files:**
   - While on the `gh-pages` branch
   - Click "Add file" → "Upload files"
   - Drag and drop `index.html` and `annotator.js`
   - Commit directly to `gh-pages` branch

3. **Delete unnecessary files** (if any were copied from main branch)
   - You only need `index.html` and `annotator.js` on this branch
   - Delete everything else via GitHub interface

### Enable GitHub Pages

1. **Go to Mothra repository Settings**
   - Click "Settings" tab
   - Scroll to "Pages" in left sidebar

2. **Configure Pages:**
   - **Source:** Select branch `gh-pages`
   - **Folder:** Select `/ (root)`
   - Click **Save**

3. **Wait for deployment** (~1-2 minutes)
   - GitHub will show: "Your site is published at `https://USERNAME.github.io/mothra/`"

### Your Tool URL

The annotation tool will be live at:
```
https://USERNAME.github.io/mothra/
```

Send this URL to your colleague - she can bookmark it for easy access.

### Branch Structure

Your Mothra repo will now have:
- `main` (or `master`) - Your main research code
- `gh-pages` - The web annotation tool (only 2 files)

These branches are completely independent. Changes to `main` won't affect the tool, and vice versa.

### Updating the Tool

**To make changes to the annotation tool:**

```bash
# Switch to the gh-pages branch
git checkout gh-pages

# Make your edits to index.html or annotator.js
# (or pull updated files)

# Commit and push
git add .
git commit -m "Updated annotation tool"
git push origin gh-pages

# Switch back to main branch
git checkout main
```

GitHub Pages will automatically rebuild (takes ~1 minute).

### Workflow for Your Colleague

1. **She visits:** `https://USERNAME.github.io/mothra/`
2. **Annotates** images in the browser
3. **Downloads** annotations in her preferred format:
   - **JSON** - Compatible with Python conversion script
   - **YOLO** - Direct YOLO format `.txt` file
   - **Both** - Downloads JSON + YOLO + `classes.txt` (3 files)
4. **Saves files** to shared folder (Dropbox/Google Drive/etc.)

### File Collection Workflow

#### If she downloads JSON:
```bash
# You convert them to YOLO format
python convert_to_yolo.py --input-dir shared_folder/ --output-dir yolo_labels/
```

#### If she downloads YOLO directly:
```bash
# Files are already in YOLO format - just copy to your dataset
cp shared_folder/*.txt dataset/labels/train/
```

#### If she downloads "Both":
She gets 3 files per image:
- `manuscript_001_annotations.json` - Full annotation data
- `manuscript_001.txt` - YOLO format
- `classes.txt` - Class name mapping

You can use either format directly!

### Customizing Classes

To add or modify classes, edit `annotator.js` in the `gh-pages` branch:

```javascript
// Find these lines:
const CLASS_NAMES = {
    1: "text",
    2: "music",
    3: "staves",
    // Add more:
    4: "decoration",
    5: "neume",
};

const CLASS_COLORS = {
    1: "#2196F3",  // Blue
    2: "#4CAF50",  // Green
    3: "#f44336",  // Red
    4: "#FF9800",  // Orange
    5: "#9C27B0",  // Purple
};
```

And update `index.html` to add corresponding buttons:

```html
<!-- Find class-buttons div and add: -->
<button class="class-btn class-4" data-class="4">4 - Decoration</button>
<button class="class-btn class-5" data-class="5">5 - Neume</button>
```

Commit and push to `gh-pages` branch.

### Troubleshooting

**Tool not loading:**
- Check you're on the `gh-pages` branch in GitHub
- Verify `index.html` and `annotator.js` are in the root of that branch
- Check Settings → Pages shows the correct branch and folder

**404 Error:**
- Wait 2-3 minutes after enabling Pages
- Clear browser cache and try again
- Verify the URL is correct: `https://USERNAME.github.io/mothra/`

**Changes not appearing:**
- GitHub Pages can take 1-2 minutes to rebuild
- Hard refresh the browser (Ctrl+Shift+R or Cmd+Shift+R)
- Check you committed to the `gh-pages` branch, not `main`

**Multiple downloads not working:**
- Some browsers block multiple automatic downloads
- Your colleague may need to allow multiple downloads in browser settings
- Or download formats separately (JSON, then YOLO)

### Branch Management Tips

```bash
# See all branches
git branch -a

# Switch between branches
git checkout main              # Back to main code
git checkout gh-pages   # Back to tool

# Pull latest changes from remote
git checkout gh-pages
git pull origin gh-pages

# Delete local copy of branch (if needed)
git branch -d gh-pages
```

### Privacy & Security

- Everything runs client-side in the browser
- No data sent to any server
- Images and annotations stay on your colleague's computer
- Only downloaded files leave her browser
- GitHub Pages only serves the static HTML/JS files

### Download Format Reference

**JSON Format** (matches Python tool):
```json
{
  "image_path": "manuscript.png",
  "image_width": 2000,
  "image_height": 3000,
  "timestamp": "2026-02-10T...",
  "class_names": {...},
  "annotations": [
    {
      "class_id": 1,
      "class_name": "text",
      "bbox": [100, 200, 300, 400],
      "timestamp": "..."
    }
  ]
}
```

**YOLO Format** (`.txt` file):
```
0 0.100000 0.200000 0.050000 0.066667
1 0.523000 0.445000 0.120000 0.088889
2 0.678000 0.567000 0.089000 0.045678
```

**classes.txt**:
```
text
music
staves
```

All three files work together for YOLO training!