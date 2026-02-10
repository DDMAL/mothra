# Mothra 🦋

**YOLO-based Optical Music Recognition for Medieval Manuscripts**

A research project exploring whether YOLO object detection can outperform our traditional OMR pipeline (Rodan) for medieval musical manuscript analysis, with a custom web-based annotation tool for rapid dataset creation.

---

## Project Overview

Mothra is an experimental (eventual) OMR system that applies modern object detection techniques (YOLO) to the challenging domain of medieval music manuscripts. The project consists of two main components:

1. **YOLO OMR Pipeline** - Machine learning experiments using DocLayout-YOLO for manuscript layout analysis
2. **Mothra Annotator** - Browser-based bounding box annotation tool for creating training datasets

### Why "Mothra"?

This project challenges the existing [Rodan](https://github.com/DDMAL/Rodan) workflow engine (kaiju fight, anyone? 🦖 vs 🦋). Where Rodan uses traditional document analysis and segmentation pipelines, Mothra explores whether end-to-end deep learning can successfully execute OMR based from object detection.

---

## Research Context
### Research Questions

1. Can YOLO detect manuscript layout elements (text, staves, neumes) more robustly than pixel-level methods?
2. Does DocLayout-YOLO's synthetic data generation translate to parchment patterns?
3. Can object detection provide a faster annotation → training → inference cycle than traditional (CNN-/deparation)OMR?
4. How does overlapping box detection (neumes on staves) perform vs. segmentation approaches?

---

## Repository Structure
Assume the branch is `main` unless otherwise specified.
```
mothra/
├── README.md                 # This file
├── annotator/                # Web-based annotation tool
│   ├── index.html           # BRANCH: `mothra-ghpages` Main annotator interface
│   ├── annotator.js         # BRANCH: `mothra-ghpages` Core annotation logic
│   ├── annotate_yolo.py     # Locally run YOLO annotation tool
│   ├── README.md            # Details on how to clone, YOLO annotation requirements, and how to move annotations forward for splitting
│   ├── SAMPLE_WORKFLOW.md   # Sample head-to-tail annotation workflow
│   ├── setup.sh             # Bash script for running the annotator using ios
│   ├── setup.bat            # Bash script for running the annotator using windows
│   ├── QUICK_REFERENCE.md   # Annotation guide 
│   └── THEME_NOTES.md       # Design documentation
├── data/                     # Datasets (gitignored)
│   ├── raw/                 # Original manuscript images
│   ├── annotations/         # JSON and YOLO format labels
│   └── splits/              # Train/val/test splits
├── samples/                 # Sample images of a fully annotated image in mothra-annotator, a correct YOLO.txt file, and a screenshot of the complete UI
├── experiments/              # YOLO training experiments
│   ├── doclayout-yolo/      # DocLayout-YOLO implementation
│   ├── configs/             # Training configurations
│   └── results/             # Model outputs, metrics
├── scripts/                  # Utility scripts
│   ├── convert_to_yolo.py   # JSON → YOLO format converter
│   └── split_dataset.py     # Manuscript-aware data splitting
└── allons-y/                     # Additional documentation
    ├── ANNOTATION_GUIDE.md  # How to annotate manuscripts
    ├── PLAN.md              # Central team docs for phases of process, needs, and questions
    └── ARCHITECTURE.md      # Technical decisions & comparisons
```

---

## Part 1: Mothra Annotator

A browser-based tool for creating YOLO training data. Designed for speed and precision when annotating hundreds of manuscript pages.
Lives on the branch `mothra-ghpages`, inside `mothra/annotator`.

### Features

- **Three-class annotation**: Text regions, music systems, staff lines
- **Zoom & pan**: Up to 500% zoom for tiny neumes
- **Opacity controls**: Adjust box transparency to see underlying content
- **Label toggle**: Hide labels when they obscure details
- **Multiple export formats**: JSON, YOLO .txt, or both
- **Keyboard shortcuts**: Rapid annotation workflow
- **Client-side processing**: Zero backend required, runs entirely in browser
- **Session persistence**: Annotations saved in browser storage

### Quick Start (Annotator)
User annotators may proceed to **Annotation Workflow**. This documentation covers the available options for hosting and deploying the Mothra Annotator tool.

#### Option 1: GitHub Pages (Recommended for Collaborators)

1. **Deploy to GitHub Pages:**
   ```bash
   cd mothra
   git checkout --orphan mothra-ghpages
   git rm -rf .
   cp annotator/index.html annotator/annotator.js .
   git add index.html annotator.js
   git commit -m "Deploy annotation tool"
   git push origin mothra-ghpages
   ```

2. **Enable Pages:**
   - Go to Settings → Pages
   - Source: `mothra-ghpages` branch, `/ (root)` folder
   - Access at: `https://USERNAME.github.io/mothra/`

3. **Share with annotators** - they just need the URL, no installation required

#### Option 2: Local Development

```bash
cd annotator
python3 -m http.server 8000
# Open http://localhost:8000 in browser
```

### Annotation Workflow

1. **Load image**: Click "Choose File" or drag & drop
2. **Select class**: Press `1` (text), `2` (music), or `3` (staves)
3. **Draw boxes**: Click and drag on canvas
4. **Zoom for precision**: Use `+`/`-` keys or Ctrl+wheel
5. **Adjust visibility**: Toggle labels and opacity as needed
6. **Download**: Press `Z` for both JSON and YOLO formats

**Pro Tips:**
- Zoom in to 300-500% for tiny neumes
- Use Shift+drag to pan when zoomed
- Lower opacity to 30-50% when checking coverage
- Use manuscript-aware splits (don't mix pages from same manuscript across train/val)

### Output Formats

**JSON** (session backup, human-readable):
```json
{
  "image_path": "AM_194_8vo_01r.png",
  "image_width": 2400,
  "image_height": 3200,
  "annotations": [
    {
      "class_id": 2,
      "class_name": "music",
      "bbox": [120, 450, 890, 620],
      "timestamp": "2026-02-10T15:23:41Z"
    }
  ]
}
```

**YOLO** (training format):
```
# classes.txt
text
music
staves

# image_name.txt (normalized coordinates)
1 0.523000 0.445000 0.120000 0.088889
```

### Class Definitions

| Class | ID | Color | Description |
|-------|----|----|-------------|
| **Text** | 1 | Muted blue | Text with or without music; rubrics and initials included |
| **Music** | 2 | Olive green | Complete staff systems including neumes |
| **Staves** | 3 | Bronze/gold | Staff lines (may overlap with music) |

**Overlapping boxes are intentional**: YOLO handles this well, and we use spatial overlap post-processing to assign neumes to specific staves for pitch inference.

---

## Part 2: YOLO OMR Pipeline

### Architecture Selection

We are currently experimenting with using **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)** as our base model because:

1. **Document-specific optimizations** - Designed for complex layouts, small text, varied scales
2. **GL-CRM module** - Global-to-Local Controllable Receptive fields handle neumes → initials range
3. **Synthetic data pipeline** - DocSynth-300K approach adaptable to parchment degradation
4. **YOLO11 foundation** - C2PSA attention for fine-grained symbol distinction

**Alternative architectures considered:**
- YOLO26 (STAL module for small targets) - may experiment if neume detection fails
- Segmentation models (U-Net, Mask R-CNN) - deferred; boxes faster to annotate

### Training Strategy

**Phase 1: Layout Detection** (*current focus*)
- Train on text/music/staves classes
- Goal: Robust region detection despite degradation
- Success metric: mAP@0.5 > 0.85 on held-out manuscripts

**Phase 2: Symbol Classification** (future)
- Expand to neume-level classes if Phase 1 succeeds
- Challenge: 30+ neume types, severe class imbalance
- May require weighted loss, augmentation, or hierarchical classification

### Dataset Requirements

**Minimum viable:**
- 50-100 annotated pages for initial experiments
- Manuscript-aware splits (entire manuscripts stay in one split)
- Balanced representation of degradation types

**Full dataset:**
- 200-500 pages for production model
- Multiple scribes, time periods, repositories
- Synthetic augmentation of rarer neume types

### Data Splits

**Critical: Manuscript-aware splitting**
```python
# scripts/split_dataset.py
# DON'T: Random split (data leakage!)
# DO: Split by manuscript ID
manuscripts = group_by_manuscript(annotations)
train_mss, val_mss, test_mss = split_manuscripts(0.7, 0.15, 0.15)
```

Why? Pages from the same manuscript have correlated characteristics (scribe, ink, parchment). Random splitting inflates validation metrics.

---

## Installation & Setup

### Prerequisites

- **For annotation only**: Modern web browser (Chrome, Firefox, Safari)
- **For YOLO training**: Python 3.8+, CUDA-capable GPU (recommended)

### YOLO Environment Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mothra.git
cd mothra

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install --break-system-packages -r requirements.txt

# Clone DocLayout-YOLO
cd experiments
git clone https://github.com/opendatalab/DocLayout-YOLO.git
cd DocLayout-YOLO
pip install --break-system-packages -r requirements.txt
```

### Training Your First Model
**_subject to change_**
```bash
# Organize data
python scripts/convert_to_yolo.py --input data/annotations/ --output data/yolo/
python scripts/split_dataset.py --input data/yolo/ --output data/splits/

# Configure training
# Edit experiments/configs/base_config.yaml with your paths

# Train
cd experiments/DocLayout-YOLO
python train.py --config ../configs/base_config.yaml

# Evaluate
python val.py --weights runs/train/exp/weights/best.pt
```

---

## Comparison to Rodan

| Feature | Rodan | Mothra |
|---------|-------|--------|
| **Approach** | Multi-stage pipeline | End-to-end detection |
| **Layout Analysis** | Pixel.js, Gamera | YOLO object detection |
| **Training Data** | Pixel-level masks | Bounding boxes |
| **Annotation Time** | High (pixel painting) | Low (box drawing) |
| **Degradation Handling** | Rule-based preprocessing | Learned robustness |
| **Staff Detection** | Hough transform | Attention mechanism |
| **Neume Classification** | Interactive Classifier | Direct detection (future) |
| **Human in Loop** | Multiple stages | Annotation phase only |
| **Inference Speed** | Slow (multi-stage) | Fast (single pass) |
| **Deployment** | Complex (Docker stack) | Simple (model file) |

**Hypothesis**: Mothra trades Rodan's interpretability for speed and potentially better handling of degraded manuscripts through learned feature extraction.

---

## Evaluation Metrics

### Layout Detection (Phase 1)

- **mAP@0.5**: Primary metric (COCO standard)
- **Per-class AP**: Ensure balanced performance across text/music/staves
- **Degradation robustness**: Test on manuscripts excluded from training with varying damage levels
- **Cross-repository generalization**: Train on one archive, test on another

### Symbol Classification (Phase 2, future)

- **Top-1 accuracy**: For neume type classification  
- **Confusion matrix**: Identify systematic errors
- **Manuscript-level MEI accuracy**: End-to-end OMR quality

### Qualitative Assessment

- **Visual inspection**: Do boxes align with human perception?
- **Failure mode analysis**: Where does it break? (extreme damage, rare neumes, unusual layouts)
- **Comparison to Rodan**: Side-by-side on same manuscripts

---

## Known Limitations & Future Work

### Current Limitations

1. **No end-to-end OMR yet** - Layout detection only, no pitch inference
2. **Small initial dataset** - 50-100 pages insufficient for production
3. **No temporal modeling** - Doesn't exploit sequential structure of music
4. **Overlapping boxes** - Require post-processing for pitch assignment

### Future Directions

1. **Expand to symbol-level** - 30+ neume classes
2. **Pitch inference pipeline** - Combine with staff line detection for vertical position
3. **Sequence modeling** - Transformer or LSTM for musical context
4. **Synthetic data** - DocSynth-style generation for rare neumes
5. **Multi-modal** - Combine visual detection with music theory constraints
6. **Active learning** - Prioritize uncertain predictions for annotation

---

## Contributing

This is a research project, but contributions welcome:

- **Annotations**: Help annotate manuscripts (use GitHub Pages tool)
- **Code**: Bug fixes, optimizations, experiments
- **Documentation**: Improve guides, add examples
- **Data**: Share medieval manuscript datasets (with appropriate permissions)

### Annotation Guidelines

See `docs/ANNOTATION_GUIDE.md` for detailed instructions on:
- What counts as a text region vs. music system
- Handling decorated initials
- Edge cases (marginal annotations, fragmentary staves)
- Quality control procedures

---

**Key papers this draws on:**

- DocLayout-YOLO: [arXiv:2410.12628](https://arxiv.org/abs/2410.12628)
- YOLO for Medieval Music: "Optical Medieval Music Recognition Using Background Knowledge" (MDPI 2022)
- Rodan: [ddmal.github.io/Rodan](https://ddmal.github.io/Rodan)

