# Staffline Detection — Comparative Experiments

This directory contains five alternative approaches to staffline centerline
detection, set up for head-to-head comparison using the evaluation framework
in `staff-finding/scripts/eval_page.py` / `eval_batch.py`.

All approaches consume the same inputs (page image + YOLO staffline detections)
and emit the same JSOMR JSON output format, so they can be evaluated identically
and compared directly.

---

## Approaches

### 1. DP Tracing (`dp_tracing/`)
**Status: implemented**

Traces each staffline as the minimum-cost path through the raw grayscale page
image using dynamic programming.  Dark pixels carry low cost; bright pixels
carry high cost.  A sliding-window smoothness constraint limits the path's
y-movement between adjacent columns.

**What makes it different from the existing pipeline:**
The existing pipeline binarizes each YOLO box crop, runs connected-component
analysis, and fits a polynomial.  DP tracing skips all of that — it works
directly on the continuous grayscale image, column by column, across the
full page width.  There is no binarization threshold, no component filtering,
and no polynomial fit.  It follows whatever is darkest within the search band.

**Data used:** page image (grayscale) + YOLO box y-center as the starting hint.
No training data, no model weights.

**Run:**
```bash
python staff-finding/experiments/dp_tracing/run_dp_page.py \
    --page  staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \
    --yolo  staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \
    --staffline-class 0 \
    --output staff-finding/e2e_tests/29may/Gent15_17_right/run_page/dp_tracing
```

**Key parameters:**
| Flag | Default | Meaning |
|------|---------|---------|
| `--band-half-multiplier` | 1.5 | Search band = ± multiplier × h |
| `--max-step-px` | 3 | Max y-shift allowed between adjacent columns |
| `--blur-sigma-multiplier` | 0.2 | Pre-blur to reduce noise sensitivity |
| `--no-extend` | off | Restrict trace to YOLO box x-range (default: full page) |

---

### 2. GP Centerlines (`gp_centerlines/`)
**Status: implemented**

Runs the existing Sauvola binarization + component filter, then replaces the
quadratic Huber polynomial fit with a Gaussian Process regression (Matérn-5/2
kernel).  The GP gives a smooth, flexible curve that adapts to arbitrary
parchment warp and provides per-column uncertainty estimates as a free output.

**What makes it different from the existing pipeline:**
The existing fit is a fixed-degree polynomial (quadratic, or cubic after
line-following), which may under-fit complex warp.  A GP has a learned,
data-driven length scale and can represent a wider class of smooth curves.
Crucially, the GP also returns uncertainty: high std at a given column flags
unreliable fits, useful for QA and downstream pitch finding.

**Data used:** same as the existing pipeline (YOLO boxes + binarized crops).
No training data — GP hyperparameters are optimised per-staffline via
marginal likelihood.

**Run:**
```bash
python staff-finding/experiments/gp_centerlines/run_gp_page.py \
    --page  staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \
    --yolo  staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt \
    --staffline-class 0 \
    --output staff-finding/e2e_tests/29may/Gent15_17_right/run_page/gp_centerlines
```

**Key parameters:**
| Flag | Default | Meaning |
|------|---------|---------|
| `--length-scale-init` | 100.0 | Initial Matérn length scale (px); optimised per-line |
| `--n-restarts` | 3 | Kernel hyperparameter optimisation restarts |
| `--binarization` | sauvola | Binarization method for component filter step |

---

### 3. Implicit Neural Representation (`implicit_neural/`)
**Status: planned — see NOTES.md**

Fits a tiny MLP per staffline by gradient descent directly on the page image.
No training phase, no binarization — the MLP *is* the curve, refined at
test time to land on dark pixels.

---

### 4. Column-wise Heatmap Regression (`heatmap_regression/`)
**Status: planned — see NOTES.md**

Trains a CNN to predict a per-column y-probability heatmap, supervised by
Gaussian blobs at YOLO box y-centers.  Equivalent to pose estimation for
stafflines.  Requires a training loop and labelled data.

---

### 5. Periodicity Self-Supervision (`periodicity/`)
**Status: planned — see NOTES.md**

Detects stafflines by exploiting their periodic vertical structure via
autocorrelation — no annotations required.  Phase tracking across columns
recovers the staffline curves.  Potential foundation for a self-supervised
pre-training approach.

---

## Evaluation

All runners produce a `*_stafflines.json` in JSOMR format compatible with
`eval_page.py` and `eval_batch.py`.

**Single-page eval:**
```bash
python staff-finding/scripts/eval_page.py \
    --gt    path/to/gt.txt \
    --pred  path/to/experiment_output/page_stafflines.json \
    --image path/to/page.jpg \
    --gt-source corrected \
    --variant dp_tracing \
    --output results.csv
```

**Batch eval across approaches** — build a manifest CSV with one row per
(page × variant) and run:
```bash
python staff-finding/scripts/eval_batch.py \
    --manifest eval_manifest.csv \
    --output   eval_results.csv \
    --summarize
```

## Key differences at a glance

| Approach | Binarization | Model | Training data | Uncertainty |
|----------|-------------|-------|--------------|-------------|
| Existing (sauvola/otsu + Huber) | yes | no | no | no |
| DP tracing | no | no | no | no |
| GP centerlines | yes (sauvola) | no | no | yes |
| Implicit neural rep | no | per-page MLP | no | via ensemble |
| Heatmap regression | no | CNN | yes | via heatmap spread |
| Periodicity | no | optional | no | via peak height |

## Images used

Current test set:
- **Gent right page**: `staff-finding/image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg`
  - YOLO GT: `staff-finding/image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt`
  - Manuscript: Ghent Antiphoner 1475, folio 17, right crop

Additional pages should be added to the test set as annotations are prepared,
prioritising manuscripts that cover the range of degradation types and staffline
counts present in the corpus.
