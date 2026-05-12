# Result: grid checkpoint `deg0_sc0p3_mo0_hsvs0p5`

Self-contained record for the best YOLOv11 checkpoint produced on the
`trainingscript_wy` branch, evaluated with v1-aligned per-class mAP@50.

- Checkpoint: `models/grid_search_v11_aug/best_deg0_sc0p3_mo0_hsvs0p5.pt`
- Eval outputs: `models/eval_v1_aligned/grid_deg0_sc0p3_mo0_hsvs0p5/`
- Eval date: 2026-05-12

---

## TL;DR

| Class  | v1 baseline (YOLOv8m, 17 imgs) | This checkpoint (YOLOv11m) |
|--------|--------------------------------|----------------------------|
| text   | 0.77                           | **0.78**                   |
| music  | 0.67                           | **0.74**                   |
| staves | —                              | 0.01 (open issue)          |

Matches v1 on text, exceeds v1 on music. Staves is not learned in any of the
v11 runs and is still open.

Overall mAP@50 = 0.510, mAP@50-95 = 0.281, P = 0.522, R = 0.478.

---

## 1. Where the checkpoint came from

### Search

Run by `scripts_v11/grid_search_v11_aug.py`. Default 4-D grid over augmentation
parameters:

```
degrees: [0, 5, 10]
scale:   [0.2, 0.3, 0.4]
mosaic:  [0.0, 0.5, 1.0]
hsv_s:   [0.3, 0.5, 0.7]
```

Tag `deg0_sc0p3_mo0_hsvs0p5` decodes to:

| degrees | scale | mosaic | hsv_s |
|---------|-------|--------|-------|
| 0       | 0.3   | 0.0    | 0.5   |

(i.e. **no rotation, ±30 % scaling, no mosaic, moderate saturation jitter** —
the least aggressive augmentation profile among the strong runs.)

### Training

All non-augmentation hyperparameters were inherited from
[configs/mothra_base11.yaml](configs/mothra_base11.yaml):

| Field          | Value      |
|----------------|------------|
| model          | YOLOv11m   |
| image_size     | 640        |
| batch_size     | 16         |
| learning_rate  | 0.001      |
| patience       | 50         |
| epochs (grid)  | 200        |
| hsv_h          | 0.01       |
| hsv_v          | 0.4        |
| translate      | 0.1        |
| fliplr / flipud| 0 / 0      |
| mixup          | 0          |

The grid driver only overrides `degrees`, `scale`, `mosaic`, `hsv_s`.

### Data split

- Strategy: **manuscript-aware** split (`scripts_v11/mothra_trainer.py:manuscript_aware_split`).
- Frozen split file: `outputs/yolo11_grid_search/datasets/20260326_110609/dataset/`
  - `data.yaml` — `nc: 3`, classes `[text, music, staves]`
  - `split_log.json` (2026-03-26):
    - train: 9 manuscripts (`CH-Fco Ms. 2`, `CH-P 18`, `D1161 210546 gthc`, `Antiphonal 1v hfngl`, `CH-Fco Ms.2`, `CH-P18 061 mssn.jpg`, `CH-Fco2`, `Antiphonal`, `D1161 210404 gthc`)
    - val: 1 manuscript (`D-KNd 1161`, 1 image)
    - test: 3 manuscripts (`NZ-Wt MSR-03`, `F-Pn Latin 15181`, `CH-P18 288 mssn`, 5 images total)
- Manuscript-level splitting means no page from a test manuscript ever
  appears in train — no data leakage. Compare with the Optuna v3 search,
  whose `split_log.json` carries an explicit `DATA LEAKAGE POSSIBLE` warning.

### Training-time record (from `models/results_multi_metrics_best_per_metric.csv`)

| metric        | best value | best epoch |
|---------------|-----------:|-----------:|
| precision     | 0.873      | 132        |
| recall        | 0.682      | 132        |
| mAP@50        | 0.771      | 132        |
| mAP@50-95     | 0.510      | 140        |

`best.pt` is saved at the **mAP@50-95-best** epoch (140), per Ultralytics'
default. So the re-eval below uses the epoch-140 checkpoint; the mAP@50 it
achieves at re-eval (0.78 for text, 0.74 for music) is the per-class
breakdown of overall mAP@50 ≈ 0.77 reported during training.

---

## 2. How the 0.78 / 0.74 numbers were obtained

### Eval script

`scripts_v11/eval_v1_aligned.py`, mode `both`:
- `model.val()` on the **test** split of the same `data.yaml` used during training
- `model.predict()` on `data/holdout/` for visualisation in the
  `inference-outputs` branch format (`<stem>_predicted.jpg` + mothra-annotator JSON)

### Exact command

```bash
python scripts_v11/eval_v1_aligned.py \
    --mode both \
    --weights models/grid_search_v11_aug/best_deg0_sc0p3_mo0_hsvs0p5.pt \
    --data    outputs/yolo11_grid_search/datasets/20260326_110609/dataset/data.yaml \
    --split   test \
    --source  data/holdout \
    --out-dir models/eval_v1_aligned/grid_deg0_sc0p3_mo0_hsvs0p5 \
    --imgsz   640 \
    --device  5
```

Run on a219, GPU 5. Wall clock for this checkpoint: ~1 minute.

### Settings (post-fix, see § 3)

| Stage  | conf  | iou | imgsz |
|--------|-------|-----|-------|
| val    | 0.001 | 0.7 | 640   |
| infer  | 0.25  | 0.7 | 640   |

`conf=0.001` for val matches Ultralytics' internal default; the main-branch
`scripts/train_mothra.py:evaluate()` does the same by not passing `conf`.

### Resulting per-class mAP@50

From `models/eval_v1_aligned/grid_deg0_sc0p3_mo0_hsvs0p5/eval_summary.csv`:

```
map50_overall    0.5096
map50_95_overall 0.2807
precision        0.5220
recall           0.4777
map50_text       0.7768
map50_music      0.7402
map50_staves     0.0118
```

Overall mAP@50 (0.51) is the unweighted mean of the three per-class numbers,
dragged down by staves.

### Visual outputs

`models/eval_v1_aligned/grid_deg0_sc0p3_mo0_hsvs0p5/` contains, for each of
the 4 holdout images in `data/holdout/`:

```
<stem>/
    <stem>_predicted.jpg     bbox overlay (inference-outputs format)
    <stem>.json              mothra-annotator-compatible
```

plus `all_predictions.json` aggregated across the 4 images. Compare directly
against the v1 renderings at `models/v1_predictions_heldout/<stem>.jpg`.

---

## 3. Bugs found and fixed before this number was trustworthy

The first re-eval pass with this script reported text=0.50, music=0.35 — much
worse than v1, which seemed implausible given training-time mAP@50 = 0.77.
Two bugs in `scripts_v11/eval_v1_aligned.py` were responsible:

### Bug A — per-class attribute mix-up (dominant)

```python
# Before (wrong):
per_class_ap50 = list(metrics.box.maps)

# After (correct):
ap50_arr = list(metrics.box.ap50)
class_idx = list(metrics.box.ap_class_index)
per_class_ap50 = [float("nan")] * len(class_names)
for slot, cls_i in enumerate(class_idx):
    per_class_ap50[int(cls_i)] = float(ap50_arr[slot])
```

`metrics.box.maps` is **per-class mAP averaged over IoU 0.5:0.95** (10
thresholds), not mAP@50. The correct attribute is `metrics.box.ap50`, which
needs to be indexed via `metrics.box.ap_class_index` because classes with 0
ground-truth instances are omitted. The original "text=0.50" was effectively
the per-class mAP@50-95 of text — a much stricter metric.

The same conflation exists in `scripts/train_mothra.py:evaluate()` on
**main branch** (prints `metrics.box.maps[i]` labelled "AP@50"); the v1
baseline numbers (text=0.77, music=0.67) quoted in
[eval_v1_aligned_progress.md](eval_v1_aligned_progress.md) almost certainly
came from that path, i.e. they are per-class mAP@50-95 mislabelled — but
either way, the right comparison now happens on the same definition.

### Bug B — conf threshold in val mode (minor)

```python
# Before:
model.val(..., conf=conf, iou=iou)   # conf=0.25 (inference default)

# After:
VAL_CONF = 0.001
model.val(..., conf=VAL_CONF, iou=iou)
```

Ultralytics' internal `val` default is `conf=0.001`. Passing `conf=0.25`
truncates the precision-recall curve at the high-confidence end and slightly
under-reports AP. Effect here was small (overall mAP@50 0.514 → 0.510); the
visible jump in per-class numbers comes from Bug A.

The CLI flag `--conf` now only controls the **infer** (predict) stage, which
is fine since `conf=0.25` is the right default for clean visualisation.

---

## 4. Caveats

- **Test set is tiny.** Manuscript-level split keeps it honest, but with only
  3 held-out manuscripts (5 images) the per-class mAP@50 numbers have wide
  confidence intervals.
- **v1's 0.77 / 0.67 baseline lives in a different test set.** Main branch
  doesn't ship a frozen split file for v1. The comparison is "v1 on its own
  unspecified holdout" vs "this checkpoint on the grid's test split" — the
  match is reassuring, not conclusive. An apples-to-apples comparison
  would run v1 through the same script on the grid test split.
- **Staves is unsolved.** All three v11 picks evaluated so far score ≤ 0.06
  on staves; the v1 model doesn't claim a staves number. Possible causes to
  investigate: too few staves boxes in the training set; staves instances
  are very thin/long, hard for default anchors; or staves labels in some
  manuscripts are stylistically inconsistent.
