# Next Grid Search Plan (v2)

> Successor to `grid_search_v11_aug.py`. Not yet executed. Captures the design
> decisions before any GPU time is spent.

## Motivation

Three concerns drive a second round:

1. **More data.** The recent merge added ~12 new manuscripts under
   `data/images/` plus corrected annotations under
   `inference-outputs/First round of corrections - zips/`. The frozen
   2026-03-26 split (17 images, 9 train / 1 val / 3 test manuscripts) no
   longer reflects what we have.
2. **Methodology alignment with main branch.** Eval uses `model.val()` with
   Ultralytics defaults (`conf=0.001`) and reports per-class mAP@50 from
   `metrics.box.ap50` (not the mis-labelled `maps` attribute on main).
   See `result_grid_deg0_sc0p3_mo0_hsvs0p5.md` § 3 for the bugs that mattered.
3. **Lessons from v1 search**:
   - Best augmentation profile was the *least* aggressive
     (`degrees=0, mosaic=0, scale=0.3, hsv_s=0.5`). Holding those fixed.
   - `staves` class collapses to ~0 mAP@50 across every checkpoint
     evaluated so far. Underlying cause is the bbox geometry:
     mean width/height ratio = **36** (vs 2.8 for text, 0.9 for music) —
     mean staves height is ~0.9 % of image height. At `imgsz=640` a staves
     box is ≈ 6 px tall, which is at or below YOLO's smallest feature-map
     stride. **Higher input resolution is the most plausible fix.**
   - Optuna v3's top trials clustered around `yolo11s @ img1280` and
     `yolo11m @ img1152`, hinting that resolution may matter more than
     capacity — but those results came from a split with explicit data
     leakage and aren't directly trustworthy.

## Goals

- Identify a single checkpoint that beats `deg0_sc0p3_mo0_hsvs0p5`
  on **all three classes**, with particular attention to staves.
- Keep the search small and trustworthy (clean split, not many runs).
- End with a per-class mAP@50 number and holdout visualisations.

## Scripts (already written, not run)

| Script | Purpose |
|--------|---------|
| [scripts_v11/build_split_v2.py](scripts_v11/build_split_v2.py) | Builds the new manuscript-aware split with staves coverage |
| [scripts_v11/grid_search_v11_size_res.py](scripts_v11/grid_search_v11_size_res.py) | The grid driver itself (size × imgsz axes) |

Sanity-checked locally with `build_split_v2.py --dry-run`: 19 image-label
pairs across 14 manuscripts, candidate split has staves counts
train=589 / val=104 / test=133.

## Search axes

Fixed (taken from prior winners):

| Param         | Value |
|---------------|-------|
| `degrees`     | 0     |
| `mosaic`      | 0.0   |
| `scale`       | 0.3   |
| `hsv_s`       | 0.5   |
| `epochs`      | 300   |
| Other aug     | as `configs/mothra_base11.yaml` |

Searched:

| Axis         | Values              | Rationale |
|--------------|---------------------|-----------|
| `model_size` | `s`, `m`            | s+high-res vs m+low-res tradeoff |
| `image_size` | `640`, `1024`, `1280`, `1600` | staves geometry argues higher is better |

Total runs: **8**. If GPU memory caps a (model, imgsz) pair, the script
auto-drops batch size before falling back to skip.

Optional follow-up axis if any of the 8 looks bad: `lr0 ∈ {0.001, 0.003}`
(Optuna v3 found ~0.0026 useful) — defer to a second pass.

## Data and split

### Prep (do this once before the grid runs)

1. Unzip every `inference-outputs/First round of corrections - zips/*.zip`
   into a temp folder. Each archive contains a `<stem>.txt` already in
   YOLO format and a `<stem>.json` Mothra-annotator export.
2. For each `<stem>`, prefer the corrected `<stem>.txt` from the zip over
   any pre-existing `data/yolo_labels/<stem>.txt` (these zips are *human
   corrections of model predictions*, i.e. ground truth).
3. Confirm there is a matching image in `data/images/<stem>.{jpg,png,…}`.

### Split policy

`scripts_v11/build_split_v2.py` does the following:

1. Group all `(image, label)` pairs by manuscript ID
   (existing `MothraTrainer.extract_manuscript_id`).
2. Manuscript-aware shuffle with `split_ratios=(0.7, 0.15, 0.15)`.
3. **Re-sample seeds until both val and test contain ≥ `min_staves_test`
   staves bbox instances** (default `min_staves_test=50`). This prevents
   the `staves`-class metric from being undefined.
4. Persist the split under
   `outputs/yolo11_grid_search_v2/datasets/<timestamp>/dataset/` with
   `data.yaml`, `split_log.json`, and per-split `images/`,`labels/` dirs.

The script aborts cleanly if the staves constraint cannot be met after a
budget of attempts.

### Why not random page-level split?

Same reason as v1: pages from the same manuscript share writing style,
parchment, and stave layout. Random page-level splits leak that style and
inflate metrics — exactly what tainted Optuna v3 (see its `split_log.json`
warning "DATA LEAKAGE POSSIBLE").

## Eval methodology (aligned with main branch)

Post-training, each checkpoint goes through `scripts_v11/eval_v1_aligned.py`
in `--mode both`. That script (after bug fix from 2026-05-12):

- Calls `model.val(conf=0.001, ...)` — Ultralytics' internal val default
  (same as main-branch `scripts/train_mothra.py:evaluate()`, which doesn't
  pass `conf` and so gets the same default).
- Extracts per-class AP@50 from `metrics.box.ap50` indexed via
  `metrics.box.ap_class_index`. Not `metrics.box.maps`, which is per-class
  mAP@50-95.
- Also reports overall mAP@50, mAP@50-95, P, R.
- Writes `eval_summary.csv` + per-image `_predicted.jpg` + mothra-annotator
  JSON, matching the `inference-outputs` branch format for direct visual
  comparison with v1.

The grid script invokes `eval_v1_aligned.py` automatically at the end of
each run.

## Wall-clock estimate

Per run, single A100:
- YOLOv11s @ 640: ~20 min for 300 epochs
- YOLOv11s @ 1280: ~50 min
- YOLOv11s @ 1600: ~80 min (memory permitting)
- YOLOv11m @ 640: ~30 min
- YOLOv11m @ 1280: ~75 min
- YOLOv11m @ 1600: ~130 min (likely needs `batch=8` or `4`)

Rough total for 8 runs: **6–10 hours** of single-GPU time. Parallelise by
distributing runs across a219's free GPUs.

## Open questions

1. **Should `staves` get its own loss weight?** Ultralytics has no
   per-class loss weight by default. If higher resolution alone doesn't
   crack staves, options are:
   - Train a separate model on `staves` only and ensemble.
   - Redefine staves as one box per system instead of one per line.
2. **`rect=True` (rectangular training)?** Manuscripts are tall portrait
   format. Rectangular training preserves the aspect ratio instead of
   letterboxing to square, which may help thin horizontal staves boxes.
   Worth trying as an axis if the 8-run grid leaves headroom.
3. **Test-set staves count.** The split helper enforces a minimum, but
   if the natural distribution puts few staves manuscripts in any split,
   we may need to oversample or hand-curate.

## Suggested run order

```bash
# 1. Freeze the split (CPU, < 1 min)
python scripts_v11/build_split_v2.py \
    --extra-labels-zips 'inference-outputs/First round of corrections - zips/*.zip' \
    --seed 42

# 2. Grab a219 GPU 5 (or any free GPU; check first)
ssh ruihan@192.168.2.12 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'

# 3. Run the grid (8 runs, 6–10 hours on one A100)
DATA=outputs/yolo11_grid_search_v2/datasets/<timestamp>/dataset/data.yaml
ssh ruihan@192.168.2.12 \
    "cd /datapool/data3/storage/ruihan/debug/mothra && \
     nohup python scripts_v11/grid_search_v11_size_res.py \
        --data-yaml $DATA \
        --epochs 300 \
        --device 5 \
     > /tmp/grid_v2.log 2>&1 & disown"

# 4. Read the per-tag CSV
cat models/grid_search_v11_size_res/results.csv
ls  models/eval_v1_aligned_v2/        # per-tag v1-aligned summaries
```

## Deliverables

When this grid runs (eventually), the outputs land in:

- `outputs/yolo11_grid_search_v2/runs/<tag>/` — Ultralytics training dirs
- `models/grid_search_v11_size_res/best_<tag>.pt` — 8 checkpoints
- `models/grid_search_v11_size_res/results.csv` — per-run best mAP50-95
- `models/eval_v1_aligned_v2/<tag>/` — per-class mAP@50 + holdout viz

The winner gets a per-checkpoint result file analogous to
[result_grid_deg0_sc0p3_mo0_hsvs0p5.md](result_grid_deg0_sc0p3_mo0_hsvs0p5.md).
