# Column-wise Heatmap Regression — Staffline Detection

## Concept

Frame staffline detection as a pose estimation problem: at every x-column of
the page image, predict a probability distribution over y (a heatmap), where
the peak of the distribution indicates the staffline's y-position.  This is
directly analogous to joint heatmap prediction in human pose estimation (e.g.
HRNet, ViTPose).

Unlike bounding-box detection (which gives coarse positions) or pixel-level
segmentation (which requires a binarization threshold), heatmap regression
produces a soft, sub-pixel accurate y-estimate at every column.  The output is
inherently a curve, not a blob.

## Architecture sketch

```
Input:  full page image or per-stave strip (H × W × 3)
       → Encoder (ResNet-34 or HRNet backbone)
       → Decoder head:  for each detected staffline, output a (1 × W) heatmap
Output: n_lines heatmaps of shape (H_reduced × W), one per staffline
        argmax along H_reduced → y-position per column
```

The number of output heatmaps is not fixed — a detection head (similar to
CenterNet) predicts the count and instantiates one heatmap channel per line.

Alternatively: a simpler per-stave-strip model that takes the YOLO box crop as
input and outputs a single (H_small × W_strip) heatmap for that one staffline.
This avoids the variable-count problem and is easier to train on the existing
YOLO-level annotations.

## Supervision signal

Prefer per-column centerline annotations where available (the main pipeline's
JSOMR `centerline`/`centerline_page` output, or corrected GT) as ground truth:
build the target heatmap as a Gaussian blob of width σ≈h/2 centred on the
annotated y at each x-column, following the actual curve.

Flat box-centre targets (a single Gaussian blob at the box y-center, repeated
unchanged across the box's x-range) are only a coarse fallback when no
centerline annotation exists — they reward a horizontal line and actively
penalise the warped stafflines this model needs to fit. If used, treat the
resulting model explicitly as coarse first-stage/pretraining supervision,
followed by a separate curve-refinement step against real centerline data,
not as the final output.

## What's needed to implement

- [ ] Training dataset: YOLO GT annotations → heatmap targets (conversion script)
- [ ] Train/val split across manuscripts (already partially set up in the broader
      project; ensure no page from the same manuscript appears in both splits)
- [ ] Model architecture (suggest starting with a simple UNet encoder-decoder)
- [ ] Training loop with MSE or focal loss on heatmaps
- [ ] Post-processing: argmax or soft-argmax along y-axis per column → curve
- [ ] Connection to existing eval_page / JSOMR output format

## Reference work

- Calvo-Zaragoza et al. (Alicante group) — column-wise regression for printed
  music staff detection (ICDAR/ISMIR proceedings)
- HRNet: *Deep High-Resolution Representation Learning* (Sun et al., CVPR 2019)
- CenterNet: *Objects as Points* (Zhou et al., CVPR 2019) — for variable line count

## Key considerations for historical manuscripts

- Parchment warp means the heatmap must accommodate significant y-variation
  across columns — a model trained on printed music will not transfer directly
- Input normalisation matters: parchment colour varies dramatically; consider
  per-image histogram equalisation or local contrast normalisation
- Data augmentation: elastic deformation, brightness jitter, ink dropout

## Expected output format

Same ExperimentFitResult / JSOMR JSON as other experiments.
