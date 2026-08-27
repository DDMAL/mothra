# paco_neume_masking_exp1 — paco stafflines layer + simulated 3-class neume masking

Test bed: 3 new hand-corrected-GT pages under `staff-finding/addtl-gt/` —
`Antiphonal_1v_hfngl`, `CH-E611_15r`, `l'Arsenal_Ms-5198_11` (single-class,
`0`=staffline, GT under `addtl-gt/labels/train/`). Driver script:
`scripts/eval_paco_neume_masking.py`.

## What was tested and why

`paco-classifier-service` currently separates staffline ink from background
with a **2-class** pixel classifier (background vs. stafflines) — neume ink
has no dedicated class and stays in the stafflines-only layer handed to the
stave YOLO model. Kyrie is training a real 3-class (background/staff/neume)
model in Rodan; this is a local mock-up of what removing neume ink from that
layer might do to stave detection, ahead of that model being ready — using
the phase-1 music-class YOLO boxes as a stand-in mask for what a real 3-class
model would (ideally) segment out on its own.

This extends an earlier "glyph masking" experiment
(`e2e_tests/{gent_right_exp1,ms234_064_exp1}/`, both untracked) that masked
the same kind of boxes out of the **raw page** before component filtering —
`band`-limited masking (preserve a thin strip around the estimated line) came
back safe there; `naive` (mask the whole box) measurably degraded one page
and lost a line on the other. That experiment's driver script and writeup
are gone (confirmed not in git history/stash/any worktree), so this script
rebuilds the same `naive`/`band` masking logic from the two READMEs'
descriptions — see `eval_paco_neume_masking.py`'s own docstring for exactly
how, and for the one place (`_band_center_y()`) that's a reconstruction
rather than a verified match to the original algorithm. The actual new
variable this experiment adds is the **base image**: the paco stafflines
layer, not the raw page.

Three variants per page:

- `paco_baseline` — the real, current 2-class stafflines layer, unmodified.
- `paco_neume_masked_naive` — every phase-1 music-class box painted solid
  white in the layer before stave detection.
- `paco_neume_masked_band` — same boxes, but a thin strip around the nearest
  baseline-detected stave line is preserved unmasked inside each box.

## Results (`eval_batch_metrics.csv`)

Not yet run end-to-end in this environment — see the note at the bottom of
this README. Run:

```bash
cd staff-finding/scripts
python eval_paco_neume_masking.py \
    --manifest ../e2e_tests/paco_neume_masking_exp1/manifest.csv \
    --tm-weights ../../landing-page/scripts/assets/models/medieval/text_music_detector_fulldata.pt \
    --stave-weights ../../landing-page/scripts/assets/models/medieval/stave_detector_fulldata.pt \
    --paco-models-dir ../../paco-classifier/models_v4 \
    --output ../e2e_tests/paco_neume_masking_exp1
```

then fill in this table from the resulting `eval_batch_metrics.csv`:

| page | variant | precision | recall | F1 | mean y-MAE |
|---|---|---|---|---|---|
| Antiphonal_1v_hfngl | paco_baseline | | | | |
| Antiphonal_1v_hfngl | paco_neume_masked_naive | | | | |
| Antiphonal_1v_hfngl | paco_neume_masked_band | | | | |
| CH-E611_15r | paco_baseline | | | | |
| CH-E611_15r | paco_neume_masked_naive | | | | |
| CH-E611_15r | paco_neume_masked_band | | | | |
| l'Arsenal_Ms-5198_11 | paco_baseline | | | | |
| l'Arsenal_Ms-5198_11 | paco_neume_masked_naive | | | | |
| l'Arsenal_Ms-5198_11 | paco_neume_masked_band | | | | |

## Diagnostics

Per page, per variant, 5 files (not one per detected line — see the script's
docstring):

1. `..._1_layer.png` — the separated layer itself (baseline / naive-masked /
   band-masked), so the masking's effect on the ink is visible directly.
2. `..._2_boxes.png` — the layer with that variant's stave-YOLO boxes (green)
   and the music-class boxes that drove masking (orange) overlaid.
3. `..._3_component_filter.png` — kept pixels (green) + discarded components'
   bounding boxes (red), composited at page location.
4. `..._4_centerlines.png` — fitted centerlines overlaid on the page.
5. `..._5_grouping.png` — final stave grouping (which lines joined which
   stave), via `group_staves`'s own existing diagnostic.

## Known caveats

- Only 2 classes are actually trained/deployed today — no real neume-ink
  model exists yet. Rectangular-box masking is a proxy for what a real
  3-class classifier might achieve, not identical to it. This is a mock
  test, not a benchmark of the real (not-yet-trained) model.
- Small corpus (3 pages, no overlap with the prior 2-page raw-page
  comparison) — a first read, not a rigorous validation.
- **Not yet run end-to-end in this session**: `torch`/`tensorflow` imports
  hung indefinitely (10+ minutes, ~0% CPU — not just slow, actually stalled)
  in the sandboxed shell this was built in, reproduced twice with the sandbox
  both on and off. Most likely cause is macOS Gatekeeper's online
  notarization check stalling against freshly-installed native libraries
  with no network reachable from this session — a known slow-first-import
  failure mode, not a bug in the script itself — but that's a guess, not
  confirmed. Run the command above from a normal terminal (not this session)
  to execute it; if it also hangs there, that's a real environment problem
  worth chasing separately from this script.
