# ms234_064_exp1 — BGR and glyph-masking experiments

Test bed: McGill_MS234-064 (`image-sets/ms234_064/`), 32 hand-corrected
staffline boxes / 8 staves, plus model-predicted text (70) and music (146)
glyph boxes from a hand-labeled multi-class annotation file. Prep script:
`scripts/prepare_ms234_testdata.py`.

## What was tested and why

1. **BGR (background removal) ink separation** — `run_page.py` has a
   `--bgr-model` flag that's never been exercised against a page with ground
   truth in this repo (every prior test used `--no-bgr`). Ran both to see
   whether the existing ML ink-separation step helps.
2. **Glyph masking** — mask upstream music-class YOLO boxes out of each
   staffline crop before component filtering, to test whether removing
   competing neume ink helps or hurts. See
   `experiments/glyph_masking/NOTES.md` for the full design/writeup.

## Results (`eval_batch_metrics.csv`)

| variant | precision | recall | F1 | mean y-MAE |
|---|---|---|---|---|
| no_bgr (baseline) | 1.000 | 1.000 | 1.000 | 2.34 px |
| bgr | 1.000 | 0.469 | 0.638 | 3.51 px |
| glyphmask_baseline (sanity check, = no_bgr) | 1.000 | 1.000 | 1.000 | 2.34 px |
| glyphmask_naive | 1.000 | 1.000 | 1.000 | 2.50 px |
| glyphmask_band | 1.000 | 1.000 | 1.000 | 2.34 px |

- **BGR hurt badly**: 17/32 lines lost entirely (`no_components_survived`).
  Visually confirmed the BGR model erased almost all staffline ink in the
  failing boxes. The checkpoint used (`best_model_14april.pth`) was trained
  on a different set of manuscripts and doesn't generalize to this one — a
  checkpoint-domain issue, not necessarily a verdict on BGR in general.
- **Glyph masking**: baseline here already hits perfect P/R/F1, so there was
  no accuracy problem to fix. Naive full-bbox masking measurably degraded
  y-MAE and introduced new merge-ambiguity flags on lines that were clean
  before masking (confirmed via diagnostic PNGs — the music bbox spanned
  most of the crop width in the affected lines, so masking deleted real
  on-line ink). Band-limited masking (preserve a thin strip around the
  estimated line) came back statistically identical to baseline, confirming
  it avoids that risk.

## Next steps

Neither experiment had a *failing* case to fix on this page. Next: run the
same two experiments (BGR and glyph masking) on the Gent pages
(`image-sets/gent/`), which are known to be more complex and are more likely
to expose an actual case where glyph-masking or BGR change the outcome
instead of just confirming safety on an easy page.
