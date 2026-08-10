# gent_right_exp1 — BGR and glyph-masking experiments (Gent right)

Same experiments as `e2e_tests/ms234_064_exp1/`, run on the established Gent
right test page (86 lines / 17 staves, corrected GT). Glyph boxes generated
via `scripts/infer_glyph_boxes.py` using the local `text_music_detector_split.pt`
checkpoint (141 text + 159 music boxes; no `staves` detections from this
checkpoint on this page, so the existing corrected staffline GT was used as
the pipeline input/eval GT as before).

See `experiments/glyph_masking/NOTES.md` for the full writeup. Headline
(`eval_batch_metrics.csv`):

| variant | F1 | mean y-MAE |
|---|---|---|
| no_bgr (baseline) | 1.000 | 1.69 px |
| bgr | 0.151 | 2.71 px |
| glyphmask_naive | 0.994 (1 line lost) | 1.72 px |
| glyphmask_band | 1.000 | 1.71 px |

BGR is even worse here than on MS234 (F1 0.15 vs 0.64) -- confirms the
checkpoint issue isn't manuscript-specific. Glyph masking: naive masking
lost `line0045` outright (5 note-groups sitting on one line, each individual
fragment after masking too small/low-scoring to survive the component
filter or be reassembled by merge/companion logic) -- the first concrete
case of naive masking causing real damage, not just accuracy drift.
Band-limited masking stayed safe (no lost lines, only single-digit-pixel
changes on 13/86 lines).
