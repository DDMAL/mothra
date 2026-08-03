# Staffline integration — follow-ups

Written after wiring staff-finding's staffline detection into the
landing-page predict/encode pipeline (see `CLAUDE.md`'s **Staffline
detection** section for how it works today). Nothing below is started;
this is a punch list, not a plan.

## Deferred by design (not started)

- **Ink-separation ("BGR")** — `staff-finding/scripts/bgr_adapter.py` wraps
  an external, unvendored `muscrat/layer_sep` repo reachable only via
  hardcoded paths on specific dev machines. No tests, no Docker/LFS story,
  and its checkpoint (`best_model.pth`) doesn't match the repo's `*.pt`
  Git-LFS rule. Staffline detection runs on raw page crops instead; Sauvola
  binarization is the interim mitigation. Real next step: give
  `muscrat/layer_sep` a stable, team-owned location (vendor it or add it as
  a submodule) and at least one test, before wiring it into
  `staffline_stage.py`.
- **`interpolate_staves.py`** — implemented and unit-tested, but off by
  default (`interpolate_missing=False`) per `staff-finding/dox/STATUS.md`'s
  "not yet validated across the corpus" caveat. It's a plumbed-through
  parameter already, so turning it on is a default flip, not new code —
  but do the corpus validation first.
- **`fallback_redetect.py`** — not wired into `staffline_stage.py`. Needs an
  already-loaded stave-detector YOLO model instance; the medieval preset has
  one resident, but custom uploaded models don't have a dedicated one to
  re-probe with.
- **Frontend UI** — staffline QA data (rhythm anomalies, low-confidence
  fits, reconciliation flags) only surfaces in `job_events` log lines today.
  No frontend references staffline/stave concepts at all yet. Natural home
  for a future overlay: `AnnotationsTab.tsx`.
- **`batch_api.py`'s text-batch-run path** — a second, independent code path
  that also runs YOLO + `write_annotation()`, outside the Celery job queue.
  Images processed only through it won't get a `staffline_detections` row;
  the 3-tier fallback in `tasks_encode.py` already degrades gracefully, but
  nobody's made it call `staffline_stage.py` either.

## Found along the way, not fixed

- **`encode_to_mei.py`'s `extract_ms_id` import** already silently falls
  back to `Path(filename).stem` in the deployed Docker image (the file it
  wants lives at the repo root, outside `landing-page/`'s build context —
  same class of gap as the one just fixed for `staff-finding/`, just never
  addressed for this one). Needs the same treatment: either add it as
  another named build context, or move/vendor it.
- **Two `.pyc` files are tracked in git**:
  `staff-finding/scripts/__pycache__/{bgr_adapter,yolo_io}.cpython-312.pyc`.
  They show as "modified" on essentially every local session that imports
  those modules. Worth a `git rm --cached` + a `__pycache__/` gitignore rule
  scoped to `staff-finding/` (the existing repo-wide `.gitignore` doesn't
  seem to cover this path).
- **`test_run_page.py`/`test_run_pageOG.py`'s fake-module stubbing has no
  teardown** — they replace `sys.modules["torch"]`/`["ultralytics"]`/
  `["inference_simple"]` at module level and never restore them. CI now
  works around this by running them as isolated `pytest` steps (see
  `.github/workflows/tests.yml`'s comments), but the underlying fragility
  in those two files is still there for anyone running them ad hoc alongside
  other test files. Proper fix: move the stubbing into a fixture with
  teardown (`monkeypatch.setitem`, scoped to the test module).
  **Update**: `test_run_page.py` specifically also stubbed the wrong
  module — `sys.modules["inference_simple"]` never took effect because
  `bgr_adapter.py` raises its own `ModuleNotFoundError` before ever reaching
  that import (an unconditional `os.path.isfile()` check across hardcoded
  developer paths). This only surfaced in real CI, not local runs, since
  the local dev machine happened to have the real dependency at one of
  those paths. Fixed by also stubbing `sys.modules["bgr_adapter"]` itself —
  the no-teardown concern above still applies to both files, unfixed.
- **`auth_api.py` had a duplicate `ALTER TABLE project_images ADD COLUMN
  folio TEXT` migration** (around two different line numbers in
  `_migrate_db()`), most likely from the same large `main` merge that also
  broke `init_db()` (see below). Harmless — `DuplicateColumn` makes the
  second one a no-op — but worth deleting the redundant block.
- **`staffline_detections` has no retention/archival plan** (CodeRabbit,
  PR #53) — it accumulates forever by design (see the schema comment), which
  is deliberate for the interpolate_staves before/after comparison use case,
  but storage still grows unbounded per re-run. Same category of gap as
  `job_uploads`/`job_sessions` (CLAUDE.md's own "things that don't exist
  yet" list) — plan a cleanup job alongside those, not in isolation.
  **Explicit deferral** (CodeRabbit followed up asking for either a bound or
  an explicit deferral statement): treat this as blocking, not a nice-to-have
  — do not point staffline detection at a production database until a
  retention bound (time- or count-based, scoped so it doesn't discard rows
  `interpolate_staves`' before/after comparison still needs) ships alongside
  the `job_uploads`/`job_sessions` cleanup job.
- **Three checked-in e2e `stave_grouping_report.txt` baselines show genuinely
  fragmented grouping — confirmed still-reproducible on current code, not
  stale fixtures** (CodeRabbit, PR #53: Gent right,
  `F-Pn-Latin-15181_107r`, `F-Pn-Latin-15181_221`). Re-ran all three through
  the current pipeline rather than assuming the checked-in numbers just
  predated a later fix:
  - `F-Pn-Latin-15181_107r`/`_221` (`test_model.sh --fallback-redetect`)
    reproduce their checked-in `reconciled_duplicate_fits` /
    `staves_with_unexpected_count` numbers almost exactly (gap values match
    to ~0.1px), even after the fallback-redetect existing-line-distance fix
    (`ecaa401`) — that fix had nothing to reject on either page
    (`fallback_redetect_report.txt`: 0 candidates / "no under-populated
    staves found"), so it isn't the relevant bottleneck here. These two are
    an accurate record of a real, current `group_staves.py`
    duplicate/companion-reconciliation limitation on dense, tightly-spaced
    staves (35-45% of fits end up marked duplicate/companion; mode line
    count collapses to 1 or 4 instead of this manuscript's real stave size).
  - Gent right didn't reproduce cleanly enough to even compare: two
    plausible source YOLO `.txt` files exist
    (`image-sets/gent/right/inference/corrected/...` vs
    `e2e_tests/29may/Gent15_17_right/Gent15_17_right_corrected.txt` — they
    differ in content and line count), and re-running `run_page.py` against
    either one produces box_index 0 at different coordinates than the
    checked-in fixture, so neither is confirmed as the original input.
    Both attempts still show heavy fragmentation (mode 5 with 8/17 staves
    unexpected, or mode 1 with 14/20 unexpected) — same symptom as the
    other two pages, just via an input this session couldn't pin down
    precisely.
  - Not attempting a fix here: this is a real algorithmic gap in
    `group_staves.py`'s reconciliation logic on dense staves, not a
    one-line bug, and forcing a low-confidence change (or regenerating the
    fixtures as-is) to make these three pages look clean risks either
    overfitting to them or just checking in a different flavor of the same
    fragmentation. CodeRabbit's ask — regenerate, then assert acceptable
    per-stave counts — is the right ask; it's blocked on the reconciliation
    fix landing first.
- **`component_filter.py`'s `discarded`/`score_breakdown[...]["kept"]`
  bookkeeping doesn't account for companion retention** (CodeRabbit, PR #53,
  `component_filter.py:280-299`, Minor) — companions folded into the active
  output via the retention pass (lines 337-401) still show up in `discarded`
  (reason `"not_top_scoring"`) and never get `"kept": True`, since the two
  passes don't talk to each other. Affects diagnostic/summary-CSV accuracy
  (`run_page.py`'s `n_discarded` column), not the actual detection/fit
  output. Deferred rather than fixed in the same pass as the other findings
  since the correct fix needs to reconcile against whichever companion-label
  set matches the caller's actual `merge_components` mode, and this file's
  companion-selection logic (811 lines total) deserved more careful reading
  than the time available for a Minor-severity bookkeeping issue.

## Resolved since this was written

- **Rebase-onto-main evaluated, decided against**: checked what rebasing
  `kyrie/staff-finding` onto `main` would take. `origin/main` hasn't moved
  since the 2026-07-30 merge (`15061d1`) — this branch already fully
  contains it (`git merge-base --is-ancestor origin/main kyrie/staff-finding`
  is true) — so a rebase would only replay this branch's own 54 commits
  against the merge commit's absence, re-hitting the same `auth_api.py`/
  `CLAUDE.md` conflicts already resolved once, and force-pushing over PR
  #53's live review thread for no content gain. Decided: no rebase, no
  force-push; `main` gets a clean single commit via squash-merge when PR #53
  lands instead (repo already has `allow_squash_merge: true`).
- **`CLAUDE.md`'s Deployment section repaired**: the 2026-07-30 merge kept
  only this branch's own `### Deployment (Docker)` section and silently
  dropped both of `main`'s — `### Deployment (Kubernetes, CI/CD via GitHub
  Actions)` (the real production path — auto-deploy to a k8s cluster on
  push to `main`) and `### Local/manual container runs`, which explicitly
  scopes `docker-compose.yml` as local-only ("the k8s manifests were
  modeled on it, not the other way around"). This branch's text had claimed
  Compose "is now the only deployment path," which was wrong — the actual
  `k8s/` manifests and CI/CD deploy job were untouched throughout (confirmed
  byte-identical to `main`), only the docs were stale. Restored both
  sections, folding this branch's genuinely new local-testing content
  (staff-finding's Docker build-context wiring, the buildx/OOM/Tridis/
  redeploy-together notes) into the restored `Local/manual` section. Also
  fixed a second, smaller staleness bug found in the same paragraph: it
  still said `pip install -e` for staff-finding's install, which an earlier
  commit this same session had already changed to non-editable.
  Also refreshed three `## Key files` table rows (`job_store.py`,
  `jobs_api.py`, `tasks_predict.py`/`tasks_encode.py`) that had fallen
  behind `main`'s versions — missing mentions of job cancel/retry and
  `tasks_text_batch.py` — table only; the full prose elsewhere in the file
  already covered this correctly. And added the `staffline_detections` row
  to the Database schema table itself — a previous session summary had
  claimed this row already existed; it didn't, this is the actual fix.
- **CI job added**: `.github/workflows/tests.yml` runs the staff-finding
  suite and the new `landing-page/scripts/tests/` on every push.
- **`main` merged in and briefly broke `auth_api.py`**: a large merge from
  `main` (bringing in real, unrelated feature work — job cancellation,
  image resizing, a new `tasks_text_batch.py`, ~45 files) collided with the
  `staffline_detections` addition to `init_db()`, leaving a duplicated
  `annotations` table block and code dedented outside its `try:` block —
  a genuine syntax error, not just a style issue. Repaired by comparing
  directly against both merge parents and reconstructing `init_db()` with
  the other side's new error-handling (`except DuplicateTable/
  DuplicateObject/UniqueViolation`) preserved. Worth noting for the eventual
  main-rejoin: `main` is moving fast, re-diff before assuming anything about
  its current state.
- **`tasks_predict.py` scoping bug**: the same merge's independent
  `has_annotation`/`has_text_alignment` skip logic left `yolo_txt`/
  `img_arr`/`ann_id` unassigned when an image already had an annotation but
  not yet a text-alignment — exactly the case that logic exists to support.
  Fixed by decoding the image unconditionally and fetching the existing
  annotation when YOLO is skipped.

## CodeRabbit review pass (PR #53), 2026-07-30

Went through every open CodeRabbit comment on
[PR #53](https://github.com/DDMAL/mothra/pull/53) (`kyrie/staff-finding` →
`main`) individually, including several that a GitHub posting failure left
only in the review-body text, never as inline comments. Fixed the ones
agreed with; two items above (`component_filter.py` bookkeeping,
`staffline_detections` retention) were agreed with but deferred rather than
rushed. Declined one on purpose: the `detect_stafflines.py`/`test_model.sh`/
`scripts/run_inference.py` same-stem-filename-collision finding is real, but
`test_model.sh`'s own help text already documents it as a known, accepted
limitation across all three files — not an oversight to fix without that
being a deliberate decision.

Fixed, each independently verified (a real test proving the specific
behavior, not just a passing compile):
- `staffline_stage.py`: the docstring promised a `status='failed'` row on
  error; the code never wrote one. Implemented it.
- `staffline_stage.py` / `tasks_predict.py`: cooperative cancellation was
  only checked once per image, not inside `run_staffline_detection`'s own
  per-box loop — exactly the gap CLAUDE.md warns new long-running stages
  need to avoid. Added a `check_cancelled` call per box, with `JobCancelled`
  explicitly re-raised past the stage's own broad `except Exception`.
- `tasks_predict.py`: the `has_text_alignment` skip's `continue` sat before
  the staffline-detection block, so an image with fresh YOLO boxes but an
  already-existing text alignment never got a `staffline_detections` row at
  all. Reordered so staffline detection is gated only on `has_class`.
- `run_page.py` / `bgr_adapter.py`: `run_page.py` imported `bgr_adapter` at
  module top, so `--no-bgr` (and even `--help`) still needed the external
  `inference_simple` dependency to resolve. Made the import lazy, reached
  only inside the `use_bgr` branch.
- `run_page.py`'s fallback-redetect path: `box_index` was computed before
  any candidate was accepted, so multiple candidates in one region collided
  on the same diagnostic filename (each overwriting the last), and the
  summary row's recorded path didn't match what was actually written
  anyway. Gave each candidate its own per-region, per-candidate path,
  carried through to its summary row via the candidate's own identity.
- `fallback_redetect.py`: candidates that just re-detected an existing line
  could win a `max_new_lines` slot before `group_staves`' own duplicate
  reconciliation ever got a chance to run, displacing a genuinely missing
  line. Added an existing-line-distance rejection before ranking.
- `eval_page.py`: crashed on interpolated JSOMR records missing
  `centerline_page` (no `bounding_box` to fall back to). Now degrades to an
  empty result instead of crashing. CodeRabbit followed up asking for the
  fixture-regeneration side too (the double-page-offset risk for stale
  pre-`centerline_page` fixtures) — regenerated all four tracked
  `implicit_neural*` Gent-right fixtures via `run_implicit_neural_page.py`;
  every record now carries `centerline_page`, verified against
  `eval_page.py`'s own `_pred_page_y`/`_pred_page_x` helpers.
- `dp_tracer.py` (experiment): the DP cost only rewarded dark pixels, so a
  trace could drift onto a darker neighbouring staffline within the same
  search band. Added a distance-from-hint penalty, weight chosen by actually
  reproducing the drift on a synthetic adversarial case and finding the
  tipping point, not guessed.
- `periodicity_detector.py` / `run_periodicity_page.py` (experiment): the
  full-page Gaussian blur was recomputed on every staffline call instead of
  once per page. Hoisted into `compute_dark_field()`; confirmed
  byte-identical output before/after.
- `scripts/run_inference.py`: failed images were silently skipped with no
  structural record and no non-zero exit code, so a partial batch run looked
  identical to a complete one to any automation checking the exit code.
  Now writes `failed_images.json` and exits 1 on partial failure;
  `all_predictions.json`'s shape is unchanged for annotator compatibility.
- `group_staves.py`: docstring claimed `interpolation_max_gap` defaults to
  `cut_threshold * INTERPOLATION_GAP_MULTIPLIER`, but the adaptive
  gap-distribution path is what actually resolves it in the normal case.
- `.gitignore`: `!models/**/*.pt` only rescues the repo-root `models/`
  directory, not `landing-page/scripts/assets/models/medieval/` (that
  path's `.pt` files are tracked today purely because gitignore doesn't
  apply retroactively — a future accidental untrack would silently need
  `-f` to re-add them). Added a path-specific negation. Left
  `staff-finding/models/` alone — confirmed those files are genuinely,
  intentionally untracked, not a gap.
- `staff-finding/pyproject.toml`: `gp_fitter.py` needs `scikit-learn` and
  `implicit_neural_fitter.py` needs `torch`, neither declared anywhere.
  Added both as a new `experiments` extra.
- `landing-page/Dockerfile`: switched the staff-finding install from
  editable to normal, since the whole source tree is already baked into
  the image regardless.

## Needs a manual step, not a code change

- **Branch protection**: `.github/workflows/tests.yml` runs on every push,
  but nothing requires it to pass before merging yet. Turn that on under
  the repo's Settings → Branches → branch protection rule for `main` →
  "Require status checks to pass before merging" → select `tests / test`.
