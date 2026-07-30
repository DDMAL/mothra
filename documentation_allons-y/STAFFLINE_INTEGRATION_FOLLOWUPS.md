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

## Needs a manual step, not a code change

- **Branch protection**: `.github/workflows/tests.yml` runs on every push,
  but nothing requires it to pass before merging yet. Turn that on under
  the repo's Settings → Branches → branch protection rule for `main` →
  "Require status checks to pass before merging" → select `tests / test`.
