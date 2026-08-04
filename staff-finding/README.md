# staff-finding
this repo contains the vagaries associated with my efforts at line detection, completion, and grouping. 
oy vey. 

## Packaging

This directory is a local pip distribution (`pyproject.toml`) so
`landing-page/`'s backend can `pip install -e staff-finding/` and import the
six core algorithmic modules (`component_filter`, `fit_centerline`,
`group_staves`, `interpolate_staves`, `fallback_redetect`, `yolo_io`)
directly — see `landing-page/scripts/staffline_stage.py` and CLAUDE.md's
**Staffline detection** section at the repo root. CLI drivers (`run_page.py`,
`bgr_adapter.py`, `detect_stafflines.py`, the `experiments/` runners) aren't
part of the package and keep working exactly as before for standalone/offline
use. Install the `diagnostics` extra (`pip install -e ".[diagnostics]"`) to
also run the tests that exercise `save_path` (matplotlib-based rendering).