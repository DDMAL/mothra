# Mothra

**YOLO-based Optical Music Recognition for Medieval Manuscripts**

A DDMAL (McGill) research project exploring whether YOLO object detection can outperform
the traditional OMR pipeline (Rodan) for medieval musical manuscript analysis — from raw
manuscript image, through human-in-the-loop correction, to a corrected MEI file ready for
Cantus Ultimus.

The live site: https://mothra.simssa.ca/.

> **New here?** [`CLAUDE.md`](CLAUDE.md) is the up-to-date technical reference for the
> web app (architecture, local setup, deployment). This README is the map of the whole
> project and how its pieces fit together; `CLAUDE.md` is where you go once you're
> working in `landing-page/`.

---

## Why "Mothra"?

This project challenges the existing [Rodan](https://github.com/DDMAL/Rodan) workflow
engine (kaiju fight, anyone?). Where Rodan uses a multi-stage, pixel-level
document analysis and segmentation pipeline, Mothra explores whether end-to-end deep
learning object detection can drive OMR instead — trading some interpretability for
speed and a simpler annotation → training → inference cycle.

---

## What's live today

The center of gravity of this repo is **[`landing-page/`](landing-page/)** — a full
web application (React + FastAPI + Celery + Postgres) that carries a manuscript through
the whole pipeline:

```
upload images → Interactive Classifier (IC) → encode to MEI → Neon.js correction → export for Cantus Ultimus
```

YOLO models detect text/music/stave regions; a dedicated staffline-detection stage fits
and groups stave lines; a text-finding service (HTR) locates chant text; the Interactive
Classifier is where a human corrects and classifies neume detections; the result is
encoded to MEI and hand-corrected in an embedded Neon.js editor before being bundled for
manual hand-off into Cantus Ultimus.

**To run it locally:**

```bash
git submodule update --init --recursive   # pulls in ic/, mothra-text/, landing-page/neon
./dev.sh                                  # starts all five services together
```

See [`CLAUDE.md`](CLAUDE.md) for the full architecture, manual per-service startup,
configuration, Docker/Kubernetes deployment, and database schema — that file is kept
current as the app evolves and is the source of truth, not this section.

---

## The ecosystem

Mothra isn't one codebase — related pieces live as sub-packages in this repo, as git
submodules, as sibling DDMAL repos, and as active-but-unmerged branches. Here's the map:

| Piece | What it does | Where it lives |
|---|---|---|
| **Landing page** | The web app tying the whole pipeline together | [`landing-page/`](landing-page/) (this repo) |
| **Staffline detection** | Component filtering → centerline fitting → stave grouping on YOLO stave boxes | [`staff-finding/`](staff-finding/) (this repo, pip-installed by `landing-page/`) — see its [status doc](staff-finding/dox/STATUS.md) |
| **Text-finding** | Kraken/HTR pipeline that locates and reads chant text on the page | [`text-service/`](text-service/) (this repo) wraps the [`mothra-text`](https://github.com/DDMAL/mothra-text) submodule |
| **Interactive Classifier (IC)** | Human-in-the-loop correction of detections, including neume classification | [`ic/`](ic/) submodule → [DDMAL/Standalone-Interactive-Classifier](https://github.com/DDMAL/Standalone-Interactive-Classifier) |
| **Pitch finding** | Turning stave + neume detections into actual pitches | not in this repo — [DDMAL/Standalone-Pitch-Finder](https://github.com/DDMAL/Standalone-Pitch-Finder) |
| **Mothra Print** | Running the pipeline on printed (not manuscript) chant books, starting with the Liber Usualis | inference artifacts in [`inference-outputs/liber_usualis_print/`](inference-outputs/liber_usualis_print/); in final tweaks for a Cantus Ultimus music-search upload |
| **paco-classifier integration** *(in progress, unmerged)* | An alternate/experimental staffline-detection classifier being wired in as its own service | branch [`gianna/calvo-integration`](https://github.com/DDMAL/mothra/tree/gianna/calvo-integration) |
| **Mothra Annotator** | Browser-based bounding-box annotation tool for building YOLO training data | now hosted externally at https://ddmal.ca/mothra-annotator/ (no longer part of this repo) |

Pitch finding and the IC's neume classification are deliberately out-of-repo — they're
developed and versioned as their own standalone tools rather than folded into
`landing-page/`.

---

## Repository layout

```
mothra/
├── README.md                    # This file
├── CLAUDE.md                    # Technical reference for landing-page/ (architecture, setup, deployment)
├── landing-page/                # The live web app (React + FastAPI + Celery + Postgres)
│   ├── scripts/                 # FastAPI backend, Celery tasks
│   ├── src/                     # React frontend
│   ├── neon/                    # Neon.js MEI editor (submodule)
│   └── ...
├── staff-finding/                # Staffline detection module (pip-installable, used by landing-page)
│   └── dox/                     # Design notes, status, pitch-finding notes
├── ic/                           # Interactive Classifier (submodule)
├── mothra-text/                  # Text-finding pipeline used by text-service/ (submodule)
├── text-service/                 # FastAPI wrapper around mothra-text
├── k8s/                          # Kubernetes manifests for deployment
├── docker-compose.yml            # Local multi-container stack (mirrors k8s/)
├── OLD-annotator/                 # Legacy local annotation script — superseded by the hosted annotator, kept for reference only
├── documentation_allons-y/        # Project notes: original brief, staff-finding integration follow-ups, WIP docs
├── configs/                       # YOLO training configs (original layout-detection experiments)
├── data/                          # Manuscript images/annotations for the YOLO layout-detection experiments (gitignored where large)
└── scripts/                       # Standalone training/inference/conversion scripts for the above experiments
```

`configs/`, `data/`, and the root-level training/inference scripts are the original
YOLO layout-detection experiment track that predates `landing-page/` — still around and
occasionally used, but less active than the web app.

---

## Comparison to Rodan

| Feature | Rodan | Mothra |
|---|---|---|
| **Approach** | Multi-stage pipeline | End-to-end object detection |
| **Layout analysis** | Pixel.js, Gamera | YOLO |
| **Training data** | Pixel-level masks | Bounding boxes |
| **Staff detection** | Hough transform | Component filter + centerline fit + stave grouping (`staff-finding/`) |
| **Neume classification / correction** | Interactive Classifier | Interactive Classifier (same tool, now standalone) |
| **Pitch finding** | Built into the Rodan job graph | Standalone tool ([Standalone-Pitch-Finder](https://github.com/DDMAL/Standalone-Pitch-Finder)), not yet wired into the Mothra pipeline |
| **Human in the loop** | Multiple pipeline stages | IC correction + Neon.js MEI correction |
| **Deployment** | Docker stack | Kubernetes (see [`k8s/`](k8s/)) or Docker Compose |

**Hypothesis:** Mothra trades some of Rodan's interpretability for speed and,
hopefully, better handling of degraded manuscripts through learned feature extraction.

---

## Status & known gaps

- **No end-to-end pitch-to-MEI in this repo yet** — staffline detection produces the
  grid pitches would be read against, but pitch assignment itself lives in the separate
  [Standalone-Pitch-Finder](https://github.com/DDMAL/Standalone-Pitch-Finder) and isn't
  integrated into the `landing-page/` pipeline.
- **Staffline detection** has a few features implemented but intentionally not yet
  enabled by default (ink separation, gap interpolation, fallback re-detection) — see
  [`staff-finding/dox/STATUS.md`](staff-finding/dox/STATUS.md).
- **Mothra Print** currently covers one printed source (the Liber Usualis); a second
  candidate manuscript is under consideration but not yet started.
- See [`CLAUDE.md`](CLAUDE.md)'s "Things that don't exist yet" section for gaps specific
  to the web app (job cleanup, health checks, IIIF import).

---

## Contributing / where to look next

- Working on the web app? Start with [`CLAUDE.md`](CLAUDE.md).
- Working on staffline detection? Start with
  [`staff-finding/dox/STATUS.md`](staff-finding/dox/STATUS.md) and
  [`staff-finding/README.md`](staff-finding/README.md).
- Curious about the project's original scope and open questions?
  [`documentation_allons-y/PLAN.md`](documentation_allons-y/PLAN.md) is the original
  brief; [`documentation_allons-y/STAFFLINE_INTEGRATION_FOLLOWUPS.md`](documentation_allons-y/STAFFLINE_INTEGRATION_FOLLOWUPS.md)
  tracks live follow-ups from wiring staffline detection into the pipeline.
- Interactive Classifier or pitch finding? Those are developed in their own repos —
  see the ecosystem table above.

---

**Key papers this draws on:**

- DocLayout-YOLO: [arXiv:2410.12628](https://arxiv.org/abs/2410.12628)
- "Optical Medieval Music Recognition Using Background Knowledge" (MDPI, 2022)
- Rodan: [ddmal.github.io/Rodan](https://ddmal.github.io/Rodan)
