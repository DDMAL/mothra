# ---- stage 1: frontend (tsc + vite) + embedded Neon.js editor ----
# Prerequisite: git submodules must be checked out before `docker build`
# (`git submodule update --init --recursive`) — `neon/` is a submodule and
# this stage's `npm run build` builds it as part of `build:neon`.
FROM node:20-slim AS frontend-build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# neon/webpack.config.js shells out to `git rev-parse --short HEAD` to embed
# a build version stamp. neon/'s real .git lives in the superproject's
# `.git/modules/`, outside this build context (context is landing-page/, not
# the repo root) — so we seed a throwaway repo just so that command
# succeeds. The embedded hash in a Docker-built image is synthetic, not the
# real commit; threading the real SHA in as a build-arg from CI (where full
# git history exists) would be the "real" fix if that watermark ever matters.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f neon/.git \
    && git -C neon init -q \
    && git -C neon -c user.email=docker@build -c user.name=docker commit --allow-empty -q -m init
# build:neon shells out to `yarn` — node:20-slim already ships it, no install needed
RUN npm run build

# ---- stage 2: python runtime ----
FROM python:3.11-slim AS runtime
# opencv (pulled in transitively by ultralytics) needs libGL/glib at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# The NVIDIA container runtime reads NVIDIA_DRIVER_CAPABILITIES to decide which
# driver features to inject; "compute" is REQUIRED for CUDA. Without it you get
# nvidia-smi but cuInit -> CUDA_ERROR_NO_DEVICE. (Same gotcha as Rodan's GPU image.)
# Harmless on CPU-only nodes — torch just reports cuda unavailable and runs on CPU.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY scripts/requirements.txt ./scripts/requirements.txt
# ultralytics pulls in torch/torchvision transitively; install the CUDA 12 build
# first (cu121 ships sm_90/Hopper cubins so no PTX JIT) so the requirements.txt
# install below sees torch already satisfied. CUDA runtime libs come from this
# wheel — no CUDA base image needed; the host driver is injected by the k8s
# `nvidia` RuntimeClass. On CPU-only nodes this same image runs on CPU via the
# app's auto-device (torch.cuda.is_available() -> False).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r scripts/requirements.txt
# staff-finding/ is a sibling repo dir, outside this build context (context is
# landing-page/, not the repo root) — provided via the "staff_finding"
# additional build context (docker-compose.yml's build.additional_contexts /
# build-images.yml's build-contexts input). staffline_stage.py imports its
# packaged modules (component_filter, fit_centerline, group_staves, ...)
# directly. Landed at /app/staff-finding so its own pyproject.toml's relative
# package-dir="scripts" mapping resolves the same way it does in a bare-metal
# checkout. .dockerignore there excludes large test-fixture/experiment dirs
# not needed at runtime.
# Non-editable: the whole source tree is already baked into the image via
# the COPY above regardless, so -e buys nothing at runtime (no live-edit
# workflow makes sense inside a built container) beyond an unnecessary
# editable-install pointer back at that same source tree.
COPY --from=staff_finding . /app/staff-finding/
RUN pip install --no-cache-dir /app/staff-finding
# pitch-finding/ is the DDMAL/Standalone-Pitch-Finder submodule, likewise a
# sibling of this build context -- provided via the "pitch_finding" additional
# build context. Only its scripts/ dir is copied: the repo also carries ~70MB
# of sample page images and debug renders that nothing at runtime reads.
#
# NOT pip-installed, unlike staff-finding: that repo carries its own
# pyproject.toml and this one doesn't (it's a prototype driven by its CLIs,
# with flat cross-imports like `import clef_rules`). pitch_stage.py puts this
# directory on sys.path instead, which is the same flat-module resolution its
# own CLIs rely on. PITCH_FINDING_DIR is how config.py finds it here -- the
# config.yaml default is a bare-metal-checkout-relative path, which this
# layout is deliberately not.
COPY --from=pitch_finding scripts/ /app/pitch-finding/scripts/
ENV PITCH_FINDING_DIR=/app/pitch-finding/scripts
# Prerequisite: `git lfs pull` before `docker build` — the medieval model
# checkpoints under scripts/assets/models/medieval/ are Git-LFS pointers
# until that's run, and this COPY would otherwise ship stub files.
COPY scripts/ ./scripts/
COPY --from=frontend-build /app/dist ./dist
COPY --from=frontend-build /app/public ./public
WORKDIR /app/scripts
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]