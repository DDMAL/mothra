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
WORKDIR /app
COPY scripts/requirements.txt ./scripts/requirements.txt
RUN pip install --no-cache-dir -r scripts/requirements.txt
# Prerequisite: `git lfs pull` before `docker build` — the medieval model
# checkpoints under scripts/assets/models/medieval/ are Git-LFS pointers
# until that's run, and this COPY would otherwise ship stub files.
COPY scripts/ ./scripts/
COPY --from=frontend-build /app/dist ./dist
COPY --from=frontend-build /app/public ./public
WORKDIR /app/scripts
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]