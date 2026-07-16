#!/usr/bin/env bash
#
# Mothra local dev launcher — starts all five web-app servers together and
# tears them all down on Ctrl-C. See CLAUDE.md for the architecture.
#
#   :5173  landing-page frontend  (Vite)      — open this in the browser
#   :8001  landing-page backend   (uvicorn)   — /api/*; proxied from :5173
#   :8000  Interactive Classifier (ic-api)    — IC REST API + IC SPA (iframe)
#   :8002  text-finding service   (uvicorn)   — mothra-text wrapper; reached
#                                                server-to-server from :8001
#   —      celery worker (predict/encode jobs) — no port; job state lives in
#                                                Postgres, Redis is only the
#                                                task broker/transport
#
# Usage:
#   ./dev.sh              start all five
#   ./dev.sh -b           rebuild the IC frontend bundle first (do this when
#                         the IC submodule's frontend changed; otherwise the
#                         iframe serves a stale build — see the binarize bug)
#   ./dev.sh -f           free the ports first if something is already on them
#   ./dev.sh -bf          both
#   ./dev.sh -h           help
#
# Ports are overridable: WEB_PORT=3000 ./dev.sh
# Requires a running Redis reachable at CELERY_BROKER_URL (default
# redis://localhost:6379/0) — e.g. `brew services start redis`. Redis is
# only the Celery task broker; job state/progress lives in Postgres.
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WEB_PORT="${WEB_PORT:-5173}"
API_PORT="${API_PORT:-8001}"
IC_PORT="${IC_PORT:-8000}"
TEXT_PORT="${TEXT_PORT:-8002}"
CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/0}"

BUILD_IC=0
FORCE=0

# --- colours (skipped when not a tty) --------------------------------------
if [ -t 1 ]; then
  C_IC=$'\033[36m'; C_API=$'\033[33m'; C_WEB=$'\033[35m'; C_TEXT=$'\033[34m'; C_WORKER=$'\033[92m'
  C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_IC=''; C_API=''; C_WEB=''; C_TEXT=''; C_WORKER=''; C_OK=''; C_ERR=''; C_DIM=''; C_RST=''
fi

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while getopts ":bfh" opt; do
  case "$opt" in
    b) BUILD_IC=1 ;;
    f) FORCE=1 ;;
    h) usage ;;
    *) echo "unknown flag: -$OPTARG (try -h)" >&2; exit 2 ;;
  esac
done

die() { echo "${C_ERR}error:${C_RST} $*" >&2; exit 1; }

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

free_port() {
  local p=$1 pids
  pids=$(lsof -nP -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null)
  [ -n "$pids" ] && kill $pids 2>/dev/null
}

# --- preflight: required venvs / deps --------------------------------------
IC_BIN="$ROOT/ic/api/.venv/bin/ic-api"
API_UVICORN="$ROOT/landing-page/scripts/.venv/bin/uvicorn"
TEXT_BIN="$ROOT/text-service/.venv/bin/uvicorn"
WORKER_BIN="$ROOT/landing-page/scripts/.venv/bin/celery"

[ -x "$IC_BIN" ]      || die "missing IC venv: $IC_BIN
  → cd ic/api && uv sync"
[ -x "$API_UVICORN" ] || die "missing landing-page venv: $API_UVICORN
  → cd landing-page/scripts && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
[ -x "$WORKER_BIN" ]  || die "missing celery in landing-page venv: $WORKER_BIN
  → cd landing-page/scripts && source .venv/bin/activate && pip install -r requirements.txt"
[ -d "$ROOT/landing-page/node_modules" ] || die "landing-page deps not installed
  → cd landing-page && npm install"
[ -f "$ROOT/landing-page/scripts/.env" ] || echo "${C_ERR}warning:${C_RST} landing-page/scripts/.env not found — backend may fail (DATABASE_URL/MOTHRA_SECRET)" >&2
[ -x "$TEXT_BIN" ] || die "missing text-finding venv: $TEXT_BIN
    → cd text-service && python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt \
      && pip install git+https://github.com/DDMAL/volpiano-display-utilities.git"

if command -v redis-cli >/dev/null 2>&1; then
  redis-cli -u "$CELERY_BROKER_URL" ping >/dev/null 2>&1 || echo "${C_ERR}warning:${C_RST} Redis (Celery broker) not reachable at $CELERY_BROKER_URL — predict/encode jobs will fail to enqueue" >&2
else
  echo "${C_ERR}warning:${C_RST} redis-cli not found — skipping Redis reachability check" >&2
fi

# --- preflight: ports ------------------------------------------------------
for pair in "web:$WEB_PORT" "backend:$API_PORT" "ic:$IC_PORT" "text:$TEXT_PORT"; do
  name=${pair%%:*}; port=${pair##*:}
  if port_busy "$port"; then
    if [ "$FORCE" -eq 1 ]; then
      echo "${C_DIM}freeing $name port $port…${C_RST}"; free_port "$port"; sleep 1
    else
      die "port $port ($name) is already in use. Stop it, or re-run with -f to free it.
$(lsof -nP -iTCP:"$port" -sTCP:LISTEN | sed 's/^/    /')"
    fi
  fi
done

# --- optional: rebuild the IC frontend bundle the iframe serves ------------
if [ "$BUILD_IC" -eq 1 ]; then
  echo "${C_DIM}building IC frontend → ic/api/src/ic_api/static …${C_RST}"
  npm --prefix "$ROOT/ic/frontend" run build || die "IC frontend build failed"
  rm -rf "$ROOT/ic/api/src/ic_api/static"
  cp -r "$ROOT/ic/frontend/dist" "$ROOT/ic/api/src/ic_api/static"
fi

# --- launch + teardown -----------------------------------------------------
PIDS=()

# Recursively kill a process and all its descendants (uvicorn --reload spawns
# a worker child; npm spawns vite — a plain kill on the parent can orphan them).
kill_tree() {
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  kill "$pid" 2>/dev/null
}

cleanup() {
  trap '' INT TERM EXIT          # disarm so this runs once
  echo; echo "${C_DIM}shutting down…${C_RST}"
  local pid
  for pid in "${PIDS[@]}"; do kill_tree "$pid"; done
  wait 2>/dev/null
  echo "${C_OK}all stopped.${C_RST}"
}
trap cleanup INT TERM EXIT

# start <label> <colour> <cmd...> — prefixes the child's output with a
# coloured tag; $! is the child's real PID (process-sub leaves it intact).
start() {
  local label=$1 colour=$2; shift 2
  "$@" > >(awk -v p="${colour}[${label}]${C_RST} " '{ print p $0; fflush() }') 2>&1 &
  PIDS+=($!)
}

echo "${C_OK}Mothra dev${C_RST}  web:${C_WEB}$WEB_PORT${C_RST}  backend:${C_API}$API_PORT${C_RST}  ic:${C_IC}$IC_PORT${C_RST}  text:${C_TEXT}$TEXT_PORT${C_RST}   ${C_DIM}(Ctrl-C to stop all)${C_RST}"

# Share the landing-page's Neon DATABASE_URL with the IC process so IC
# sessions are persisted with the mothra project (see ic/api db_store.py).
# Empty → IC uses its in-memory store and sessions vanish on restart.
IC_DB_URL=""
ENV_FILE="$ROOT/landing-page/scripts/.env"
if [ -f "$ENV_FILE" ]; then
  IC_DB_URL="$(sed -n 's/^[[:space:]]*DATABASE_URL[[:space:]]*=[[:space:]]*//p' "$ENV_FILE" | head -1)"
  IC_DB_URL="${IC_DB_URL%\"}"; IC_DB_URL="${IC_DB_URL#\"}"
  IC_DB_URL="${IC_DB_URL%\'}"; IC_DB_URL="${IC_DB_URL#\'}"
fi
[ -n "$IC_DB_URL" ] || echo "${C_DIM}note: no DATABASE_URL found — IC sessions won't persist across restarts${C_RST}"

start ic  "$C_IC"  env HOST=127.0.0.1 PORT="$IC_PORT" DATABASE_URL="$IC_DB_URL" "$IC_BIN"
start text "$C_TEXT" "$TEXT_BIN" main:app --app-dir "$ROOT/text-service" --port "$TEXT_PORT"
start backend "$C_API" "$API_UVICORN" main:app --app-dir "$ROOT/landing-page/scripts" --reload --port "$API_PORT"
start worker "$C_WORKER" env PYTHONPATH="$ROOT/landing-page/scripts" CELERY_BROKER_URL="$CELERY_BROKER_URL" "$WORKER_BIN" -A celery_app.celery_app worker --loglevel=info --pool=threads --concurrency=2
start web "$C_WEB" npm --prefix "$ROOT/landing-page" run dev -- --port "$WEB_PORT" --strictPort

echo "${C_DIM}→ open http://localhost:$WEB_PORT${C_RST}"
wait