#!/usr/bin/env bash
#
# Mothra local dev launcher — starts all three web-app servers together and
# tears them all down on Ctrl-C. See CLAUDE.md for the architecture.
#
#   :5173  landing-page frontend  (Vite)      — open this in the browser
#   :8001  landing-page backend   (uvicorn)   — /api/*; proxied from :5173
#   :8000  Interactive Classifier (ic-api)    — IC REST API + IC SPA (iframe)
#
# Usage:
#   ./dev.sh              start all three
#   ./dev.sh -b           rebuild the IC frontend bundle first (do this when
#                         the IC submodule's frontend changed; otherwise the
#                         iframe serves a stale build — see the binarize bug)
#   ./dev.sh -f           free the ports first if something is already on them
#   ./dev.sh -bf          both
#   ./dev.sh -h           help
#
# Ports are overridable: WEB_PORT=3000 ./dev.sh
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WEB_PORT="${WEB_PORT:-5173}"
API_PORT="${API_PORT:-8001}"
IC_PORT="${IC_PORT:-8000}"

BUILD_IC=0
FORCE=0

# --- colours (skipped when not a tty) --------------------------------------
if [ -t 1 ]; then
  C_IC=$'\033[36m'; C_API=$'\033[33m'; C_WEB=$'\033[35m'
  C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_IC=''; C_API=''; C_WEB=''; C_OK=''; C_ERR=''; C_DIM=''; C_RST=''
fi

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

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

[ -x "$IC_BIN" ]      || die "missing IC venv: $IC_BIN
  → cd ic/api && uv sync"
[ -x "$API_UVICORN" ] || die "missing landing-page venv: $API_UVICORN
  → cd landing-page/scripts && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
[ -d "$ROOT/landing-page/node_modules" ] || die "landing-page deps not installed
  → cd landing-page && npm install"
[ -f "$ROOT/landing-page/scripts/.env" ] || echo "${C_ERR}warning:${C_RST} landing-page/scripts/.env not found — backend may fail (DATABASE_URL/MOTHRA_SECRET)" >&2

# --- preflight: ports ------------------------------------------------------
for pair in "web:$WEB_PORT" "backend:$API_PORT" "ic:$IC_PORT"; do
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

echo "${C_OK}Mothra dev${C_RST}  web:${C_WEB}$WEB_PORT${C_RST}  backend:${C_API}$API_PORT${C_RST}  ic:${C_IC}$IC_PORT${C_RST}   ${C_DIM}(Ctrl-C to stop all)${C_RST}"

start ic  "$C_IC"  env HOST=127.0.0.1 PORT="$IC_PORT" "$IC_BIN"
start backend "$C_API" "$API_UVICORN" main:app --app-dir "$ROOT/landing-page/scripts" --reload --port "$API_PORT"
start web "$C_WEB" npm --prefix "$ROOT/landing-page" run dev -- --port "$WEB_PORT" --strictPort

echo "${C_DIM}→ open http://localhost:$WEB_PORT${C_RST}"
wait
