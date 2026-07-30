#!/usr/bin/env bash
#
# staff-finding end-to-end pipeline test — detect stafflines with the
# bundled YOLO stave detector, fit/group them into staves via one of 5
# interchangeable methods, and optionally evaluate against ground truth.
#
# Usage:
#   ./test_model.sh <input image|dir> <output dir> [flags]
#
# Flags:
#   --method {main,gp_centerlines,dp_tracing,implicit_neural,periodicity,all}
#                        centerline-fitting method to run (default: main)
#   --weights PATH       stave-detector .pt (default: staff-finding/models/stave_detector_fulldata.pt)
#   --conf FLOAT         YOLO confidence threshold (default: 0.25)
#   --device STR         forwarded only to the detection step (cuda/cpu/cuda:N)
#   --gt-dir PATH        ground-truth YOLO .txt directory; enables the eval step
#   --gt-class INT       YOLO class id for stafflines in --gt-dir (default: 2 —
#                        this varies by GT source in this repo; verify with
#                        `awk '{print $1}' <gt-dir>/<any-file>.txt | sort -u`)
#   --python PATH        skip env auto-resolution, use this interpreter directly
#   -h, --help           this help
#
# Note: same-named images in different subdirectories of the input dir will
# collide (results are keyed by filename stem) — a pre-existing limitation
# shared with scripts/run_inference.py, not specific to this script.
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- colours (skipped when not a tty) --------------------------------------
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_OK=''; C_ERR=''; C_DIM=''; C_RST=''
fi

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
die() { echo "${C_ERR}error:${C_RST} $*" >&2; exit 1; }

# --- defaults ----------------------------------------------------------------
METHOD="main"
WEIGHTS="$ROOT/models/stave_detector_fulldata.pt"
CONF="0.25"
DEVICE=""
GT_DIR=""
GT_CLASS="2"
PYTHON_OVERRIDE=""

# --- manual long-form flag parsing (not getopts — need long flags) ---------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)   METHOD="$2"; shift 2 ;;
    --weights)  WEIGHTS="$2"; shift 2 ;;
    --conf)     CONF="$2"; shift 2 ;;
    --device)   DEVICE="$2"; shift 2 ;;
    --gt-dir)   GT_DIR="$2"; shift 2 ;;
    --gt-class) GT_CLASS="$2"; shift 2 ;;
    --python)   PYTHON_OVERRIDE="$2"; shift 2 ;;
    -h|--help)  usage ;;
    --) shift; break ;;
    -*) die "unknown flag: $1 (try -h)" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"
[ $# -eq 2 ] || die "usage: $0 <input image|dir> <output dir> [flags]  (try -h)"
INPUT="$1"; OUTPUT="$2"

[ -e "$INPUT" ]   || die "input not found: $INPUT"
[ -f "$WEIGHTS" ] || die "weights not found: $WEIGHTS"
mkdir -p "$OUTPUT" || die "cannot create output dir: $OUTPUT"

case "$METHOD" in
  main|gp_centerlines|dp_tracing|implicit_neural|periodicity|all) ;;
  *) die "unknown --method: $METHOD (expected main|gp_centerlines|dp_tracing|implicit_neural|periodicity|all)" ;;
esac

# --- python interpreter resolution ------------------------------------------
resolve_python() {
  local candidates=() tried=() c base

  [ -n "$PYTHON_OVERRIDE" ] && candidates+=("$PYTHON_OVERRIDE")
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null)"
    if [ -n "$base" ]; then
      candidates+=("$base/envs/mothrav8/bin/python" "$base/envs/annorlad/bin/python")
    fi
  fi
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")

  for c in "${candidates[@]}"; do
    [ -x "$c" ] || continue
    tried+=("$c")
    if "$c" -c "import ultralytics, torch, cv2, numpy, scipy, sklearn, matplotlib" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done

  die "no interpreter has the full package set (ultralytics, torch, cv2, numpy, scipy, sklearn, matplotlib). Tried:
$(printf '  - %s\n' "${tried[@]}")
Fix: conda run -n mothrav8 pip install scikit-learn   (or use -n annorlad, which already has everything, or pass --python)"
}
PYTHON="$(resolve_python)" || exit 1
echo "${C_DIM}using python: $PYTHON${C_RST}"

export MPLBACKEND=Agg  # defensive: no backend is configured anywhere in the
                        # pipeline's diagnostic-plotting code; avoids a class
                        # of flaky-when-headless matplotlib failures.

# --- method dispatch table ---------------------------------------------------
# A case statement rather than a lookup table: bash 3.2 (this machine's
# stock /bin/bash) has no associative arrays, and each method's flags
# genuinely differ enough (see staff-finding/experiments/README.md) that a
# generic flag-forwarding scheme across methods isn't the right shape anyway.
# All methods get --staffline-class 0, matching the bundled single-class
# stave detector's native output class (tied to that specific model — if
# --weights ever points at a different, multi-class model, this hardcode
# would need revisiting).
dispatch() {  # $1=method $2=page $3=yolo_txt $4=out_dir
  case "$1" in
    main)
      "$PYTHON" "$ROOT/scripts/run_page.py" \
        --page "$2" --yolo "$3" --output "$4" --staffline-class 0 --no-bgr ;;
    gp_centerlines)
      "$PYTHON" "$ROOT/experiments/gp_centerlines/run_gp_page.py" \
        --page "$2" --yolo "$3" --output "$4" --staffline-class 0 ;;
    dp_tracing)
      "$PYTHON" "$ROOT/experiments/dp_tracing/run_dp_page.py" \
        --page "$2" --yolo "$3" --output "$4" --staffline-class 0 ;;
    implicit_neural)
      "$PYTHON" "$ROOT/experiments/implicit_neural/run_implicit_neural_page.py" \
        --page "$2" --yolo "$3" --output "$4" --staffline-class 0 ;;
    periodicity)
      "$PYTHON" "$ROOT/experiments/periodicity/run_periodicity_page.py" \
        --page "$2" --yolo "$3" --output "$4" --staffline-class 0 ;;
  esac
}

if [ "$METHOD" = "all" ]; then
  METHODS=(main gp_centerlines dp_tracing implicit_neural periodicity)
else
  METHODS=("$METHOD")
fi

# --- Step 1: detect once for the whole input --------------------------------
YOLO_DIR="$OUTPUT/yolo_txt"
mkdir -p "$YOLO_DIR"

if [ -f "$INPUT" ]; then
  DETECT_ARG=(--image "$INPUT")
else
  DETECT_ARG=(--images-dir "$INPUT")
fi
DEVICE_ARG=()
[ -n "$DEVICE" ] && DEVICE_ARG=(--device "$DEVICE")

echo "${C_DIM}--- detecting stafflines ---${C_RST}"
"$PYTHON" "$ROOT/scripts/detect_stafflines.py" "${DETECT_ARG[@]}" \
  --weights "$WEIGHTS" --output "$YOLO_DIR" --conf "$CONF" "${DEVICE_ARG[@]}" \
  || die "detect_stafflines.py failed"

IMAGES=()
while IFS= read -r line; do
  [ -n "$line" ] && IMAGES+=("$line")
done < "$YOLO_DIR/manifest.txt"
[ ${#IMAGES[@]} -gt 0 ] || die "no images found in $INPUT"

# --- Step 2: per-page x per-method dispatch, eval manifest, GT sanity check -
MANIFEST="$OUTPUT/eval_manifest.csv"
GT_LABEL=""
if [ -n "$GT_DIR" ]; then
  echo "page_name,image,gt_txt,gt_source,pred_json,variant" > "$MANIFEST"
  GT_LABEL="$(basename "$GT_DIR")"
fi
GT_CHECKED_COUNT=0
GT_ZERO_MATCH_COUNT=0

N_OK=0
N_SKIPPED=0
N_FAILED=0

echo "${C_DIM}--- running pipeline (method: $METHOD) on ${#IMAGES[@]} page(s) ---${C_RST}"
for img in "${IMAGES[@]}"; do
  stem="$(basename "$img")"; stem="${stem%.*}"
  yolo_txt="$YOLO_DIR/$stem.txt"

  if [ ! -s "$yolo_txt" ]; then
    echo "warning: no detections for $stem (missing/empty $yolo_txt); skipping pipeline step" >&2
    N_SKIPPED=$((N_SKIPPED + 1))
    continue
  fi

  first_method="${METHODS[0]}"
  for method in "${METHODS[@]}"; do
    # Nested by <method> too (not just <stem>): each method's script hardcodes
    # its own output subdirectory suffix (e.g. gp_centerlines -> <stem>_gp/,
    # run_page.py -> <stem>_no_bgr/), so once --method all has run 2+ methods
    # for the same page, a `find` scoped only to <stem>/ would return multiple
    # ambiguous *_stafflines.json matches.
    method_out="$OUTPUT/pipeline/$stem/$method"
    mkdir -p "$method_out"

    if dispatch "$method" "$img" "$yolo_txt" "$method_out"; then
      N_OK=$((N_OK + 1))
    else
      echo "warning: $method failed for $stem; continuing" >&2
      N_FAILED=$((N_FAILED + 1))
      continue
    fi

    pred_json="$(find "$method_out" -name '*_stafflines.json' | head -1)"
    [ -n "$pred_json" ] || { echo "warning: $method produced no *_stafflines.json for $stem" >&2; continue; }

    if [ -n "$GT_DIR" ]; then
      gt_txt="$GT_DIR/$stem.txt"
      if [ -f "$gt_txt" ]; then
        echo "$stem,$img,$gt_txt,$GT_LABEL,$pred_json,$method" >> "$MANIFEST"

        # Only tally the GT file's own class-id content once per page (it
        # doesn't depend on which method ran).
        if [ "$method" = "$first_method" ]; then
          GT_CHECKED_COUNT=$((GT_CHECKED_COUNT + 1))
          n_match=$(awk -v c="$GT_CLASS" '$1==c' "$gt_txt" | wc -l | tr -d ' ')
          [ "$n_match" -eq 0 ] && GT_ZERO_MATCH_COUNT=$((GT_ZERO_MATCH_COUNT + 1))
        fi
      fi
    fi
  done
done

# --- GT class-id sanity-check report (fires before eval_batch.py runs) ------
if [ "$GT_CHECKED_COUNT" -gt 0 ]; then
  if [ "$GT_ZERO_MATCH_COUNT" -eq "$GT_CHECKED_COUNT" ]; then
    echo "${C_ERR}WARNING:${C_RST} --gt-class $GT_CLASS matched ZERO ground-truth lines in ALL $GT_CHECKED_COUNT GT file(s) under $GT_DIR." >&2
    echo "  This almost always means --gt-class is wrong for this GT source (the class-id" >&2
    echo "  convention genuinely differs across GT directories in this repo). Check with:" >&2
    echo "    awk '{print \$1}' $GT_DIR/<any_file>.txt | sort -u" >&2
    echo "  then re-run with the correct --gt-class. Proceeding anyway; expect n_gt=0 everywhere." >&2
  elif [ "$GT_ZERO_MATCH_COUNT" -gt 0 ]; then
    echo "warning: --gt-class $GT_CLASS matched zero lines in $GT_ZERO_MATCH_COUNT of $GT_CHECKED_COUNT GT file(s) — some pages may use a different class-id or have no stafflines." >&2
  fi
fi

# --- Step 3: batch eval ------------------------------------------------------
EVAL_RAN=0
if [ -n "$GT_DIR" ] && [ -f "$MANIFEST" ] && [ "$(wc -l < "$MANIFEST" | tr -d ' ')" -gt 1 ]; then
  echo "${C_DIM}--- evaluating against $GT_DIR ---${C_RST}"
  "$PYTHON" "$ROOT/scripts/eval_batch.py" --manifest "$MANIFEST" --output "$OUTPUT/eval_results.csv" \
    --staffline-class "$GT_CLASS" --summarize
  EVAL_RAN=1
fi

# --- final summary ------------------------------------------------------------
echo
echo "${C_OK}done.${C_RST} pages: ${#IMAGES[@]}  dispatched-ok: $N_OK  failed: $N_FAILED  skipped(no detections): $N_SKIPPED"
echo "  detections:  $YOLO_DIR"
echo "  pipeline:    $OUTPUT/pipeline/"
if [ "$EVAL_RAN" -eq 1 ]; then
  echo "  eval:        $OUTPUT/eval_results.csv"
elif [ -n "$GT_DIR" ]; then
  echo "  eval:        skipped (no page had both a detection and a matching GT file)"
fi
