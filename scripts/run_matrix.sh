#!/usr/bin/env bash
# Convert and benchmark a list of configurations, one at a time, continuing after a failure.
# Usage: scripts/run_matrix.sh WORK_DIR RESULTS_DIR "DEVICE LABEL" "model width height window precision" ...
# Optional environment:
#   DP2_PRECHECK    command run before each job; a non-zero exit stops the matrix
#   DP2_WATCH       space-separated process names recorded as other GPU clients
#   DP2_CONVERT_ARGS  extra arguments for convert.py
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$1"; RESULTS_DIR="$2"; DEVICE="$3"; shift 3
PY="$WORK_DIR/venv/bin/python"
export TORCH_HOME="${TORCH_HOME:-$WORK_DIR/cache/torch}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
ENGINES="$WORK_DIR/engines"
mkdir -p "$RESULTS_DIR" "$ENGINES"
cd "$REPO_DIR" || exit 1

WATCH_ARGS=()
for name in ${DP2_WATCH:-}; do WATCH_ARGS+=(--watch-process "$name"); done

for job in "$@"; do
  read -r model width height window precision <<< "$job"
  stem="${model}_${width}x${height}_w${window}_${precision}"
  if [ -n "${DP2_PRECHECK:-}" ] && ! bash -c "$DP2_PRECHECK"; then
    echo "[$stem] precheck failed; stopping the matrix"
    exit 4
  fi
  echo "[$stem] start $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086
  "$PY" convert.py --model "$model" --width "$width" --height "$height" --window "$window" \
    --precision "$precision" --upstream "$WORK_DIR/ml-depth-pro-2" --out "$ENGINES" ${DP2_CONVERT_ARGS:-}
  code=$?
  [ -f "$ENGINES/$stem.report.json" ] && cp "$ENGINES/$stem.report.json" "$RESULTS_DIR/$stem.conversion.json"
  if [ "$code" -eq 0 ]; then
    "$PY" benchmark.py --engine "$ENGINES/$stem.engine" --device-label "$DEVICE" \
      --upstream "$WORK_DIR/ml-depth-pro-2" --output "$RESULTS_DIR/$stem.json" "${WATCH_ARGS[@]}" \
      || echo "[$stem] benchmark failed"
  else
    echo "[$stem] conversion exit status $code"
  fi
done
echo "MATRIX DONE"
