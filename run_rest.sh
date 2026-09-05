#!/usr/bin/env bash
# Finish the n=250 extension. Everything resumes: completed item ids are
# skipped, so re-running this after any interruption is always safe.
#
# Excluded on purpose:
#   tigerllm-1b  already at 500 items
#   titulm-1b    base variant dropped (0-2% accuracy, degenerate output,
#                ~2h to go from n=100 to n=250 for no new information)
set -uo pipefail
cd /d/791
PY=C:/anaconda/envs/bengali-rq2/python.exe
export PYTHONIOENCODING=utf-8
LOG=logs/rest250.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== qwen3-1.7b-easy (after only; before is the shared base) ==="
"$PY" -u eval_math.py --model qwen3-1.7b-easy --condition after >> "$LOG" 2>&1
say "qwen3-1.7b-easy done"

for m in qwen3-0.6b titulm-1b-it titulm-3b-it qwen3-4b; do
  say "=== $m ==="
  "$PY" -u eval_math.py --model "$m" >> "$LOG" 2>&1
  say "$m done"
done
say "ALL DONE"
