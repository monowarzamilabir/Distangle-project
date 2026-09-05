#!/usr/bin/env bash
# Last model in the n=250 extension.
#
# The first attempt died silently at batch 2 about 8 items in: GPU released,
# no traceback, log ended mid-run. eval_math.py now pins this model to
# batch 1. Resumes from completed item ids, so re-running after any
# interruption is safe and never repeats work.
set -uo pipefail
cd /d/791
PY=C:/anaconda/envs/bengali-rq2/python.exe
export PYTHONIOENCODING=utf-8
LOG=logs/qwen4b_250.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== qwen3-4b before (batch 1 after the batch-2 OOM) ==="
"$PY" -u eval_math.py --model qwen3-4b --condition before >> "$LOG" 2>&1
say "qwen3-4b before done (exit $?)"

say "=== qwen3-4b after ==="
"$PY" -u eval_math.py --model qwen3-4b --condition after >> "$LOG" 2>&1
say "qwen3-4b after done (exit $?)"

say "=== ALL EVALUATION COMPLETE ==="
