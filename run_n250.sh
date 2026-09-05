#!/usr/bin/env bash
# Extend the two Qwen3-1.7B adapters from n=100 to n=250.
# Items 0-99 are already scored and are skipped automatically; only 100-249 run.
set -uo pipefail
cd /d/791
PY=C:/anaconda/envs/bengali-rq2/python.exe
export PYTHONIOENCODING=utf-8
LOG=logs/n250.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "qwen3-1.7b (olympiad-trained) -- before + after"
"$PY" -u eval_math.py --model qwen3-1.7b >> "$LOG" 2>&1
say "qwen3-1.7b-easy -- after only (before is the shared base model)"
"$PY" -u eval_math.py --model qwen3-1.7b-easy --condition after >> "$LOG" 2>&1
say "ALL DONE"
