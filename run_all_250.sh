#!/usr/bin/env bash
# Bring every model to n=250 held-out items.
# Waits for the in-flight n250 job, then evaluates the rest. Items 0-99 (or
# 0-249 for TigerLLM's before/after, already at 500) are skipped automatically.
set -uo pipefail
cd /d/791
PY=C:/anaconda/envs/bengali-rq2/python.exe
export PYTHONIOENCODING=utf-8
LOG=logs/all250.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for the running n250 job to finish ..."
while pgrep -f "eval_math.py" > /dev/null 2>&1 || \
      powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -match 'eval_math'} | Measure-Object).Count" 2>/dev/null | grep -qv '^0'; do
  sleep 30
done
say "GPU free -- starting remaining models"

# fastest first, so an interruption costs the least
# titulm-1b (base) dropped: scores 0-2% with 100% degenerate non-terminating
# output, so n=100 -> 250 only tightens a bound on zero, at ~2h cost.
for m in qwen3-0.6b tigerllm-1b titulm-1b-it titulm-3b-it qwen3-4b; do
  say "=== $m ==="
  "$PY" -u eval_math.py --model "$m" >> "$LOG" 2>&1
  say "$m done"
done
say "ALL DONE -- every model at n=250"
