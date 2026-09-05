#!/usr/bin/env bash
# Easy-subset experiment, chained so it runs unattended.
#
#   1. wait for TituLM-3B-Instruct to finish its 100 'after' items
#   2. stop the eval BEFORE it starts Qwen3-4B
#   3. train Qwen3-1.7B on the easy subset -> adapters/qwen3-1.7b-easy
#   4. re-run the train-set memorisation check against the easy subset
#   5. leave the GPU free; the main eval is resumed manually
#
# Everything is resumable: the eval skips completed items on restart.
set -uo pipefail
cd /d/791

PY=C:/anaconda/envs/bengali-rq2/python.exe
export PYTHONIOENCODING=utf-8
LOG=logs/easy_experiment.log
: > "$LOG"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for titulm-3b-it [after] to reach 100 ..."
while true; do
  n=$(wc -l < results_v2/preds_titulm-3b-it_after.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 100 ] && break
  sleep 20
done
say "titulm-3b-it complete ($n items)"

say "stopping eval before Qwen3-4B starts"
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*eval_math*'} | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
sleep 8

say "training Qwen3-1.7B on the EASY subset"
"$PY" -u train_lora.py --model qwen3-1.7b-easy \
      --data curated/ganit_easy_n1000_c28-32_seed42.jsonl >> "$LOG" 2>&1
say "training done"

say "train-set memorisation check (easy subset)"
"$PY" -u memorisation_check.py --model qwen3-1.7b-easy --n 50 \
      --data curated/ganit_easy_n1000_c28-32_seed42.jsonl >> "$LOG" 2>&1
say "memorisation check done"

say "ALL DONE -- main eval NOT restarted (Qwen3-4B still pending)"
