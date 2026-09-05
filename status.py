# -*- coding: utf-8 -*-
"""One-shot project status. Run any time:  python status.py

Shows what is trained, what is evaluated, and what the GPU is doing, without
needing to know where any log lives.
"""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent

print("=" * 74)
print("TRAINED ADAPTERS")
print("=" * 74)
metas = sorted(glob.glob(str(ROOT / "adapters/*/training_meta.json")))
if not metas:
    print("  none yet")
for f in metas:
    m = json.load(open(f, encoding="utf-8"))
    t, r = m["training"], m["results"]
    flag = "ok " if t["max_seq_length"] == 4096 and m["n_truncated"] == 0 else "!! "
    print(f"  {flag}{m['model_label']:<22} loss={r['final_train_loss']:<8.4f} "
          f"len={t['max_seq_length']:<5} trunc={m['n_truncated']:<4} "
          f"{r['train_runtime_seconds']/60:>5.1f}min  {r['peak_vram_gb']:>5.2f}GB")

print()
print("=" * 74)
print("EVAL PROGRESS")
print("=" * 74)
preds = sorted(glob.glob(str(ROOT / "results_v2/preds_*.jsonl")))
if not preds:
    print("  not started (run: python eval_math.py --smoke)")
for f in preds:
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if not rows:
        continue
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    cap = sum(r.get("hit_cap", False) for r in rows) / n
    name = Path(f).stem.replace("preds_", "")
    print(f"  {name:<26} n={n:<5} acc={acc:<7.3f} cap={cap:.2f}")

print()
print("=" * 74)
print("RUNNING NOW")
print("=" * 74)
# train_live.log is the running job. Other files in logs/ are snapshots whose
# mtimes reflect when they were copied, so sorting by mtime picks the wrong one.
_live = ROOT / "logs" / "train_live.log"
logs = [_live] if _live.exists() else sorted(
    ROOT.glob("logs/*.log"), key=lambda p: p.stat().st_mtime)
if logs:
    newest = logs[-1]
    txt = newest.read_text(encoding="utf-8", errors="replace")
    steps = re.findall(r"\| (\d+)/(\d+)", txt)
    model = re.findall(r"^(.+?)\s+\(.+?\)\s+prompt_style=", txt, re.M)
    print(f"  log   : {newest.name}")
    if model:
        print(f"  model : {model[-1]}")
    if steps:
        cur, tot = steps[-1]
        pct = 100 * int(cur) / int(tot)
        print(f"  step  : {cur}/{tot}  ({pct:.0f}%)")
    loss = re.findall(r"'loss': ([\d.]+)", txt)
    if loss:
        print(f"  loss  : {loss[-1]}")
else:
    print("  no logs")

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    print(f"  gpu   : {out}")
except Exception as e:
    print(f"  gpu   : unavailable ({e})")
