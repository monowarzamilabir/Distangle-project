# -*- coding: utf-8 -*-
"""Live project monitor.

    python watch.py           # refresh every 10s
    python watch.py 5         # every 5s

Ctrl+C stops watching; whatever is running keeps running.

Reads the ACTUAL artifacts on disk -- prediction files, adapter metadata, the
running process list -- rather than scraping progress bars out of a log. Log
scraping was the previous design and it repeatedly showed stale numbers: log
lines print only every N items, and a log that ever held output from two
processes keeps reporting whichever wrote last.

Stall detection is based on whether the output files are actually growing, not
on GPU utilisation. On this machine a saturated card reports 100% utilisation
while spilling to host memory and making no progress at all -- the giveaway is
power draw collapsing (~145W -> ~45W) with the item count frozen.
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
RESULTS = ROOT / "results_v2"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10

# Read the live config instead of hardcoding it. Hardcoded expectations went
# stale the moment n_items changed, which is exactly the staleness this rewrite
# was meant to remove.
sys.path.insert(0, str(ROOT))
try:
    from eval_math import EVAL_META, ORDER
    from train_lora import MODELS as _M
    MODELS = [(k, _M[k]["label"], _M[k]["adapter"],
               EVAL_META[k].get("n_items", 500)) for k in ORDER]
except Exception as e:                                  # pragma: no cover
    print(f"could not import config ({e}); falling back to defaults")
    MODELS = [("tigerllm-1b", "TigerLLM-1B", "tigerllm-1b", 500)]


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, used, tot, p, t = [x.strip() for x in out.split(",")]
        return int(u), int(used), int(tot), float(p), int(t)
    except Exception:
        return None


def running():
    """What is actually executing: (kind, pid) or (None, None)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            if "eval_math" in line:
                return "eval", line.split(",")[0].strip('"')
            if "train_lora" in line:
                return "train", line.split(",")[0].strip('"')
    except Exception:
        pass
    return None, None


def count(path):
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def bar(done, total, width=22):
    if total <= 0:
        return " " * width
    f = min(1.0, done / total)
    k = int(f * width)
    return "#" * k + "-" * (width - k)


prev_total = prev_time = stall_since = None

try:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        now = datetime.now()
        kind, pid = running()

        print(f"  Bengali RQ1                                   {now:%H:%M:%S}")
        print("=" * 72)

        trained = sum(1 for _, _, a, _ in MODELS
                      if (ROOT / "adapters" / a / "training_meta.json").exists())
        print(f"  adapters trained : {trained}/{len(MODELS)}")

        print()
        print(f"  {'model':<20} {'before':>11} {'after':>11}   progress")
        print("  " + "-" * 68)
        total_done = total_want = 0
        for key, label, _, want in MODELS:
            b = count(RESULTS / f"preds_{key}_before.jsonl")
            a = count(RESULTS / f"preds_{key}_after.jsonl")
            total_done += b + a
            total_want += 2 * want
            mark = " OK" if (b >= want and a >= want) else "   "
            print(f"  {label:<20} {b:>6}/{want:<4} {a:>6}/{want:<4} "
                  f"[{bar(b + a, 2 * want)}]{mark}")
        print("  " + "-" * 68)
        pct = 100 * total_done / total_want if total_want else 0
        print(f"  {'TOTAL':<20} {total_done:>5}/{total_want:<5} item-passes"
              f"            {pct:5.1f}%")

        print()
        if kind is None:
            print("  running : NOTHING  (no eval_math / train_lora process)")
        else:
            print(f"  running : {kind}  (pid {pid})")

        if prev_total is not None and total_done > prev_total:
            per = (time.time() - prev_time) / (total_done - prev_total)
            left = timedelta(seconds=int(per * (total_want - total_done)))
            print(f"  rate    : {per:.1f}s/item   ~{left} left "
                  f"(~{(now + left):%H:%M})")
            stall_since = None
        elif prev_total is not None and total_done == prev_total and kind:
            stall_since = stall_since or time.time()
            print(f"  rate    : NO new items for "
                  f"{(time.time() - stall_since)/60:.1f} min")
        if prev_total is None or total_done != prev_total:
            prev_total, prev_time = total_done, time.time()

        g = gpu()
        if g:
            u, used, tot, p, t = g
            print(f"  gpu     : {u:3d}% util  {used}/{tot} MiB "
                  f"({100*used/tot:.0f}%)  {p:5.1f}W  {t}C")
            if kind and p < 70 and u > 80:
                print()
                print("  !! STALLED: 100% util at low power = VRAM spilled to host")
                print("     memory. Kill and restart; it resumes from disk:")
                print("     python eval_math.py --all")
            elif kind and 100 * used / tot > 95:
                print("  !  VRAM >95% -- spill risk; lower this model's batch size")

        print()
        print(f"  refresh {INTERVAL}s   Ctrl+C stops watching (jobs keep running)")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nstopped watching; jobs keep running")
