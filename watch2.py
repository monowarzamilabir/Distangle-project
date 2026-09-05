# -*- coding: utf-8 -*-
"""Progress monitor for the easy-subset experiment.

    python watch2.py          # refresh every 5s
    python watch2.py 10

Follows logs/easy_experiment.log, which chains three phases in one file:
waiting for the main eval to reach a safe stopping point, then LoRA training on
the easy subset, then the train-set memorisation check.

Separate from watch.py because neither phase writes prediction files, so the
"count artifacts on disk" approach watch.py uses has nothing to count here.
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
LOG = ROOT / "logs" / "easy_experiment.log"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5

PHASE_RE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (.+)$", re.M)
STEP_RE = re.compile(r"(\d+)/(\d+) \[[\d:]+<[\d:]+,\s*([\d.]+)s?/?i?t?\]")
LOSS_RE = re.compile(r"'loss': ([\d.]+).*?'epoch': ([\d.]+)")
TRAINDONE_RE = re.compile(r"Finished in ([\d.]+) min \| loss ([\d.]+)")
MEM_PROG_RE = re.compile(r"\[(before|after)\]\s+(\d+)/(\d+)")
MEM_ROW_RE = re.compile(r"^(before|after)\s+(\d+)/(\d+)\s+=\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
                        re.M)
VERDICT_RE = re.compile(r"VERDICT: (.+?)$", re.M)


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, used, tot, p = [x.strip() for x in out.split(",")]
        return int(u), int(used), int(tot), float(p)
    except Exception:
        return None


def procs():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=15).stdout
        return ("train_lora" in out, "memorisation_check" in out)
    except Exception:
        return (False, False)


def bar(done, total, width=26):
    if total <= 0:
        return " " * width
    k = int(min(1.0, done / total) * width)
    return "#" * k + "-" * (width - k)


start = time.time()
try:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"  easy-subset experiment                     {datetime.now():%H:%M:%S}")
        print("=" * 64)

        txt = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
        flat = txt.replace("\r", "\n")
        training, memchecking = procs()

        phases = PHASE_RE.findall(txt)
        if phases:
            print(f"  phase   : {phases[-1][1]}   (since {phases[-1][0]})")
        else:
            print("  phase   : starting")

        # ---- training ------------------------------------------------------
        tdone = TRAINDONE_RE.findall(txt)
        steps = STEP_RE.findall(flat)
        losses = LOSS_RE.findall(flat)
        print()
        if tdone:
            mins, loss = tdone[-1]
            print(f"  1 train : DONE   {mins} min   final loss {loss}")
        elif steps:
            cur, tot, rate = steps[-1]
            cur, tot = int(cur), int(tot)
            eta = timedelta(seconds=int(float(rate) * (tot - cur)))
            l = f"   loss {losses[-1][0]} (ep {losses[-1][1]})" if losses else ""
            print(f"  1 train : {cur:>3}/{tot:<3} [{bar(cur, tot)}] "
                  f"{100*cur/tot:5.1f}%")
            print(f"            {rate}s/step   eta {eta}{l}")
        else:
            print("  1 train : waiting")

        # ---- memorisation check --------------------------------------------
        rows = MEM_ROW_RE.findall(txt)
        prog = MEM_PROG_RE.findall(flat)
        print()
        print("  2 memcheck (accuracy on problems it was TRAINED on)")
        for cond in ("before", "after"):
            fin = [r for r in rows if r[0] == cond]
            if fin:
                _, k, n, acc, cap, tk = fin[-1]
                print(f"      {cond:<7} DONE  {k}/{n} = {float(acc):.2f}   "
                      f"cap {float(cap):.2f}  {tk} tok")
            else:
                cur = [p for p in prog if p[0] == cond]
                if cur:
                    _, d, t = cur[-1]
                    print(f"      {cond:<7} {d:>3}/{t:<3} "
                          f"[{bar(int(d), int(t))}] {100*int(d)/int(t):5.1f}%")
                else:
                    print(f"      {cond:<7} waiting")

        v = VERDICT_RE.findall(txt)
        if v:
            print()
            print(f"  VERDICT : {v[-1].strip()}")

        # ---- baseline for comparison ---------------------------------------
        print()
        print("  reference (olympiad-trained adapter, same backbone):")
        print("      before  3/50 = 0.06     after  4/50 = 0.08")

        print()
        alive = training or memchecking
        print(f"  running : {'train_lora' if training else 'memorisation_check' if memchecking else 'NOTHING'}"
              f"   elapsed {timedelta(seconds=int(time.time()-start))}")
        g = gpu()
        if g:
            u, used, tot, p = g
            print(f"  gpu     : {u:3d}% util  {used}/{tot} MiB "
                  f"({100*used/tot:.0f}%)  {p:5.1f}W")
            if alive and p < 70 and u > 80:
                print("  !! STALLED: high util at low power = VRAM spilled to host")

        if "ALL DONE" in txt:
            print()
            print("  === experiment complete ===")
            print(f"  full log: {LOG}")
            break

        print()
        print(f"  refresh {INTERVAL}s   Ctrl+C to stop watching")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nstopped watching; the experiment keeps running")
