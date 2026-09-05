# -*- coding: utf-8 -*-
"""Follow any training or evaluation log.

    python track.py                       # newest log in logs/
    python track.py logs/eval_easy.log
    python track.py logs/eval_easy.log 10 # 10s refresh

Ctrl+C stops watching; the job keeps running.

Replaces the one-off watchers. It detects what kind of job a log belongs to --
LoRA training prints "step/total [elapsed<eta, Ns/it]", evaluation prints
"done/total  Ns/item  eta N min" -- and where an eval writes prediction files it
counts those instead, since the log only prints every N items and can go stale.
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
LOGDIR = ROOT / "logs"

argv = sys.argv[1:]
paths = [a for a in argv if not a.isdigit()]
nums = [a for a in argv if a.isdigit()]
INTERVAL = int(nums[0]) if nums else 5

if paths:
    LOG = Path(paths[0])
    if not LOG.is_absolute() and not LOG.exists():
        LOG = ROOT / paths[0]
else:
    logs = sorted(LOGDIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    LOG = logs[-1] if logs else LOGDIR / "eval_live.log"

TRAIN_RE = re.compile(r"(\d+)/(\d+) \[[\d:]+<([\d:]+),\s*([\d.]+)s?/?i?t?\]")
LOSS_RE = re.compile(r"'loss': ([\d.]+).*?'epoch': ([\d.]+)")
TRAINDONE_RE = re.compile(r"Finished in ([\d.]+) min \| loss ([\d.]+)")
EVAL_RE = re.compile(r"^\s+(\d+)/(\d+)\s+([\d.]+)s/item\s+eta (\d+) min", re.M)
EVAL_HDR_RE = re.compile(r"^(.+?)\s+\[(before|after)\]\s+(\d+) to do, (\d+) done", re.M)
MEM_ROW_RE = re.compile(r"^(before|after)\s+(\d+)/(\d+)\s+=\s+([\d.]+)", re.M)
VERDICT_RE = re.compile(r"VERDICT: (.+?)$", re.M)


def gpu():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, used, tot, p = [x.strip() for x in o.split(",")]
        return int(u), int(used), int(tot), float(p)
    except Exception:
        return None


def job():
    try:
        o = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=15).stdout
        for name in ("train_lora", "eval_math", "memorisation_check"):
            if name in o:
                return name
    except Exception:
        pass
    return None


def bar(d, t, w=26):
    if t <= 0:
        return " " * w
    k = int(min(1.0, d / t) * w)
    return "#" * k + "-" * (w - k)


def preds_count(model_key, cond):
    f = ROOT / "results_v2" / f"preds_{model_key}_{cond}.jsonl"
    if not f.exists():
        return None
    return sum(1 for ln in f.open(encoding="utf-8", errors="replace") if ln.strip())


KEY_BY_LABEL = {}
try:
    sys.path.insert(0, str(ROOT))
    from train_lora import MODELS as _M
    KEY_BY_LABEL = {v["label"]: k for k, v in _M.items()}
except Exception:
    pass

prev = prevt = None
try:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"  {LOG.name:<34}            {datetime.now():%H:%M:%S}")
        print("=" * 62)

        if not LOG.exists():
            print(f"  no such log: {LOG}")
        else:
            txt = LOG.read_text(encoding="utf-8", errors="replace")
            flat = txt.replace("\r", "\n")
            age = time.time() - LOG.stat().st_mtime

            hdr = EVAL_HDR_RE.findall(txt)
            tdone = TRAINDONE_RE.findall(txt)
            steps = TRAIN_RE.findall(flat)

            if hdr:                                    # ---- evaluation ----
                label, cond, todo, done = hdr[-1]
                key = KEY_BY_LABEL.get(label.strip())
                print(f"  eval    : {label.strip()}  [{cond}]")
                n = preds_count(key, cond) if key else None
                total = int(todo) + int(done)
                if n is not None:
                    print(f"  items   : {n}/{total} [{bar(n, total)}] "
                          f"{100*n/total:5.1f}%")
                    if prev is not None and n > prev:
                        per = (time.time() - prevt) / (n - prev)
                        left = timedelta(seconds=int(per * (total - n)))
                        print(f"  rate    : {per:.1f}s/item  ~{left} left "
                              f"(~{(datetime.now()+left):%H:%M})")
                    if prev is None or n != prev:
                        prev, prevt = n, time.time()
                ev = EVAL_RE.findall(txt)
                if ev and n is None:
                    d, t, r, e = ev[-1]
                    print(f"  items   : {d}/{t}  {r}s/item  eta {e} min")

            elif steps or tdone:                       # ---- training ----
                if tdone:
                    m, l = tdone[-1]
                    print(f"  train   : DONE  {m} min  final loss {l}")
                else:
                    cur, tot, eta, rate = steps[-1]
                    cur, tot = int(cur), int(tot)
                    print(f"  train   : {cur}/{tot} [{bar(cur, tot)}] "
                          f"{100*cur/tot:5.1f}%")
                    ls = LOSS_RE.findall(flat)
                    extra = f"  loss {ls[-1][0]} (ep {ls[-1][1]})" if ls else ""
                    print(f"  rate    : {rate}s/step   eta {eta}{extra}")

            rows = MEM_ROW_RE.findall(txt)             # ---- memcheck ----
            if rows:
                print()
                for c, k, n_, acc in rows:
                    print(f"  memcheck {c:<7} {k}/{n_} = {float(acc):.2f}")
            v = VERDICT_RE.findall(txt)
            if v:
                print(f"  VERDICT : {v[-1].strip()}")

            print()
            print(f"  log     : updated {age:.0f}s ago")

        running = job()
        print(f"  running : {running or 'NOTHING'}")
        g = gpu()
        if g:
            u, used, tot, p = g
            print(f"  gpu     : {u:3d}% util  {used}/{tot} MiB "
                  f"({100*used/tot:.0f}%)  {p:5.1f}W")
            if running and p < 70 and u > 80:
                print("  !! STALLED: high util at low power = VRAM spilled to host")

        if not running:
            print()
            print("  (no job running -- this log may be finished)")

        print()
        print(f"  refresh {INTERVAL}s   Ctrl+C to stop watching")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nstopped watching; the job keeps running")
