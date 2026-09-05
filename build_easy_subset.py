# -*- coding: utf-8 -*-
"""Build an EASY-difficulty LIMO-style training subset from Ganit.

    python build_easy_subset.py

Why this exists
---------------
The original curated set used LIMO's difficulty window, `1 <= correct_counts
<= 16` -- problems that Qwen3-32B solved between 1 and 16 times out of 32.
That is calibrated to a 32B model. Applied to 0.6-4B backbones it selected
almost entirely olympiad-tier problems (74.6% olympiad, 25.3% hard), and the
train-set memorisation check showed the result: after fine-tuning, Qwen3-1.7B
answered only 4/50 (8%) of the problems it had been TRAINED on, against 3/50
(6%) for the base model. The models never fit their own training data, so the
held-out null result cannot distinguish "elicitation does not transfer" from
"the training signal was unreachable".

This subset holds everything else constant -- same source dataset, same n, same
seed, same decontamination, same fields -- and changes only the difficulty band
to `correct_counts >= 28`, i.e. problems the 32B judge solved almost every time.
If a model trained on this DOES fit its training data, the difficulty-calibration
explanation is demonstrated rather than asserted.

Available pool (16,868 SFT rows), by Qwen3-32B pass rate over 32 attempts:
    0        never solved      5,645  (33.5%)
    1-16     LIMO window       1,025  ( 6.1%)   <- original set drawn from here
    17-27                        348  ( 2.1%)
    28-31                      1,288  ( 7.6%)   <- easy set drawn from
    32       always solved     8,562  (50.8%)   <- easy set drawn from
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_TARGET = 1000
MIN_CORRECT = 28          # of 32 attempts by Qwen3-32B
OUT = Path("curated/ganit_easy_n1000_c28-32_seed42.jsonl")
HELDOUT = Path("heldout_math_eval.jsonl")


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(t).lower())).strip()


def numeric_fingerprint(t):
    return tuple(re.findall(r"[\d০-৯]+", str(t))[:12])


print("loading Ganit SFT ...")
ds = load_dataset("dipta007/Ganit", "SFT")
ds = ds[list(ds.keys())[0]]
print(f"  pool: {len(ds)} rows")

rows = [r for r in ds if (r.get("correct_counts") or 0) >= MIN_CORRECT]
print(f"  correct_counts >= {MIN_CORRECT}: {len(rows)}")

# --- decontamination against the held-out evaluation set --------------------
held = [json.loads(l) for l in HELDOUT.open(encoding="utf-8") if l.strip()]
held_ids = {str(h.get("id")) for h in held}
held_text = {norm(h["problem"]) for h in held}
held_fp = {numeric_fingerprint(h["problem"]) for h in held}

kept, drop_id, drop_text, drop_fp = [], 0, 0, 0
for r in rows:
    if str(r.get("id")) in held_ids:
        drop_id += 1;   continue
    if norm(r.get("problem", "")) in held_text:
        drop_text += 1; continue
    if numeric_fingerprint(r.get("problem", "")) in held_fp:
        drop_fp += 1;   continue
    kept.append(r)
print(f"  decontamination: -{drop_id} id, -{drop_text} exact text, "
      f"-{drop_fp} numeric fingerprint  -> {len(kept)}")

# --- internal dedup ---------------------------------------------------------
seen, deduped = set(), []
for r in kept:
    k = norm(r.get("problem", ""))
    if k and k not in seen:
        seen.add(k)
        deduped.append(r)
print(f"  internal dedup: -{len(kept) - len(deduped)} -> {len(deduped)}")

# --- sample -----------------------------------------------------------------
rng = random.Random(SEED)
sel = rng.sample(deduped, min(N_TARGET, len(deduped)))
print(f"  sampled {len(sel)} with seed {SEED}")

# --- must have a usable target and answer -----------------------------------
final = []
for r in sel:
    tgt = ""
    for m in reversed(r.get("messages") or []):
        if m.get("role") == "assistant":
            tgt = (m.get("content") or "").strip()
            break
    if r.get("problem") and tgt and str(r.get("bengali_solution") or "").strip():
        final.append(r)
print(f"  usable (problem + trace + answer): {len(final)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps({
            "id": r.get("id"),
            "problem": r["problem"],
            "bengali_solution": r.get("bengali_solution"),
            "messages": r.get("messages"),
            "difficulty": r.get("difficulty"),
            "correct_counts": r.get("correct_counts"),
            "source_name": r.get("source_name"),
        }, ensure_ascii=False) + "\n")

print(f"\nwrote {OUT}  ({len(final)} rows)")
print("\nby difficulty tag:")
for k, v in Counter(r.get("difficulty") for r in final).most_common():
    print(f"  {str(k):<12} {v:>5}  ({100*v/len(final):5.1f}%)")
print("\nby source:")
for k, v in Counter(r.get("source_name") for r in final).most_common():
    print(f"  {str(k):<26} {v:>5}  ({100*v/len(final):5.1f}%)")
cc = [r.get("correct_counts") for r in final]
print(f"\ncorrect_counts: min {min(cc)}  max {max(cc)}  "
      f"mean {sum(cc)/len(cc):.1f}")
