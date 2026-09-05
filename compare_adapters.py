# -*- coding: utf-8 -*-
"""Compare two adapters trained on the same backbone against their shared base.

    python compare_adapters.py qwen3-1.7b qwen3-1.7b-easy

Both adapters were trained from the same base model, with the same LoRA config,
seed, epochs and prompt, and are evaluated on the same held-out items. The only
difference is the training data's difficulty band:

    qwen3-1.7b       correct_counts 1-16/32   (mean 3.3)   74.6% olympiad
    qwen3-1.7b-easy  correct_counts 28-32/32  (mean 31.8)  100% easy

The `before` condition is the base model and is therefore identical for both, so
it is read once from the first adapter's file rather than recomputed.

Reports exact McNemar for each paired contrast. With accuracy near 10-20% and
n=100, differences of two or three items are noise -- the test says which
comparisons can actually carry a claim.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
RESULTS = ROOT / "results_v2"
TIERS = ["easy", "medium", "hard", "olympiad"]


def load(key, cond):
    f = RESULTS / f"preds_{key}_{cond}.jsonl"
    if not f.exists():
        return None
    return {str(json.loads(l)["id"]): json.loads(l)
            for l in f.open(encoding="utf-8") if l.strip()}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(a, b, ids):
    """Exact McNemar on the discordant pairs between two result dicts."""
    g = sum(1 for i in ids if not a[i]["correct"] and b[i]["correct"])
    l = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    m = g + l
    if m == 0:
        return g, l, 1.0
    k = min(g, l)
    return g, l, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


ap = argparse.ArgumentParser()
ap.add_argument("a", help="first adapter key (also supplies the shared 'before')")
ap.add_argument("b", help="second adapter key")
args = ap.parse_args()

base = load(args.a, "before")
A = load(args.a, "after")
B = load(args.b, "after")
missing = [n for n, d in [(f"{args.a}_before", base), (f"{args.a}_after", A),
                          (f"{args.b}_after", B)] if not d]
if missing:
    raise SystemExit(f"missing predictions: {', '.join(missing)}")

ids = sorted(set(base) & set(A) & set(B))
if not ids:
    raise SystemExit("no overlapping items")

label = {"base": A[ids[0]].get("model_label", args.a).replace("-easy", ""),
         "a": args.a, "b": args.b}

print()
print("=" * 72)
print(f"  {args.a}  vs  {args.b}      n = {len(ids)} shared held-out items")
print("=" * 72)

sets = [("base (no LoRA)", base), (f"{args.a} (olympiad-trained)", A),
        (f"{args.b} (easy-trained)", B)]
print(f"  {'condition':<30} {'correct':>9}  {'acc':>7}   95% CI")
print("  " + "-" * 66)
for name, d in sets:
    k = sum(d[i]["correct"] for i in ids)
    lo, hi = wilson(k, len(ids))
    print(f"  {name:<30} {k:>4}/{len(ids):<4} {k/len(ids):>7.3f}   "
          f"[{lo:.3f}, {hi:.3f}]")

print()
print("  paired contrasts (exact McNemar)")
print("  " + "-" * 66)
for n1, d1, n2, d2 in [("base", base, args.a, A),
                       ("base", base, args.b, B),
                       (args.a, A, args.b, B)]:
    g, l, p = mcnemar(d1, d2, ids)
    verdict = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"  {n1:<16} -> {n2:<18} +{g} / -{l}   p = {p:.4f}  {verdict}")

print()
print(f"  {'accuracy by difficulty':<24}" + "".join(f"{t:>11}" for t in TIERS))
print("  " + "-" * 68)
for name, d in sets:
    buckets = defaultdict(lambda: [0, 0])
    for i in ids:
        t = d[i].get("difficulty", "unknown")
        buckets[t][1] += 1
        buckets[t][0] += bool(d[i]["correct"])
    cells = ""
    for t in TIERS:
        c, n = buckets.get(t, (0, 0))
        cells += f"{c/n:>11.3f}" if n else f"{'-':>11}"
    print(f"  {name[:24]:<24}{cells}")

print()
print(f"  {'behaviour':<30} {'tokens':>9} {'cap%':>7} {'bn':>7} {'fmt fail':>9}")
print("  " + "-" * 66)
for name, d in sets:
    n = len(ids)
    tok = sum(d[i]["n_generated_tokens"] for i in ids) / n
    cap = sum(bool(d[i].get("hit_cap")) for i in ids) / n
    bn = sum(d[i]["bengali_ratio"] for i in ids) / n
    fmt = sum(bool(d[i]["format_failure"]) for i in ids) / n
    print(f"  {name:<30} {tok:>9.0f} {cap:>7.2f} {bn:>7.3f} {fmt:>9.2f}")

# items only one adapter gets right -- where the difficulty band actually mattered
only_a = [i for i in ids if A[i]["correct"] and not B[i]["correct"]]
only_b = [i for i in ids if B[i]["correct"] and not A[i]["correct"]]
both = [i for i in ids if A[i]["correct"] and B[i]["correct"]]
print()
print(f"  only {args.a} correct : {len(only_a)}")
print(f"  only {args.b} correct : {len(only_b)}")
print(f"  both correct                : {len(both)}")
