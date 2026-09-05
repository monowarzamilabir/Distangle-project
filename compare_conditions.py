# -*- coding: utf-8 -*-
"""Before vs after LoRA elicitation, for one model or all of them.

    python compare_conditions.py tigerllm-1b
    python compare_conditions.py --all

Reports more than accuracy. On this task most models sit near zero, so a single
accuracy number hides whether elicitation did anything at all. What actually
moves is behaviour: whether the model terminates, how long it generates, whether
it stays in Bengali, and whether it produces a parseable answer. Those are
measured over all 500 items and are informative even when accuracy is 0.

The flip analysis (wrong->right vs right->wrong) matters because a net accuracy
change of zero can hide substantial churn in both directions.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
RESULTS = ROOT / "results_v2"
TIERS = ["easy", "medium", "hard", "olympiad"]


def load(key, cond):
    f = RESULTS / f"preds_{key}_{cond}.jsonl"
    if not f.exists():
        return None
    rows = {}
    for line in f.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def mean(rows, field):
    vals = [r[field] for r in rows if field in r]
    return sum(vals) / len(vals) if vals else float("nan")


def rate(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else float("nan")


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct for proportions near 0, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_p(gained, lost):
    """Exact McNemar (binomial) on discordant pairs.

    The right test for paired before/after on the SAME items. With accuracy
    near zero, a raw difference in counts is dominated by noise: TigerLLM went
    19/500 -> 16/500, which looks like a decline but is p=0.69.
    """
    m = gained + lost
    if m == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def paired_diff_ci(B, A, f, z=1.96):
    """Mean paired difference with a 95% CI. Significant if the CI excludes 0."""
    ds = [f(a) - f(b) for b, a in zip(B, A)]
    n = len(ds)
    md = sum(ds) / n
    if n < 2:
        return md, md, md
    sd = math.sqrt(sum((d - md) ** 2 for d in ds) / (n - 1))
    se = sd / math.sqrt(n)
    return md, md - z * se, md + z * se


def arrow(before, after, higher_better=True):
    if before != before or after != after:      # NaN
        return ""
    d = after - before
    if abs(d) < 1e-9:
        return "  (no change)"
    good = (d > 0) if higher_better else (d < 0)
    return f"  {'+' if d > 0 else ''}{d:.3f}  {'IMPROVED' if good else 'WORSE'}"


def report(key, label=None):
    b, a = load(key, "before"), load(key, "after")
    if not b or not a:
        print(f"{key}: missing {'before' if not b else 'after'} predictions")
        return
    common = sorted(set(b) & set(a))
    if not common:
        print(f"{key}: no overlapping items yet")
        return

    B = [b[i] for i in common]
    A = [a[i] for i in common]
    label = label or B[0].get("model_label", key)

    print()
    print("=" * 72)
    print(f"  {label}   before vs after LoRA      n = {len(common)} items")
    if len(b) != 500 or len(a) != 500:
        print(f"  (partial: before={len(b)}, after={len(a)} of 500)")
    print("=" * 72)

    rows = [
        ("accuracy",           rate(B, lambda r: r["correct"]),
                               rate(A, lambda r: r["correct"]), True),
        ("format failures",    rate(B, lambda r: r["format_failure"]),
                               rate(A, lambda r: r["format_failure"]), False),
        ("hit token cap",      rate(B, lambda r: r.get("hit_cap", False)),
                               rate(A, lambda r: r.get("hit_cap", False)), False),
        ("Bengali ratio",      mean(B, "bengali_ratio"),
                               mean(A, "bengali_ratio"), True),
    ]
    print(f"  {'metric':<20} {'before':>9} {'after':>9}   change")
    print("  " + "-" * 66)
    for name, bv, av, hb in rows:
        print(f"  {name:<20} {bv:>9.3f} {av:>9.3f}{arrow(bv, av, hb)}")

    tb, ta = mean(B, "n_generated_tokens"), mean(A, "n_generated_tokens")
    print(f"  {'mean tokens':<20} {tb:>9.1f} {ta:>9.1f}"
          f"  {ta-tb:+.1f}  ({'shorter' if ta < tb else 'longer'})")

    # Accuracy by difficulty
    print()
    print(f"  {'difficulty':<12} {'n':>5} {'before':>9} {'after':>9}   change")
    print("  " + "-" * 66)
    for t in TIERS:
        bi = [r for r in B if r.get("difficulty") == t]
        ai = [r for r in A if r.get("difficulty") == t]
        if not bi:
            continue
        bv = sum(r["correct"] for r in bi) / len(bi)
        av = sum(r["correct"] for r in ai) / len(ai)
        print(f"  {t:<12} {len(bi):>5} {bv:>9.3f} {av:>9.3f}{arrow(bv, av)}")

    # Item-level churn: a net zero can hide movement in both directions.
    gained = [i for i in common if not b[i]["correct"] and a[i]["correct"]]
    lost = [i for i in common if b[i]["correct"] and not a[i]["correct"]]
    both = [i for i in common if b[i]["correct"] and a[i]["correct"]]
    print()
    print("  significance (paired, same items)")
    print("  " + "-" * 66)
    nb = sum(r["correct"] for r in B)
    na = sum(r["correct"] for r in A)
    lo_b, hi_b = wilson(nb, len(B))
    lo_a, hi_a = wilson(na, len(A))
    print(f"  accuracy before : {nb}/{len(B)} = {nb/len(B):.4f}  "
          f"95% CI [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  accuracy after  : {na}/{len(A)} = {na/len(A):.4f}  "
          f"95% CI [{lo_a:.4f}, {hi_a:.4f}]")
    _g = sum(1 for i in common if not b[i]["correct"] and a[i]["correct"])
    _l = sum(1 for i in common if b[i]["correct"] and not a[i]["correct"])
    _p = mcnemar_p(_g, _l)
    print(f"  exact McNemar p = {_p:.3f}  ->  "
          f"{'NOT significant' if _p > 0.05 else 'SIGNIFICANT'} at alpha=0.05")
    print()
    for nm, fn in [("mean tokens", lambda r: r["n_generated_tokens"]),
                   ("Bengali ratio", lambda r: r["bengali_ratio"]),
                   ("hit cap", lambda r: float(r.get("hit_cap", False))),
                   ("format failure", lambda r: float(r["format_failure"]))]:
        md, lo, hi = paired_diff_ci(B, A, fn)
        flag = "" if lo <= 0 <= hi else "  SIGNIFICANT"
        print(f"    {nm:<16} diff {md:+9.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{flag}")

    print()
    print(f"  item-level flips")
    print("  " + "-" * 66)
    print(f"  wrong -> right : {len(gained):>4}")
    print(f"  right -> wrong : {len(lost):>4}")
    print(f"  right in both  : {len(both):>4}")
    print(f"  net            : {len(gained) - len(lost):>+4}")
    if gained:
        print(f"  gained ids: {', '.join(gained[:8])}"
              + (" ..." if len(gained) > 8 else ""))
    if lost:
        print(f"  lost ids  : {', '.join(lost[:8])}"
              + (" ..." if len(lost) > 8 else ""))

    # One concrete example, since near-zero accuracy makes numbers hard to read.
    ex = (gained or lost or common)[0]
    print()
    print(f"  example item {ex}   gold = {b[ex]['gold']}")
    for cond, r in (("before", b[ex]), ("after", a[ex])):
        txt = " ".join(str(r["completion"]).split())[:220]
        print(f"    [{cond:<6}] pred={r['pred_answer']!r} correct={r['correct']} "
              f"tokens={r['n_generated_tokens']}")
        print(f"             {txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    keys = ["tigerllm-1b", "titulm-1b", "titulm-3b",
            "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]
    targets = keys if args.all else ([args.model] if args.model else keys)
    for k in targets:
        report(k)
