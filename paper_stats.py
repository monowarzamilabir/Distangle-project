r"""
Every statistic reported in the paper, and the LaTeX for every table.

Run this, then make_figs.py, then merge_paper.py, then build_main.py.

Writes:
    paper/tab_main.tex        before/after with effect size and 95% CI
    paper/tab_retention.tex   answer retention
    paper/tab_difficulty.tex  per-difficulty accuracy, all models
    paper/tab_adherence.tex   Bengali character ratio with paired CI
    paper/draft_numbers.json  machine-readable, consumed by make_figs.py
    paper/draft_results.txt   console report

Statistical choices, and why:
  paired binary (before vs after, same items)  -> exact McNemar on discordant
      pairs, plus the risk difference and its 95% CI by the standard paired
      formula. The CI is the effect size; the p-value alone is not.
  unpaired binary (model A vs model B)         -> Fisher's exact test
  single proportion                            -> Wilson score interval, which
      behaves correctly near zero where the normal approximation does not
  paired continuous (Bengali character ratio)  -> mean paired difference with
      a 95% t interval, plus an exact sign test on the per-item direction
"""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

PRED = Path("results_v2")
OUT = Path("paper")
OUT.mkdir(exist_ok=True)

ORDER = ["titulm-1b", "titulm-1b-it", "tigerllm-1b", "titulm-3b-it",
         "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]

META = {
    "titulm-1b":       ("TituLLM-1B (base)",   1.0, "deep"),
    "titulm-1b-it":    ("TituLLM-1B-Instruct", 1.0, "deep"),
    "titulm-3b-it":    ("TituLLM-3B-Instruct", 3.2, "deep"),
    "tigerllm-1b":     ("TigerLLM-1B",         1.0, "deep"),
    "qwen3-0.6b":      ("Qwen3-0.6B",          0.6, "shallow"),
    "qwen3-1.7b":      ("Qwen3-1.7B",          1.7, "shallow"),
    "qwen3-4b":        ("Qwen3-4B",            4.0, "shallow"),
    "qwen3-1.7b-easy": ("Qwen3-1.7B (easy)",   1.7, "shallow"),
}
DIFFS = ["easy", "medium", "hard", "olympiad"]


# ----------------------------------------------------------------- helpers
def load(key, cond):
    f = PRED / f"preds_{key}_{cond}.jsonl"
    if not f.exists():
        return None
    return {json.loads(l)["id"]: json.loads(l)
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    obs = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(obs + 1)) / 2 ** n)


def paired_diff_ci(b, c, n, z=1.96):
    """
    Risk difference for paired binary data and its 95% CI.

    d = (c - b) / n, where b = lost and c = gained. The variance of a paired
    difference depends only on the discordant counts, which is why two models
    with the same net change can have very different intervals.
    """
    if n == 0:
        return 0.0, (0.0, 0.0)
    d = (c - b) / n
    var = (b + c - (c - b) ** 2 / n) / (n * n)
    h = z * math.sqrt(max(var, 0.0))
    return d, (max(-1.0, d - h), min(1.0, d + h))


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    def p_tab(a, b, c, d):
        return (math.comb(a + b, a) * math.comb(c + d, c)
                / math.comb(a + b + c + d, a + c))
    obs = p_tab(a, b, c, d)
    n1, n2, k = a + b, c + d, a + c
    tot = 0.0
    for i in range(max(0, k - n2), min(n1, k) + 1):
        p = p_tab(i, n1 - i, k - i, n2 - (k - i))
        if p <= obs * (1 + 1e-9):
            tot += p
    return min(1.0, tot)


def sign_test(diffs):
    down = sum(1 for x in diffs if x < 0)
    up = sum(1 for x in diffs if x > 0)
    n = down + up
    if n == 0:
        return down, up, 1.0
    obs = min(down, up)
    return down, up, min(1.0, 2 * sum(math.comb(n, i) for i in range(obs + 1)) / 2 ** n)


def mean_ci(xs, z=1.96):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, (m, m)
    se = st.stdev(xs) / math.sqrt(len(xs))
    return m, (m - z * se, m + z * se)


# ----------------------------------------------------------------- collect
rows = []
for key in ORDER + ["qwen3-1.7b-easy"]:
    bef, aft = load(key, "before"), load(key, "after")
    if not bef or not aft:
        rows.append(dict(key=key, status="missing"))
        continue
    ids = sorted(set(bef) & set(aft))
    n = len(ids)
    kb = sum(bef[i]["correct"] for i in ids)
    ka = sum(aft[i]["correct"] for i in ids)
    lost = sum(1 for i in ids if bef[i]["correct"] and not aft[i]["correct"])
    gain = sum(1 for i in ids if aft[i]["correct"] and not bef[i]["correct"])
    d, dci = paired_diff_ci(lost, gain, n)
    bn = [aft[i].get("bengali_ratio", 0) - bef[i].get("bengali_ratio", 0) for i in ids]
    bnm, bnci = mean_ci(bn)
    down, up, bnp = sign_test(bn)
    label, pb, depth = META[key]
    bd = defaultdict(lambda: [0, 0, 0])
    for i in ids:
        dd = bef[i].get("difficulty", "?")
        bd[dd][0] += 1
        bd[dd][1] += bef[i]["correct"]
        bd[dd][2] += aft[i]["correct"]
    rows.append(dict(
        key=key, label=label, params_b=pb, depth=depth, n=n, status="ok",
        acc_b=kb / n, acc_a=ka / n, k_b=kb, k_a=ka,
        ci_b=wilson(kb, n), ci_a=wilson(ka, n),
        delta=d, delta_ci=dci, lost=lost, gained=gain, p=mcnemar_exact(lost, gain),
        cap_b=sum(bef[i]["hit_cap"] for i in ids) / n,
        cap_a=sum(aft[i]["hit_cap"] for i in ids) / n,
        bn_b=sum(bef[i].get("bengali_ratio", 0) for i in ids) / n,
        bn_a=sum(aft[i].get("bengali_ratio", 0) for i in ids) / n,
        bn_delta=bnm, bn_ci=bnci, bn_down=down, bn_up=up, bn_p=bnp,
        retained=sum(1 for i in ids if bef[i]["correct"] and aft[i]["correct"]),
        flipped=lost + gain,
        bydiff=dict(bd),
    ))

by = {r["key"]: r for r in rows if r["status"] == "ok"}
json.dump(rows, open(OUT / "draft_numbers.json", "w"), indent=1, default=list)


# ----------------------------------------------------------------- report
L = []
A = L.append
A("=" * 100)
A("PAPER STATISTICS   (paired on the ids each model finished in both conditions)")
A("=" * 100)
A(f"{'model':22} {'n':>4} {'before':>15} {'after':>15} {'delta [95% CI]':>22} {'McNemar':>8}")
A("-" * 100)
for r in rows:
    if r["status"] != "ok":
        continue
    sb = f"{r['acc_b']*100:5.1f} [{r['ci_b'][0]*100:4.1f},{r['ci_b'][1]*100:4.1f}]"
    sa = f"{r['acc_a']*100:5.1f} [{r['ci_a'][0]*100:4.1f},{r['ci_a'][1]*100:4.1f}]"
    sd = f"{r['delta']*100:+5.1f} [{r['delta_ci'][0]*100:+5.1f},{r['delta_ci'][1]*100:+5.1f}]"
    A(f"{r['label']:22} {r['n']:>4} {sb:>15} {sa:>15} {sd:>22} {r['p']:>8.3f}"
      + ("*" if r["p"] < 0.05 else ""))
A("-" * 100)

A("")
A("SCALE LADDER, unpaired Fisher exact (untuned condition)")
ladder = [("qwen3-0.6b", "qwen3-1.7b"), ("qwen3-1.7b", "qwen3-4b"),
          ("qwen3-0.6b", "qwen3-4b")]
fisher_p = {}
for x, y in ladder:
    rx, ry = by[x], by[y]
    p = fisher(rx["k_b"], rx["n"] - rx["k_b"], ry["k_b"], ry["n"] - ry["k_b"])
    fisher_p[(x, y)] = p
    A(f"  {rx['label']:12} {rx['acc_b']*100:5.1f}%  vs  {ry['label']:12} "
      f"{ry['acc_b']*100:5.1f}%   p = {p:.3g}")

A("")
A("DEPTH COMPARISONS, unpaired Fisher exact (untuned condition)")
depth_p = {}
for x, y in [("tigerllm-1b", "titulm-1b"), ("tigerllm-1b", "titulm-1b-it"),
             ("titulm-3b-it", "qwen3-1.7b")]:
    rx, ry = by[x], by[y]
    p = fisher(rx["k_b"], rx["n"] - rx["k_b"], ry["k_b"], ry["n"] - ry["k_b"])
    depth_p[(x, y)] = p
    A(f"  {rx['label']:20} {rx['acc_b']*100:5.1f}% (n={rx['n']})  vs  "
      f"{ry['label']:20} {ry['acc_b']*100:5.1f}% (n={ry['n']})   p = {p:.4g}")

A("")
A("BENGALI CHARACTER RATIO, mean paired difference with 95% CI and sign test")
for k in ORDER:
    r = by[k]
    A(f"  {r['label']:22} {r['bn_b']*100:5.1f}% -> {r['bn_a']*100:5.1f}%   "
      f"delta {r['bn_delta']*100:+5.1f} [{r['bn_ci'][0]*100:+5.1f},"
      f"{r['bn_ci'][1]*100:+5.1f}]   {r['bn_down']:3}down/{r['bn_up']:3}up  "
      f"sign p = {r['bn_p']:.2g}")

A("")
A("TOKEN-CAP RATE")
for k in ORDER:
    r = by[k]
    A(f"  {r['label']:22} {r['cap_b']*100:5.1f}% -> {r['cap_a']*100:5.1f}%")

A("")
A("ANSWER RETENTION")
for k in ["qwen3-4b", "qwen3-1.7b", "qwen3-0.6b", "tigerllm-1b",
          "titulm-3b-it", "titulm-1b-it"]:
    r = by[k]
    pct = f"{r['retained']/r['k_b']*100:.0f}%" if r["k_b"] else "n/a"
    A(f"  {r['label']:22} right {r['k_b']:3}  kept {r['retained']:3}  "
      f"retention {pct:>4}  flipped {r['flipped']:3}/{r['n']}")

(OUT / "draft_results.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))


# ----------------------------------------------------------------- tables
def w(name, lines):
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / name)


# Table: main results, now carrying the effect size and its interval
T = [r"\begin{table*}[t]",
     r"\caption{Accuracy before and after LIMO-style fine-tuning on 873 curated",
     r"Bengali reasoning traces, on the same 250 held-out items per model",
     r"(TituLLM-1B base on 100). Intervals on each accuracy are Wilson 95\%.",
     r"$\Delta$ is the paired risk difference in percentage points with its 95\%",
     r"confidence interval; it is the effect size, and it is reported because a",
     r"$p$-value alone does not bound how large the change could be. $p$ is a",
     r"two-sided exact McNemar test on the discordant pairs. Notice that no",
     r"comparison reaches significance and that six of the seven $\Delta$",
     r"intervals contain zero. The exception is TituLLM-1B-Instruct, whose",
     r"interval is computed from only eight discordant pairs; the Wald interval",
     r"is anti-conservative at that count and the exact test ($p=0.070$) is the",
     r"reliable reading.}",
     r"\label{tab:main}", r"\centering\small",
     r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|}", r"\hline",
     r"\textbf{Model} & \textbf{Params} & \textbf{Bn depth} & \textbf{$n$} & "
     r"\textbf{Before} & \textbf{After} & \textbf{$\Delta$ (pp)} & "
     r"\textbf{95\% CI} & \textbf{$p$} \\", r"\hline"]
for k in ORDER:
    r = by[k]
    T.append("%s & %.1fB & %s & %d & %.1f\\%% & %.1f\\%% & %+.1f & "
             "[%+.1f, %+.1f] & %.3f \\\\"
             % (r["label"], r["params_b"], r["depth"], r["n"],
                r["acc_b"] * 100, r["acc_a"] * 100, r["delta"] * 100,
                r["delta_ci"][0] * 100, r["delta_ci"][1] * 100, r["p"]))
T += [r"\hline", r"\end{tabular}", r"\end{table*}"]
w("tab_main.tex", T)


# Table: answer retention
R = [r"\begin{table}[htbp]",
     r"\caption{Answer retention. Of the items each model answered correctly",
     r"before fine-tuning, how many it still answered correctly afterwards, and",
     r"how many items changed outcome in either direction. Notice that no model",
     r"retained even half of what it already knew, which is why the small net",
     r"changes in Table~\ref{tab:main} understate how much the adapter moved.}",
     r"\label{tab:retention}", r"\centering\small",
     r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
     r"\textbf{Model} & \textbf{Correct before} & \textbf{Retained} & "
     r"\textbf{Retention} & \textbf{Items flipped} \\", r"\hline"]
for k in ["qwen3-4b", "qwen3-1.7b", "qwen3-0.6b", "tigerllm-1b",
          "titulm-3b-it", "titulm-1b-it"]:
    r = by[k]
    pct = "%.0f\\%%" % (100 * r["retained"] / r["k_b"]) if r["k_b"] else "n/a"
    R.append("%s & %d & %d & %s & %d / %d \\\\"
             % (r["label"], r["k_b"], r["retained"], pct, r["flipped"], r["n"]))
R += [r"\hline", r"\end{tabular}", r"\end{table}"]
w("tab_retention.tex", R)


# Table: per-difficulty accuracy, every model
P = [r"\begin{table*}[t]",
     r"\caption{Accuracy by difficulty tier, before and after fine-tuning, for",
     r"every backbone on the same held-out items. Tier counts are 69 easy, 69",
     r"medium, 67 hard and 45 olympiad for the models evaluated on 250 items,",
     r"and 30/24/29/17 for TituLLM-1B base on 100. Notice that the Qwen3",
     r"advantage holds in every tier rather than coming from one band, and that",
     r"no tier shows a consistent post-tuning gain.}",
     r"\label{tab:difficulty}", r"\centering\small",
     r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
     r"\textbf{Model} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & "
     r"\textbf{Olympiad} \\", r"\hline"]
for k in ORDER:
    r = by[k]
    cells = []
    for d in DIFFS:
        if d in r["bydiff"]:
            nn, bb, aa = r["bydiff"][d]
            cells.append("%.1f $\\rightarrow$ %.1f" % (100 * bb / nn, 100 * aa / nn))
        else:
            cells.append("--")
    P.append("%s & %s \\\\" % (r["label"], " & ".join(cells)))
P += [r"\hline", r"\end{tabular}",
      r"\multicolumn{1}{l}{\small All values are percentages, before "
      r"$\rightarrow$ after.}",
      r"\end{table*}"]
w("tab_difficulty.tex", P)


# Table: Bengali adherence
B = [r"\begin{table}[htbp]",
     r"\caption{Bengali character ratio in generated output, before and after",
     r"fine-tuning on Bengali reasoning traces. $\Delta$ is the mean paired",
     r"difference in percentage points with its 95\% confidence interval; $p$ is",
     r"a two-sided exact sign test on the per-item direction of change. Notice",
     r"that training on Bengali traces moved six of seven models away from",
     r"Bengali, and moved them furthest in the models that started most Bengali.}",
     r"\label{tab:adherence}", r"\centering\small",
     r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
     r"\textbf{Model} & \textbf{Before} & \textbf{After} & "
     r"\textbf{$\Delta$ (pp), 95\% CI} & \textbf{$p$} \\", r"\hline"]
for k in ORDER:
    r = by[k]
    B.append("%s & %.1f\\%% & %.1f\\%% & %+.1f [%+.1f, %+.1f] & %s \\\\"
             % (r["label"], r["bn_b"] * 100, r["bn_a"] * 100,
                r["bn_delta"] * 100, r["bn_ci"][0] * 100, r["bn_ci"][1] * 100,
                ("$<10^{-6}$" if r["bn_p"] < 1e-6 else "%.4f" % r["bn_p"])))
B += [r"\hline", r"\end{tabular}", r"\end{table}"]
w("tab_adherence.tex", B)


# machine-readable numbers the prose quotes, so the text can be checked
facts = {
    "fisher_scale": {f"{a}_vs_{b}": p for (a, b), p in fisher_p.items()},
    "fisher_depth": {f"{a}_vs_{b}": p for (a, b), p in depth_p.items()},
}
json.dump(facts, open(OUT / "draft_facts.json", "w"), indent=1)
print("wrote", OUT / "draft_facts.json")
