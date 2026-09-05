"""
Computes every number in the draft results section and writes the figures.

Reads results_v2/preds_*.jsonl. Pairs each model's before/after on the
item ids they share, so a model still mid-run is reported on its completed
overlap rather than being silently mixed with a longer run.
"""
import json, math, itertools
from pathlib import Path
from collections import defaultdict

PRED = Path("results_v2")
FIG  = Path("paper/figs"); FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["titulm-1b", "titulm-1b-it", "tigerllm-1b", "titulm-3b-it",
         "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-1.7b-easy"]

META = {  # label, params_b, pretraining depth on Bengali
 "titulm-1b":       ("TituLLM-1B (base)",      1.0, "deep"),
 "titulm-1b-it":    ("TituLLM-1B-Instruct",    1.0, "deep"),
 "titulm-3b-it":    ("TituLLM-3B-Instruct",    3.2, "deep"),
 "tigerllm-1b":     ("TigerLLM-1B",            1.0, "deep"),
 "qwen3-0.6b":      ("Qwen3-0.6B",             0.6, "shallow"),
 "qwen3-1.7b":      ("Qwen3-1.7B",             1.7, "shallow"),
 "qwen3-4b":        ("Qwen3-4B",               4.0, "shallow"),
 "qwen3-1.7b-easy": ("Qwen3-1.7B (easy data)", 1.7, "shallow"),
}

def load(key, cond):
    f = PRED / f"preds_{key}_{cond}.jsonl"
    if not f.exists(): return None
    out = {}
    for line in f.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line); out[r["id"]] = r
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def mcnemar_exact(b, c):
    """two-sided exact binomial on the discordant pairs"""
    n = b + c
    if n == 0: return 1.0
    obs = min(b, c)
    tail = sum(math.comb(n, i) for i in range(obs+1)) / (2**n)
    return min(1.0, 2*tail)

rows = []
for key in ORDER:
    bef, aft = load(key, "before"), load(key, "after")
    if not bef or not aft: 
        rows.append(dict(key=key, status="missing")); continue
    ids = sorted(set(bef) & set(aft))
    n = len(ids)
    kb = sum(bef[i]["correct"] for i in ids)
    ka = sum(aft[i]["correct"] for i in ids)
    b = sum(1 for i in ids if bef[i]["correct"] and not aft[i]["correct"])  # lost
    c = sum(1 for i in ids if aft[i]["correct"] and not bef[i]["correct"])  # gained
    label, pb, depth = META[key]
    rows.append(dict(
        key=key, label=label, params_b=pb, depth=depth, n=n, status="ok",
        n_before=len(bef), n_after=len(aft),
        acc_b=kb/n, acc_a=ka/n, k_b=kb, k_a=ka,
        ci_b=wilson(kb, n), ci_a=wilson(ka, n),
        delta=(ka-kb)/n, lost=b, gained=c, p=mcnemar_exact(b, c),
        cap_b=sum(bef[i]["hit_cap"] for i in ids)/n,
        cap_a=sum(aft[i]["hit_cap"] for i in ids)/n,
        bn_b=sum(bef[i].get("bengali_ratio",0) for i in ids)/n,
        bn_a=sum(aft[i].get("bengali_ratio",0) for i in ids)/n,
    ))
    # difficulty split
    bd = defaultdict(lambda: [0,0,0])
    for i in ids:
        d = bef[i].get("difficulty","?")
        bd[d][0] += 1; bd[d][1] += bef[i]["correct"]; bd[d][2] += aft[i]["correct"]
    rows[-1]["bydiff"] = dict(bd)

json.dump(rows, open("paper/draft_numbers.json","w"), indent=1, default=list)

# ---------------- text report ----------------
L = []
A = L.append
A("="*96)
A("DRAFT RESULTS  (paired on the ids each model has finished in both conditions)")
A("="*96)
A(f"{'model':24} {'n':>4} {'before':>16} {'after':>16} {'delta':>8} {'McNemar':>9}  {'-/+':>7}")
A("-"*96)
for r in rows:
    if r["status"] != "ok":
        A(f"{r['key']:24} {'--- not yet evaluated in both conditions ---'}"); continue
    sb = f"{r['acc_b']*100:5.1f} [{r['ci_b'][0]*100:4.1f},{r['ci_b'][1]*100:4.1f}]"
    sa = f"{r['acc_a']*100:5.1f} [{r['ci_a'][0]*100:4.1f},{r['ci_a'][1]*100:4.1f}]"
    star = "*" if r["p"] < 0.05 else " "
    A(f"{r['label']:24} {r['n']:>4} {sb:>16} {sa:>16} {r['delta']*100:+7.1f} "
      f"{r['p']:>8.3f}{star} {r['lost']:>3}/{r['gained']:<3}")
A("-"*96)
A("* p < 0.05.  delta is percentage points.  -/+ = items lost / gained after tuning.")
A("")
A("TOKEN-CAP RATE (fraction of items where generation hit max_new_tokens)")
for r in rows:
    if r["status"]=="ok":
        A(f"  {r['label']:24} before {r['cap_b']*100:5.1f}%   after {r['cap_a']*100:5.1f}%")
A("")
A("ACCURACY BY DIFFICULTY (before -> after, n in parens)")
for r in rows:
    if r["status"]!="ok": continue
    parts=[]
    for d in ["easy","medium","hard","olympiad"]:
        if d in r["bydiff"]:
            n_,b_,a_ = r["bydiff"][d]
            parts.append(f"{d[:4]} {b_/n_*100:4.1f}->{a_/n_*100:4.1f} ({n_})")
    A(f"  {r['label']:24} " + "  ".join(parts))
Path("paper/draft_results.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

# ---------------- LaTeX table ----------------
T = []
T.append(r"\begin{table}[htbp]")
T.append(r"\caption{Accuracy before and after LIMO-style fine-tuning on 873 curated")
T.append(r"Bengali traces. Intervals are Wilson 95\%. $p$ is an exact McNemar test on")
T.append(r"the discordant pairs. Every comparison is paired on the same items.}")
T.append(r"\label{tab:main}")
T.append(r"\centering\small")
T.append(r"\begin{tabular}{|l|c|c|c|c|c|c|}")
T.append(r"\hline")
T.append(r"\textbf{Model} & \textbf{Params} & \textbf{Bn depth} & \textbf{$n$} & "
         r"\textbf{Before} & \textbf{After} & \textbf{$p$} \\")
T.append(r"\hline")
for r in rows:
    if r["status"] != "ok":
        continue
    lab = r["label"].replace("TituLLM", "TituLLM").replace("_", r"\_")
    T.append("%s & %.1fB & %s & %d & %.1f\%% & %.1f\%% & %.3f \\\\" % (
        lab, r["params_b"], r["depth"], r["n"],
        r["acc_b"]*100, r["acc_a"]*100, r["p"]))
T.append(r"\hline")
T.append(r"\end{tabular}")
T.append(r"\end{table}")
Path("paper/tab_main.tex").write_text("\n".join(T), encoding="utf-8")
print("\nwrote paper/tab_main.tex")


# ---------------- answer retention ----------------
# How much of what a model already knew survives fine-tuning. A method that
# teaches reasoning should retain correct answers and add to them; churn on
# this scale means the adapter rewrote the output distribution instead.
R = []
R.append(r"\begin{table}[htbp]")
R.append(r"\caption{Answer retention. Of the items each model answered correctly")
R.append(r"before fine-tuning, how many it still answered correctly afterwards.}")
R.append(r"\label{tab:retention}")
R.append(r"\centering\small")
R.append(r"\begin{tabular}{|l|c|c|c|c|}")
R.append(r"\hline")
R.append(r"\textbf{Model} & \textbf{Correct before} & \textbf{Retained} & "
         r"\textbf{Retention} & \textbf{Items flipped} \\")
R.append(r"\hline")
ret_rows = []
for key in ["qwen3-4b", "qwen3-1.7b", "qwen3-0.6b", "tigerllm-1b",
            "titulm-3b-it", "titulm-1b-it"]:
    bef, aft = load(key, "before"), load(key, "after")
    if not bef or not aft:
        continue
    ids = sorted(set(bef) & set(aft))
    rb = sum(bef[i]["correct"] for i in ids)
    kept = sum(1 for i in ids if bef[i]["correct"] and aft[i]["correct"])
    flip = sum(1 for i in ids if bef[i]["correct"] != aft[i]["correct"])
    pct = "%.0f\%%" % (100 * kept / rb) if rb else "n/a"
    label = META[key][0]
    R.append("%s & %d & %d & %s & %d / %d \\\\" % (label, rb, kept, pct, flip, len(ids)))
    ret_rows.append((label, rb, kept, kept / rb if rb else 0, flip, len(ids)))
R.append(r"\hline")
R.append(r"\end{tabular}")
R.append(r"\end{table}")
Path("paper/tab_retention.tex").write_text("\n".join(R), encoding="utf-8")

print("\nANSWER RETENTION")
print(f"  {'model':22} {'right':>6} {'kept':>5} {'retention':>10} {'flipped':>10}")
for label, rb, kept, r, flip, n in ret_rows:
    print(f"  {label:22} {rb:>6} {kept:>5} {r*100:>9.0f}% {flip:>6}/{n}")
print("\nwrote paper/tab_retention.tex")
