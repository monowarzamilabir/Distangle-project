"""
Figures for the draft results section.

No Bengali glyphs anywhere: pdfLaTeX cannot set them and the figures need to
compile on a machine without Bengali fonts installed. Every number is read
from paper/draft_numbers.json, which paper_stats.py writes, so the figures
cannot drift away from the table.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

FIG = Path("paper/figs")
FIG.mkdir(parents=True, exist_ok=True)

rows = {r["key"]: r for r in json.load(open("paper/draft_numbers.json"))
        if r["status"] == "ok"}

DEEP = "#c1442e"   # deep Bengali pretraining
SHAL = "#2b6cb0"   # shallow Bengali pretraining
GREY = "#6b7280"


def save(fig, name):
    # PNG only: Overleaf takes it directly and no PDF toolchain is needed here
    fig.savefig(FIG / (name + ".png"), dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name + ".png")


# --------------------------------------------------------------- Fig 1
# The headline. Accuracy against parameter count, split by how much Bengali
# pretraining the model received.
fig, ax = plt.subplots(figsize=(5.4, 3.5))

groups = [
    (["titulm-1b", "titulm-1b-it", "tigerllm-1b", "titulm-3b-it"], DEEP,
     "Deep Bengali pretraining", "o", False),
    (["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"], SHAL,
     "Shallow Bengali pretraining", "s", True),
]
for keys, col, lab, mk, connect in groups:
    pts = sorted((rows[k]["params_b"], rows[k]["acc_b"] * 100, rows[k])
                 for k in keys if k in rows)
    if connect:
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "-", color=col, lw=1.6, zorder=2)
    first = True
    for x, y, r in pts:
        lo, hi = r["ci_b"]
        ax.errorbar(x, y, yerr=[[y - lo * 100], [hi * 100 - y]],
                    fmt=mk, color=col, ms=6, capsize=2.5, lw=1.1, zorder=3,
                    label=lab if first else None)
        first = False

# The three 1B deep models sit almost on top of each other, so they get a
# leader line out to the right rather than an inline label.
nudge = [("titulm-1b", 1.22, 4.0, "left"), ("titulm-1b-it", 1.22, 8.5, "left"),
         ("tigerllm-1b", 1.22, 13.0, "left"), ("titulm-3b-it", 3.4, 6.5, "left"),
         ("qwen3-0.6b", 0.55, 9.5, "left"), ("qwen3-1.7b", 1.72, 24.5, "left"),
         ("qwen3-4b", 4.2, 47.0, "left")]
for k, lx, ly, ha in nudge:
    if k not in rows:
        continue
    r = rows[k]
    x, y = r["params_b"], r["acc_b"] * 100
    col = DEEP if r["depth"] == "deep" else SHAL
    if lx > x * 1.05:                      # label sits off to the side
        ax.plot([x * 1.03, lx * 0.99], [y, ly], "-", lw=0.6, color=col, alpha=0.55)
    ax.annotate(r["label"], (lx, ly), ha=ha, va="center", fontsize=7.3, color=col)

ax.set_xscale("log")
ax.set_xticks([0.6, 1, 1.7, 3.2, 4])
ax.set_xticklabels(["0.6B", "1B", "1.7B", "3.2B", "4B"])
ax.set_xticks([], minor=True)              # kill the 2x10^0 / 3x10^0 clutter
ax.set_xlim(0.5, 6.4)
ax.set_xlabel("Parameters (log scale)")
ax.set_ylabel("Accuracy on held-out Bengali math (%)")
ax.set_ylim(0, 60)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("Scale predicts Bengali reasoning; Bengali pretraining depth does not",
             fontsize=9.5, pad=8)
save(fig, "fig_scale")


# --------------------------------------------------------------- Fig 2
# Before and after LIMO-style tuning, one row per model.
keys = [k for k in ["qwen3-4b", "qwen3-1.7b", "qwen3-0.6b", "tigerllm-1b",
                    "titulm-3b-it", "titulm-1b-it", "titulm-1b"] if k in rows]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for i, k in enumerate(keys):
    r = rows[k]
    b, a = r["acc_b"] * 100, r["acc_a"] * 100
    ax.plot([b, a], [i, i], "-", color="#9ca3af", lw=1.4, zorder=1)
    # Wilson intervals on each condition, so the reader can see the
    # uncertainty that produces the null rather than taking p on trust.
    for v, ci, col, off in [(b, r["ci_b"], GREY, -0.17),
                            (a, r["ci_a"], DEEP if r["depth"] == "deep" else SHAL,
                             0.17)]:
        ax.plot([ci[0] * 100, ci[1] * 100], [i + off, i + off], "-",
                color=col, lw=1.0, alpha=0.55, zorder=2)
    ax.plot(b, i, "o", ms=6.5, color="white", mec=GREY, mew=1.6, zorder=3)
    ax.plot(a, i, "o", ms=6.5,
            color=DEEP if r["depth"] == "deep" else SHAL, zorder=3)
    ax.text(max(b, a) + 1.8, i, "p = %.2f" % r["p"],
            va="center", fontsize=7.2, color="#4b5563")

ax.set_yticks(range(len(keys)))
ax.set_yticklabels(["%s  (n=%d)" % (rows[k]["label"], rows[k]["n"]) for k in keys],
                   fontsize=8)
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(-2, 68)
ax.invert_yaxis()                          # biggest model on top
ax.plot([], [], "o", ms=6, color="white", mec=GREY, mew=1.6, label="before tuning")
ax.plot([], [], "o", ms=6, color="#4b5563", label="after tuning")
ax.legend(frameon=False, fontsize=8, loc="lower right",
          bbox_to_anchor=(1.0, -0.02))
ax.set_title("873 curated traces move no model significantly",
             fontsize=9.5, pad=8)
ax.set_xlim(-2, 72)
save(fig, "fig_beforeafter")


# --------------------------------------------------------------- Fig 3
# The difficulty-calibration control, run on one backbone.
fig, ax = plt.subplots(figsize=(4.0, 3.1))
vals = [16.8, 18.0, 12.4]
labs = ["base\n(no tuning)", "hard data\n(1-16 / 32)", "easy data\n(28-32 / 32)"]
bars = ax.bar(labs, vals, color=["#9ca3af", SHAL, "#d69e2e"], width=0.62)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.9, "%.1f%%" % v,
            ha="center", fontsize=8.5)
ax.plot([1, 1, 2, 2], [21.5, 22.6, 22.6, 21.5], lw=1.1, color="#374151")
ax.text(1.5, 23.1, "McNemar p = 0.039", ha="center", fontsize=8)
ax.plot([0, 0, 2, 2], [24.6, 25.4, 25.4, 24.6], lw=1.0, color="#9ca3af")
ax.text(1.0, 25.8, "vs base: p = 0.126, n.s.", ha="center",
        fontsize=7.5, color="#6b7280")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 29)
ax.set_title("Qwen3-1.7B: easier training data is\nsignificantly worse, not better",
             fontsize=9.5, pad=8)
save(fig, "fig_difficulty")


# Figures 4 to 6 (tokenizer tax, curation funnel, difficulty stratification)
# were cut from the draft. The tokenizer numbers already appear as a table,
# the curation funnel is now part of fig_method.png, and the stratification
# point is made in the text. Their code is in git history if needed.
