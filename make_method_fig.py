"""
The methodology diagram: the whole study design in one figure.

Five stages as horizontal bands, each with its own colour. Shape carries
meaning: rounded rectangles are artefacts (data, models, adapters) and
hexagons are processes that transform them. The split after decontamination
is where the main experiment and its difficulty control diverge.

PNG only, 200 dpi. No Bengali glyphs, so it renders anywhere.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans"})

FIG = Path("paper/figs")
FIG.mkdir(parents=True, exist_ok=True)

# muted, print-safe, and still separable in greyscale by lightness
INK, MUTE = "#1f2937", "#6b7280"
BANDS = {
    "data":  ("#2563eb", "#eff5ff"),   # blue
    "model": ("#7c3aed", "#f5f1fe"),   # violet
    "adapt": ("#d97706", "#fff8ec"),   # amber
    "eval":  ("#0d9488", "#effcfa"),   # teal
    "stats": ("#be185d", "#fdf2f7"),   # rose
}
DEEP, SHAL = "#dc2626", "#2563eb"

fig, ax = plt.subplots(figsize=(11.4, 6.9))
ax.set_xlim(0, 22)
ax.set_ylim(0, 13.3)
ax.axis("off")


def band(y, h, key, num, name):
    """Coloured background strip with a numbered stage label down the left."""
    edge, fill = BANDS[key]
    ax.add_patch(FancyBboxPatch(
        (0.15, y), 21.7, h, boxstyle="round,pad=0,rounding_size=0.18",
        fc=fill, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch(
        (0.15, y), 0.16, h, boxstyle="square,pad=0",
        fc=edge, ec="none", zorder=1))
    ax.text(0.62, y + h - 0.40, num, fontsize=15, weight="bold",
            color=edge, va="top", ha="left", zorder=2)
    ax.text(0.62, y + h - 1.00, name, fontsize=8.3, weight="bold",
            color=edge, va="top", ha="left", zorder=2, linespacing=1.25)


def box(x, y, w, h, title, sub=None, edge=INK, fill="white",
        tsize=8.6, ssize=7.0, tcol=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
        fc=fill, ec=edge, lw=1.15, zorder=3))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 0.26, title, ha="center", va="center",
                fontsize=tsize, weight="bold", color=tcol or INK, zorder=4)
        ax.text(x + w / 2, cy - 0.33, sub, ha="center", va="center",
                fontsize=ssize, color=MUTE, zorder=4, linespacing=1.3)
    elif title:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=tsize, weight="bold", color=tcol or INK,
                zorder=4, linespacing=1.3)


def hexbox(cx, cy, w, h, title, sub, edge, tsize=8.4, ssize=6.7):
    """Hexagon marks a process rather than an artefact."""
    k = 0.22 * w
    pts = [(cx - w / 2, cy), (cx - w / 2 + k, cy + h / 2),
           (cx + w / 2 - k, cy + h / 2), (cx + w / 2, cy),
           (cx + w / 2 - k, cy - h / 2), (cx - w / 2 + k, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=edge,
                         lw=1.15, zorder=3))
    ax.text(cx, cy + 0.30, title, ha="center", va="center", fontsize=tsize,
            weight="bold", color=edge, zorder=4, linespacing=1.2)
    ax.text(cx, cy - 0.36, sub, ha="center", va="center", fontsize=ssize,
            color=MUTE, zorder=4, linespacing=1.35)


def arrow(x1, y1, x2, y2, col="#9ca3af", lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        color=col, lw=lw, zorder=2, shrinkA=1, shrinkB=1))


# ------------------------------------------------------------ 1. DATA
band(9.30, 3.85, "data", "1", "DATA")
E = BANDS["data"][0]
box(3.50, 10.50, 2.55, 1.45, "Ganit SFT",
    "16,868 problems\n4 upstream sources", edge=E, tsize=8.6, ssize=6.8)
hexbox(9.20, 11.22, 5.10, 1.85, "Decontaminate",
       "against 4,080 items from Bn-MGSM,\nBn-MSVAMP and BenNumEval\n"
       "numeric fingerprint or Jaccard >= 0.7", E)
box(12.15, 10.62, 1.85, 1.20, "13,553", "clean pool", edge=E,
    tsize=9.0, ssize=6.8)
hexbox(15.95, 11.22, 3.20, 1.55, "Split by\ndifficulty",
       "solve rate out of 32\nby Qwen3-32B", E)
box(17.90, 11.32, 3.85, 1.16, "873 traces",
    "solved 1 to 16 of 32  |  MAIN", edge=E, fill="#dbeafe",
    tsize=8.6, ssize=6.7)
box(17.90, 9.92, 3.85, 1.16, "1,000 traces",
    "solved 28 to 32  |  CONTROL", edge="#d97706", fill="#fef3c7",
    tsize=8.6, ssize=6.7)

arrow(6.05, 11.22, 6.65, 11.22, E)
arrow(11.75, 11.22, 12.15, 11.22, E)
arrow(14.00, 11.22, 14.35, 11.22, E)
arrow(17.55, 11.48, 17.90, 11.90, E)
arrow(17.55, 10.96, 17.90, 10.50, "#d97706")

# ------------------------------------------------------------ 2. MODELS
band(5.95, 3.15, "model", "2", "MODELS\n7 backbones")
box(3.50, 6.30, 8.40, 2.40, "", None, edge=DEEP, fill="#fef2f2")
ax.text(7.70, 8.42, "DEEP Bengali pretraining", ha="center", fontsize=8.2,
        weight="bold", color=DEEP, zorder=5)
for i, (nm, pr) in enumerate([("TituLLM-1B", "base"), ("TituLLM-1B", "Instruct"),
                              ("TituLLM-3B", "Instruct"), ("TigerLLM-1B", "Gemma3 base")]):
    box(3.74 + i * 2.03, 6.50, 1.87, 1.55, nm, pr, edge=DEEP,
        tsize=7.5, ssize=6.4, tcol=DEEP)

box(12.55, 6.30, 6.60, 2.40, "", None, edge=SHAL, fill="#eff6ff")
ax.text(15.85, 8.42, "SHALLOW Bengali pretraining", ha="center", fontsize=8.2,
        weight="bold", color=SHAL, zorder=5)
for i, nm in enumerate(["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]):
    box(12.79 + i * 2.09, 6.50, 1.93, 1.55, nm, "no Bengali\ncontinued PT",
        edge=SHAL, tsize=7.5, ssize=6.4, tcol=SHAL)

ax.text(20.50, 7.50, "scale and depth\nvaried\nindependently", ha="center",
        va="center", fontsize=7.1, color=MUTE, style="italic",
        linespacing=1.4, zorder=5)

# ------------------------------------------------------------ 3. ADAPTATION
band(4.20, 1.55, "adapt", "3", "ADAPTATION")
E = BANDS["adapt"][0]
box(3.50, 4.40, 15.65, 1.15,
    "LoRA    r = 16,   alpha = 32,   dropout 0.05,   3 epochs,   seed 42",
    "applied to q, k, v, o, gate, up and down projections    |    "
    "identical configuration for every backbone    |    single 16GB consumer GPU",
    edge=E, tsize=8.6, ssize=6.8)
arrow(7.70, 6.30, 7.70, 5.63, "#9ca3af")
arrow(15.85, 6.30, 15.85, 5.63, "#9ca3af")
arrow(19.80, 9.92, 19.80, 8.72, "#9ca3af")

# ------------------------------------------------------------ 4. EVALUATION
band(1.75, 2.30, "eval", "4", "EVALUATION\npaired")
E = BANDS["eval"][0]
box(3.50, 2.65, 3.55, 1.25, "250 held-out items",
    "GANIT-DEV, stratified,\nidentical across models", edge=E,
    tsize=8.2, ssize=6.7)
box(8.05, 3.25, 4.70, 0.92, "BEFORE    base weights only", None,
    edge=MUTE, tsize=8.0)
box(8.05, 2.10, 4.70, 0.92, "AFTER    base + LoRA adapter", None,
    edge=E, fill="#ccfbf1", tsize=8.0)
box(13.75, 2.65, 5.40, 1.25, "Shared answer scorer",
    "answer region located first,\nthen the first number taken\n"
    "one implementation for all models",
    edge=E, tsize=8.2, ssize=6.5)
arrow(7.05, 3.45, 8.05, 3.71, E)
arrow(7.05, 3.10, 8.05, 2.56, E)
arrow(12.75, 3.71, 13.75, 3.45, E)
arrow(12.75, 2.56, 13.75, 3.10, E)
arrow(5.28, 4.40, 5.28, 3.90, "#9ca3af")

# ------------------------------------------------------------ 5. ANALYSIS
band(0.20, 1.35, "stats", "5", "ANALYSIS")
E = BANDS["stats"][0]
for i, (t, s) in enumerate([
        ("Wilson 95% interval", "on every accuracy"),
        ("Exact McNemar", "paired, on the discordant items"),
        ("Difficulty stratification", "easy / medium / hard / olympiad")]):
    box(3.50 + i * 5.25, 0.40, 4.95, 0.95, t, s, edge=E,
        tsize=8.0, ssize=6.6)
arrow(16.45, 2.65, 16.45, 1.35, "#9ca3af")

fig.savefig(FIG / "fig_method.png", dpi=200, bbox_inches="tight",
            pad_inches=0.06, facecolor="white")
plt.close(fig)
print("wrote fig_method.png")
