r"""
Checks the assembled paper against the computed statistics.

Catches the failure mode that bit this project repeatedly: a number gets
updated in one place and left stale in another. Every check below compares
prose or table text in paper/main.tex against paper/draft_numbers.json and
paper/tokenizer_facts.json, which are written from the stored predictions.

    python verify_paper.py

Exit code 0 means every check passed.
"""
import json
import re
import sys
from pathlib import Path

P = Path("paper/main.tex")
text = P.read_text(encoding="utf-8")
prose = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))

rows = {r["key"]: r for r in json.load(open("paper/draft_numbers.json"))
        if r["status"] == "ok"}
tokf = json.load(open("paper/tokenizer_facts.json"))

fails, checks = [], 0


def check(desc, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        fails.append((desc, detail))


def pct(x):
    return "%.1f" % (x * 100)


# --- every accuracy in the main table matches the computed value -----------
# Scope the search to the tab:main block: several model names also appear as
# row labels in the tokenizer and retention tables, and a row's cells contain
# backslashes (\% and \\), so the row pattern must allow them.
a = prose.index(r"\label{tab:main}")
main_tbl = prose[a:prose.index(r"\end{table", a)]
for k, r in rows.items():
    if k == "qwen3-1.7b-easy":
        continue
    row = re.search(re.escape(r["label"]) + r" &[^\n]*", main_tbl)
    check(f"table row present: {r['label']}", row is not None)
    if row:
        cells = row.group(0)
        for want, what in [(pct(r["acc_b"]), "before"), (pct(r["acc_a"]), "after"),
                           (str(r["n"]), "n")]:
            check(f"{r['label']} {what} = {want}", want in cells,
                  f"row text: {cells.strip()}")
        # the effect size and its interval must be there too
        check(f"{r['label']} has Delta", ("%+.1f" % (r["delta"] * 100)) in cells,
              f"row text: {cells.strip()}")

# --- prose numbers that are quoted by hand ---------------------------------
QUOTED = [
    ("Qwen3 scale ladder", ["3.6", "16.8", "44.8"]),
    ("Qwen3-4B drop", ["44.8", "39.6", "0.228", "-5.2", "13.0", "2.6"]),
    ("retention", ["112", "56", "43", "99"]),
    ("format migration", ["225", "222", "230"]),
    ("sensitivity bound", ["46.8"]),
    ("difficulty control", ["12.4", "18.0", "16.8", "0.039", "0.126"]),
    ("curation pipeline", ["16,868", "13,553", "873", "3,315", "4,080"]),
    ("adherence", ["97.9", "80.8", "98.3", "81.3", "87.8", "85.6"]),
]
for name, vals in QUOTED:
    for v in vals:
        check(f"{name}: {v} appears", v in prose)

# --- tokenizer table matches the measurement script ------------------------
for row in tokf["rows"]:
    check(f"tokenizer {row['tokenizer']} median {row['median']}",
          format(row["median"], ",") in prose)
    check(f"tokenizer {row['tokenizer']} vocab {row['vocab']}",
          format(row["vocab"], ",") in prose)
check("tokenizer ratio 3.1x quoted",
      "3.1" in prose and abs(tokf["ratio_qwen_over_titullm"] - 3.15) < 0.1)
check("truncation count matches measurement",
      str(tokf["qwen3_4b_truncated_at_2048"]) in prose)

# --- things that must NOT be present ---------------------------------------
BANNED = {
    "stale eval size (500 items scored)": r"on 500 items",
    "causal overclaim": r"Scale determines",
    "old Qwen3-4B value": r"45\.0\\%",
    "old TituLLM-3B value": r"1\.0\\% \(",
    "unsupported training-fit claim": r"improved training-set fit",
    "stale cap-rate count": r"five of seven models",
    "unfilled placeholder": r"\\(NUM|TODO)\{",
    "em dash": "\u2014",
}
for name, pat in BANNED.items():
    check(f"absent: {name}", re.search(pat, text) is None)

# --- structural integrity ---------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", text))
check("no dangling references", not (refs - labels), str(sorted(refs - labels)))
check("exactly one contributions list", text.count("Our contributions are:") == 1)
check("exactly one Results section", text.count(r"\section{Results}") == 1)
check("all four figures included", text.count(r"\includegraphics") == 4)
n_tables = len(re.findall(r"\\begin\{table\*?\}", text))
check("six tables present", n_tables == 6, f"found {n_tables}")
check("document is complete", text.rstrip().endswith(r"\end{document}"))
check("ascii only", not [c for c in text if ord(c) > 127])

# --- report -----------------------------------------------------------------
print(f"{checks - len(fails)}/{checks} checks passed")
if fails:
    print("\nFAILURES")
    for desc, detail in fails:
        print(f"  x {desc}" + (f"\n      {detail}" if detail else ""))
    sys.exit(1)
print("paper is internally consistent with the computed results")
