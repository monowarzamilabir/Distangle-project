r"""
Measures the Bengali tokenizer tax and writes Table 4.

This existed only as hand-recorded numbers in the draft, which made it the one
central claim in the paper with no path back to code. It now reconstructs the
exact training sequence that train_lora.py tokenizes: the shared Bengali
instruction, the problem, the model's own prompt formatting, and the target
trace. That is the sequence that actually competes for the context budget, so
it is the one the truncation and generation-budget arguments depend on.

    C:\anaconda\envs\bengali-rq2\python.exe measure_tokenizer.py
"""
import json
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

from train_lora import MODELS, BENGALI_INSTRUCTION, ENABLE_THINKING

DATA = Path("curated/ganit_limo_n1000_c1-16_seed42.jsonl")
OUT = Path("paper")

# One representative per tokenizer family. TituLLM-1B and 3B share a tokenizer,
# as do all three Qwen3 sizes, so the other checkpoints would just repeat these
# three distributions.
FAMILIES = [("TituLLM-1B", "titulm-1b"),
            ("TigerLLM-1B", "tigerllm-1b"),
            ("Qwen3", "qwen3-1.7b")]


def pairs():
    """(problem, target) exactly as load_pairs in train_lora.py builds them."""
    out = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        problem = str(r.get("problem") or "").strip()
        target = ""
        for m in reversed(r.get("messages") or []):
            if isinstance(m, dict) and m.get("role") == "assistant":
                target = str(m.get("content") or "").strip()
                break
        if problem and target:
            out.append((problem, target))
    return out


def sequence_lengths(key, data):
    cfg = MODELS[key]
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    chat = cfg.get("prompt_style") == "chat"

    def prompt(problem):
        user = f"{BENGALI_INSTRUCTION}\n\nসমস্যা: {problem}"
        if not chat:
            return user + "\n\nসমাধান:"
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
        except TypeError:                 # Gemma/Llama reject enable_thinking
            return tok.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True)

    lens = [len(tok(prompt(p) + t, add_special_tokens=False)["input_ids"])
            for p, t in data]
    # len(tokenizer) counts added tokens; tok.vocab_size does not. The released
    # sizes people quote, and TigerLLM's "Gemma3 plus exactly one token", are
    # the len() figures.
    return len(tok), lens


def pctile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


data = pairs()
print(f"{len(data)} training sequences from {DATA.name}\n")

rows = []
for label, key in FAMILIES:
    vocab, lens = sequence_lengths(key, data)
    rows.append((label, vocab, int(st.median(lens)), int(pctile(lens, 0.95)),
                 max(lens)))
    print("%-14s vocab %-8s median %5d  p95 %5d  max %5d"
          % (label, format(vocab, ","), st.median(lens), pctile(lens, 0.95),
             max(lens)))

med = {r[0]: r[2] for r in rows}
ratio = med["Qwen3"] / med["TituLLM-1B"]
print(f"\nQwen3 / TituLLM median ratio: {ratio:.2f}x")
# how many Qwen3 sequences the 2,048 training limit actually truncated
_, qlens = sequence_lengths("qwen3-4b", data)
over = sum(1 for x in qlens if x > 2048)
print(f"Qwen3-4B sequences longer than its 2,048-token limit: {over} of {len(qlens)}")

T = [r"\begin{table}[htbp]",
     r"\caption{Tokens required to represent the same 873 training sequences",
     r"under each tokenizer, measured on the full sequence train\_lora.py",
     r"builds: shared Bengali instruction, problem, model-specific prompt",
     r"formatting, and reasoning trace. Vocabulary sizes include added tokens.",
     r"Notice that the two Bengali-adapted tokenizers need roughly a third as",
     r"many tokens as Qwen3 for identical content, and that this efficiency is",
     r"not reflected in the accuracies of Table~\ref{tab:main}.}",
     r"\label{tab:tokenizer}", r"\centering",
     r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
     r"\textbf{Tokenizer} & \textbf{Vocab} & \textbf{Median} & \textbf{p95} & "
     r"\textbf{Max} \\", r"\hline"]
for label, v, m, p95, mx in rows:
    T.append("%s & %s & %s & %s & %s \\\\"
             % (label, format(v, ","), format(m, ","), format(p95, ","),
                format(mx, ",")))
T += [r"\hline", r"\end{tabular}", r"\end{table}"]
(OUT / "tab_tokenizer.tex").write_text("\n".join(T), encoding="utf-8")

json.dump({"ratio_qwen_over_titullm": round(ratio, 2),
           "qwen3_4b_truncated_at_2048": over,
           "n_sequences": len(data),
           "rows": [{"tokenizer": a, "vocab": b, "median": c, "p95": d, "max": e}
                    for a, b, c, d, e in rows]},
          open(OUT / "tokenizer_facts.json", "w"), indent=1)
print("wrote paper/tab_tokenizer.tex and paper/tokenizer_facts.json")
