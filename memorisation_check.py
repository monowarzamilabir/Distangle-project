# -*- coding: utf-8 -*-
"""Did the LoRA actually learn anything? Evaluate on the TRAINING set.

    python memorisation_check.py --model qwen3-1.7b --n 50

Every backbone showed a statistically insignificant accuracy change on held-out
data. That leaves two indistinguishable explanations:

  (a) LIMO-style elicitation genuinely does not transfer to Bengali maths at
      these scales -- the paper's claim; or
  (b) the fine-tuning was misconfigured and taught the models nothing.

Held-out accuracy cannot separate them. Training-set accuracy can. A model that
has been trained on a problem and still cannot answer it did not learn. A model
that answers its training problems but not held-out ones learned fine and simply
does not generalise -- which is the interesting result, and is (a).

This is the first question a reviewer asks about a null result, so it is worth
ten minutes.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from scorer import extract_answer, is_correct, bengali_ratio
from train_lora import MODELS, BENGALI_INSTRUCTION, ENABLE_THINKING
from eval_math import EVAL_META, collect_stop_ids, make_prompt_fn

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--data", default="curated/ganit_limo_n1000_c1-16_seed42.jsonl")
ap.add_argument("--adapter-root", default="adapters")
args = ap.parse_args()

cfg, meta = MODELS[args.model], EVAL_META[args.model]

rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
items = [{"id": r.get("id"), "problem": r["problem"].strip(),
          "gold": str(r.get("bengali_solution") or "").strip()}
         for r in rows if r.get("problem") and r.get("bengali_solution")][:args.n]

print(f"{cfg['label']}  --  {len(items)} problems FROM THE TRAINING SET")
print(f"(these exact problems were in the LoRA training data)\n")

results = {}
for condition in ("before", "after"):
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16,
        attn_implementation=cfg["attn"]).cuda()
    if condition == "after":
        model = PeftModel.from_pretrained(
            model, str(Path(args.adapter_root) / cfg["adapter"])).merge_and_unload()
    model.eval()

    build = make_prompt_fn(args.model, tok)
    stop_ids = collect_stop_ids(tok)
    stop_set = set(stop_ids)
    bs, max_new = meta["batch"], meta["max_new"]

    correct = capped = 0
    toks, sample = [], None
    for s in range(0, len(items), bs):
        batch = items[s:s + bs]
        enc = tok([build(it["problem"]) for it in batch], return_tensors="pt",
                  padding=True,
                  add_special_tokens=(cfg["prompt_style"] != "chat")).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=1.1,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=stop_ids)
        for it, row in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            ids = row.tolist()
            n_tok = len(ids)
            for j, t in enumerate(ids):
                if t in stop_set:
                    n_tok = j
                    break
            text = tok.decode(row[:n_tok], skip_special_tokens=True)
            if is_correct(extract_answer(text), it["gold"]):
                correct += 1
            if n_tok >= max_new:
                capped += 1
            toks.append(n_tok)
            if sample is None:
                sample = (it, text)
        print(f"  [{condition}] {min(s+bs, len(items))}/{len(items)}", end="\r")

    results[condition] = dict(correct=correct, n=len(items), capped=capped,
                              tokens=sum(toks) / len(toks), sample=sample)
    del model
    torch.cuda.empty_cache()
    print(" " * 40, end="\r")

print(f"{'condition':<10} {'train-set acc':>15} {'cap%':>7} {'tokens':>8}")
print("-" * 44)
for c in ("before", "after"):
    r = results[c]
    print(f"{c:<10} {r['correct']:>6}/{r['n']:<3} = {r['correct']/r['n']:<4.2f} "
          f"{r['capped']/r['n']:>7.2f} {r['tokens']:>8.0f}")

b, a = results["before"]["correct"], results["after"]["correct"]
print()
if a > b + max(2, 0.1 * len(items)):
    print("  VERDICT: the adapter LEARNED its training data.")
    print("  Training is fine; the held-out null result reflects failure to")
    print("  GENERALISE, which is the paper's actual claim.")
elif a <= b:
    print("  VERDICT: no gain even on data it was trained on.")
    print("  The fine-tuning did not take. Investigate before reporting the")
    print("  held-out null -- undertraining, LoRA capacity, or target formatting.")
else:
    print("  VERDICT: marginal gain on training data. Weak learning --")
    print("  consistent with undertraining (loss was still falling at epoch 3).")

it, text = results["after"]["sample"]
print(f"\n  sample training problem (gold = {it['gold']}):")
print(f"  {' '.join(str(text).split())[:300]}")
