# -*- coding: utf-8 -*-
"""Bengali math evaluation: before vs after LoRA elicitation.

Every number in the paper comes from this script. Supersedes
04_eval_math.ipynb and 07_eval_math_v2.ipynb.

The model registry and the Bengali instruction are imported from train_lora.py
rather than restated, so the prompt a model is evaluated on is by construction
the prompt it was trained on.

    python eval_math.py --smoke                  # 20 items, one model
    python eval_math.py --model tigerllm-1b
    python eval_math.py --all
    python eval_math.py --summary                # re-score, no generation

Safe to interrupt: every batch is fsync'd and finished items are skipped on
restart.
"""
import argparse
import csv
import gc
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from scorer import extract_answer, is_correct, bengali_ratio
# Single source of truth: same registry and same instruction text as training.
from train_lora import MODELS as TRAIN_MODELS, BENGALI_INSTRUCTION, ENABLE_THINKING

# ---- eval-only per-model metadata ------------------------------------------
# depth/params_b are for the RQ1 tables; batch is tuned to a 16GB card.
# max_new is per-model, matched to the 99th percentile of that model's OWN
# training targets. An equal token budget is not an equal opportunity when
# tokenizers differ 3x: the same Bengali reasoning trace is
#
#   tokenizer      median   p90    p99    max    >1024
#   TigerLLM          333   682   1089   1385    1%
#   TituLM            309   657    997   1458    1%
#   Qwen3             899  1768   2375   3104   42%
#
# so 1024 leaves the Bengali models room to spare while cutting off 42% of what
# Qwen3 was trained to produce. Qwen3 gets 2048; raising the Bengali models
# would only buy more degenerate text, since TituLM never emits EOS and runs to
# whatever cap it is given.
EVAL_META = {
    # All models report at n=250 for a coherent results table. TigerLLM was
    # originally run to 500; those extra items are preserved in
    # results_v2/full500/ but are not used, so every comparison in the paper
    # is over the identical 250 held-out problems.
    "tigerllm-1b": dict(depth="deep",    params_b=1.0, batch=8, max_new=1024, n_items=250),
    # n_items=100: TituLM never emits EOS, so every sequence runs the full
    # 1024-token budget -- 22s/item, i.e. ~19h for both TituLM models at
    # n=500. It scored 0/28 with a 100% cap rate and degenerate output
    # (repetition loops, hallucinated Q&A pairs, English filler), so a
    # larger sample only tightens a bound on zero. 0/100 bounds accuracy
    # below 3% (95% CI, rule of three), which supports the same conclusion.
    # Shortening max_new was rejected instead: with dozens of "উত্তর:"
    # markers in a loop, truncation changes WHICH value is extracted
    # (only 54-68% agreement), so it would alter the measurement itself.
    "titulm-1b":   dict(depth="deep",    params_b=1.2, batch=4, max_new=1024,
                        n_items=250),
    "titulm-3b":   dict(depth="deep",    params_b=3.2, batch=2, max_new=1024,
                        n_items=100),
    # Instruction-tuned: expected to emit EOS properly, unlike the base
    # models, so full n=500 and a larger batch are affordable here.
    "titulm-1b-it": dict(depth="deep",    params_b=1.2, batch=4, max_new=1024, n_items=250),
    # batch=1: at batch 2 this model saturated the card mid-run (97% VRAM,
    # 50W, frozen at item 158). A 3B model with a 170k vocabulary makes the
    # LM-head logits tensor the binding cost.
    "titulm-3b-it": dict(depth="deep",    params_b=3.2, batch=1, max_new=1024, n_items=250),
    "qwen3-0.6b":  dict(depth="shallow", params_b=0.6, batch=8, max_new=2048, n_items=250),
    # n_items=250 for the two 1.7B variants only: the olympiad-vs-easy
    # adapter contrast is the study's most interesting mechanism and its
    # least supported (p=0.077 at n=100). Extending re-uses items 0-99.
    "qwen3-1.7b":  dict(depth="shallow", params_b=1.7, batch=4, max_new=2048, n_items=250),
    "qwen3-1.7b-easy": dict(depth="shallow", params_b=1.7, batch=4,
                        max_new=2048, n_items=250),
    # batch=1: at batch 2 this died silently ~8 items in, GPU released, no
    # traceback. Same OOM signature as titulm-3b-it. Qwen3's tokenizer needs
    # ~3x more tokens for Bengali (median 1,245 vs 397), so a 2,048-token
    # generation budget on a 4B model does not leave room for two sequences.
    "qwen3-4b":    dict(depth="shallow", params_b=4.0, batch=1, max_new=2048, n_items=250),
}

# titulm-3b (base) deliberately omitted for now -- the base variants may be
# dropped from the paper, and its 100-item pass costs ~1.2h that is better
# spent on the instruction-tuned and Qwen models.
#
# n_items=100 is a FIRST PASS to get the shape of the results. Raising it
# later re-uses everything already done: items are taken as a prefix of the
# held-out file and completed ids are skipped, so n_items=250 evaluates only
# items 100-249.
ORDER = ["tigerllm-1b", "titulm-1b", "titulm-1b-it", "qwen3-0.6b",
         "qwen3-1.7b", "titulm-3b-it", "qwen3-4b"]

SEED = 42
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.1      # 04 had none; 500/500 generations ran to the cap
DO_SAMPLE = False             # greedy: deterministic and reproducible
TRUST_REMOTE_CODE = False

# Every plausible end-of-sequence marker across the three model families.
# TituLM reports eos "<|eot_id|>" -- a *chat* turn terminator -- but is prompted
# in raw completion format where that token never follows free text. Passing all
# candidates makes stopping robust regardless of the format a model expects.
STOP_TOKEN_CANDIDATES = [
    "<|end_of_text|>", "<|eot_id|>",      # Llama 3.x
    "<end_of_turn>", "<eos>",             # Gemma
    "<|im_end|>", "<|endoftext|>",        # Qwen / GPT-style
]

TIERS = ["easy", "medium", "hard", "olympiad"]


def collect_stop_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    for tok in STOP_TOKEN_CANDIDATES:
        if tok in vocab:
            ids.add(int(vocab[tok]))
    return sorted(ids)


def load_eval_items(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path} line {i} is not valid JSON: {e}")
    if not rows:
        raise SystemExit(f"{path} is empty.")

    items = [{
        "id": r.get("id", f"heldout-{i}"),
        "problem": str(r["problem"]).strip(),
        "gold": str(r["answer"]).strip(),
        "difficulty": r.get("difficulty", "unknown"),
    } for i, r in enumerate(rows)]
    return items[:limit] if limit else items


def check_overlap(items, train_path):
    """Memorisation guard: any 'after' gain on a seen problem is not elicitation."""
    p = Path(train_path)
    if not p.exists():
        print(f"WARN  training file not found ({p}); overlap check skipped.")
        return
    train_rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    train_ids = {str(r.get("id")) for r in train_rows if r.get("id") is not None}
    eval_ids = {str(it["id"]) for it in items}

    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(t).lower())).strip()

    train_problems = {norm(r.get("problem", "")) for r in train_rows}
    text_overlap = [it["id"] for it in items if norm(it["problem"]) in train_problems]
    id_overlap = train_ids & eval_ids

    print(f"Overlap check: {len(train_rows)} train rows | "
          f"id overlap {len(id_overlap)} | text overlap {len(text_overlap)}")
    if id_overlap or text_overlap:
        print("*** STOP: eval overlaps training. Any 'after' gain is partly memorisation. ***")
    else:
        print("PASS  no exact overlap (will not catch paraphrases)")


def make_prompt_fn(key, tokenizer):
    """Identical Bengali instruction for every model; only the wrapper differs."""
    style = TRAIN_MODELS[key]["prompt_style"]

    def content(problem):
        return f"{BENGALI_INSTRUCTION}\n\nসমস্যা: {problem}"

    if style == "chat":
        def fn(problem):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": content(problem)}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=ENABLE_THINKING)
            except TypeError:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": content(problem)}],
                    tokenize=False, add_generation_prompt=True)
    else:
        def fn(problem):
            return content(problem) + "\n\nসমাধান:"
    return fn


def load_done_ids(path):
    if not Path(path).exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
    return done


def append_jsonl(path, records):
    """Append + fsync. This machine loses power; a crash must not cost the batch."""
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def evaluate(key, condition, items, results_dir, adapter_root, resume=True):
    cfg = TRAIN_MODELS[key]
    meta = EVAL_META[key]
    # A model may be evaluated on a prefix of the held-out set; the file order
    # is already difficulty-balanced (28/29/26/17% vs 29/25/26/20% at n=500).
    if meta.get("n_items"):
        items = items[:meta["n_items"]]
    out_path = Path(results_dir) / f"preds_{key}_{condition}.jsonl"
    done = load_done_ids(out_path) if resume else set()
    todo = [it for it in items if str(it["id"]) not in done]

    print(f"\n{'='*66}")
    print(f"{cfg['label']}  [{condition}]   {len(todo)} to do, {len(done)} done"
          + (f"   (subsample n={meta['n_items']})" if meta.get("n_items") else ""))
    print(f"{'='*66}")
    if not todo:
        print("Nothing to do.")
        return

    adapter_dir = Path(adapter_root) / cfg["adapter"]
    if condition == "after" and not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter at {adapter_dir}. Train it first.")

    # Always the BASE tokenizer unless the adapter actually trained embeddings.
    # TigerLLM's saved adapter tokenizer carries a token at id 262144, one past
    # Gemma3-1B's embedding matrix -- loading base weights with it is what
    # crashed the original eval.
    tok_src = cfg["model_id"]
    if condition == "after":
        acfg = adapter_dir / "adapter_config.json"
        saved = json.loads(acfg.read_text(encoding="utf-8")).get("modules_to_save")
        if saved and any("embed" in str(m) or "lm_head" in str(m) for m in saved):
            tok_src = str(adapter_dir)
            print(f"  adapter trained embeddings {saved}; using its tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"     # required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation=cfg["attn"]).cuda()

    if condition == "after":
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
        print("  LoRA adapter merged.")

    model.eval()
    model.config.use_cache = True

    build_prompt = make_prompt_fn(key, tokenizer)
    stop_ids = collect_stop_ids(tokenizer)
    stop_set = set(stop_ids)
    bs = meta["batch"]
    max_new = meta["max_new"]
    print(f"  prompt_style={cfg['prompt_style']}  batch={bs}  "
          f"max_new={max_new}  "
          f"stop={[tokenizer.decode([i]) for i in stop_ids]}")

    t0, n_done = time.time(), 0
    for start in range(0, len(todo), bs):
        batch = todo[start:start + bs]
        prompts = [build_prompt(it["problem"]) for it in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=(cfg["prompt_style"] != "chat")).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
        gen = out[:, enc["input_ids"].shape[1]:]

        records = []
        for it, row in zip(batch, gen):
            ids = row.tolist()
            n_tok = len(ids)
            for j, t in enumerate(ids):
                if t in stop_set:
                    n_tok = j
                    break
            text = tokenizer.decode(row[:n_tok], skip_special_tokens=True)
            pred = extract_answer(text)
            records.append({
                "id": it["id"], "model_key": key, "model_label": cfg["label"],
                "condition": condition, "depth": meta["depth"],
                "params_b": meta["params_b"], "difficulty": it["difficulty"],
                "gold": it["gold"], "pred_answer": pred,
                "correct": bool(is_correct(pred, it["gold"])),
                "format_failure": pred is None,
                "n_generated_tokens": int(n_tok),
                "max_new_tokens": max_new,
                "hit_cap": bool(n_tok >= max_new),
                "bengali_ratio": round(bengali_ratio(text), 4),
                "completion": text,
            })

        append_jsonl(out_path, records)
        n_done += len(batch)
        if n_done % (bs * 5) < bs or n_done >= len(todo):
            rate = (time.time() - t0) / n_done
            print(f"  {n_done}/{len(todo)}  {rate:.1f}s/item  "
                  f"eta {rate*(len(todo)-n_done)/60:.0f} min")

    # Thorough teardown. A plain `del model` left enough referenced for the
    # next model to load on top of it: TituLM-1B started at 98% VRAM, spilled
    # into shared host memory and stalled before writing a single item.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  freed; VRAM now {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Wrote {out_path}")


def summarise(results_dir, keys):
    rows, strat = [], []
    for key in keys:
        cfg, meta = TRAIN_MODELS[key], EVAL_META[key]
        for condition in ("before", "after"):
            path = Path(results_dir) / f"preds_{key}_{condition}.jsonl"
            if not path.exists():
                continue
            preds = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            if not preds:
                continue
            n = len(preds)
            rows.append({
                "model_key": key, "model_label": cfg["label"],
                "depth": meta["depth"], "params_b": meta["params_b"],
                "condition": condition, "n": n,
                "n_correct": sum(1 for p in preds if p["correct"]),
                "accuracy": round(sum(1 for p in preds if p["correct"]) / n, 4),
                "format_failures": sum(1 for p in preds if p["format_failure"]),
                "hit_cap_rate": round(sum(1 for p in preds if p.get("hit_cap")) / n, 4),
                "mean_generated_tokens": round(sum(p["n_generated_tokens"] for p in preds) / n, 1),
                "mean_bengali_ratio": round(sum(p["bengali_ratio"] for p in preds) / n, 4),
            })
            buckets = defaultdict(lambda: [0, 0])
            for p in preds:
                b = buckets[p.get("difficulty", "unknown")]
                b[1] += 1
                b[0] += bool(p["correct"])
            srow = {"model": cfg["label"], "depth": meta["depth"],
                    "params_b": meta["params_b"], "condition": condition}
            for t in TIERS:
                c, m = buckets.get(t, (0, 0))
                srow[t] = round(c / m, 4) if m else None
                srow[f"{t}_n"] = m
            strat.append(srow)

    if not rows:
        print("No prediction files yet.")
        return

    outdir = Path(results_dir)
    with open(outdir / "eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(outdir / "accuracy_by_difficulty.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(strat[0].keys()))
        w.writeheader(); w.writerows(strat)

    print(f"\n{'model':<22} {'depth':<8} {'cond':<7} {'n':>5} {'acc':>7} "
          f"{'cap%':>6} {'fmt':>5} {'bn':>6}")
    print("-" * 74)
    for r in rows:
        print(f"{r['model_label']:<22} {r['depth']:<8} {r['condition']:<7} "
              f"{r['n']:>5} {r['accuracy']:>7.3f} {r['hit_cap_rate']:>6.2f} "
              f"{r['format_failures']:>5} {r['mean_bengali_ratio']:>6.2f}")

    print(f"\n{'model':<22} {'cond':<7}" + "".join(f"{t:>10}" for t in TIERS))
    print("-" * 74)
    for r in strat:
        cells = "".join(f"{r[t]:>10.3f}" if r[t] is not None else f"{'-':>10}"
                        for t in TIERS)
        print(f"{r['model']:<22} {r['condition']:<7}{cells}")
    print(f"\nWrote {outdir/'eval_summary.csv'} and {outdir/'accuracy_by_difficulty.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(TRAIN_MODELS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="20 items on tigerllm-1b only; verifies the path end to end")
    ap.add_argument("--summary", action="store_true",
                    help="re-score existing predictions, no generation")
    ap.add_argument("--condition", choices=["before", "after", "both"], default="both")
    ap.add_argument("--heldout", default="heldout_math_eval.jsonl")
    ap.add_argument("--train-data", default="curated/ganit_limo_n1000_c1-16_seed42.jsonl")
    ap.add_argument("--results-dir", default="results_v2")
    ap.add_argument("--adapter-root", default="adapters")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    keys = ORDER if (args.all or args.summary) else \
        (["tigerllm-1b"] if args.smoke else [args.model] if args.model else None)
    if keys is None:
        ap.error("pass --model KEY, --all, --smoke or --summary")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.summary:
        summarise(args.results_dir, [k for k in ORDER])
        return

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible.")

    limit = 20 if args.smoke else args.limit
    items = load_eval_items(args.heldout, limit)
    print(f"GPU {torch.cuda.get_device_name(0)} | transformers {transformers.__version__}")
    print(f"Eval items: {len(items)} | by difficulty: "
          f"{dict(Counter(it['difficulty'] for it in items))}")
    check_overlap(items, args.train_data)

    conditions = ["before", "after"] if args.condition == "both" else [args.condition]
    for key in keys:
        for condition in conditions:
            try:
                evaluate(key, condition, items, args.results_dir,
                         args.adapter_root, resume=not args.no_resume)
            except Exception as e:
                # One broken backbone must not abort the remaining passes.
                print(f"\n!!! {key} [{condition}] FAILED: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()

    summarise(args.results_dir, keys)


if __name__ == "__main__":
    main()
