# -*- coding: utf-8 -*-
"""LoRA elicitation training for every backbone in the RQ1 grid.

Supersedes train_qwen3.py and notebooks 01/02/03: one script, one config, so
"all six models were trained identically" is true by construction rather than
by careful copy-paste across four files.

The only per-model differences are the ones that MUST differ -- the prompt
wrapper each model was trained for, and the attention kernel its architecture
needs. The Bengali instruction text, LoRA hyperparameters, optimiser settings,
seed and sequence length are shared.

    python train_lora.py --model titulm-1b
    python train_lora.py --model tigerllm-1b
    python train_lora.py --all
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from torch.utils.data import Dataset

# ===========================================================================
# MODEL REGISTRY -- keep in sync with MODELS in 07_eval_math_v2.ipynb
# prompt_style: "raw" for completion/base models, "chat" for instruction-tuned.
#   Chosen empirically per model (see the probe results in the paper's
#   methods section), not assumed.
# ===========================================================================
MODELS = {
    "tigerllm-1b": dict(model_id="md-nishat-008/TigerLLM-1B-it",
                        adapter="tigerllm-1b", label="TigerLLM-1B",
                        prompt_style="raw", attn="eager"),
    "titulm-1b":   dict(model_id="hishab/titulm-llama-3.2-1b-v2.0",
                        adapter="titulm-1b-v2", label="TituLLM-1B (v2.0)",
                        prompt_style="raw", attn="sdpa"),
    "titulm-3b":   dict(model_id="hishab/titulm-llama-3.2-3b-v2.0",
                        adapter="titulm-3b", label="TituLLM-3B (v2.0)",
                        prompt_style="raw", attn="sdpa"),
    # Instruction-tuned TituLM. TigerLLM-1B-it is instruction-tuned (100k
    # GPT-4o/Claude pairs) while TituLM base is not, so a TigerLLM-vs-base
    # comparison confounds pretraining depth with instruction tuning.
    # These variants remove that confound; keeping the base models too
    # gives an instruction-tuning ablation within the TituLM family.
    "titulm-1b-it": dict(model_id="hishab/titulm-llama-3.2-1b-v2.0-Instruct-v1.0",
                        adapter="titulm-1b-instruct",
                        label="TituLLM-1B-Instruct",
                        prompt_style="chat", attn="sdpa"),
    # max_len=2048: a 3B model at 4096 saturates the 16GB card and spills
    # (observed 7.2s/it -> 21.5s/it, power 145W -> 41W). Unlike Qwen3-4B
    # this costs NOTHING: TituLM's longest prompt+target is 1500 tokens
    # (p99 = 997), so the cap never binds and 0 examples are truncated at
    # either setting -- the training data is byte-identical to a 4096 run.
    "titulm-3b-it": dict(model_id="hishab/titulm-llama-3.2-3b-v2.0-Instruct-v1.0",
                        adapter="titulm-3b-instruct",
                        label="TituLLM-3B-Instruct",
                        prompt_style="chat", attn="sdpa", max_len=2048),
    "qwen3-0.6b":  dict(model_id="Qwen/Qwen3-0.6B",
                        adapter="qwen3-0.6b", label="Qwen3-0.6B",
                        prompt_style="chat", attn="sdpa"),
    "qwen3-1.7b":  dict(model_id="Qwen/Qwen3-1.7B",
                        adapter="qwen3-1.7b", label="Qwen3-1.7B",
                        prompt_style="chat", attn="sdpa"),
    # Same backbone, same recipe, DIFFERENT training-data difficulty band.
    # Trained on curated/ganit_easy_n1000_c28-32_seed42.jsonl (100% easy,
    # mean 31.8/32 pass rate) instead of the LIMO window (74.6% olympiad,
    # mean 3.3/32). Tests whether the failure to fit the training data is
    # caused by difficulty calibrated to a 32B judge. Separate adapter so
    # the original result is preserved.
    # max_len=3072, not the global 4096. Qwen3-1.7B genuinely saturates a
    # 16GB card at 4096: with NO other process on the GPU it still reached
    # 15.6GB / 39W and froze at step 19/189. The binding cost is the LM-head
    # logits tensor, seq_len x 151936 vocab x 4 bytes -- 2.5GB at 4096 versus
    # 1.9GB at 3072 -- plus its backward pass.
    #
    # 3072 truncates 0 of 1000 (longest prompt+target is 2347 tokens), so the
    # training data is byte-identical to a 4096 run and remains directly
    # comparable to the olympiad-trained adapter, which used 4096 but also
    # needed two checkpoint resumes to get through it.
    "qwen3-1.7b-easy": dict(model_id="Qwen/Qwen3-1.7B",
                        adapter="qwen3-1.7b-easy", label="Qwen3-1.7B-easy",
                        prompt_style="chat", attn="sdpa", max_len=3072),
    # max_len override: Qwen3-4B will not train reliably at 4096 on a 16GB
    # card. Peak VRAM during the backward pass on the longest sequences (3544
    # tokens) exceeds capacity, at which point Windows' WDDM driver spills into
    # shared host memory and throughput collapses -- observed twice, 45s/it ->
    # 199s/it and then a full stall, with power draw falling from 154W to 41W
    # while nvidia-smi still reported 100% utilisation. Periodic
    # torch.cuda.empty_cache() delayed it but did not prevent it.
    #
    # At 2048 this model truncates 104 of 873 traces (12%). That is a real
    # per-model deviation and MUST be reported: Qwen3-4B trained on slightly
    # less of each long trace than the other five backbones. It is the honest
    # trade for keeping its before/after elicitation delta on this hardware.
    "qwen3-4b":    dict(model_id="Qwen/Qwen3-4B",
                        adapter="qwen3-4b", label="Qwen3-4B",
                        prompt_style="chat", attn="sdpa", max_len=2048),
}

# Fastest first, so a power cut costs the least finished work.
ORDER = ["titulm-1b", "tigerllm-1b", "titulm-1b-it", "qwen3-0.6b",
         "qwen3-1.7b", "titulm-3b", "titulm-3b-it", "qwen3-4b"]

# Why MAX_SEQ_LENGTH = 4096 and not 2048.
#
# Token counts for the SAME 873 Bengali traces differ enormously by tokenizer:
#
#   tokenizer      vocab    median   p95    max    >2048
#   TituLM-1B     170,497      397   873   1500        0
#   TigerLLM-1B   262,145      428   933   1434        0
#   Qwen3         151,669     1245  2396   3544      104
#
# Qwen3 needs ~3.1x more tokens for identical Bengali text -- the tokenizer tax
# Yong et al. report as ~3.5x. At 2048 that silently truncated 104 of Qwen3's
# traces (12%) while truncating none of the Bengali models', so the two arms of
# the comparison would have trained on different amounts of data. 4096 clears
# Qwen3's longest trace (3544), giving every model zero truncation.
#
# This costs the Bengali models nothing: per_device_batch_size=1 with dynamic
# padding means the cap only truncates, it never pads up to itself.

# ===========================================================================
# SHARED CONFIG -- identical for every model
# ===========================================================================
SEED = 42
TRAIN_TARGET = "messages"        # full Bengali reasoning trace, not bare answer

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16            # effective batch = 16
MAX_SEQ_LENGTH = 4096            # see note below
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 5
# 10, not 50: mains power here is unreliable, and a checkpoint costs ~300MB and
# 1-3s to write (<1% overhead) while capping worst-case lost work at ~10 steps.
# save_total_limit=2 bounds disk use regardless of frequency.
SAVE_STEPS = 10
GRADIENT_CHECKPOINTING = True
BF16 = True
TRUST_REMOTE_CODE = False
ENABLE_THINKING = False          # Qwen3 only; identical intervention across models

BENGALI_INSTRUCTION = "নিচের গাণিতিক সমস্যাটি সমাধান করো। বাংলায় উত্তর দাও।"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_pairs(data_path):
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{data_path} line {i} is not valid JSON: {e}")

    ids = [r["id"] for r in rows if r.get("id") is not None]
    if len(ids) - len(set(ids)):
        raise SystemExit("Duplicate IDs in the training set.")

    pairs, skipped = [], 0
    for r in rows:
        problem = str(r.get("problem") or "").strip()
        target = ""
        if TRAIN_TARGET == "bengali_solution":
            target = str(r.get("bengali_solution") or "").strip()
        else:
            for msg in reversed(r.get("messages") or []):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    target = str(msg.get("content") or "").strip()
                    break
        if not problem or not target:
            skipped += 1
            continue
        pairs.append({"id": r.get("id"), "problem": problem, "target": target})

    assert all("english_solution" not in p for p in pairs), \
        "english_solution must never enter the training pairs"
    return pairs, len(rows), skipped


class ElicitationDataset(Dataset):
    """Prompt tokens masked to -100; loss on the target only."""

    def __init__(self, pairs, tokenizer, build_prompt, max_len, chat):
        self.features = []
        self.truncated = 0
        for p in pairs:
            # A chat template emits its own special tokens; a raw prompt needs
            # the tokenizer to add BOS.
            prompt_ids = tokenizer(build_prompt(p["problem"]),
                                   add_special_tokens=not chat)["input_ids"]
            target_ids = tokenizer(("" if chat else " ") + p["target"],
                                   add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [tokenizer.eos_token_id]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            if len(input_ids) > max_len:
                self.truncated += 1
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

            if all(l == -100 for l in labels):
                continue

            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class ReclaimVRAM(transformers.TrainerCallback):
    """Release cached-but-unused CUDA blocks every N steps.

    Qwen3-4B at 4096 sits at ~97% of a 16GB card. Variable sequence lengths
    fragment the caching allocator, and once Windows' WDDM driver starts
    spilling into shared (host) memory the run degrades and never recovers:
    observed 38s/it -> 199s/it over ~40 steps, with power draw falling from
    149W to 45W (the GPU reports 100% utilisation while actually waiting on
    PCIe transfers). Freeing the cache periodically keeps the allocator inside
    real VRAM. Costs a little reallocation per call, far less than the spill.
    """

    def __init__(self, every=5):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            torch.cuda.empty_cache()


def train_one(key, data_path, adapter_root):
    cfg = MODELS[key]
    chat = cfg["prompt_style"] == "chat"

    print(f"\n{'='*66}")
    print(f"{cfg['label']}  ({cfg['model_id']})   prompt_style={cfg['prompt_style']}")
    print(f"{'='*66}")

    max_len = cfg.get("max_len", MAX_SEQ_LENGTH)
    if max_len != MAX_SEQ_LENGTH:
        print(f"NOTE  max_seq_length={max_len} for this model "
              f"(global is {MAX_SEQ_LENGTH}) -- see registry comment")

    set_all_seeds(SEED)
    pairs, n_rows, skipped = load_pairs(data_path)
    print(f"Loaded {n_rows} rows -> {len(pairs)} usable (skipped {skipped})")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"      # training; eval uses left

    def user_content(problem):
        return f"{BENGALI_INSTRUCTION}\n\nসমস্যা: {problem}"

    def build_prompt(problem):
        if not chat:
            return user_content(problem) + "\n\nসমাধান:"
        kw = {}
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content(problem)}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING)
        except TypeError:
            # Gemma/Llama templates do not accept enable_thinking.
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content(problem)}],
                tokenize=False, add_generation_prompt=True, **kw)

    _probe = "৩ + ৫ = কত?"
    print(f"Vocab {len(tokenizer)} | eos {tokenizer.eos_token!r} "
          f"({tokenizer.eos_token_id}) | Bengali probe "
          f"{len(tokenizer(_probe)['input_ids'])} tokens")

    set_all_seeds(SEED)
    ds = ElicitationDataset(pairs, tokenizer, build_prompt, max_len, chat)
    print(f"Training examples: {len(ds)} | truncated at {max_len}: {ds.truncated}")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(torch.cat([b["input_ids"],
                                        torch.full((n,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"],
                                     torch.full((n,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long),
                                   torch.zeros(n, dtype=torch.long)]))
        return {"input_ids": torch.stack(input_ids),
                "labels": torch.stack(labels),
                "attention_mask": torch.stack(attn)}

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16 if BF16 else torch.float16,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation=cfg["attn"], device_map=None,
    ).cuda()
    print(f"Loaded in {time.time()-t0:.1f}s | {model.config.model_type} | "
          f"{sum(p.numel() for p in model.parameters())/1e9:.3f}B params")

    from peft import LoraConfig, get_peft_model, TaskType
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
        task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise SystemExit("No trainable parameters -- LORA_TARGET_MODULES mismatch.")

    adapter_dir = Path(adapter_root) / cfg["adapter"]
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_dir / "_checkpoints"),
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO, lr_scheduler_type=LR_SCHEDULER,
            bf16=BF16, fp16=not BF16,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            logging_steps=LOGGING_STEPS, save_strategy="steps",
            save_steps=SAVE_STEPS, save_total_limit=2,
            report_to=[], seed=SEED, data_seed=SEED,
            remove_unused_columns=False),
        train_dataset=ds,
        data_collator=collate,
        callbacks=[ReclaimVRAM(every=5)],
    )

    set_all_seeds(SEED)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Resume from the newest checkpoint if one survived an interrupted run.
    # This machine loses power often and a 3-hour run restarting from zero is
    # the difference between finishing tonight and not.
    ckpts = sorted(adapter_dir.glob("_checkpoints/checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    resume = str(ckpts[-1]) if ckpts else None
    if resume:
        print(f"Resuming from {resume}")

    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nFinished in {elapsed/60:.1f} min | loss {result.training_loss:.4f} "
          f"| peak VRAM {peak_gb:.2f} GB")

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "model_label": cfg["label"], "model_id": cfg["model_id"],
        "adapter_name": cfg["adapter"],
        "method": "LoRA (PEFT) on bf16 base weights -- not QLoRA, no 4-bit",
        "quantization": None, "seed": SEED,
        "train_target_field": TRAIN_TARGET, "data_path": str(data_path),
        "n_examples": len(ds), "n_truncated": ds.truncated,
        "prompt_style": cfg["prompt_style"],
        "bengali_instruction": BENGALI_INSTRUCTION,
        "enable_thinking": ENABLE_THINKING if chat else None,
        "example_prompt": build_prompt(pairs[0]["problem"]),
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGET_MODULES},
        "training": {
            "learning_rate": LEARNING_RATE, "num_epochs": NUM_EPOCHS,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS,
            "max_seq_length": max_len, "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler": LR_SCHEDULER, "weight_decay": WEIGHT_DECAY,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "precision": "bf16" if BF16 else "fp16"},
        "results": {
            "final_train_loss": float(result.training_loss),
            "train_runtime_seconds": round(elapsed, 1),
            "peak_vram_gb": round(peak_gb, 3)},
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "trainable_params": int(trainable), "total_params": int(total),
            "model_type": model.config.model_type},
    }
    with open(adapter_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Adapter saved to {adapter_dir}")

    del model, trainer
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--all", action="store_true",
                    help=f"train, in order: {ORDER}")
    ap.add_argument("--data", default="curated/ganit_limo_n1000_c1-16_seed42.jsonl")
    ap.add_argument("--adapter-root", default="adapters")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a matching adapter already exists")
    args = ap.parse_args()

    if not args.model and not args.all:
        ap.error("pass --model KEY or --all")

    _ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if _ver < (4, 51):
        raise SystemExit(f"transformers {transformers.__version__} cannot load Qwen3.")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")

    print(f"GPU {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB | "
          f"transformers {transformers.__version__}")
    print(f"Seed {SEED} | max_len {MAX_SEQ_LENGTH} | target {TRAIN_TARGET}")

    todo = ORDER if args.all else [args.model]
    for key in todo:
        # training_meta.json is written only after a run completes and saves,
        # so its presence is a reliable "this one is done" marker. Lets the
        # whole queue be restarted after a power cut without redoing work.
        done_marker = Path(args.adapter_root) / MODELS[key]["adapter"] / "training_meta.json"
        if done_marker.exists() and not args.force:
            meta = json.loads(done_marker.read_text(encoding="utf-8"))
            want_len = MODELS[key].get("max_len", MAX_SEQ_LENGTH)
            if meta.get("training", {}).get("max_seq_length") == want_len:
                print(f"\nSKIP {key}: already trained at max_len={MAX_SEQ_LENGTH} "
                      f"(loss {meta['results']['final_train_loss']:.4f}). "
                      f"Use --force to retrain.")
                continue
            print(f"\n{key}: existing adapter was trained at max_len="
                  f"{meta.get('training', {}).get('max_seq_length')}, "
                  f"retraining at {MAX_SEQ_LENGTH}")
        try:
            train_one(key, data_path, args.adapter_root)
        except Exception as e:
            # One failed backbone must not abort the rest of the queue.
            print(f"\n!!! {key} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    print("\nAll requested training runs finished.")


if __name__ == "__main__":
    main()
