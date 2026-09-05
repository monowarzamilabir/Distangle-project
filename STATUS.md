# Project status — 2026-09-03

Everything is idle. Nothing is running; the GPU is free.

## Headline results

### RQ1: scale beats targeted depth (H1 refuted)

Held-out Bengali maths, base models, `before` condition:

| model | Bengali pretraining | params | accuracy | 95% CI |
|---|---|---|---|---|
| TituLM-1B-Instruct | 37B tokens | 1.2B | 0/100 = 0.000 | [0.000, 0.037] |
| TituLM-3B-Instruct | 37B tokens | 3.2B | 1/100 = 0.010 | [0.002, 0.054] |
| TituLM-1B | 37B tokens | 1.2B | 2/100 = 0.020 | [0.006, 0.070] |
| Qwen3-0.6B | incidental | 0.6B | 2/100 = 0.020 | [0.006, 0.070] |
| TigerLLM-1B | 10M tokens | 1.0B | 19/500 = 0.038 | [0.024, 0.059] |
| Qwen3-1.7B | incidental | 1.7B | 14/100 = 0.140 | [0.085, 0.221] |
| **Qwen3-4B** | incidental | 4.0B | **45/100 = 0.450** | [0.356, 0.548] |

Fisher exact, two-sided:

    Qwen3-4B    vs TigerLLM-1B          p < 0.00001  SIGNIFICANT
    Qwen3-4B    vs Qwen3-1.7B           p < 0.00001  SIGNIFICANT
    Qwen3-1.7B  vs Qwen3-0.6B           p = 0.0029   SIGNIFICANT
    TigerLLM-1B vs TituLM-1B            p = 0.5534   n.s.   <- depth does nothing

Monotonic scale curve 2% -> 14% -> 45%, every step significant. A 3,700x
difference in Bengali pretraining depth (10M vs 37B tokens) produces no
measurable difference. Contradicts TigerLLM's published claim that targeted
depth substitutes for scale, using their own model.

### RQ1b: LIMO-style elicitation produced no gain anywhere

Seven backbones, 0.6B-4B, two pretraining philosophies, two training-difficulty
regimes. No significant accuracy change in any (exact McNemar, paired):

| model | before | after | p |
|---|---|---|---|
| TigerLLM-1B | 0.038 | 0.032 | 0.690 |
| TituLM-1B | 0.020 | 0.000 | 0.500 |
| TituLM-1B-Instruct | 0.000 | 0.010 | — |
| TituLM-3B-Instruct | 0.010 | 0.010 | 1.000 |
| Qwen3-0.6B | 0.020 | 0.020 | 1.000 |
| Qwen3-1.7B | 0.140 | 0.160 | 0.791 |
| Qwen3-4B | 0.450 | 0.380 | 0.382 |

Includes Qwen3-4B, where LIMO's precondition (latent capability) demonstrably
holds at 45% baseline. LIMO was demonstrated on a 32B model; this establishes a
scale boundary on its applicability.

### What elicitation DID do: change form, not capability

- TituLM-1B: token-cap rate 100% -> 47%, format failures 17% -> 1% (both SIG)
- TigerLLM-1B: 0% -> 100% `<think>` adoption, +144 tokens (SIG)
- Qwen3-0.6B lengthened (+610 tok), Qwen3-1.7B shortened (-248 tok) — opposite
  directions from identical training
- **Bengali adherence fell significantly in 5 of 7 models** (up to -0.17)

An intervention meant to elicit Bengali reasoning made models measurably less
Bengali without making them more correct.

### Objections ruled out empirically

| objection | evidence against it |
|---|---|
| benchmark broken | Qwen3-4B scores 45% on it |
| scorer broken | validated on 12 verbatim model outputs (`python scorer.py`) |
| training didn't run | loss 1.6 -> 0.28; behaviour changed significantly |
| training data too hard | easy-subset experiment: models DID fit it (28% -> 38% train-set), held-out still fell 14% -> 8% |
| instruction-tuning confound | base AND Instruct variants both evaluated |

## The easy-subset experiment (weakest link)

Same backbone, same recipe, only the training-data difficulty band differs:

    base (no LoRA)                   14/100 = 0.140
    qwen3-1.7b      (olympiad)       16/100 = 0.160
    qwen3-1.7b-easy (easy)            8/100 = 0.080

    base       -> olympiad   p = 0.79   n.s.
    base       -> easy       p = 0.18   n.s.
    olympiad   -> easy       p = 0.077  n.s.  <- most interesting, least supported

Training on difficulty-matched easy data made held-out accuracy WORSE, with
olympiad-tier accuracy falling 0.176 -> 0.000. In-distribution learning
succeeded (train-set 28% -> 38%) and transferred negatively.

**This is the number worth strengthening.** Extending both adapters to n=250
costs ~1h and might push p below 0.05.

## Next steps, in priority order

1. `python eval_math.py --model qwen3-1.7b --condition after` and
   `--model qwen3-1.7b-easy --condition after` after raising both to
   `n_items=250` in `EVAL_META` — re-uses the first 100, evaluates only 100-249.
2. Optional: raise other models to n=250 for tighter intervals.
3. TituLM-3B (base) was never evaluated — only its Instruct variant. Include or
   drop deliberately.

## Known limitations to report

- **Qwen3-4B trained at `max_len=2048`**, truncating 104/873 traces (12%). The
  only model with a real truncation deviation. Others: 0 truncated.
- **n=100 for most models** (TigerLLM has n=500). Mixed sample sizes; CIs differ.
- **3 epochs**, loss still falling for 4/6 models at epoch 3. LIMO used ~15,
  s1 used 5. No epoch ablation was run.
- **LoRA r=16**, not full fine-tuning as in LIMO/s1.
- **Architecture confounded with depth** across families (Gemma3/Llama vs Qwen3)
  — pre-registered in the P3 validity-threats plan as a "directional comparison".
- **Training targets keep GANIT-SFT's `<think>/<answer>` markup.** For Qwen3
  `<think>` is a control token, and combining `enable_thinking=False` with a
  target that reopens one is non-canonical. Applied identically at train and
  eval, so internally valid, but may understate Qwen3.
- **Easy subset shifted source composition** (SOMADHAN 13.6% -> 31.0%), so
  difficulty and source are not fully separable in that experiment.
- **Qwen3-1.7B loss (0.1262) and TituLM-3B-Instruct loss (0.4506)** are from
  resumed runs and average only post-resume steps. NOT comparable to the others.
  Recover full curves from `_checkpoints/*/trainer_state.json` before tabulating.

## Files

    train_lora.py           training, all backbones, one shared config
    eval_math.py            generation + scoring, before/after, resumable
    scorer.py               answer extraction; `python scorer.py` self-tests
    build_easy_subset.py    builds the easy-difficulty training set
    memorisation_check.py   train-set accuracy diagnostic
    compare_conditions.py   before/after for one model, with significance
    compare_adapters.py     two adapters vs shared base
    watch.py                whole-project grid view
    track.py                follow any single log
    status.py               one-shot status

    results_v2/             predictions + eval_summary.csv + accuracy_by_difficulty.csv
    logs/                   all run logs, incl. final_summary.txt
    adapters/               8 trained adapters + training_meta.json each
    superseded/             retired notebooks + why
    environment.json        exact stack; requirements_lock.txt pins it
