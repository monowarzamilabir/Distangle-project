# P7 Update — CSE791 Group 11

Copy each section into the corresponding form field.

---

## P7 Evidence Collected or Implementation Completed

Built and ran the complete RQ1 experimental pipeline end to end.

**Data.** Froze two LIMO-style training sets from the curated Ganit pool, both
decontaminated against the held-out evaluation set by ID, exact text and numeric
fingerprint (0 overlaps in both):
- **Olympiad set** — 873 traces, LIMO's difficulty window (Qwen3-32B pass rate
  1–16 of 32; mean 3.3), 74.6% olympiad / 25.3% hard.
- **Easy set** — 1,000 traces, pass rate 28–32 of 32 (mean 31.8), 100% easy.
  Built specifically to test whether LIMO's difficulty window transfers across
  model scale.
- **Held-out evaluation set** — 500 problems from GANIT-DEV, stratified across
  easy (145) / medium (127) / hard (129) / olympiad (99).

**Models trained.** 9 LoRA adapters across 6 backbones, 4.4 GPU-hours total on a
single RTX 5060 Ti (16GB), all with an identical configuration (r=16, alpha=32,
lr 2e-4, 3 epochs, effective batch 16, seed 42):

| backbone | Bengali pretraining | final train loss |
|---|---|---|
| TigerLLM-1B-it | ~10M curated textbook tokens | 0.891 |
| TituLM-1B v2.0 | ~37B general Bengali tokens | 1.607 |
| TituLM-1B-Instruct | ~37B, instruction-tuned | 1.542 |
| TituLM-3B v2.0 | ~37B general Bengali tokens | 1.353 |
| TituLM-3B-Instruct | ~37B, instruction-tuned | 0.451* |
| Qwen3-0.6B | incidental only | 0.454 |
| Qwen3-1.7B | incidental only | 0.126* |
| Qwen3-1.7B (easy set) | incidental only | 0.291 |
| Qwen3-4B | incidental only | 0.276 |

\*resumed from checkpoint after a power interruption; these two losses average
only post-resume steps and are not comparable to the others.

**Benchmark runs.** 2,474 item-passes scored so far across 15 prediction files
(every model in both before-LoRA and after-LoRA conditions). Extension from
n=100 to n=250 per model is in progress.

**Instruments built.** A reproducible script pipeline replacing the earlier
notebooks: shared training script, evaluation harness with per-item resume,
shared answer scorer with self-tests, difficulty-stratified reporting, paired
significance testing, and an adapter-comparison tool. Environment pinned
(`requirements_lock.txt`, `environment.json`).

---

## P7 Preliminary Analysis or Results Summary

**RQ1 is answered, and H1 is refuted.**

Accuracy on the held-out set (n=100 per model at time of writing; TigerLLM
n=500):

| model | Bengali pretraining | params | before | after |
|---|---|---|---|---|
| TituLM-1B-Instruct | 37B | 1.2B | 0.000 | 0.010 |
| TituLM-3B-Instruct | 37B | 3.2B | 0.010 | 0.010 |
| TigerLLM-1B | 10M | 1.0B | 0.030 | 0.010 |
| Qwen3-0.6B | — | 0.6B | 0.020 | 0.020 |
| Qwen3-1.7B | — | 1.7B | 0.140 | 0.160 |
| **Qwen3-4B** | — | 4.0B | **0.450** | **0.380** |

**1. Scale determines performance; targeted Bengali depth does not.** The Qwen3
scale curve is monotonic and every step is significant in both conditions
(0.6B→1.7B p=0.003 before / p=0.0008 after; 1.7B→4B p<0.00001 / p=0.0007).
Meanwhile a 3,700x difference in targeted Bengali pretraining — TigerLLM's 10M
curated textbook tokens versus TituLM's 37B general tokens — produces **no
significant difference** (Fisher exact p=0.553). Qwen3-4B, with no Bengali-
specific pretraining at all, outperforms the best Bengali-adapted model by 12x
(p<0.00001). This directly contradicts TigerLLM's published claim that targeted
depth can substitute for scale, tested on a Bengali benchmark using their own
released model.

**2. LIMO-style elicitation produced no significant accuracy gain on any
backbone.** Seven models spanning 0.6B–4B, two pretraining philosophies and two
training-difficulty regimes; every before/after contrast is non-significant by
exact McNemar. This includes Qwen3-4B, where LIMO's stated precondition (latent
capability) demonstrably holds at 45% baseline accuracy. LIMO was demonstrated
on a 32B model; this indicates a scale boundary on its applicability.

**3. Elicitation reliably changes output form without changing capability.**
TituLM-1B's token-cap rate fell 100%→47% and format failures 17%→1%; TigerLLM
adopted the training traces' `<think>` format on 100% of items and generated 144
more tokens; Qwen3-0.6B lengthened by 610 tokens while Qwen3-1.7B shortened by
248 from identical training. Most notably, **Bengali-character adherence fell
significantly in 5 of 7 models** (up to −0.17) — an intervention intended to
elicit Bengali reasoning made models measurably less Bengali without making them
more correct.

**4. Difficulty-matched training does not rescue the null.** Training Qwen3-1.7B
on the easy set instead of the olympiad set improved training-set accuracy
(28%→38%, versus 6%→8% for the olympiad set, confirming the models can fit
properly-calibrated data) but held-out accuracy *fell* (14%→8%), with
olympiad-tier accuracy dropping to zero. Successful in-distribution learning
transferred negatively. (Currently p=0.077; being extended to n=250 for power.)

**5. A measured tokenizer tax.** Qwen3 requires ~3.1x more tokens than TituLM
for identical Bengali text (median 1,245 vs 397 for the same traces), close to
the ~3.5x reported by Yong et al. TituLM's extended Bengali vocabulary delivers
real representational efficiency even on the task where it fails outright — and
that cost determines which models can be trained on consumer hardware at all.

---

## P7 Current Problems and Mitigation Plan

**RQ2 blocked — B-REASO still unavailable.** No work has begun on the
cross-domain transfer question. Mitigation: attempt direct download; if it
remains unavailable, scope the paper to RQ1 and report RQ2 as future work rather
than presenting it thinly.

**RQ3 not started.** The confidence-based stopping experiments have no
implementation. Given remaining time, the realistic plan is to defend RQ1
thoroughly rather than present three partial research questions.

**Statistical power.** Most models are currently evaluated at n=100, giving wide
confidence intervals (Qwen3-4B: 0.356–0.548). Extension to n=250 for every model
is running now. The easy-vs-olympiad adapter contrast (p=0.077) is the specific
result most in need of the additional items.

**Undertraining not ruled out.** Training loss was still falling at epoch 3 for
four of six backbones; LIMO used ~15 epochs and s1 used 5. No epoch ablation was
run, so "elicitation does not work at this scale" cannot yet be fully separated
from "3 epochs was insufficient." This is stated as a limitation.

**One per-model deviation.** Qwen3-4B was trained at max_seq_length 2048 rather
than 4096 because a 4B model at 4096 exhausted the 16GB card; this truncated 104
of 873 traces (12%). Every other model truncated 0. Reported explicitly.

**Architecture confounded with depth across families** (Gemma3/Llama-3.2 vs
Qwen3). Pre-registered in the P3 validity-threats plan and reported as a
directional comparison rather than a causal claim.

**Two training losses are not comparable** (Qwen3-1.7B, TituLM-3B-Instruct):
both resumed from checkpoints after power interruptions, so the reported figure
averages only post-resume steps. Full curves are recoverable from the saved
trainer states before tabulation.

---

## P7 Reproducibility or Documentation Status

**Strong.**

- Every result is produced by version-controlled scripts, not notebooks. The
  earlier notebooks are retired to `superseded/` with a README explaining why
  each was replaced, including two defects that invalidated the original
  evaluation numbers.
- The evaluation script imports its model registry and prompt text from the
  training script, so a model is evaluated on exactly the prompt it was trained
  on by construction rather than by manual synchronisation.
- Each adapter carries a `training_meta.json` recording its full configuration,
  final loss, runtime, peak VRAM and rendered example prompt.
- The answer scorer has an executable self-test suite built from verbatim model
  outputs (`python scorer.py`), added after two extraction bugs were found by
  inspecting real generations rather than synthetic cases.
- Environment pinned to exact versions; the GPU, driver and CUDA build are
  recorded in `environment.json`.
- Known limitation: `torch.use_deterministic_algorithms()` is not enabled, so
  losses reproduce to roughly three decimal places rather than exactly. Reported
  differences are orders of magnitude larger than this variance.

---

## P7 Next Steps before Full Draft

1. **Complete the n=250 extension** for all models (running; ~2:30 remaining),
   then regenerate all tables and significance tests at uniform sample size.
2. **Recover comparable training losses** for Qwen3-1.7B and TituLM-3B-Instruct
   from their saved trainer states before any loss table is finalised.
3. **Decide RQ2/RQ3 scope** — pursue B-REASO access, or formally narrow the
   paper to RQ1 and reframe the remaining questions as future work.
4. **Draft the results and discussion sections** around the three findings that
   are already statistically supported: the scale-versus-depth result, the
   universal elicitation null, and the form-without-capability characterisation.
5. **Optional if time permits** — an epoch ablation (6 epochs on one backbone,
   ~30 min) to close the undertraining objection, and evaluating GanitLLM's
   released checkpoint on the same held-out set for a direct comparison between
   full fine-tuning with RL and minimal-data LoRA.
