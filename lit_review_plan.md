# Literature review restructure — dropping RQ2/RQ3, adding for the new findings

> **Verify every citation below before using it.** The "add" suggestions are from
> memory: titles, authors, years and venues must be checked against the actual
> papers. I am flagging my confidence per entry, but confidence is not
> verification.

---

## 1. DROP — RQ3-only (test-time scaling, confidence, calibration)

These exist solely to motivate RQ3. With RQ3 cut, Sections 2.6 and 2.7 of your
current draft go entirely.

| # | paper | why it was there |
|---|---|---|
| 1 | Atom of Thoughts | RQ3 — Markovian test-time scaling |
| 2 | Towards Reasoning Era (survey) | RQ3 — long-CoT survey |
| 3 | CaTS | RQ3 — calibrated test-time scaling |
| 6 | Revisiting Test-Time Scaling of o1-like Models | RQ3 — overthinking |
| 7 | Snell et al., Scaling Test-Time Compute | RQ3 — compute allocation |
| 8 | TOPS (Thinking-Optimal Scaling) | RQ3 — optimal CoT length |
| 18 | MetaFaith | RQ3 — verbalised confidence |
| 19 | Thought Calibration | RQ3 — internal probes |
| 20 | Kadavath et al., LMs (Mostly) Know What They Know | RQ3 — self-evaluation |

**9 papers dropped.**

One judgement call: **Snell et al.** and **Revisiting Test-Time Scaling** could
be retained in a single sentence if you want to note that "more compute at
inference" is a separate and also-contested lever you did not test. Optional.

---

## 2. KEEP — all 12 remain load-bearing

| # | paper | now supports |
|---|---|---|
| 4 | **LIMO** | The hypothesis under test. Its precondition (latent capability) is what Qwen3-4B satisfies at 45% and the Bengali models do not. |
| 5 | **s1** | Second instance of minimal-data elicitation; your epoch/method comparison point (5 epochs, full FT vs your 3 epochs, LoRA). |
| 9 | Alnazi et al., low-resource LLM eval | Bengali task-performance baseline; CoT instability. |
| 10 | **GanitLLM** | Your dataset source, and the full-FT + CGRPO contrast to your minimal-data LoRA. |
| 11 | **TigerLLM** | The targeted-depth claim your results contradict. |
| 12 | **TituLLM** | The 37B general-corpus arm. Its "helps System-1 not System-2" split now reads as a weaker version of your result. |
| 13 | Qi et al., XReasoning | The elicitation tax — directly supports your Bengali-adherence drop across 5 of 7 models. |
| 14 | LESS | Data selection by functional relevance; frames your difficulty-band experiment. |
| 15 | Yong et al., Crosslingual Test-Time Scaling | Scale claim **and** the ~3.5x Bengali tokenizer cost your 3.1x measurement corroborates. |
| 16 | SOMADHAN | Bengali math CoT; also 31% of your easy subset. |
| 17 | BenNumEval | Bengali numerical reasoning; the 4.7% distilled-70B collapse is a useful precedent for near-floor results. |
| 21 | MathMist | Qwen scaling 0.6B→14B shrinking the Bengali gap — the closest published analogue to your scale curve. |

---

## 3. ADD — grouped by which of your findings they support

Target: 12 keep + 10 add = **22 papers**.

### 3a. Scale determines performance (your headline)

Your scale curve (2% → 14% → 45%) needs theoretical grounding.

- **Kaplan et al. 2020, "Scaling Laws for Neural Language Models"** — the
  canonical scaling-law reference. *(confident)*
- **Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models"
  (Chinchilla)** — compute-optimal scaling; relevant because TigerLLM's 10M
  tokens is far off any compute-optimal frontier. *(confident)*
- **Wei et al. 2022, "Emergent Abilities of Large Language Models"** — your
  0.6B→1.7B→4B jump from 2% to 45% looks like an emergence curve. *(confident)*
- *Optional counterpoint:* **Schaeffer et al. 2023, "Are Emergent Abilities of
  Large Language Models a Mirage?"** — argues emergence is a metric artifact.
  Worth citing precisely because exact-match accuracy is the metric they
  critique. *(confident)*

### 3b. LoRA vs full fine-tuning (your main methodological limitation)

You used LoRA r=16; LIMO and s1 used full fine-tuning. You must address this.

- **Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"** —
  the method itself. *(certain)*
- **Biderman et al. 2024, "LoRA Learns Less and Forgets Less"** — the single
  most useful addition to your paper. It reports that LoRA underperforms full
  fine-tuning on learning new capabilities while better preserving base
  behaviour. That is *exactly* your result: form learned, capability not, base
  performance largely intact. *(fairly confident — verify authors/venue)*

### 3c. Tokenizer cost for low-resource languages (your 3.1x measurement)

- **Ahia et al. 2023, "Do All Languages Cost the Same? Tokenization in the Era
  of Commercial Language Models"** — token-count disparity across languages.
  *(confident)*
- **Petrov et al. 2023, "Language Model Tokenizers Introduce Unfairness Between
  Languages"** — quantifies the same inequity. Together these turn your
  tokenizer finding from an observation into a contribution to an existing
  line of work. *(confident)*

### 3d. Continued pretraining and forgetting (the TituLM story)

TituLM's 37B tokens of Bengali continued pretraining coincides with near-zero
maths, and your base-vs-Instruct ablation showed the degeneracy was largely an
instruction-tuning absence rather than forgetting. Both need a frame.

- **Gururangan et al. 2020, "Don't Stop Pretraining: Adapt Language Models to
  Domains and Tasks"** — the DAPT/TAPT framework TituLM and TigerLLM both sit
  inside. *(certain)*
- **Luo et al. 2023, "An Empirical Study of Catastrophic Forgetting in Large
  Language Models During Continual Fine-tuning"** — supports the forgetting
  hypothesis you raise and then partially rule out. *(moderately confident —
  verify; several similar-titled papers exist)*

### 3e. Data selection and difficulty (your easy-subset experiment)

- **Zhou et al. 2023, "LIMA: Less Is More for Alignment"** — the direct
  ancestor of LIMO's claim. Citing it strengthens the framing that
  "less is more" is a family of claims, not one paper. *(certain)*

### 3f. Model provenance (you must cite what you actually used)

At minimum cite the technical reports for the backbones:

- **Qwen3 technical report (Yang et al., 2025)** *(confident it exists —
  verify exact citation)*
- **Gemma 3 technical report (Google DeepMind, 2025)** — TigerLLM's base
  *(confident)*
- **Llama 3 (Dubey et al., 2024, "The Llama 3 Herd of Models")** — TituLM's
  base *(certain)*

These three are not optional. Your central claim is about pretraining
differences between these families; you cannot compare them without citing
what they are.

### 3g. Benchmark provenance (optional but cheap)

- **Shi et al. 2022, "Language Models are Multilingual Chain-of-Thought
  Reasoners"** — the MGSM paper that Bn-MGSM derives from. *(certain)*
- **Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems"
  (GSM8K)** — SOMADHAN derives from GSM8K, which is 31% of your easy subset.
  *(certain)*

---

## 4. Proposed new section structure

Replace Sections 2.6 and 2.7 (the RQ3 material) with two sections built around
what you actually found.

**2.1 The Low-Resource Reasoning Baseline**
Alnazi, BenNumEval, SOMADHAN, MathMist — unchanged, minus Kadavath.

**2.2 The Backbone Debate: Scale vs Targeted Depth**
TigerLLM, TituLLM, Yong et al., MathMist — plus Kaplan, Hoffmann, Wei
(+ Schaeffer as counterpoint). This is now your central section and should
grow, not shrink. Add the base-model technical reports here.

**2.3 The Less-Is-More Elicitation Paradigm**
LIMO, s1, LESS — plus LIMA as the ancestor. Note that all demonstrations are at
>=32B and in high-resource languages: that gap is your contribution.

**2.4 The Elicitation Tax: Native Reasoning vs English-Pivot**
Qi et al., Yong et al., BenNumEval, SOMADHAN — unchanged. Now also the frame for
your Bengali-adherence decline.

**2.5 (NEW) Adaptation Method and Its Limits**
Hu et al. (LoRA), Biderman et al. (LoRA learns less), Gururangan (DAPT), Luo
(forgetting), GanitLLM (full FT + RL). This section justifies your method,
states its limitation, and sets up the GanitLLM contrast.

**2.6 (NEW) The Cost of Bengali Tokenization**
Ahia, Petrov, Yong et al., TituLLM's extended vocabulary. Short section, but it
is a genuine measured contribution and gives your accessibility argument a home.

**2.7 Gap Statement**
Rewrite for RQ1 only.

---

## 5. Count check

    keep                    12
    scale (3a)               3   (+1 optional counterpoint)
    LoRA (3b)                2
    tokenizer (3c)           2
    continued PT (3d)        2
    data selection (3e)      1
    model reports (3f)       3
    benchmarks (3g)          2   (optional)
    ------------------------------
    core total              25   (28 with all optionals)

Comfortably over 20 with room to drop the ones you cannot verify.

---

## 6. What to do about the dropped RQ2/RQ3 papers

Do not delete them from the tracker — the tracker documents your search, and
the effort was real. In the paper, either:

- **Cut them entirely** and note in Limitations that confidence-based stopping
  (RQ3) and cross-domain transfer (RQ2) were scoped out; or
- **Keep one paragraph** in Future Work citing 2–3 of them (Snell, CaTS,
  Thought Calibration) to show the questions were considered and deliberately
  deferred rather than overlooked.

The second reads better to a supervisor who saw your P3 proposal listing three
RQs.
