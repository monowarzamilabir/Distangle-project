# Data dictionary

Every field in every file the analysis reads or writes.

---

## 1. Source corpus — `dipta007/Ganit`, SFT split (16,868 rows)

| Field | Type | Allowed values | Unit | Notes |
|---|---|---|---|---|
| `problem` | string | non-empty | — | Bengali problem statement. Numerals appear in both Bengali (০–৯) and Western (0–9) script depending on upstream source |
| `source_name` | categorical | `numina-math-cot-bn`, `SOMADHAN`, `mCoT-MATH-bn`, `s1k-32-Bangla` | — | 13,160 / 3,487 / 193 / 28 rows respectively |
| `id` | integer | ≥ 0 | — | Upstream identifier, unique within the split |
| `bengali_solution` | string | may be very short | — | Final answer only, not a worked solution. Median length 2 characters |
| `english_solution` | string | may be very short | — | Same value in Western numerals |
| `correct_counts` | integer | 0–32 | passes | **The difficulty measure.** Number of times Qwen3-32B solved the problem in 32 sampled attempts. Lower is harder |
| `difficulty` | categorical | `easy`, `medium`, `hard`, `olympiad` | — | Upstream tag. 10,065 / 134 / 258 / 6,411 rows. Note this disagrees with `correct_counts`: a third of the corpus is tagged easy or olympiad yet scores 0/32 |
| `messages` | list of dict | roles `user`, `assistant` | — | The assistant turn is the full Bengali reasoning trace and is the training target. Formatted as `<think>…</think>` then `<answer>…</answer>` |
| `row_id` | integer | 0–16,867 | — | Position in the split |

**Missing values:** none. The corpus was checked for nulls and for empty
strings in `problem`, `bengali_solution` and `english_solution`; all clean.

---

## 2. Curated training sets — `curated/*.jsonl`

`ganit_limo_n1000_c1-16_seed42.jsonl` (873 rows, main experiment)
`ganit_easy_n1000_c28-32_seed42.jsonl` (1,000 rows, difficulty control)

Same fields as the source corpus, minus `row_id`, plus nothing new. The two
files differ only in the `correct_counts` band they draw from:

| File | Band | Mean pass rate | Difficulty mix |
|---|---|---|---|
| `..._c1-16_...` | 1 ≤ `correct_counts` ≤ 16 | 3.3 / 32 | 74.6% olympiad, 25.3% hard |
| `..._c28-32_...` | `correct_counts` ≥ 28 | 31.8 / 32 | 100% easy |

`correct_counts = 0` is excluded from both. A third of the corpus sits in that
band and was never solved by a 32B model in 32 attempts, so we treat it as
unreliable rather than merely hard.

---

## 3. Held-out evaluation set — `heldout/heldout_ganit_dev.jsonl` (500 rows)

| Field | Type | Allowed values | Notes |
|---|---|---|---|
| `id` | string | `ganit_dev-<n>` | Unique |
| `problem` | string | ≥ 15 characters | Bengali |
| `gold` | string | ≤ 60 characters | Short final value, Bengali or Western numerals |
| `difficulty` | categorical | `easy`, `medium`, `hard`, `olympiad` | 145 / 127 / 129 / 99 across the full 500 |

Sampled with seed 42 and shuffled. **Only the first 250 rows are scored** for
every model (first 100 for TituLLM-1B base). Because the file is shuffled, that
subset preserves the difficulty mix: 69 easy, 69 medium, 67 hard, 45 olympiad.

Decontaminated against the 873 training traces by identifier match, exact
normalised text match, and numeric fingerprint plus Jaccard overlap. One item
matched and was removed.

---

## 4. Model predictions — `results_v2/preds_<model>_<condition>.jsonl`

One row per evaluated item. This is the file every reported number derives from.

| Field | Type | Allowed values | Unit | Notes |
|---|---|---|---|---|
| `id` | string | matches held-out `id` | — | Join key across conditions |
| `model_key` | string | see `eval_math.py` `MODELS` | — | e.g. `qwen3-4b` |
| `model_label` | string | — | — | Display name used in tables |
| `condition` | categorical | `before`, `after` | — | `before` = released weights; `after` = weights + LoRA adapter |
| `depth` | categorical | `deep`, `shallow` | — | Whether the backbone had Bengali-specific continued pretraining |
| `params_b` | float | 0.6–4.0 | billions | Total parameters, not trainable |
| `difficulty` | categorical | `easy`, `medium`, `hard`, `olympiad` | — | Carried from the held-out set |
| `gold` | string | — | — | Reference answer |
| `pred_answer` | string or null | — | — | What the scorer extracted. `null` means nothing parseable was found |
| `correct` | boolean | `true`, `false` | — | Exact match after normalisation |
| `format_failure` | boolean | `true`, `false` | — | `true` iff `pred_answer` is null. **Tracked separately from a wrong answer.** 0 occurrences in the reported runs |
| `n_generated_tokens` | integer | ≥ 0 | tokens | Length of the completion |
| `max_new_tokens` | integer | 1024 or 2048 | tokens | Per-model generation budget. Qwen3 gets 2048 because its tokenizer needs ~3× more tokens for the same Bengali text |
| `hit_cap` | boolean | `true`, `false` | — | `true` iff `n_generated_tokens == max_new_tokens`, i.e. the model was cut off rather than stopping |
| `bengali_ratio` | float | 0.0–1.0 | proportion | Share of characters in the completion in the Bengali Unicode block (U+0980–U+09FF). **The language-adherence measure** |
| `completion` | string | — | — | The raw generation, stored verbatim so every claim can be re-checked |

**Missing values:** `pred_answer` may be null; every other field is always
present. No imputation is performed anywhere.

---

## 5. Derived quantities used in the paper

| Name | Definition | Where computed |
|---|---|---|
| Accuracy | mean of `correct` over scored items | `paper_stats.py` |
| Wilson 95% interval | score interval on a single proportion | `paper_stats.py: wilson()` |
| Δ (risk difference) | `(gained − lost) / n` on paired items | `paper_stats.py: paired_diff_ci()` |
| Δ 95% CI | Wald interval from the discordant counts | same |
| McNemar p | two-sided exact binomial on discordant pairs | `paper_stats.py: mcnemar_exact()` |
| Fisher p | two-sided exact test, unpaired model-vs-model | `paper_stats.py: fisher()` |
| Retention | of items correct before, the share still correct after | `paper_stats.py` |
| Token-cap rate | mean of `hit_cap` | `paper_stats.py` |
| Adherence Δ | mean paired change in `bengali_ratio`, with t interval | `paper_stats.py: mean_ci()` |
| Sign test p | exact binomial on the direction of per-item change | `paper_stats.py: sign_test()` |

---

## 6. Environment record — `environment.json`

| Field | Meaning |
|---|---|
| `python`, `torch`, `transformers`, `peft`, `cuda_runtime`, `cudnn` | Exact versions |
| `gpu` | Device, driver version, VRAM |
| `seed` | 42, applied to Python, NumPy and PyTorch |
| `deterministic_algorithms` | `false`. See `reproducibility_notes` for what that costs |
| `reproducibility_notes` | Known sources of run-to-run variation, stated plainly |
