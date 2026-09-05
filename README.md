# Disentangling Scale from Targeted Pretraining Depth

Testing the less-is-more hypothesis for Bengali mathematical reasoning.

CSE791 Group 11, BRAC University.

## Research question

When a small model is given a curated set of Bengali mathematical reasoning
traces, what determines how well it performs: parameter scale, or the amount of
Bengali it saw during pretraining?

Two recent Bengali model families, TigerLLM and TituLLM, are built on the
implicit premise that targeted language depth can stand in for scale. We test
that premise directly, and we test whether LIMO-style minimal-data elicitation,
demonstrated at 32B parameters in English, survives the move to Bengali at sizes
a student can actually train.

**Headline result.** Scale predicts performance and Bengali pretraining depth
does not. Accuracy rises 3.6% → 16.8% → 44.8% across Qwen3 0.6B → 4B, while four
Bengali-specialised models spanning 1B to 3.2B all sit between 0.4% and 4.4%.
Fine-tuning on 873 curated traces moved no model significantly, though it
changed 40% of one model's answers.

## Data

| What | Where | Rows |
|---|---|---|
| Source corpus | `dipta007/Ganit`, SFT split (HuggingFace) | 16,868 |
| Decontamination targets | Bn-MGSM, Bn-MSVAMP, BenNumEval (HuggingFace) | 4,080 |
| Main training set | `curated/ganit_limo_n1000_c1-16_seed42.jsonl` | 873 |
| Difficulty control set | `curated/ganit_easy_n1000_c28-32_seed42.jsonl` | 1,000 |
| Held-out evaluation set | `heldout/heldout_ganit_dev.jsonl` | 500 (first 250 scored) |
| Model predictions | `results_v2/preds_<model>_<before\|after>.jsonl` | 250 each |

All source data is public. No personal or sensitive data is involved; the
corpus is school and olympiad mathematics word problems.

See `DATA_DICTIONARY.md` for every field and its allowed values.

## Running it

Two Python environments are involved. The evaluation needs PyTorch; the
analysis does not.

```bash
conda create -n bengali-rq2 python=3.11
conda activate bengali-rq2
pip install -r requirements.txt
```

Then, in order:

```bash
python data_curation.ipynb        # or run the notebook; builds curated/
python train_lora.py --all        # trains every adapter into adapters/
python eval_math.py --all         # writes results_v2/preds_*.jsonl
python paper_stats.py             # all statistics + every LaTeX table
python measure_tokenizer.py       # tokenizer table
python make_figs.py               # fig_scale, fig_beforeafter, fig_difficulty
python make_method_fig.py         # fig_method
python merge_paper.py             # assembles paper/paper_body.tex
python build_main.py              # assembles paper/main.tex
```

The last five are cheap and need no GPU. `eval_math.py` resumes from completed
item ids, so interrupting it is safe and never repeats work.

## Outputs

| File | Contents |
|---|---|
| `paper/main.tex` | The complete paper, self-contained |
| `paper/tab_main.tex` | Table 2: before/after accuracy, Δ with 95% CI, McNemar p |
| `paper/tab_difficulty.tex` | Table 3: per-difficulty accuracy, all models |
| `paper/tab_retention.tex` | Table 4: answer retention |
| `paper/tab_adherence.tex` | Table 5: Bengali character ratio with paired CIs |
| `paper/tab_tokenizer.tex` | Table 6: tokenizer cost |
| `paper/figs/*.png` | The four figures |
| `paper/draft_results.txt` | Console report of every statistic |
| `paper/draft_numbers.json` | Machine-readable, consumed by `make_figs.py` |

Every number in the paper regenerates from `results_v2/` by running
`paper_stats.py`. Nothing is hand-entered.

## Environment

Recorded in `environment.json`: Python 3.11.16, PyTorch 2.8.0+cu128,
Transformers 4.57.6, PEFT 0.20.0, single NVIDIA RTX 5060 Ti (16GB), seed 42.

Deterministic CUDA algorithms are **not** enabled. Greedy generation is stable,
but training losses reproduce to roughly three decimal places rather than
exactly. `environment.json` records the other caveats, including that two
adapters were resumed from checkpoints after power interruptions and their
reported training losses average only post-resume steps.

## Repository layout

```
train_lora.py          LoRA training, one config for every backbone
eval_math.py           evaluation, resumable, writes results_v2/
scorer.py              shared answer extraction; run it to see its self-tests
paper_stats.py         every statistic and LaTeX table in the paper
measure_tokenizer.py   tokenizer cost table
make_figs.py           result figures
make_method_fig.py     methodology diagram
merge_paper.py         assembles the paper body
build_main.py          assembles the complete main.tex
data_curation.ipynb    curation and decontamination
build_easy_subset.py   the difficulty control set
compare_conditions.py  per-model before/after detail
compare_adapters.py    three-way shared-base comparison
```

`scorer.py` runs its own test suite when executed directly. The tests use
verbatim model outputs, not invented strings, because two real scoring bugs
were found that synthetic cases would have missed.

## Authors

Monowar Zamil Abir, Department of Computer Science and Engineering,
BRAC University. CSE791 Group 11.
