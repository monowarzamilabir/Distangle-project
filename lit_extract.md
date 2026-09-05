## Models
**Cite:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020). Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.0836112.

**Problem:** This paper empirically investigates how language modeling cross-entropy loss depends on model parameter count, dataset size, and training compute13. Understanding these relationships is critical because it provides a predictable framework to optimize training efficiency, avoid overfitting, and determine the optimal allocation of compute budgets before training massive models1more_horiz.

**Results:** The paper reports several key exact power-law relations:Power-law scaling for non-embedding parameters: $L(N) = (N_c/N)^{\alpha_N}$ with $\alpha_N \sim 0.076$ and $N_c \sim 8.8 \times 10^{13}$1524.Power-law scaling for tokens: $L(D) = (D_c/D)^{\alpha_D}$ with $\alpha_D \sim 0.095$ and $D_c \sim 5.4 \times 10^{13}$1624 (or $\alpha_D = 0.103$ and $D_c = 1.8 \times 10^{13}$ when fitting the full $L(N,D)$2425).Power-law scaling for optimal compute: $L(C_{\rm min}) = (C_{\rm min}^c/C_{\rm min})^{\alpha_{\rm min}^C}$ with $\alpha_{\rm min}^C \sim 0.050$ and $C_{\rm min}^c \sim 3.1 \times 10^8$ PF-days1624 (and $\alpha_{\rm min}^C \approx 0.054$ predicted from learning curves26).To avoid overfitting within $0.02$ nats of convergence, the required dataset size scales sublinearly as $D \gtrsim (5 \times 10^3) N^{0.74}$2728.Under a fixed compute budget $C_{\rm min}$, optimal performance is achieve

**Model scale:** The model sizes evaluated range from $768$ to $1.5$ billion non-embedding parameters13 (with some figures displaying runs from $10^3$ to $10^9$ parameters3738). This studied scale overlaps with the $0.6\text{B}–4\text{B}$ range, as it goes up to $1.5$ billion non-embedding parameters13.

**Languages:** The paper evaluated datasets in English (WebText27, Books Corpus6, Common Crawl6, English Wikipedia6, and public Internet Books6). Bengali or other Indic languages are not included or mentioned.Releva

**Relevance:** 

**Quotes:** "Performance depends strongly on scale, weakly on model shape:" (Section 1.1, Page 3)9"Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width." (Section 1.1, Page 3)9"The precise numerical values of Nc, Cmin c, and Dc depend on the vocabulary size and tokenization and hence do not have a fundamental meaning." (Section 1.2,

---

## Models
**Cite:** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus (2022). Emergent Abilities of Larg

**Problem:** The paper addresses the unpredictable phenomenon of "emergent abilities" in large language models—capabilities that are absent in smaller models but appear sharply in larger models4243. This matters because it means downstream performance on certain tasks cannot be predicted by simply extrapolating small-scale trends, which raises the possibility that further scaling could unlock entirely new capabilities or unexpect

**Results:** The paper documents several headline emergent ability thresholds:Performance on 3-digit arithmetic (addition/subtraction) and 2-digit multiplication jumps to substantially above random at $2 \times 10^{22}$ training FLOPs ($13\text{B}$ parameters) for GPT-3 and $10^{23}$ training FLOPs ($68\text{B}$ parameters) for LaMDA46.TruthfulQA accuracy jumps more than $20\%$ above random for Gopher at $5 \times 10^{23}$ training FLOPs ($280\text{B}$ parameters)47.Above-random performance on the Word in Context (WiC) benchmark emerges when PaLM is scaled to $2.5 \times 10^{24}$ FLOPs ($540\text{B}$ parameters)50.Chain-of-thought prompting on GSM8K only surpasses standard prompting when scaled to $10^{23}$ training FLOPs ($\sim 100\text{B}$ parameters, specifically emerging at $1.3 \times 10^{23}$ FLOPs and $68\text{B}$ parameters for LaMDA)6875.Instruction following hurts performance at $7 \times 1

**Model scale:** The model sizes evaluated range from $2.1\text{M}$ parameters (LaMDA) up to $540\text{B}$ parameters (PaLM)5282. Studied dense families include GPT-3 ($125\text{M}$ to $175\text{B}$), LaMDA ($2.1\text{M}$ to $137\text{B}$), Gopher ($417\text{M}$ to $280\text{B}$), Chinchilla ($417\text{M}$ to $70\te

**Languages:** The paper evaluated tasks in English, Hinglish, Persian, Russian, German, Swahili, Spanish, Hindi, Kannada, Chinese, and "non-English" languages46more_horiz. Bengali is not explicitly studied, though 

**Relevance:** 

**Quotes:** "Emergence is when quantitative changes in a system result in qualitative changes in behavior." (Section 1, Page 2)43"An ability is emergent if it is not present in smaller models but is present in larger models." (Section 2, Page 2)59"We consider an ability to be emergent if it is not present in smaller models but is present in larger models." (Abstract, Page 1)42CROSS-SOURCE OBSERVATIONS(a) Model scales represented across the papers, and whether either paper demonstrates its main claim at ≤4B parameters.Model scales represented: Across both sources, the model scales range from extremely small models ($10^3$ to $2.1\text{M}$ parameters) to massive models ($1.5\text{B}$ parameters in Kaplan 

---

## MODELS
**Cite:** Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021). Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.0968512.

**Problem:** The paper addresses the challenge of full fine-tuning for massive pre-trained language models (like GPT-3 175B), which is prohibitively expensive because it updates and saves all model parameters13. The authors argue this matters because full-scale retuning creates severe GPU memory and storage bottlenecks during both training and deployment in latency-sensitive production environments3more_horiz.

**Results:** headline results include:RoBERTa-base (125M) average GLUE score: Full Fine-Tuning (FT): 86.4; LoRA (0.3M trainable params): 87.227.RoBERTa-large (355M) average GLUE score: Full FT: 88.9; LoRA (0.8M trainable params): 89.028.DeBERTa-XXL (1.5B) average GLUE score: Full FT: 91.1; LoRA (4.7M trainable params): 91.343.GPT-2 Medium E2E NLG Challenge: Full FT (354.92M parameters): BLEU = 68.2, CIDEr = 2.47; LoRA (0.35M parameters): BLEU = 70.4±.1, CIDEr = 2.53±.021045.GPT-3 175B logical form validation accuracy on WikiSQL: Full FT: 73.8%; LoRA (37.7M parameters): 74.0%; LoRA (4.7M parameters): 73.4%44.GPT-3 175B MultiNLI-matched accuracy: Full FT: 89.5%; LoRA (4.7M parameters): 91.7%; LoRA (37.7M parameters): 91.6%44.GPT-3 175B SAMSum R1/R2/RL: Full FT: 52.0/28.0/44.5; LoRA (4.7M parameters): 53.8/29.8/45.944.Low-data regime (GPT-3 175B on MNLI-100): Full FT: 60.2%; LoRA: 63.8%41.Inference late

**Model scale:** RoBERTa-base (125M)2757, RoBERTa-large (355M)2857, DeBERTa-XXL (1.5B)1243, GPT-2 Medium (354.92M)10, GPT-2 Large (774.03M)45, and GPT-3 (175,255.8M)4458. This studied scale overlaps with the 0.6B–4B range, as it includes DeBERTa-XXL (1.5B) and GPT-2 Large (774.03M).

**Languages:** English datasets (GLUE, WikiSQL, E2E NLG, SAMSum, DART, WebNLG) are evaluated8more_horiz. Bengali or other Indic languages are not stated in the source.Relevance to my research:Finding 1: Contextualis

**Relevance:** 

**Quotes:** "we freeze the pretrained model weights and inject trainable rank decomposition matrices into each layer of the Transformer architecture" (Abstract, Page 1)1"We hypothesize that the change in weights during model adaptation also has a low “intrinsic rank”" (Section 1, Page 1)17"this guarantees that we do not introduce any additional latency during inference compared to a fine-tuned model by construction." (Section 4.1,

---

## Forgets Less
**Cite:** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham (2024). LoRA Learns Less and Forgets Less. Reviewed on OpenRe

**Problem:** This paper investigates whether LoRA compromises model performance compared to full finetuning when specializing on challenging target domains like code and mathematics, and to what extent LoRA mitigates the catastrophic forgetting of base model capabilities61more_horiz. This matters because while LoRA is widely adopted under hardware constraints, its relative performance limits and regularization characteristics are

**Results:** Exact numerical headline results include:Code CPT (HumanEval pass@1 at 20B tokens): Full Finetuning: 0.263; LoRA (r=256): 0.224; LoRA (r=64): 0.196; LoRA (r=16): 0.162103104.Code CPT Forgetting Average (at 20B tokens): Full Finetuning: 0.545; LoRA (r=256): 0.617; LoRA (r=16): 0.635105.Code IFT (HumanEval pass@1 at Epoch 4): Full Finetuning: 0.470 (peaks at 0.497 at epoch 8); LoRA (r=256): 0.498; LoRA (r=64): 0.417; LoRA (r=16): 0.35893106.Code IFT Forgetting Average (at Epoch 4): Full Finetuning: 0.512 (degrades to 0.414 at epoch 16); LoRA (r=256): 0.631; LoRA (r=16): 0.65297107.Math CPT (GSM8K strict match at 20B tokens): Full Finetuning: 0.293; LoRA (r=256): 0.202 (peaks at 0.203 at 16B); LoRA (r=16): 0.158105108.Math CPT Forgetting Average (at 20B tokens): Full Finetuning: 0.618; LoRA (r=256): 0.616; LoRA (r=16): 0.637106.Math IFT (GSM8K strict match at Epoch 4): Full Finetuning: 0.64

**Model scale:** Empirically trains Llama-2-7B (7 billion parameters)6683, with theoretical scaling projections provided for 70B113114 and 405B scales113114. Overlap with the 0.6B–4B range: No. The only model trained and evaluated is 7B, which is outside the 0.6B–4B range.

**Languages:** English datasets (OpenWebMath is classified as 99.7% English, and HellaSwag, ARC, MetaMathQA are English) are evaluated68more_horiz. Programming languages include 80+ programming languages in the Star

**Relevance:** 

**Quotes:** "our results show that, in the standard low-rank settings, LoRA substantially underperforms full finetuning." (Abstract, Page 1)61"full finetuning finds perturbations with a rank that is 10-100× greater than typical LoRA configurations, possibly explaining some of the reported gaps." (Abstract, Page 1)61"LoRA – even with higher rank – mitigates forgetting more aggressively than classic regularization techniques" (Section 1, Page 2)76CROSS-SOURCE OBSERVATIONS(a) Model scales represented across the papers, and whether either paper demonstrates its main claim at ≤4B parameters.Model scales represented:Hu et al. evaluate model scales spanning from 125M (RoBERTa-base), 355M (RoBERTa-large), 354M–

---

## Models
**Cite:** Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, and Yulia Tsvetkov (2023). Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models. arXiv preprint arXiv:2305.1370712.

**Problem:** The paper addresses the unfairness of commercial language model API pricing across languages due to the nonuniformity of subword tokenizers, where underrepresented languages require significantly more tokens to convey the same information as English1more_horiz. This matters because it forces speakers of low-resource languages (often in less economically developed regions) to be overcharged up to 5 times more for poor

**Results:** Languages with their own scripts, such as Telugu and Georgian, require up to 5× more tokens than English to convey the exact same information9.Running experiments in Telugu and Amharic costs up to 4× more for prompt + generation on XLSUM compared to English18.For most mid-resourced Indic languages with non-Latin scripts, there is close to a 5× increase in cost compared to English6.ChatGPT supports a maximum context of 4,096 tokens33, and BLOOMZ’s Huggingface API is capped at 1,000 tokens33. Due to high fragmentation, Telugu and Amharic struggle to fit even a single in-context example, restricting evaluations to zero-shot2034.Table 1 exact correlation numbers between pairs of variables (Cost, HDI, and Utility)25:XFACT: Cost–HDI Spearman is -0.41, Cost–HDI Pearson is -0.60; HDI–Utility Spearman is 0.34, HDI–Utility Pearson is 0.38; Cost–Utility Spearman is -0.61, Cost–Utility Pearson is -0

**Model scale:** ChatGPT (gpt-3.5-turbo)26 (parameter count not stated in the source33).BLOOMZ26 (parameter count: 175B33).Overlap with the 0.6B–4B range: No. ChatGPT's parameters are unstated, and BLOOMZ is 175B.

**Languages:** 22 typologically diverse languages evaluated1. The paper explicitly lists evaluated languages in Figure 10, Figure 14, and Appendix C, including English, French, Portuguese, Spanish, German, Russian, 

**Relevance:** 

**Quotes:** "Latin-script languages are represented with substantially fewer tokens compared to languages in other scripts." (Section 4.1, Page 4 or Passage 147)"our findings highlight disparities in the utility of LLMs, as well as socio-economic disparities and increased costs in using commercial APIs" (Abstract/Figure 1, Page 1 or Passage 128)"the lower the HDI index, the higher the fragmentation rate and vice versa." (Section 4.4,

---

## Languages
**Cite:** Aleksandar Petrov, Emanuele La Malfa, Philip H.S. Torr, Adel Bibi (2023). Language Model Tokenizers Introduce Unfairness Between Languages. 37th Conference on Neural Information Processing Systems (NeurIPS 2023).3940

**Problem:** The paper addresses the systemic disparity in the treatment of different languages at the tokenization stage, where the exact same text translated into different languages results in drastically different tokenization lengths39. This matters because these tokenization length differences induce unfair treatment for some language communities in regard to the financial cost of commercial services, longer processing late

**Results:** ChatGPT/GPT-4 (cl100k_base) uses 1.6 times more tokens for Italian than English, 2.6 times for Bulgarian, 3 times for Arabic, and 15 times for Shan40.Byte-level representation of the same text is over 4 times longer for Burmese or Tibetan than Chinese40.On ChatGPT/GPT-4, the cheapest non-English languages (Portuguese, Pangasinan, German) still carry a premium of 50% relative to English69.More than half of Japanese kanji characters require three tokens in GPT-252, and cl100k_base requires two tokens to represent some Cyrillic letters and three tokens for more than 65% of kanji characters52.With only one-third of the vocabulary, English sequences become just 10% longer for ChatGPT/GPT-4, and a 10-fold vocabulary reduction results in only 30% longer English sequences70.Table 1 exact premiums for GPT-2: Standard Arabic is 4.40, Bulgarian is 5.51, Chinese (Simplified) is 3.21, Burmese is 16.8

**Model scale:** GPT-2, RoBERTa, ChatGPT, GPT-4, FlanT5, GottBERT, CamemBERT, PhoBERT, RoCBert, BERT Japanese, ArabicBERT, MuRIL, XLM-R, NLLB, mT5, M2M100, CANINE, ByT5 (parameter counts are not explicitly stated in the source, though they note that RoBERTa block size is 512, GPT-2 has context size 768, 1024, 1280, 

**Languages:** 200 different languages from FLORES-200 evaluated42. Bengali, Hindi, Tamil, Telugu, Kannada, Gujarati, Malayalam, Marathi, Kashmiri, Assamese, Urdu, Nepali, Sanskrit, Sindhi, and Eastern Panjabi are e

**Relevance:** 

**Quotes:** "disparities persist even for tokenizers that are intentionally trained for multilingual support." (Abstract, Page 1 or Passage 221)"unequal treatment of languages arises at the tokenization stage, well before the language model sees any data at all." (Section 1, Page 2 or Passage 223)"tokenizers should produce similar encoded lengths for the same content across languages." (Section 1, Page 2 or Passage 225)CROSS-SOURCE OBSERVATIONS(a) Which model scales are represented across the papers, and whether either paper demonstrates its main claim at ≤4B parameters.Model scales represented: Ahia et al. evaluate commercial APIs like ChatGPT (gpt-3.5-turbo) and BLOOMZ (175B parameters)2633. Petrov et

---

## Alignment
**Cite:** Chunting Zhou*, Pengfei Liu*, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy (2023). LIMA: Less Is More for Alignment. Preprint. Under review.12

**Problem:** The paper addresses the relative importance of pretraining versus alignment in large language models, exploring if ChatGPT-level performance can be achieved with a tiny but high-quality instruction dataset13. This matters because standard alignment approaches (like RLHF or instruction tuning over millions of examples) require substantial compute and complex curation, whereas confirming that "less is more" suggests mo

**Results:** Exact printed results include:In human evaluation, LIMA was equivalent to or preferred over GPT-4 in 43% of cases (18% LIMA wins, 25% Tie, 57% LIMA loses)118.Human preference versus other models (win/tie/lose): versus Claude (24% / 22% / 54%), versus Bard (33% / 25% / 42%), versus DaVinci003 (44% / 21% / 35%), and versus Alpaca 65B (53% / 21% / 26%)18.GPT-4 as an annotator preferences (win/tie/lose): versus GPT-4 (19% / 15% / 66%), versus Claude (14% / 23% / 63%), versus Bard (27% / 26% / 47%), versus DaVinci003 (54% / 23% / 23%), and versus Alpaca 65B (64% / 19% / 17%)12.On absolute quality: 50% of LIMA responses were rated Excellent, 38% Pass, and 12% Fail20.Out of distribution: 45% Excellent, 35% Pass, and 20% Fail20.Safety: LIMA responded safely to 80% of sensitive prompts20.Ablation quality scores (1-6 scale): wikiHow (3.49), Unfiltered Stack Exchange (3.33), and Filtered Stack Exch

**Model scale:** LLaMa 65B17 and LLaMa 7B (for ablations)14. This scale does not overlap with the 0.6B–4B range, as the minimum evaluated model is 7B parameters.

**Languages:** English datasets were evaluated25. Bengali or other Indic languages are not stated in the source.Relevance to my research:Finding 1: Not relevant. The paper focuses on the alignment stage and does not

**Relevance:** 

**Quotes:** "almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary" (Abstract, Page 1 / Passage 125)"alignment teaches it which subdistribution of formats should be used when interacting with users." (Section 2, Page 2 / Passage 129)"alignment can be a simple process where the model learns the style or format for interacting with users" (Section 1, Page 1-2 / Passage 126)CROSS-SOURCE OBSERVATIONS(a) Which model scales are represented across the papers, and whether either paper demonstrates its main claim at ≤4B parameters.Model scales represented: Across both LIMA and LoRA papers, evaluated scales range from LLaMa 7B up to LLaM

---

## Models
**Cite:** Llama Team, AI@Meta (2024). The Llama 3 Herd of Models. Website: https://llama.meta.com/12

**Problem:** The paper introduces Llama 3, a new herd of foundation language models that natively support multilinguality, coding, reasoning, and tool usage13. The authors argue that developing high-quality foundation models rests on optimizing three key levers: data (improving quantity and quality of pre- and post-training data), scale (training at a far larger scale than previous versions), and managing complexity (opting for s

**Results:** The paper reports several key exact results:Llama 3 405B Evaluation: MMLU (5-shot) is 87.3, MMLU (0-shot, CoT) is 88.6, MMLU-Pro (5-shot, CoT) is 73.3, and IFEval is 88.628.Coding: HumanEval (0-shot) is 89.0 and MBPP EvalPlus (0-shot) is 88.628. MultiPL-E HumanEval scores: C++ is 82.0 ±5.9, Java is 80.4 ±6.2, PHP is 76.4 ±6.6, and TS is 81.1 ±6.152.Mathematics: GSM8K (8-shot, CoT) is 96.8 and MATH (0-shot, CoT) is 73.840.Reasoning: ARC Challenge (0-shot) is 96.9 and GPQA (0-shot, CoT) is 51.140.Proficiency Exams: LSAT is 81.1 ±3.8, SAT Reading is 74.8 ±3.7, SAT Math is 94.9 ±2.3, GMAT Quant is 96.0 ±7.7, GMAT Verbal is 86.6 ±8.2, AP Average is 93.5 ±1.9, GRE Quant is 162.0, and GRE Verbal is 166.043.Pre-training Verbatim Memorization (English, 50-gram weighted averages): Llama 3 8B is 0.26%, Llama 3 70B is 0.60%, Llama 3 405B is 1.13%53. Llama 3 405B (All, 1000-gram) is 3.91%53.Vision: M

**Model scale:** The released model family parameters are 8B, 70B, and 405B65. While pre-training scaling law experiments evaluated smaller proxy models between 40M and 16B parameters66, the final released model family starts at 8B parameters. The studied scale does not overlap with the 0.6B–4B range.

**Languages:** Evaluated languages include English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai natively910, with multilingual pre-training and speech evaluations covering up to 34 languages (inclu

**Relevance:** 

**Quotes:** "We believe there are three key levers in the development of high-quality foundation models: data, scale, and managing complexity." (Section 1, Page 1 / Passage 320)"Llama 3 uses a standard, dense Transformer architecture... our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale." (Section 3.2, Page 6 / Passage 343)"We follow the principle that post-training should align the model to “know what it knows” rather than add knowledge" (Section 4.3.6,

---

## Report
**Cite:** Gemma Team, Google DeepMind (2025). Gemma 3 Technical Report. arXiv preprint arXiv:2503.1978671.

**Problem:** The paper introduces Gemma 3, a multimodal addition to the Gemma family of lightweight open models ranging in scale from 1B to 27B parameters71. The authors address the challenge of bringing vision understanding, wider language coverage, and 128K context to lightweight models designed to run on consumer hardware (phones, laptops, and high-end GPUs) without the typical inference memory explosions of the KV-cache7172.

**Results:** The paper reports several key exact results:LMSYS Chatbot Arena Elo: Gemma 3 27B IT achieved an Elo rating of 1338, ranking among the top 10 best models86.Zero-Shot Benchmarks (Gemma 3 27B IT): MMLU-Pro is 67.3 (Pro) and 67.5 (Gemma 3)96; LiveCodeBench is 29.7, Bird-SQL is 54.4, GPQA Diamond is 42.4, FACTS Grounding is 74.9, Global MMLU-Lite is 75.1, MATH is 89.0, MMMU (val) is 64.996.Gemma 3 4B IT: MMLU-Pro is 43.6 and MATH is 75.696.Multilingual (Gemma 3 27B pre-trained): MGSM is 74.3, Global MMLU-Lite is 75.7, WMT24++ is 55.7, Flores is 48.8, and XQuAD is 76.876.Indic Gen Bench (Gemma 3 27B pre-trained): Average is 63.4 (XQuAD Indic is 77.8, XORQA in-en is 70.4, XORQA in-xx is 46.0, Flores Indic is 59.5)76101.Total Memorization Rates: Exact memorization rate for Gemma 3 models is roughly 0.0001% to 0.001% (significantly lower than prior models)99102.

**Model scale:** Gemma 3 dense models are: 1B (302M embedding, 698M non-embedding), 4B (417M vision, 675M embedding, 3.2B non-embedding), 12B (417M vision, 1B embedding, 10.7B non-embedding), and 27B (417M vision, 1.4B embedding, 25.6B non-embedding) parameters80. The studied scale overlaps with the 0.6B–4B range vi

**Languages:** Pre-training covers numerous languages73. The paper explicitly evaluates non-English languages including Spanish, French, Portuguese, Italian, Arabic, Japanese, Korean, Indonesian, Russian, Vietnamese

**Relevance:** 

**Quotes:** "The pre-training optimization recipe is similar to Gemma 2, with some modifications in the architecture design." (Section 1, Page 2 / Passage 722)"We also change the architecture of the model to reduce the KV-cache memory... achieved by increasing the ratio of local to global attention layers..." (Abstract, Page 1 / Passage 719)"We alternate between a local sliding window self-attention and global self-attention... 5 local layers for every global layer..." (

---

## Report
**Cite:** Qwen Team (2025). Qwen3 Technical Report. https://github.com/QwenLM/Qwen3104.

**Problem:** The paper introduces Qwen3, a new family of open-weight large language models104. The authors address the challenge of advancing model performance, efficiency, and multilingual capabilities while eliminating the need for users to switch between chat-optimized models (like GPT-4o) and dedicated reasoning models (like QwQ-32B)104. This is achieved by integrating "thinking" (multi-step reasoning) and "non-thinking" mode

**Results:** The paper reports several key exact results:Qwen3-235B-A22B (Thinking) results: MMLU-Redux is 92.7, GPQA-Diamond is 71.1, C-Eval is 89.6, LiveBench is 77.1, IFEval is 83.4, Arena-Hard is 95.6, AlignBench is 8.94, MATH-500 is 98.0, AIME'24 is 85.7, AIME'25 is 81.5, AutoLogi is 89.0, BFCL v3 is 70.8, LiveCodeBench v5 is 70.7, and CodeForces rating is 2056 (98.2 percentile)119131.Qwen3-235B-A22B (Non-thinking) results: MMLU-Redux is 89.2, LiveBench is 62.5, Arena-Hard is 96.1, AlignBench is 8.91, MATH-500 is 91.2, AIME'24 is 40.1, and CodeForces rating is 1387120132.Qwen3-32B (Thinking): MMLU-Redux is 90.9, GPQA-Diamond is 68.4, MATH-500 is 97.2, AIME'24 is 81.4, LiveCodeBench v5 is 65.7, and CodeForces rating is 1977133134.On-policy Distillation vs. RL on Qwen3-8B: Distillation achieved 74.4 on AIME'24 and 65.5 on AIME'25 using only 1,800 GPU hours, whereas RL achieved 67.6 and 55.5 using 

**Model scale:** Dense models are: 0.6B, 1.7B, 4B, 8B, 14B, and 32B138. MoE models are 30B-A3B (30B total, 3B activated) and 235B-A22B (235B total, 22B activated)138139. The studied scale overlaps with the 0.6B–4B range via the 0.6B, 1.7B, and 4B dense models.

**Languages:** Datasets cover 119 languages and dialects104107. Explicitly evaluates Spanish, French, Portuguese, Italian, Arabic, Japanese, Korean, Indonesian, Russian, Vietnamese, German, Thai140more_horiz. Yes, B

**Relevance:** 

**Quotes:** "A key innovation in Qwen3 is the integration of thinking mode... and non-thinking mode... into a unified framework." (Abstract, Page 1 / Passage 815)"Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency." (Section 1, Page 2 / Passage 819)"Qwen3 demonstrates scalable and smooth performance improvements correlated to the allocated thinking budget." (Section 4.7, Page 53 / Passage 896)CROSS-SOURCE OBSERVATIONS(a) Model scales represented across the papers, and whether either paper demonstrates its main claim at ≤4B parameters.Model scales represented: Across the three technical reports, the model scales range from lig

---

