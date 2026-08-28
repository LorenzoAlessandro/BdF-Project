# Does Institutional Provenance Shape a Model's Monetary Policy Priors?

Code and data pipeline for the paper *"Does Institutional Provenance Shape a Model's Monetary Policy Priors?"* — Jérôme Coffinet & Lorenzo Uberti Bona Blotto (August 2026).

Central bank communication is now a policy instrument in its own right, and language models are increasingly used to quantify its stance. But the corpora used to adapt those models are rarely institutionally neutral: they are dominated by the language of one particular central bank. This project asks whether that provenance leaves a measurable fingerprint on hawkish–dovish–neutral classification.

**Headline result:** it does, and the effect is asymmetric. A RoBERTa classifier trained on Federal Reserve text loses roughly 0.10–0.14 macro-F1 when applied to ECB text, while an ECB-trained classifier loses almost nothing (0.00–0.04) going the other way. The gap is concentrated almost entirely in the **dovish** class: Fed-trained models recall 62–69% of dovish sentences in-domain, but only 36–44% on ECB text, and 34–41% of true-dovish ECB sentences get flagged as *hawkish*.

---

## Contents

- [Research questions](#research-questions)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Part 1 — Generative LLMs](#part-1--do-generative-llms-classify-fed-and-ecb-text-differently)
- [Part 2 — DAPT + SFT encoders](#part-2--does-institutional-origin-shape-a-fine-tuned-models-transfer)
- [Results](#results)
- [Reproducing the study](#reproducing-the-study)
- [Status and known issues](#status-and-known-issues)
- [Citation](#citation)

---

## Research questions

1. **Do frontier generative models carry a regional or institutional prior?** Eight hosted LLMs from four providers (US, EU, China) classify the same annotated sentences under five prompt formulations. If provenance matters, misclassification should be systematic rather than noisy.
2. **Does the training corpus determine what a fine-tuned encoder can generalise to?** Seven BERT-family checkpoints are fine-tuned separately on Fed and on ECB data, then evaluated on both held-out test sets — a 2×2 transfer matrix.

|                | Fed test set     | ECB test set      |
| -------------- | ---------------- | ----------------- |
| **Fed-trained**   | in-domain        | cross-institution |
| **ECB-trained**   | cross-institution| in-domain         |

---

## Repository layout

```
.
├── scraping/            # Per-publication-type scrapers for federalreserve.gov and ecb.europa.eu
├── cleaning/
│   ├── document_clean/  # Source-specific front-ends (boilerplate, chart debris, footnotes)
│   ├── sentence_core/   # Shared sentence-processing core (identical across institutions)
│   └── chunking/        # Token-budgeted chunking + quality gate + dedup for DAPT
├── annotation/
│   ├── dictionary/      # Panel A1/B1/A2/B2/C term lists and clause segmenter
│   └── guides/          # Fed and ECB annotation guides (Tables 7 and 8)
├── llm_eval/            # Part 1: provider SDK clients, five prompts, macro-F1 decomposition
├── finetune/
│   ├── dapt/            # Continued MLM pre-training (RoBERTa-base / -large)
│   └── sft/             # Shared HF Trainer routine, 5 seeds per (model, institution)
├── analysis/            # Confusion matrices, transfer tables, stance index time series
└── data/
    ├── raw/ cleaned/    # Scraped text + manifest logs
    ├── unlabelled/      # DAPT chunks (annotated sentences removed)
    └── annotated/       # ~3,000 labelled sentences per institution
```

> Adjust paths to match your local checkout — the structure above reflects the pipeline described in the paper.

---

## Data

**Sources.** Federal Reserve Board publications and ECB communications, 1998–2026. The window is identical for both institutions and corpus sizes are matched by token count so that downstream differences reflect the institutions, not the sample. Pages are fetched with retry logic, parsed with BeautifulSoup, stripped of navigation and boilerplate, and written out as raw + cleaned text with a manifest log for reproducibility.

**Cleaning.** Every document passes a five-stage routine before chunking:

1. **Unicode / mojibake repair** — `ftfy`, then vulgar-fraction rewriting *before* NFKC (NFKC would otherwise glue fraction digits onto the preceding integer), soft-hyphen rejoining, hyphen-lookalike folding, control-character stripping.
2. **Line-level boilerplate removal** — a frequency rule (short lines repeated ≥3× in a document), NOTE/SOURCE openers and numbered-footnote bodies, plus curated patterns for chart axes, release codes, captions and web chrome. Applied as full-line matches, so an in-sentence date survives while a cover-page dateline does not.
3. **De-hyphenation across line breaks only** — preserves the suspended hyphens pervasive in Fed prose ("short- and long-term rates").
4. **Paragraph reflow** — unwraps the two-column PDF hard breaks while keeping blank lines as paragraph boundaries.
5. **Final tidy-up** — glued footnote markers, punctuation spacing, whitespace collapse.

**DAPT corpus.** Cleaned documents are greedily packed into ≤2,048-token chunks (matching the continued-pre-training config), degrading to sentence-level and then hard token cuts for oversized units. Chunks below 64 tokens are dropped. A final gate requires ≥200 characters and a letter-to-character ratio ≥0.60; duplicates are suppressed corpus-wide via a hash over whitespace-collapsed lowercase text, because Fed reports reuse whole boilerplate paragraphs year to year.

**Sentence selection.** Following Shah, Paturi & Chava (2023) and Gorodnichenko, Pham & Talavera (2021), a sentence is retained only if it matches a term from **both** a topic panel (A1: inflation, interest rate, price, employment…; B1: unemployment, growth, exchange rate, demand…) and is then passed through clause-level segmentation on `;`, *but*, *however*, *even though*, *although*, *while*. A split is accepted only if every resulting segment still carries a dictionary term — so "inflation moderated but growth remained subdued" is split, while a sentence with an off-topic second clause is kept whole.

**Annotation.** The Fed side uses the Shah et al. (2023) guide verbatim; the ECB guide is a direct adaptation holding categories fixed and substituting institution-specific ones (euro value, member-state fragmentation, APP/PEPP/TLTRO, M3). Fed speeches are hand-annotated from 2023 and back-filled with the publicly available Shah et al. labels. **All annotated sentences are removed from the DAPT corpus** to prevent look-ahead contamination.

**Splits.** Chronological, with the same cut-off for both institutions (test = 2021 onwards). This mirrors deployment, prevents leakage from recycled formulaic phrasing, and aligns both test sets on the same calendar window so that in-domain vs cross-institution differences reflect the *institution*, not the period.

| Dataset | Split | Neutral | Dovish | Hawkish | N |
|---|---|---|---|---|---|
| ECB | Train | 0.506 | 0.323 | 0.171 | 2,022 |
| ECB | Val | 0.379 | 0.290 | 0.331 | 317 |
| ECB | Test | 0.378 | 0.288 | 0.333 | 645 |
| FED | Train | 0.492 | 0.279 | 0.229 | 2,367 |
| FED | Val | 0.470 | 0.255 | 0.275 | 200 |
| FED | Test | 0.469 | 0.256 | 0.275 | 407 |

A robustness run replaces the ECB chronological split with a plain 80/20 split, since accommodative stances dominate the early-2000s and 2008-crisis ECB record and leave hawkish sentences thin in training.

---

## Part 1 — Do generative LLMs classify Fed and ECB text differently?

Eight models across four providers, chosen to span the three main centres of frontier development and to mix closed and open-weight systems. All are queried as hosted services.

| Provider (HQ) | Model | API identifier | Decoding |
|---|---|---|---|
| Anthropic (US) | Claude Sonnet 4.6 | `claude-sonnet-4-6` | T = 0 |
| Anthropic (US) | Claude Haiku 4.5 | `claude-haiku-4-5` | T = 0 |
| OpenAI (US) | GPT-5.1 | `gpt-5.1` | seed 42, reasoning off |
| OpenAI (US) | GPT-5-mini | `gpt-5-mini` | seed 42, minimal reasoning |
| Mistral (FR) | Mistral Large | `mistral-large-latest` | T = 0, seed 42 |
| Mistral (FR) | Mistral Small | `mistral-small-latest` | T = 0, seed 42 |
| DeepSeek (CN) | DeepSeek Chat | `deepseek-chat` | T = 0 |
| DeepSeek (CN) | DeepSeek-V4 | `deepseek-v4-pro` | sampling params ignored by API |

Each request is a system message fixing the role plus a user message with guidelines and the sentence verbatim. The model never sees dates or human labels — the sentence is the only foreign data. Output is capped at 20 tokens to force label-only completions; calls run over four worker threads with per-provider rate limiting and six retries with exponential back-off and jitter.

**Five prompt formulations** (a `M × P × 4` factorial, one run per triple):

| | Distinguishing element | Shot |
|---|---|---|
| **P1 Concise** | one-line guideline per class | zero-shot |
| **P2 Rubric** | full rubric: per-class indicators and phrasing cues | zero-shot |
| **P3 Rules** | if–then rules with an explicit tie-break to Neutral | zero-shot |
| **P4 JSON** | machine-readable contract `{"label": d}` | zero-shot |
| **P5 Few-shot** | one worked example per class, drawn from a *different* source | k = 3 |

**Metric.** Macro-F1 (`F_mpd`), chosen because minority stance classes matter as much as the majority. Raw levels are decomposed additively into a grand mean, model strength `α_m`, corpus difficulty `β_d`, prompt effect `γ_p`, and their interactions. Because raw scores confound "this model is better" with "this corpus is easier", the headline comparison uses **relative skill**, centred on the corpus mean:

```
RS_mb = α_m + (αβ)_mb        Δ_RS = RS_(m,ECB) − RS_(m,FED)
```

The general-ability term cancels in the difference, so `Δ_RS > 0` means model *m* is stronger on ECB material *relative to the field* — the interaction contrast that raw levels cannot license.

---

## Part 2 — Does institutional origin shape a fine-tuned model's transfer?

| Checkpoint | Params | Reference |
|---|---|---|
| RoBERTa-base | 125M | Liu et al. (2019) |
| RoBERTa-large | 355M | Liu et al. (2019) |
| DeBERTa-v3-base | 184M | He, Gao & Chen (2023) |
| FinBERT | 110M | Araci (2019) |
| FinBERT-tone | 110M | Huang, Wang & Yang (2023) |

**DAPT.** Continued MLM pre-training on the unlabelled in-domain corpus, same objective / architecture / vocabulary as the general checkpoint, with a modest budget and lower learning rate to limit catastrophic forgetting. Applied to RoBERTa-base and RoBERTa-large only (preliminary runs identified them as top performers; compute constrained the rest). The tokeniser is *not* extended — domain terms remain multi-piece and the encoder learns to compose them.

**SFT.** One shared routine across all models and both institutions, so any difference is attributable to checkpoint and data rather than procedure:

- Each model's own `AutoTokenizer`, truncation at 256 tokens, dynamic padding via `DataCollatorWithPadding`
- `AutoModelForSequenceClassification` with `num_labels=3`, shared `id2label`/`label2id`, `ignore_mismatched_sizes=True`
- Weights loaded in `float32` even under fp16 AMP (DeBERTa-v3 ships reduced-precision weights that otherwise break)
- 5 epochs · train batch 16 / eval 32 · weight decay 0.01 · warmup ratio 0.10 · fp16 on GPU
- Per-epoch evaluation and checkpointing, `load_best_model_at_end` on validation macro-F1, early stopping patience 2, `save_total_limit=1`
- Learning rate is a controlled factor across `{1e-5 … 5e-5}`; **the reported run uses config v4, lr = 3e-5**, identical for Fed and ECB so cross-institution comparisons are never confounded by a hyperparameter mismatch
- Five seeds per (model, institution): run *i* uses `42 + (i−1)`, following Dodge et al. (2020) on seed sensitivity → **70 fine-tuned models total**

Results are reported as mean ± sample SD with Bessel correction, since the five runs sample the distribution of fine-tuning outcomes rather than a population.

---

## Results

### Generative LLMs (mean macro-F1 across five prompts)

**Statements & press conferences** — every model scores higher on ECB material:

| Model | Fed | SD | ECB | SD |
|---|---|---|---|---|
| gpt-5.1 | 0.644 | 0.015 | **0.695** | 0.018 |
| mistral-large | 0.620 | 0.025 | 0.681 | 0.012 |
| claude-sonnet | 0.610 | 0.035 | 0.675 | 0.019 |
| claude-haiku | 0.636 | 0.023 | 0.669 | 0.027 |
| gpt-5-mini | 0.640 | 0.033 | 0.663 | 0.029 |
| deepseek-reasoner | 0.592 | 0.019 | 0.634 | 0.031 |
| deepseek-chat | 0.538 | 0.097 | 0.623 | 0.038 |
| mistral-small | 0.596 | 0.023 | 0.619 | 0.055 |

**Speeches** — the ECB advantage largely disappears and several models reverse. `deepseek-chat` shows by far the widest prompt sensitivity (SD 0.09), driven by a collapse under the rubric prompt (macro-F1 0.381 on Fed statements).

### Cross-institution transfer (accuracy, five seeds)

| Model | Fed→Fed | Fed→ECB | ECB→ECB | ECB→Fed |
|---|---|---|---|---|
| RoBERTa-base | 0.694 ± 0.011 | 0.557 ± 0.038 | 0.637 ± 0.025 | 0.644 ± 0.018 |
| RoBERTa-base DAPT | 0.692 ± 0.022 | 0.598 ± 0.034 | 0.643 ± 0.015 | 0.660 ± 0.010 |
| RoBERTa-large | 0.714 ± 0.025 | 0.601 ± 0.023 | 0.680 ± 0.010 | 0.659 ± 0.017 |
| RoBERTa-large DAPT | 0.710 ± 0.026 | 0.608 ± 0.038 | 0.686 ± 0.014 | 0.661 ± 0.019 |
| **mean** | **0.703** | **0.591** | **0.662** | **0.656** |
| **change** | — | **−0.112** | — | **−0.005** |

Fed→ECB costs 7–15 accuracy points (all significant at p < 0.01, paired by seed). ECB→Fed costs half a point on average, within seed variation.

### What the asymmetry is made of

- The macro-F1 gap ranges from a 10× asymmetry at base scale (0.138 vs 0.014) to roughly 3× for the large architectures. The ECB model's cross-domain gap never exceeds 0.043; the Fed model's never falls below 0.097.
- Nearly all of it sits in the **dovish** class. Fed models: 62–69% dovish recall in-domain → 36–44% on ECB, with 34–41% of true-dovish ECB sentences predicted hawkish. ECB models: 74–80% in-domain → 56–60% on Fed.
- Hawkish and neutral performance is substantially more stable in both directions.
- **DAPT helps most where it is needed most.** At base scale the Fed transfer gap shrinks 0.138 → 0.097 and the ECB gap effectively vanishes (0.014 → 0.001), driven by ECB→Fed rising 0.621 → 0.641 — beating both large architectures on that metric.
- Scaling base → large gives a consistent in-domain gain (+0.019–0.024 F1 on Fed, +0.039 on ECB), the larger ECB gain being consistent with the more jargon-dense structure of ECB language.

A stance index built from the hand annotations (`+1` hawkish, `0` neutral, `−1` dovish, averaged over annotated sentences per period) tracks inflation and unemployment through the 2008–09 crisis and the 2020–21 pandemic for both institutions — a sanity check that the labels carry real signal.

---

## Reproducing the study

```bash
# 1. Environment
pip install -r requirements.txt          # transformers, datasets, torch, beautifulsoup4, ftfy, scikit-learn

# 2. Scrape (writes raw/, cleaned/ and a manifest log)
python -m scraping.fed   --start 1998 --end 2026
python -m scraping.ecb   --start 1998 --end 2026

# 3. Build corpora
python -m cleaning.build_unlabelled --max-tokens 2048 --min-tokens 64
python -m annotation.extract_sentences --bank fed
python -m annotation.extract_sentences --bank ecb

# 4. Part 1 — generative LLM grid (8 models x 5 prompts x 4 corpora)
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... MISTRAL_API_KEY=... DEEPSEEK_API_KEY=...
python -m llm_eval.run --workers 4
python -m llm_eval.decompose            # macro-F1, relative skill, confusion matrices

# 5. Part 2 — DAPT then SFT
python -m finetune.dapt --model roberta-base --corpus data/unlabelled/fed
python -m finetune.sft  --config v4 --seeds 42 43 44 45 46 --cross-eval
```

API keys are read from the environment; never commit them. Expect the Part 1 grid to be the dominant cost and the 70 fine-tunes to be the dominant GPU time.


---

Gururangan et al. (2020), *Don't Stop Pretraining* · Shah, Paturi & Chava (2023), *Trillion Dollar Words* · Dodge et al. (2020), *Fine-Tuning Pretrained Language Models* · Gambacorta et al. (2024), *CB-LMs: Language Models for Central Banking* · Hansen & Kazinnik (2023), *Can ChatGPT Decipher Fedspeak?* · Blinder et al. (2008), *Central Bank Communication and Monetary Policy*

Macroeconomic series (CPI, HICP, unemployment) are drawn from FRED, Federal Reserve Bank of St. Louis.
