# README — `dapt-finetune-bert-models.ipynb`

**Role in the project:** produces the **DAPT'd RoBERTa-base checkpoint** — the "roberta-base DAPT" row of the transfer table. Despite the filename, this notebook does **no** supervised fine-tuning and never sees a label: it is continued masked-language-model pre-training (Gururangan et al. 2020, *Don't Stop Pretraining*) on the unlabelled FOMC corpus. The classifier head is attached later, by the SFT notebooks. Written for and run on **Kaggle** (T4 GPU); the paths are Kaggle paths.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Configuration (cell 3) [stated]

| Setting | Value |
|---|---|
| `Checkpoint` | `roberta-base` |
| `Out_directory` | `/kaggle/working/dapt/roberta` |
| `Block_size` | 512 |
| `MLM_PROB` | 0.15 |
| `Seed` | 42 |

## Input

`/kaggle/input/datasets/lorenzouberti/unlabbled-fomc-data/*.jsonl` — the DAPT corpus built by `Tokenizer/Tokenizer- FED Branch.ipynb` and uploaded to Kaggle as a dataset. Fields: `text, source, path, chunk, n_tokens` [recorded]. Empty/non-string `text` rows are filtered out, then a **1% holdout** is split off with `seed=42` purely to track perplexity [stated].

**Recorded corpus size: 18,630 train / 189 eval documents** [recorded].

## Pipeline

1. **Sentence segmentation (cell 8).** Each document is split on blank lines into paragraphs, then each paragraph is segmented with `pysbd` (`language="en", clean=False`). Paragraph pre-splitting keeps `pysbd` from running across the hard line breaks left by two-column PDF layout. All original metadata columns are dropped here (`remove_columns=raw["train"].column_names`), so `source`/`path` do **not** survive into training — a deliberate simplification, but it means a block cannot be traced back to its document afterwards [inferred]. The segmented dataset is cached to `/kaggle/working/sentences` so the `pysbd` pass is paid once [stated].

2. **Greedy sentence packing into blocks (cell 10).** The heart of the notebook. Sentences are batch-tokenized *without* special tokens, then greedily accumulated up to `BUDGET = Block_size - 2 = 510`, and each emitted block is wrapped as `[CLS] … [SEP]` with a matching `special_tokens_mask` marking those two positions as never-maskable. Two deliberate properties [stated]:
   - a block is **flushed at document end**, so no block ever spans two documents;
   - an oversized single "sentence" (tables, chart debris that survived cleaning) is hard-truncated to `BUDGET` rather than dropped.

   The inline estimate is *"~30M tokens / ~512 per block ≈ 58k blocks"* [stated].

3. **MLM training (cell 11).** `AutoModelForMaskedLM` + `DataCollatorForLanguageModeling(mlm_probability=0.15, pad_to_multiple_of=8)`. Key arguments, with the comments that explain them [stated]:

   | Argument | Value | Reason given in the code |
   |---|---|---|
   | `learning_rate` | 5e-5 | — |
   | `per_device_train_batch_size` | 8 | "16 OOMs on the 15GB T4: MLM logits are batch × 512 × vocab" |
   | `gradient_accumulation_steps` | 8 | "effective batch stays 64 → same optimization as before" |
   | `num_train_epochs` | 5 | — |
   | `warmup_ratio` / `weight_decay` | 0.06 / 0.01 | — |
   | `eval_steps` = `save_steps` | 250 | — |
   | `save_total_limit` | 1 | "keeps you under Kaggle's 20GB working-dir cap" |
   | `load_best_model_at_end` / `metric_for_best_model` | True / `loss` | — |
   | `fp16` | if CUDA | — |

   `gradient_checkpointing` is left commented out as "the next lever if it STILL OOMs (~30–40% slower)" [stated].

4. **Evaluation and save.** Reports eval loss and `exp(eval_loss)` as perplexity, then writes model + tokenizer to `{Out_directory}/final`.

## Output

A DAPT'd `roberta-base` under `/kaggle/working/dapt/roberta/final`. **This notebook does not push to the Hub** — unlike its `roberta-large` sibling, there is no `push_to_hub` cell, so the checkpoint must be downloaded from the Kaggle working directory or re-uploaded manually [stated by omission].

## Cells 5–6, and why cell 6 exists

Cell 5 already builds a proper `DatasetDict` via `train_test_split`, then does `raw = raw_unlablled_jsonl.copy()` — and `.copy()` on a `DatasetDict` returns a **plain `dict`**, which is why cell 6 prints `<class 'dict'>` next to the comment *"plain dict = the culprit"* [stated]. Cell 6 is a defensive repair: if `raw` is a bare dict it concatenates the values, and if the result is a single `Dataset` it re-creates the 1% holdout. On the recorded run it passed straight through and printed the expected `DatasetDict` [recorded]. Harmless, but cells 5–6 should be collapsed into one clean load.

## Reproducibility

- **Seed 42**, passed to the holdout split and to `TrainingArguments(seed=...)`. `set_seed` is imported but never called [stated] — the `TrainingArguments` seed covers the trainer, so this is cosmetic.
- **Environment:** Kaggle, single T4, Python 3.11-era Kaggle image; `pysbd` is `pip install`ed at the top (0.3.4 recorded). `HF_TOKEN` is not set, producing an unauthenticated-download warning [recorded].
- **Not runnable as-is off Kaggle:** `/kaggle/input/...` and `/kaggle/working/...` are hard-coded.

## Audit notes / pre-submission checks

1. **Block size is 512 here, but the root README's data section describes DAPT chunks of "≤2,048 tokens (matching the continued-pre-training config)".** One of the two is wrong. This is a *Pre Tests* notebook, so the reported run may well use 2,048 — but confirm which config produced the reported DAPT rows and make the README match.
2. **No `push_to_hub`.** Record where this checkpoint ended up, or the base-DAPT row of the transfer table has no traceable artefact.
3. **Metadata is dropped at segmentation**, so per-source ablations of the DAPT corpus are impossible after the fact. Keep `source` through `pack()` if that is wanted.
4. The saved outputs stop after cell 10 — **no training log, eval loss or perplexity is recorded in the committed notebook** [recorded]. Re-run and keep the final perplexity; it is the natural sanity check that DAPT did something.
5. `math`, `Dataset`, `concatenate_datasets` and `set_seed` are imported in cell 1 and partly unused; `Dataset`/`concatenate_datasets` are re-imported in cell 6.
