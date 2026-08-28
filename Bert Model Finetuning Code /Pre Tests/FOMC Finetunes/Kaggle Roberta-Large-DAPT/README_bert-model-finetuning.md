# README — `bert-model-finetuning.ipynb`

**Role in the project:** the **first end-to-end supervised fine-tuning run on the Fed side** — the preliminary experiment that established the shared SFT routine (256-token truncation, macro-F1 model selection, early stopping patience 2) later reused unchanged in the reported grid, and that identified RoBERTa as the strongest base checkpoint. It is Fed-only and single-seed: there is **no ECB training and no cross-institution evaluation here**. Those come from `Code Repos For Full Multimodel, Cross Eval/MultiModel with Cross Evaluation Analysis.ipynb`.

Despite living in the `Kaggle Roberta-Large-DAPT/` folder, this notebook fine-tunes **`roberta-base`, FinBERT and FinBERT-tone** — no large model and no DAPT checkpoint. Written for **Kaggle** (GPU); paths are Kaggle paths and the header comment says so [stated].

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Input (cell 3)

Three annotated spreadsheets, concatenated:

| File | Source tag | Date column |
|---|---|---|
| `fomc_statements_train_1000.xls` | `fomc_statements` | `date` |
| `lab-manual-sp-split-train-944601-augmented.xlsx` | `lab_manual_sp` | `year` → renamed to `date` |
| `presconf_to_annotate.xlsx` | `presconf` | `date` |

All are read from `/kaggle/input/datasets/lorenzouberti/fomc-annotated-dataset/`. The `lab_manual_sp` file is the Shah et al. speech data, which carries year-only dates — renamed to `date` "so it keeps the date column consistent across all sources" [stated].

## Preparation

1. **Cleaning (cell 5).** Keep `text, label, date`; strip whitespace; drop rows with any of the three missing; drop empty text; **`drop_duplicates(subset="text")`** — the comment gives the reason: *"so it doesn't overinflate results further down the line"* [stated]. FOMC statements recycle boilerplate verbatim across meetings, so without this the same sentence would appear in train and test.

2. **Label normalisation (cell 8).** Lowercased, then the 5-point annotation scale is collapsed to 3 classes: `d, md → dovish`; `n → neutral`; `mh, h → hawkish`. A stray `m` label is mapped to `neutral` in a separate line. **Recorded distribution: neutral 1,257 / hawkish 646 / dovish 613** (2,516 rows) [recorded].

3. **Label IDs (cell 10).** Assigned by `sorted(unique)` — giving **`{dovish: 0, hawkish: 1, neutral: 2}`** [recorded]. This is the same mapping the LLM-extension notebooks use, so predictions are comparable across Part 1 and Part 2.

4. **Chronological split (cell 12).** `cutoff = 2021-01-01`; everything before it is train, everything from it onward is split into val/test with `train_test_split(test_size=0.67, stratify=label_id, random_state=42)`. **Recorded: train 2,056 / val 151 / test 309**, 0 unparseable dates [recorded].

5. **Composition check (cell 14).** Recorded proportions [recorded]:

   | Split | neutral | dovish | hawkish |
   |---|---|---|---|
   | train | 0.512 | 0.244 | 0.244 |
   | val | 0.444 | 0.245 | 0.311 |
   | test | 0.447 | 0.239 | 0.314 |

   The post-2021 period is more hawkish than the training period — the expected consequence of a chronological split spanning the 2022 tightening cycle, and worth keeping as a stated feature of the design rather than a flaw.

6. **JSONL export (cell 16).** `{"text": ..., "label": int}` per line, plus `label_mapping.json`, into `/kaggle/working/data/`. `SOURCE_COL = None`, so the source tag assigned in cell 3 is **not** carried into training [stated] — per-source ablation is not possible downstream.

## The `finetune()` routine (cell 19)

This is the ancestor of the shared routine described in the paper's Part 2:

- `AutoTokenizer` per checkpoint, `truncation=True, max_length=256`, `DataCollatorWithPadding` (dynamic padding)
- `AutoModelForSequenceClassification` with `num_labels=3`, shared `id2label`/`label2id`, `ignore_mismatched_sizes=True`
- 5 epochs · train batch 16 / eval 32 · `lr=2e-5` (default) · weight decay 0.01 · warmup ratio 0.10 · fp16 when CUDA is available
- per-epoch eval and save, `load_best_model_at_end` on **`f1_macro`**, `save_total_limit=1`, `EarlyStoppingCallback(patience=2)`
- `set_seed(seed)` with `seed=42`, also passed to `TrainingArguments`
- `compute_metrics` returns accuracy, macro-F1 and weighted-F1

Cell 20 loops it over `{roberta: roberta-base, finbert: ProsusAI/finbert, finbert-tone: yiyanghkust/finbert-tone}`, with `gc.collect()` + `torch.cuda.empty_cache()` between models.

## Recorded results (cell 21)

| Model | test acc | test macro-F1 | test weighted-F1 | test loss |
|---|---|---|---|---|
| roberta | **0.615** | **0.607** | 0.616 | 1.903 |
| finbert | 0.573 | 0.529 | 0.566 | 2.081 |
| finbert-tone | 0.586 | 0.557 | 0.578 | 2.134 |

**This is the result that motivated the paper's choice to spend the DAPT budget on RoBERTa only** — the code comment in the large-DAPT notebook ("preliminary runs identified them as top performers") refers to exactly this table. Note the test losses above 1.9 alongside ~0.6 accuracy: the best-macro-F1 checkpoint is confidently wrong on a subset, which is consistent with the dovish-class difficulty the paper reports later.

## Downstream cells

- **Cell 22** reloads each `best/` checkpoint, predicts on the test set, and draws a 1×3 confusion-matrix panel plus a per-class `classification_report`, saving `/kaggle/working/confusion_matrices.png`.
- **Cell 23** pushes each model to `LorenzoAleCon29/{name}-fomc-hawkish-dovish` (**private**).

## Reproducibility

- **Seed 42** throughout: `set_seed`, `TrainingArguments(seed=...)`, and the val/test `random_state`. Single seed — the five-seed protocol arrives only in the reported grid.
- **Environment:** Kaggle GPU image; `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `huggingface_hub`.
- **Not runnable off Kaggle** — `/kaggle/input` and `/kaggle/working` are hard-coded, as the cell-3 comment warns [stated].
- Cell 23's `login(token="xxxxxxxxxx")` is a **redacted placeholder, not a live key**. Replace it with a Kaggle secret (as `Roberta-Large-DAPT.ipynb` does) rather than pasting a token back in.

## Audit notes / pre-submission checks

1. **These splits are not the reported ones.** train/val/test = 2,056/151/309 here, against 2,367/200/407 for FED in the root README's split table. Different source files and a different val/test ratio. Cite this notebook only as a preliminary run, never as the reported configuration.
2. **Cell 18 has a broken relative path**: `open("data/label_mapping.json")` while the file was written to `/kaggle/working/data/label_mapping.json`. It works only if the working directory happens to be `/kaggle/working`. Make it absolute.
3. **Cell 19 is truncated in the committed notebook** — the source ends mid-statement at `test_metrics = t`. The function evidently completed when it was run (cell 21 shows `test_*` keys), but as committed the notebook **cannot be re-executed**. Restore the full cell.
4. **`lr=2e-5` is the default used here**, while the reported grid uses 3e-5. Fine for a preliminary run; just don't mix the numbers when writing up.
5. Cell 20's recorded output stops after `========== roberta ==========`, so the FinBERT training logs are not preserved even though their metrics are.
6. Duplicate imports in cell 1 (`numpy`, `torch` twice), and `pandas`/`matplotlib` re-imported in cells 21–22.
