# README — `Result Analysis, LLM EXtension .ipynb`

**Role in the project:** the **macro-F1 decomposition for Part 1** — the notebook that turns the four per-run summary CSVs into the paper's headline LLM tables, and in particular computes the **relative-skill** measure the root README defines as

```
RS_mb = α_m + (αβ)_mb        Δ_RS = RS_(m,ECB) − RS_(m,FED)
```

This is the notebook that licenses the paper's central Part 1 claim. Raw macro-F1 confounds "this model is better" with "this corpus is easier"; the centring performed here removes the corpus term so that a positive `ECB_minus_FED` means *stronger on ECB relative to the field*, not *ECB was easier*.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Inputs (cell 3)

Four summary CSVs, one per (institution × document type), written by the Excel-edition run notebooks:

```
Fed Side/Results/summary__fed_statements_presconf.csv
Fed Side/Results/summary__fed_speeches.csv
ECB Side/results/summary__ECB_statements_presconf.csv
ECB Side/results/summary__ECB_speeches.csv
```

Columns [recorded]: `model, prompt, n, parsed, errors, accuracy, macro_f1, in_tok, out_tok, cost_usd`.

⚠ Note the case difference in the directory names — Fed is `Results/`, ECB is `results/`. Correct as committed, and a trap on a case-insensitive filesystem.

**Recorded sample rows** [recorded]:

| Dataset | model | prompt | n | parsed | errors | accuracy |
|---|---|---|---|---|---|---|
| ECB statements+presconf | claude-haiku | p1_analyst_direct | 2044 | 2043 | 0 | 0.720 |
| ECB statements+presconf | claude-haiku | p3_analyst_rules | 2044 | 2043 | 0 | 0.664 |
| ECB speeches | claude-haiku | p1_analyst_direct | 1000 | 1000 | 0 | 0.683 |

Zero API errors and one unparsed row across the ECB statements run — the pipeline's retry and self-healing cells did their job.

## Construction (cell 7)

The four frames are concatenated with `bank ∈ {FED, ECB}` and `doc_type ∈ {statements_pressconf, speeches}` tags, and a combined `dataset = bank + "_" + doc_type` key. Everything downstream is a pivot over that long frame — 8 models × 5 prompts × 4 datasets = 160 runs.

### Table A — raw picture

`macro_f1` averaged per model × bank (over the 5 prompts and both doc types), with `ECB_minus_FED`. The code's own comment marks the limitation: *"NB: ECB_minus_FED here mixes model skill with dataset difficulty — table C fixes that"* [stated]. **Table A is the descriptive warm-up, not the result.** `A2` breaks it out per dataset.

### Table B — prompt lift

```python
runs["prompt_lift"] = runs["macro_f1"] - runs.groupby(["dataset","model"])["macro_f1"].transform("mean")
```

Each run minus **that model's own average on the same dataset**. Positive = this prompt pushes the model above its usual level. Because every model is measured against itself, prompt effects become comparable across models of very different strength [stated]. `B` averages lift per prompt × bank; `B2` gives the per-model detail (a heatmap-ready `model × [bank, prompt]` pivot) — i.e. *which prompt works for which model*, which is the finer claim behind the paper's prompt-sensitivity discussion.

### Table C — relative skill (the headline)

```python
runs["rel_skill"] = runs["macro_f1"] - runs.groupby("dataset")["macro_f1"].transform("mean")
```

The same centring trick, flipped: each run minus the **dataset** average over all models and prompts. This removes dataset difficulty, so *"ECB_minus_FED > 0 really means 'relatively stronger on ECB than the field', not 'ECB was easier'"* [stated]. This is `RS_mb` in the root README's notation, and `ECB_minus_FED` is `Δ_RS`.

### Extra — best prompt per model × bank

`mp.loc[mp.groupby(["bank","model"])["macro_f1"].idxmax()]` — the computed version of the selection that `LLM Extension Plots ECB.ipynb` hard-codes. **Prefer this cell's output over that hard-coded list.**

### Tables D–F — robustness

`D1` and `D2` re-run tables A and C with `aggfunc="median"`. The reasoning is stated in the code: *"One catastrophic run (e.g. JSON parsing failure) drags a 10-run mean by drop/10 but barely moves the median (breakdown point ≈ 50% vs 0% for the mean). If the rankings below match tables A and C, the mean results are not outlier-driven"* [stated]. `D2b` goes further — demeaning by the dataset **median** and aggregating by median, so no mean enters anywhere.

Given that `deepseek-chat` has the widest recorded prompt SD on both institutions, **the median check is not optional here** — report D alongside A and C, and say whether the rankings hold.

`E` covers document weighting (statements+presconf and speeches have different `n`, so an unweighted mean over doc types silently over-weights the smaller corpus) and `F` the sanity checks implied by the construction.

## Outputs

Everything is `display`ed inline; **nothing is written to disk** [stated]. The tables must be copied out by hand, which means the numbers in the paper have no file-level provenance. Add `to_csv` calls for tables A, B, C and D at minimum.

## Reproducibility

- **No RNG**; deterministic given the four CSVs.
- `pd.set_option("display.float_format", …3f)` — displayed values are rounded to 3 dp; the underlying frames are not.
- A `display`/`print` fallback shim makes the cells runnable as a plain script [stated].
- **Environment:** Anaconda Python 3.13; `pandas`, `numpy`, `matplotlib`.
- **Path hazard:** absolute macOS paths with the trailing-space folder `BANQUE DE FRANCE `.

## Audit notes / pre-submission checks

1. **The committed outputs are truncated.** The saved output stops at `A · mean macro-F1 per model × bank` [recorded] — none of tables A through F actually appear in the notebook as committed. Re-run and commit with outputs, or the decomposition backing the headline claim is undocumented in the repo.
2. **Save the tables to CSV.** Table C in particular is the paper's central Part 1 evidence and currently exists only in a notebook cell.
3. **Report D (medians) next to A and C**, explicitly, given the known prompt sensitivity of `deepseek-chat`.
4. **Decide the document-weighting question** (table E) before averaging across doc types — statements+presconf is roughly twice the size of speeches on both sides.
5. Cells 7 and 8 carry `# %%` markers and "paste below your loading cell" instructions — leftovers from assembling the notebook out of script fragments. Tidy them so the notebook reads as a single document.
6. The filename itself (`Result Analysis, LLM EXtension .ipynb`) has a comma, inconsistent capitalisation and a trailing space, which makes it awkward to reference from a shell or another notebook.
