# README — `LLM Extension Plots ECB.ipynb`

**Role in the project:** analysis and figures for the **ECB half** of the LLM benchmark. Reads both workbooks written by `LLM_Extension_ECB_Side.ipynb`, builds a quarterly hawkish–dovish index per `model × prompt` column alongside the human `label` index, and produces per-family index plots, confusion matrices, a divergence analysis, and a **pooled statements + speeches index** that the Fed-side plots notebook does not have.

It is the ECB counterpart of `../Fed Side/LLM Extension Code repos/LLM Extension Plots.ipynb` and shares its `build_index` definition; the sections below concentrate on this notebook's own content.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## ⚠ SECURITY — fix before the repo ships

Cell 9 contains a **hard-coded FRED API key in plain text** (`Fred(api_key="88a8…77f3")`, **twice in the same cell**). The identical key also appears in `../Fed Side/LLM Extension Code repos/LLM Extension Plots.ipynb` and in `Plots/Plot_Code_Repos/FED_Hawkish_Dovish_Plots.ipynb` — three notebooks, one credential. Before submission: **(1) revoke and rotate the key at FRED, (2) purge it from all three notebooks and from git history** (`git filter-repo` / BFG), **(3) load it from `os.environ["FRED_API_KEY"]`**. FRED keys are free; a live credential in a replication package is a needless referee finding.

## Inputs

| File | Shape [recorded] |
|---|---|
| `results/ECB_statements_presconf_WITH_LLM_LABELS.xlsx` | columns `source, orig_row, id, source_file, date, doc_type, sentence, label, gold_collapsed, llm_majority` + 40 `{model}__{prompt}` columns |
| `results/ECB_speeches_WITH_LLM_LABELS.xlsx` | **1,000 × 55**, same prediction columns plus stray `Unnamed: 7/8/9` legend columns |

Prediction cells are letters `D/H/N`, with `?` = unparsed reply and `ERR` = failed call.

FRED, 2000-01-01 → 2026-01-01: `FPCPITOTLZGUSA` (US annual CPI), `T10YIEM` (10-year breakeven), `UNRATE` (unemployment), resampled month-end with forward fill.

⚠ **The macro series pulled are all US.** For an ECB stance index the natural context series are euro-area HICP and euro-area unemployment. The root README's data section says "CPI, HICP, unemployment" are used, so either the HICP pull lives elsewhere or this notebook overlays US macro on an ECB index. Resolve before any figure using these series goes in the paper.

## The index (cell 11)

```python
def build_index(df, freq, label_col):
    """Measure per period. freq: 'Q' = quarterly, 'A' = yearly."""
    counts = df.groupby([pd.Grouper(key="date", freq=freq), label_col]).size().unstack(fill_value=0)
    return (counts.get("H", 0) - counts.get("D", 0)) / counts.sum(axis=1)
```

Applied at `freq="QE"` to the human `label` column and to all 40 prediction columns, giving one comparable index per annotator.

⚠ **Denominator caveat** [inferred consequence, same as the Fed side]: `counts.sum(axis=1)` counts *every* value in the column, so `?` and `ERR` rows inflate the denominator of the affected model columns while never counting as H or D — mechanically pulling those models' indices toward zero. The confusion-matrix cells filter to `H/D/N`, so the two analyses in this notebook use **different effective samples**. Either drop `?`/`ERR` before indexing or disclose it.

## What the notebook produces, cell by cell

| Cells | Output |
|---|---|
| 4–5 | load both workbooks; print columns and shape |
| 7 | `Anthropic_Model_Analysis_df` — `date`, `label`, and the 10 Claude columns, as a first look |
| 9 | FRED pulls (inflation, unemployment) |
| 11–13 | `build_index` and `index_df_all` — quarterly index for `label` + all 40 columns |
| 14 | **Per-family index plots**: columns grouped by the prefix before `__`, one figure per family (8 figures), human `label` as the bold black reference line and the five prompts as thin lines |
| 17 | **Confusion matrices, all 40 combinations** in a 5-column grid, restricted to rows where gold *and* prediction ∈ {H, D, N} |
| 19–20 | **Best-performing prompt per model family** — index plot and an 8-panel confusion-matrix grid |
| 22–23 | `Best_Performance_Comparison_df` (`date`, `sentence`, `label`, 8 best columns) → **`best_performing_predictions.csv`** |
| 25 | **Divergence analysis** — `signed_div = model_index − label_index` per quarter, plus its absolute value, with quarter labels as periods (`2008Q4`) and a styled "worst-diverging quarter" table |
| 27–28 | **Pooled index** — statements and speeches concatenated, index rebuilt on the pooled sample |

### The "best performing" selection (cell 19)

```
claude-haiku__p1_analyst_direct     deepseek-reasoner__p4_analyst_json
claude-sonnet__p4_analyst_json      gpt-5-mini__p1_analyst_direct
deepseek-chat__p4_analyst_json      gpt-5.1__p4_analyst_json
mistral-large__p4_analyst_json      mistral-small__p4_analyst_json
```

**This list is hard-coded, not computed** [stated by the literal list]. Six of the eight are `p4_analyst_json`, which is a substantive finding — the machine-readable JSON contract is the best prompt for most models on ECB text — but as written the notebook gives no evidence for the selection. Derive it from `summary__*.csv` in code so the claim is auditable.

### Divergence (cell 25)

Sign convention, stated explicitly in the markdown: **positive = the model reads the quarter as MORE hawkish than the annotators, negative = MORE dovish** [stated]. This is the notebook's most paper-relevant output — it localises disagreement in *time* rather than in aggregate, so a systematic prior shows up as a persistent sign rather than as noise.

### Pooled index (cell 28)

Concatenates `All_Model_Analysis_df` (statements) and the speeches frame, tags each with `source`, drops rows with unparseable dates, and rebuilds the index on the pooled sample.

**Recorded: speech 1,000 / statement 986** [recorded]. Note the statements count is **986, not the 2,044 classified** by the run notebook — the gap is the date-`dropna` plus whatever the earlier frame construction dropped. Track down and state that attrition before pooling results are reported; a pooled index in which speeches outnumber statements is weighted very differently from the underlying corpus.

## Reproducibility

- **No RNG.** Everything is a deterministic function of the two workbooks plus the FRED snapshot.
- ⚠ **FRED series are revised over time.** Record the retrieval date, or cache the pulled series to CSV, or the macro overlays are not reproducible [inferred hazard].
- **Environment:** Anaconda Python 3.13, kernel `base`; `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn`, `fredapi`, `openpyxl`. A pandas `numexpr` `UserWarning` and a `concat` sort-deprecation warning appear in the saved output and are harmless [recorded]. Nothing pinned.
- **Path hazard:** absolute macOS paths with the trailing-space folder `BANQUE DE FRANCE `.

## Audit notes / pre-submission checks

1. **Rotate the FRED key and purge it from history** — see the security note above.
2. **Swap the US macro series for euro-area HICP and unemployment**, or state clearly why US series contextualise an ECB index.
3. **Compute the "best performing" list** rather than hard-coding it.
4. **Resolve the 2,044 → 986 statements attrition** before using the pooled index.
5. **Decide the `?`/`ERR` denominator question once** and apply the same rule in the index and the confusion matrices, here and in the Fed-side notebook.
6. Cells 14, 17, 25 and 28 carry `# %%` / `# %% [markdown]` markers inside code cells — leftovers from a percent-format script paste. They render as code comments; convert them to real markdown cells so the notebook reads correctly.
7. `best_performing_predictions.csv` (cell 23) is written to the **current working directory**, not to `results/`. Make it an explicit path.
8. Nothing is saved with `savefig` — every figure exists only in the notebook's stored outputs. Save any figure that reaches the paper into `figures/`.
