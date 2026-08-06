# README — `LLM_Extension_Plots.ipynb`

**Role in the project:** analysis/figures for the LLM benchmark. Reads the Excel edition's output workbook (`results/fed_statements_presconf_WITH_LLM_LABELS.xlsx`), builds a quarterly hawkish–dovish index per **model × prompt** column alongside the human `label` index, and produces per-family index plots, correlation heatmaps, rolling correlations, and confusion matrices; FRED macro series are pulled for context.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## ⚠ SECURITY — fix before the repo ships

Cell 9 contains a **hard-coded FRED API key in plain text** (`Fred(api_key="88a8…77f3")`, twice in the same cell; the same key also appears in `FED_Hawkish_Dovish_Plots.ipynb`). Before submission/publication: **(1) revoke/rotate the key at FRED, (2) purge it from the notebook and from git history** (`git filter-repo` or BFG), **(3) load it from an environment variable** (e.g. `Fred(api_key=os.environ["FRED_API_KEY"])`). FRED keys are free, but a leaked credential in a replication package is a needless finding for a referee.

## Inputs

- `results/fed_statements_presconf_WITH_LLM_LABELS.xlsx` — the merged, date-sorted statements+press-conference workbook with the human `label` (letters, MD/MH already collapsed in that output) and **40 prediction columns** `{model}__{prompt}` in `D/H/N` (plus `?` = unparsed, `ERR` = failed call), `gold_collapsed`, `llm_majority` [recorded column list]. Only this workbook is used; the speeches workbook is not analyzed here.
- FRED (2000-01-01 → 2026-01-01): `FPCPITOTLZGUSA` (US annual CPI), `T10YIEM` (10-yr breakeven), `UNRATE` (unemployment); monthly-end resample + ffill [recorded].

## Method, decision by decision

1. **Index definition** (`build_index`): per quarter (`freq="QE"`), `(#H − #D) / #Total` computed from the letter values of each column — applied to the human `label` column and to every `{model}__{prompt}` column, giving one comparable index per annotator [stated by the docstring "Measure per period"].
   ⚠ **Denominator note** [inferred consequence]: `counts.sum(axis=1)` counts *all* values in the column, so `?` (unparsed) and `ERR` rows inflate the denominator of the affected model-columns while never counting as H or D. Decide whether to drop `?`/`ERR` rows per column before indexing (cleaner) or keep and disclose; either way, state it.
2. **Anthropic-only frame** (cell 7) and the **all-model frame** (cell 12: `date`, `label`, + all 40 combo columns) [recorded: head shows e.g. gold H with Sonnet-p1 H/D/… — the raw disagreement matter of the study].
3. **Per-family index plots** (cell 14): columns grouped by the prefix before `__`; one figure per family (8 figures [recorded]) with the human `label` index as the bold black reference line and the five prompt variants as thin lines.
4. **Correlation diagnostics** (cell 16): per family, a masked (upper-triangle) seaborn heatmap of the index correlations (family columns + `label`) and a **rolling correlation with `label`, `window = 4` quarters** per prompt (8 figures [recorded]).
5. **Confusion matrices** (cell 18): sklearn `confusion_matrix` per combo on rows where **both** gold and prediction ∈ {H, D, N} (so `?`/`ERR`/blank rows are excluded here, unlike the index), 40 matrices in one 5-column grid [recorded].

## Reproducibility & environment

- **No RNG** in this notebook; everything is a deterministic function of the input workbook + the FRED snapshot. FRED series are *revised over time* — record the retrieval date (or cache the pulled series to CSV) so the macro overlays are reproducible [inferred hazard].
- **Environment:** Python 3.13.12, kernel `base` [recorded]; `pandas` (3.0.5 per the Excel notebook's recorded pip output — a `Pandas4Warning` about concat sorting appears here [recorded]), `matplotlib`, `seaborn`, `numpy`, `scikit-learn`, `fredapi`, `openpyxl`. None pinned — pin.
- **Figures are only displayed, never saved** — every plot ends in `plt.show()` with no `savefig`. For the paper, add `fig.savefig(...)` (300 dpi, tight bbox) so the exact rendered artifacts are versioned rather than living only inside the .ipynb.
- Path hazard: the results path sits under the trailing-space folder `BANQUE DE FRANCE `.

## Pre-submission checks

1. **Rotate + purge the FRED key; switch to an env var** (top priority).
2. Decide and document the `?`/`ERR` handling in the index denominator; align it with the confusion-matrix filtering or justify the difference.
3. Add `savefig` calls and commit the generated figures (or a `figures/` build step).
4. If any `?`/`ERR` cells remain in the workbook, re-run the Excel notebook's healing cells first so the plots reflect a complete grid.
5. Record the FRED retrieval date / cache the macro series; pin package versions.
6. The empty trailing cells (19–20) can be deleted.
