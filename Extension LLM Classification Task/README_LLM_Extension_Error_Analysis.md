# README — `LLM Extension Error Analysis.ipynb`

**Role in the project:** the notebook that asks **what kind of sentence each model gets wrong**, rather than how often it is wrong. It joins the LLM predictions to the human `Dissagreement` column — a topic tag on each annotated sentence (Economic Status, Labor, Money Supply, Fed/ECB Expectations, Dollar/Euro Value Change, Foreign Nations, Energy/House Prices, Key Words, General, Member States/Fragmentation) — and reports per-model error rates by topic, on both institutions, then formally tests two hypotheses about *why* the Fed→ECB gap exists.

This is where the paper's "the asymmetry is concentrated somewhere specific" claim can be checked against topic structure rather than class labels alone.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## ⚠ Broken paths — this notebook does not run as committed

Three cells point at **`/Users/lorenzouberti/Desktop/cvbnjkl`**, a scratch directory that is not part of the repository [stated]. Cell 2's saved output is a live `FileNotFoundError` [recorded]:

```
FileNotFoundError: '/Users/lorenzouberti/Desktop/cvbnjkl/fed_statements_presconf_WITH_LLM_LABELS_2021.xlsx'
```

Cell 5 uses the real path (`Fed Side/Excel Sentences Results/`) and did run. Cells 7 and 8 fall back to `cvbnjkl` again. **Repoint every path to the repository before re-running** — the recorded outputs prove the analysis worked at some point, but the committed notebook cannot reproduce them.

## Inputs

| Cell | File | Notes |
|---|---|---|
| 5 | `Fed Side/Excel Sentences Results/fed_statements_presconf_WITH_LLM_LABELS_2021.xlsx` | the run that works; columns include `label`, `Dissagreement`, `llm_majority`, and 40 `{model}__{prompt}` columns |
| 7 | `fed_..._FULL.xlsx` and `ECB_..._FULL.xlsx` | the full-history variants, for the Fed↔ECB comparison |
| 8 | `Fed Side/Excel Sentences Results/fed_statements_presconf_WITH_LLM_LABEL.xlsx` and `ECB Side/Excel Results/ECB_statements_presconf_WITH_LLM_LABELS_2021.xlsx` | the 2021+ ECB split, chosen "so the Fed vs ECB comparison is apples-to-apples (the Fed data is 2021+ only)" [stated] |

⚠ Three different filename variants of the Fed workbook appear (`_2021`, `_FULL`, `_WITH_LLM_LABEL` singular). Establish which is which and name them consistently — as written it is not possible to tell whether a given table used the 2021+ or the full-history sample.

## Method

**Cell 5 — Fed-side topic error analysis.** Prediction columns are auto-detected as anything containing `__`; rows without a ground-truth label are dropped; then a long/tidy "miss table" is built with one row per (sentence, model, prompt) where `pred != label`. Errors are summarised per model **pooled across its five prompts**, both as raw counts and as an error *rate* (misses ÷ predictions in that category) — the rate is the one to quote, since topic categories are very unevenly sized.

**Recorded — overall miss rate per model, pooled across prompts** [recorded]:

| Model | n | misses | miss rate |
|---|---|---|---|
| gpt-5-mini | 3,035 | 1,069 | 0.352 |
| gpt-5.1 | 3,035 | 1,090 | 0.359 |
| mistral-large | 3,035 | 1,090 | 0.359 |
| deepseek-reasoner | 3,030 | 1,089 | 0.359 |
| claude-sonnet | 2,908 | 1,053 | 0.362 |
| claude-haiku | 3,035 | 1,124 | 0.370 |
| mistral-small | 3,035 | 1,179 | 0.388 |
| deepseek-chat | 3,035 | 1,370 | 0.451 |

Note the unequal denominators — claude-sonnet has 2,908 and deepseek-reasoner 3,030 against 3,035 for the rest, i.e. missing predictions. Worth explaining before the table is cited.

Written to `model_disagreement_analysis.xlsx` [recorded].

**Cell 7 — ECB side and the cross-institution comparison.** Mirrors cell 5 on the ECB workbook, then maps the two institutions' topic vocabularies onto a shared set via a `CANON` dictionary [stated]:

```
Fed Expectations   ┐
ECB Expectations   ┴→ Central Bank Expectations
Dollar Value Change ┐
Euro Value Change   ┴→ Currency Value Change
Member States/Fragmentation → (ECB-only, no Fed counterpart)
```

**Recorded — worst topic per model, side by side** [recorded]:

| Model | Fed top category | rate | ECB top category | rate |
|---|---|---|---|---|
| claude-haiku | Money Supply | 0.533 | Member States/Fragmentation | 0.586 |
| claude-sonnet | Money Supply | 0.442 | Member States/Fragmentation | 0.607 |
| deepseek-chat | Money Supply | 0.511 | Member States/Fragmentation | 0.609 |
| deepseek-reasoner | Money Supply | 0.556 | Member States/Fragmentation | 0.563 |
| gpt-5-mini | Foreign Nations/Trade | 0.403 | Member States/Fragmentation | 0.602 |
| gpt-5.1 | Money Supply | 0.556 | Member States/Fragmentation | 0.591 |
| mistral-large | Currency Value Change | 0.450 | Member States/Fragmentation | 0.549 |
| mistral-small | Money Supply | 0.622 | Member States/Fragmentation | 0.549 |

**This is a strong, clean result and arguably the most quotable output in the folder: all eight models fail hardest on the one topic category that has no Fed analogue** (Member States/Fragmentation), and seven of eight fail hardest on Money Supply on the Fed side. It is direct evidence that the difficulty is institution-specific subject matter, not generic model weakness.

**Cell 8 — the two formal hypotheses.** The markdown states the design carefully [stated]:

> The hypothesis "open models understand ECB *communication* worse" is NOT the same as "open models are less accurate on ECB" — a model can simply be weaker overall. The clean test is the **Fed → ECB drop**: does a group lose *more* accuracy moving from Fed to ECB than another group? That is the openness × bank interaction.

Two groupings are tested, both defined explicitly in code:
- **Openness** — Anthropic + OpenAI = Proprietary; Mistral + DeepSeek = Open-weight
- **Region** — OpenAI + Anthropic = US; Mistral = Europe; DeepSeek = China

with the competing prediction spelled out: if regional proximity matters, European models should *gain* relative ground on ECB text. `statsmodels` (`smf` for the interaction models, `proportion_confint` for CIs) is imported inside a `try/except`, so the cell degrades to descriptive tables if it is unavailable [stated].

**Recorded result** [recorded]:

```
Mean open-vs-prop gap on ECB-specific cats: +0.018 | on generic cats: +0.017
```

Essentially **no differential** — the open-weight/proprietary gap is the same size on ECB-specific topics as on generic ones, which is a null result for the openness hypothesis and should be reported as such. Per-category accuracy differences are small and mostly within noise (Foreign Nations +0.021 on n=955; Member States/Fragmentation −0.016 on n=205).

Outputs: `open_vs_proprietary_ecb_analysis.xlsx`, `acc_openness_bank.png`, `per_model_fed_vs_ecb.png` [recorded].

## Outputs

| File | Written by |
|---|---|
| `model_disagreement_analysis.xlsx` | cell 5, into `Fed Side/Excel Sentences Results/` |
| `open_vs_proprietary_ecb_analysis.xlsx` | cell 8, into `ECB Side/Excel Results/` |
| `acc_openness_bank.png`, `per_model_fed_vs_ecb.png` | cell 8 |

## Reproducibility

- **No RNG.** Deterministic given the input workbooks.
- **Environment:** Anaconda Python 3.13; `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels` (optional). Nothing pinned.
- **The `Dissagreement` column is hand-assigned** and is not produced by any script in this repository [inferred — no code writes it]. It is a research input in its own right: document who tagged it, against what definitions, and with what reliability, or the topic results have no provenance.

## Audit notes / pre-submission checks

1. **Fix the `cvbnjkl` paths** in cells 2, 7 and 8 — three of the four analyses cannot run as committed.
2. **Resolve the three Fed workbook filename variants** (`_2021`, `_FULL`, `_WITH_LLM_LABEL`) and state which sample backs which table.
3. **Explain the unequal denominators** (2,908 / 3,030 / 3,035) in the miss-rate table.
4. **Document the `Dissagreement` taxonomy** — its categories carry the headline finding above.
5. **Report the openness null result explicitly.** A well-designed test that comes out flat is a finding; leaving it out would be selective.
6. The column is spelled `Dissagreement` throughout (two s's). Keep the spelling consistent with the spreadsheets rather than "fixing" it in one place only.
7. Cells 7 and 8 carry `# %%` percent-format markers inside code cells — convert to real markdown cells.
