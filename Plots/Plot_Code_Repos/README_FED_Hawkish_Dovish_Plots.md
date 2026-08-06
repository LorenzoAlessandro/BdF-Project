# README — `FED_Hawkish_Dovish_Plots.ipynb`

**Role in the project:** the main index-and-macro figure notebook. Builds sentence-share hawkish–dovish indices from the **hand-annotated** Fed files (statements, press conferences, speeches) and from **Claude-annotated** counterparts, overlays them on FRED macro series (CPI, unemployment; euro-area HICP and unemployment for the ECB), and computes annotator-agreement statistics (rolling and static correlations) plus index-vs-macro correlations. Also builds the ECB quarterly index from the ECB annotation files.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## ⚠ SECURITY — fix before the repo ships

Cells 4 and 6 contain a **hard-coded FRED API key in plain text** (`Fred(api_key="88a8…77f3")` — the same key also appears in `LLM_Extension_Plots.ipynb`). Revoke/rotate it, purge it from the notebooks **and git history**, and load it from an environment variable. Top-priority item.

## Inputs (all under `BdF-Project/data/...`; paths carry the trailing-space `BANQUE DE FRANCE ` folder)

**Hand-annotated Fed** [recorded shapes/columns]:
- `FOMC STATEMENTS/fomc_statements_train_1000.xls` — 1,000 × 8, columns `id, raw_text, doc_type, clean_text, sentence, label, date, exact_match`. The presence of `date` + `exact_match` shows this is the **dated** statements file — i.e., the Track-B date re-attachment output merged back into the annotated set [inferred; consistent with the matcher scripts' column additions]. Labels include the 5-point letters (H/MH/N/MD/D).
- `Press Conferences/presconf_to_annotate.xlsx` — 999 × 9 (`date, file, speaker, sentence, label, Unnamed:…`), day-level dates from 2011-04-27 (Bernanke) [recorded].
- `Speeches/Annotated with Shah Merge/lab-manual-sp-train-5768-augmented.xlsx` — 951 × 6 (`sentence, year, label, …`) — **year-only dates**, so speeches enter the *yearly* index only [stated by the index design].

**Claude-annotated Fed** (cell 28): Sonnet (xlsx), Haiku (csv), "Claude Opus 5" (csv) for statements, plus the three press-conference counterparts (`Sonnet_presconf_annotated_manual.xlsx`, `Haiku presconf_annotated_statements.csv`, `Opus presconf_annotated.xlsx`). The Sonnet statements file has the same dated columns with full-word labels ("Hawkish/Neutral/…") [recorded]. ⚠ **Haiku is loaded but never used** — the comparison `SOURCES` covers Hand / Sonnet / Opus only [recorded]; either add Haiku or drop the load. Document, in the data appendix, exactly how each of these Claude annotation files was produced (interface, model version, prompt, date) — that provenance is not in any uploaded code [inferred gap].

**ECB** (cell 16): `ecb_train-2.xls` read with `header=1` (first spreadsheet row skipped as a title band [inferred]) **concatenated with `ecb_test.csv` "just for graph"** [stated] → 1,058 × 10 [recorded] — i.e., the full labeled ECB sentence set (train 800 + test 258). Columns include **both `label` and `final_label`** plus annotation-legend `Unnamed:` columns [recorded]. `speeches_annotated_v2.xlsx` and `ecb_Press Conference Statements + Q&A_to_annotate.xlsx` (986 × 5 [recorded], dates from 1998 introductory statements) complete the ECB inputs.

**FRED** (2000-01-01 → 2026-01-01): US `FPCPITOTLZGUSA`, `T10YIEM`, `UNRATE`; euro area `EA19CPALTT01GYM` (HICP y/y), `LRHUTTTTEZM156S` (unemployment); monthly resample + ffill [recorded].

## Method, decision by decision

1. **Label normalization.** `clean_label`: `H/MH → hawkish`, `D/MD → dovish`, `N → neutral`, else `None` (dropped). `clean_label_llm` additionally accepts full words and `0/1/2` numeric codes (with 0=dovish, 1=hawkish, 2=neutral) — one mapper for every annotator format [stated by its docstring].
2. **Index.** `build_index`: per period, `(#hawkish − #dovish) / #total` via `pd.Grouper` on the date. **Quarterly** index = statements + press conferences; **yearly** index = + speeches (year-only dates) [stated in the header]. Composition charts show the H/N/D shares per period (row-normalized stacked bars).
3. **Diagnostics-first workflow** [stated]: cells print each file's columns/head and per-file sentence counts ("if a file shows 0 sentences, its column names above are wrong") before any figure — keep these in the replication package; they are the debugging path a replicator will need.
4. **Claude-vs-human comparison** (cells 30–39): per annotator, `(date, label)` rows → quarterly (`QS`) index; overlay plots; **rolling 8-quarter (2-year) pairwise correlations, `min_periods = 4`**, shown both as a heat-strip and as lines with the underlying indices below; a static Pearson matrix on the 106 shared quarters; and cell 41 correlates each index with CPI/unemployment on the joint quarterly calendar.
5. **Verification exports** (cell 26): `verification_quart.csv` / `verification_yearly.csv` — ⚠ written to the **current working directory** (relative paths); give them explicit output paths.

## Recorded results (last saved run)

- Hand-annotated: **1,996** dated sentences; Claude Sonnet: **1,998** (neutral 1,019 / dovish 509 / hawkish 470); Claude Opus: **1,998**; all spanning **2000-02-02 → 2026-06-17, 106 quarters**, all 106 with every annotator present [recorded].
- Rolling 8-quarter correlations (103 windows): **Hand × Sonnet mean r = 0.49; Hand × Opus mean r = 0.53; Sonnet × Opus mean r = 0.37** [recorded].
- Index-vs-macro panel: 105 quarters, 2000-01-01 → 2026-01-01 [recorded].
- ECB: one flagged date-parsing warning (below); quarterly index + composition figures produced [recorded].

## ⚠ Correctness flags to resolve

1. **ECB date parsing is ambiguous.** The ECB announcement dates are `dd/mm/yyyy` strings and are parsed with `pd.to_datetime(..., errors="coerce")` **without `dayfirst=True`** — the saved run even prints pandas' warning: "Parsing dates in %d/%m/%Y format when dayfirst=False (the default) was specified" [recorded]. Unambiguous values (day > 12) parse day-first, but **ambiguous ones (e.g. `03/02/2022`, `04/04/2012`-style) can be parsed month-first**, silently shifting sentences to the wrong month/quarter. Fix with `dayfirst=True` (or `format="%d/%m/%Y"`) everywhere ECB dates are parsed, then re-derive every ECB figure.
2. **`label` vs `final_label` (ECB).** The train file carries both; the code uses `label`. If `final_label` is the reconciled/adjudicated annotation, the ECB figures currently use the pre-reconciliation labels — confirm which column is authoritative and state it.
3. **Yearly comparability.** The hand yearly index includes **speeches**; the Claude yearly index (cell 30/33) is built from statements + press conferences only — the overlaid "Yearly" panel therefore compares different source mixes. Either add Claude speech annotations to the yearly Claude index or restrict the hand yearly index to statements+presconf for that panel (the panel title already says "statements + press conferences", matching the second option).
4. **Frequency-anchor consistency.** Indices are built variously on `QS` (quarter-start), `QE` (quarter-end), `AS`/`YS`, `YE` across cells; joins/overlays only align when anchors match (`macro_q` is `QS` for the US join in cell 41, `QE` for the ECB). Unify on one anchor (e.g. `QS` everywhere) to avoid off-by-one-quarter joins [inferred hazard; deprecation warnings for `'M'`/`'AS'` are recorded and should be cleaned to `'ME'`/`'YS'` for pandas 3].
5. **Figures are only displayed, never saved** — add `savefig` for the paper's build.

## Reproducibility & environment

- **No RNG** — deterministic given the input files and the FRED snapshot; FRED series revise over time, so record the retrieval date or cache the pulled series [inferred hazard].
- **Environment:** Python 3.13.12, kernel `base` [recorded]; `pandas` 3.x (Pandas-4 deprecation warnings recorded), `matplotlib`, `numpy`, `fredapi`, `openpyxl`, `xlrd` (the `.xls` inputs). None pinned — pin.

## Pre-submission checks

Rotate/purge the FRED key (item 1 above); fix `dayfirst`; resolve `label` vs `final_label`; fix the yearly-panel source mix; unify period anchors; decide Haiku in/out; give the verification CSVs explicit paths; add `savefig`; document the provenance of every Claude annotation file and of `ecb_train-2.xls` (how it relates to the guarded `ecb_train.csv` from `ECB_text_Splitter.ipynb` — the "-2" suggests a re-export with annotations [inferred]); cache/record FRED retrieval.
