# README — `Dezzy_Stats.ipynb`

**Role in the project:** the **descriptive statistics of the annotated corpora** — the notebook behind the paper's corpus-composition figures. It loads all seven annotated spreadsheets (three Fed, four ECB), builds a hawkish–dovish index per institution and document type, and draws the nested-donut label-distribution figure comparing Fed and ECB. "Dezzy" is shorthand for *descriptive*.

This is the only place in the repo where the Fed and ECB annotation files are read **side by side purely to describe them**, before any model is involved — so it is the natural source for the paper's data-section counts.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Inputs (cell 3)

**Fed side**

| Variable | File |
|---|---|
| `STATEMENTS_PATH_FED` | `data/FED/Annotated/FOMC STATEMENTS/fomc_statements_train_1000.xls` |
| `PRESCONF_PATH_FED` | `data/FED/Annotated/Press Conferences/presconf_to_annotate.xlsx` |
| `SPEECHES_PATH_FED` | `data/FED/Annotated/Speeches/Annotated with Shah Merge/lab-manual-sp-train-5768-augmented.xlsx` |

**ECB side**

| Variable | File |
|---|---|
| `STATEMENTS_PATH_ECB` | `data/ECB/Annotated/ecb_train-2.xls` — read with **`header=1`** (a title row sits in A1) |
| `STATEMENTS_PATH_ECB_2` | `data/ECB/Annotated/ecb_test.csv` |
| `PRESCONF_PATH_ECB` | `data/ECB/Annotated/ecb_Press Conference Statements + QA_to_annotate.xlsx` |
| `SPEECHES_PATH_ECB` | `data/ECB/Annotated/speeches_annotated_v3.xlsx` |

The ECB statements train and test files are concatenated into `merged_ECB_Statement` — the same merge the ECB run notebook performs, so the descriptives and the classification runs describe the same sample. These are the **same files** the LLM extension notebooks classify, which is what makes this notebook's counts the right ones to quote.

## The index (cell 5)

```python
def build_index(df, freq, label_col):
    df[label_col] = df[label_col].replace({'MD': 'D', 'MH': 'H'})
    counts = df.groupby([pd.Grouper(key="date", freq=freq), label_col]).size().unstack(fill_value=0)
    return (counts.get("H", 0) - counts.get("D", 0)) / counts.sum(axis=1)
```

Two things distinguish it from the identically-named function in the LLM plots notebooks:

1. **It collapses the 5-point scale itself** (`MD→D`, `MH→H`) before counting. The plots notebooks receive already-collapsed letters from the Excel writer, so they don't need this. Same collapse rule, applied one stage earlier.
2. There is a **second variant, `build_index_speech`**, identical except that it groups on a `year` column instead of `date` — because the Fed speeches file (the Shah et al. merge) carries year-only dates.

Six indices are built [stated]:

| Index | Frequency | Grouping column |
|---|---|---|
| FED Statements | `QE` quarterly | `date` |
| ECB Statements (merged) | `QE` | `date` |
| FED Press Conf | `QE` | `date` |
| ECB Press Conf | `QE` | `date` |
| **FED Speeches** | **`YE` yearly** | `year` |
| ECB Speeches | `QE` | `date` |

⚠ **The Fed speeches index is annual while every other index is quarterly.** That is forced by the source data, not a choice — but it means the Fed and ECB speech indices are *not* directly comparable, and any figure placing them on one axis needs to say so.

⚠ **Recorded warning** [recorded]:

```
UserWarning: Parsing dates in %d/%m/%Y format when dayfirst=False (the default)
was specified. Pass `dayfirst=True` or specify a format to silence this warning.
```

This is the European date hazard. The ECB run notebook (`LLM_Extension_ECB_Side.ipynb`) explicitly parses **day-first**; this notebook does **not**, so `07/06/2021` is read as 6 July here and 7 June there. **Some ECB dates in this notebook are silently wrong**, which shifts sentences between quarters in every ECB index above. Fix with `dayfirst=True` before quoting any quarterly ECB descriptive.

## The figure (cell 7)

`nested_donut` draws a three-ring composition chart per institution, side by side:

- **outer ring** — hawkish/dovish/neutral counts within each document type, labelled with the within-type percentage
- **middle ring** — the document types themselves (Statements, Press Conf, Speeches) as a share of the institution's total
- **centre** — the institution's overall H/D/N split as a solid pie

Colours are fixed: `H = #d62728` (red), `D = #1f77b4` (blue), `N = #7f7f7f` (grey); ring colours are a charcoal/teal/sage palette [stated]. This is the figure that corresponds to `figures/Distribution of Labels by Institution.png`.

## Outputs

**None written to disk.** The donut figure exists only as stored notebook output — there is no `savefig` anywhere in the notebook [stated by absence], so the PNG in `figures/` was exported by hand and has no traceable link back to this code.

## Reproducibility

- **No RNG**; deterministic given the seven input files.
- **Environment:** Anaconda Python 3.13, kernel `base`; `pandas`, `matplotlib`, `numpy`. `xlrd` is needed for the two legacy `.xls` files.
- **Path hazard:** absolute macOS paths with the trailing-space folder `BANQUE DE FRANCE `.

## Audit notes / pre-submission checks

1. **`fed_all` and `ecb_all` are never defined in this notebook.** Cell 7 calls `nested_donut(fed_all, ...)` and `nested_donut(ecb_all, ...)`, and the figure *did* render [recorded] — so those frames were built in a cell that has since been deleted, or typed straight into the kernel. **As committed the notebook cannot be re-run**, and there is no record of how the per-institution frames were assembled (which files, what `doc_type` tagging, whether rows were deduplicated). This is the single most important thing to fix here: the paper's corpus-composition figure currently has no reproducible construction.
2. **Set `dayfirst=True`** on the ECB date parses, then regenerate every ECB index.
3. **Add `savefig` into `figures/`** so the committed PNG is traceable.
4. **State the annual-vs-quarterly asymmetry** for the Fed speeches index wherever it appears.
5. **The six indices are computed but never plotted** in the committed notebook — only the donut figure is. Either plot them or move the index code to where it is used; as it stands, cell 5 does work whose result is discarded.
6. Cell 1 imports `matplotlib` twice (once as `plt`, immediately shadowed by `matplotlib.pyplot as plt`); cell 7 re-imports `pandas`, `numpy` and `matplotlib.pyplot`.
7. Cell 8 is an empty markdown heading ("Distribution and Descriptive Statistics") with nothing under it — the descriptive-statistics tables the title promises are not in the notebook. If they exist, they were lost; if they don't, this is unfinished work in the folder that is meant to supply them.
