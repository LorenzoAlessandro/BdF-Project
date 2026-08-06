# README — `FED_Fomc_announcements_Id_mathcher.py`

**Role in the project:** date re-attachment, **Pass 1 (id-based)**. The original FOMC pipeline (`Text_Data_Cleaning_Pipeline.ipynb`) dropped the date column before sampling the labeled holdout, so the annotated file has stable `id`s but no dates. This standalone script re-runs the full cleaning pipeline "same rules as the original run, so indices 0..N-1 line up with the `id` column in your annotated holdout" [stated], **with a date column carried through**, then attaches dates by positional lookup, verifies every row, and saves — never mutating the annotated CSV.

Its own header states the validity condition precisely [stated]: *"the id matching is only valid if the cleaning rules here are IDENTICAL to the run that generated the holdout. The exact_match check at the end is precisely the test of that."* That test subsequently failed for some rows (see `Text_Matcher.py`, Pass 2).

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from saved outputs of the companion notebook.

---

## Inputs / outputs

| | Path (under `…/FED BRANCH/POLICY DECISIONS/FOMC ANNOUNCMENTS`) |
|---|---|
| Input (raw) | `fomc_minutes_statements.csv` |
| Input (labeled, **read-only**) | `fomc_statements_test_500.csv` (`id` + `label` + sentence columns) |
| Output | `fomc_statements_test_500_with_dates.csv` (labels untouched; `id, date` moved to front) |
| Output | `id_match_report.csv` (full side-by-side verification table) |

Runs "top to bottom as a plain script OR as a notebook" [stated] (`# %%` cell markers).

## Part 1 — pipeline rebuild (identical rules + date)

Everything is line-for-line the `Text_Data_Cleaning_Pipeline.ipynb` machinery (see that README for the full decision-by-decision documentation of `clean_text`, `strip_leading_wrapper`, `END_MARKERS`/`HEADER_MARKERS`/`NOISE`/`ROSTER`, the <6-word fragment merge, `is_boilerplate_sentence`, the Table 1 dictionary with `USE_PREFIX_MATCH = False`, `split_on_connectives`, and the `(?<=[.!?])\s+` base splitter). Additions specific to this script:

1. **Date-column auto-detection `_detect_date_col`** — three tiers: (1) any datetime-dtype column; (2) any column whose **name** matches `date|release` (case-insensitive), scanning columns in order; (3) any column where >80% of a 50-row sample parses via `pd.to_datetime`. Fails loudly with instructions to set `date_col` manually if nothing matches [stated]. On the current schema (`Date, Release Date, Type, Text`) this deterministically resolves to **`Date` — the meeting date — because `Date` precedes `Release Date` in column order** [inferred from the algorithm + the recorded column order].
   **Decision consequence to confirm:** minutes are stamped with the *meeting* date, not the ≈3-weeks-later *release* date. For an event study on information arrival, minutes should arguably carry `Release Date`; if the unit of analysis is the meeting, `Date` is correct. Make the choice explicit in the paper; `date_col = "Date"` / `"Release Date"` can be set manually at the top.
2. **Text-column auto-detection** — greatest mean string length among non-numeric, non-datetime candidates; resolves to `Text`. Both detections are printed (`"text column: … | date column: …"`) — **check that line on every run**; a silent upstream schema change would otherwise change which columns are used.
3. Dates normalized to `YYYY-MM-DD` via `pd.to_datetime(errors="coerce").dt.strftime(…)` — unparseable dates become missing and propagate as NaN (the match report is where to catch them).
4. The rebuilt frame is reduced to `[date, doc_type, sentence]`; **row order is what carries the id correspondence**.

## Part 2 — id matching

Loads the annotated CSV; hard-fails (`KeyError`) if there is no `id` column; **warns** on duplicate ids; **warns** on ids outside `0..N-1` — "These get NaN dates — likely the pipeline changed since the holdout was drawn." [stated]. Dates attached via `sents.reindex(annotated["id"])`.

## Part 3 — verification

Builds a side-by-side table (`sentence_main` from the rebuild vs. `sentence_annotated` from the labeled file) and computes `exact_match` on **whitespace-normalized** text (an empty rebuilt sentence never counts as a match). Prints the match count, the first 10 rows, and then either

- *"No mismatches — every annotated id points at the identical sentence. Dates are safe to use."*, or
- the **full** mismatch list with: *"do NOT trust these dates until you work out why the ids no longer line up"* [stated].

## Part 4 — save, with guards

- `OUTPUT_CSV` is **overwrite-guarded**: `FileExistsError(" … already exists — delete/rename it to regenerate.")` [stated].
- `MATCH_REPORT` is written **unconditionally** (no guard) — inconsistency worth fixing.
- The annotated input is never modified ("labels untouched" [stated]).

## Reproducibility

- **No RNG** — fully deterministic given identical inputs and library versions. Its *correctness*, however, is conditional on rule-and-input identity with the original holdout run; that is exactly what Part 3 tests.
- **What made the test fail** [inferred — two candidate mechanisms, both consistent with the evidence]: (a) `fomc_minutes_statements.csv` is a living, **newest-first** scrape, so re-scraping after new FOMC meetings *prepends* rows and shifts every downstream sentence index; and/or (b) an edit to any cleaning rule between the original run and the rebuild. Determine which applied and state it in the replication appendix.
- **Environment:** Python 3 (script), `pandas` + stdlib only. No versions pinned. Path hazards: the `DATA_DIR` contains the trailing-space folder `BANQUE DE FRANCE ` and the misspelled `FOMC ANNOUNCMENTS` — reproduce verbatim or switch to relative paths. The config cell is marked "change for your machine!!" [stated].

## Pre-submission checks

1. Record (in the appendix) the exact `exact_match` count this script printed on the run you used, and how each mismatched row was resolved (Pass 2 / manual).
2. Decide and document `Date` vs `Release Date` for minutes.
3. Optionally add an overwrite guard to `MATCH_REPORT`.
