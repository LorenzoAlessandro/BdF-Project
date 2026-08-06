# README — `ECB_text_Splitter.ipynb`

**Role in the project:** the ECB arm of the labeled-data track. Merges the per-year scraped ECB "Monetary policy decisions" `.txt` files (1999–2026), isolates the release body from page chrome, parses the dateline, runs the **identical** FOMC sentence stage — the dictionary and connective rules are "copied verbatim from your FOMC notebook so the ECB corpus is filtered by the IDENTICAL dictionary and connective rules — that's what keeps the two banks comparable. If you later change the dictionary on the FOMC side, mirror it here." [stated] — and draws a seeded train/test split for manual labeling.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs.

---

## ⚠ Cell layout — read this first

- **Cell 0**: the pipeline through sentence extraction. **Superseded.**
- **Cell 1**: displays `ecb_sents`.
- **Cell 2**: the **same pipeline plus the train/test split appended** (a strict superset of cell 0; verified by diff). Its saved output re-ran the whole pipeline and then wrote the split. **Treat cell 2 as authoritative; delete or clearly deprecate cell 0** so the notebook can't be double-run into inconsistent states.

## Inputs / outputs

- **Input:** `…/ECB BRANCH/ECB Monetary policy decisions/txt/<year>/*.txt` — one scraped release per file, per-year subfolders `1999 … 2026` [stated]. 314 files at snapshot time [recorded].
- **Outputs** (to `…/ECB Monetary policy decisions/out/`):
  - `ecb_statements_clean.csv` + `.jsonl` — one row per release (`source_file, folder_year, date, doc_type, raw_text`), JSONL with `force_ascii=False` "keep €/–/curly quotes readable" [stated];
  - `ecb_sentences.csv` — the filtered/split sentence set (1,058 rows [recorded]);
  - `ecb_train.csv`/`.jsonl` (800) and `ecb_test.csv`/`.jsonl` (258) [recorded] — CSVs keep `id, source_file, date, doc_type, sentence, label`; JSONLs keep `id, sentence, label`.

## Method, decision by decision

The page layout is constant 1999–2026 and is diagrammed in the header [stated]: chrome above the title → `PRESS RELEASE` → `Monetary policy decisions` (title anchor) → `<dd Month yyyy>` dateline → **the release body** → "The President of the ECB will comment …" sign-off → chrome below.

1. **Merge (STEP 1) — `gather_ecb_txt`.** `rglob` over the year folders (sorted → deterministic order); per file: `source_file` = path relative to `txt/` "so identically-named files in different year folders never collide" [stated]; `folder_year` from the path via `(?:19|20)\d{2}`; raw text read with `errors="replace"`. No cleaning yet, "so you can eyeball the merged pile first" [stated].
2. **Body isolation (STEP 2) — `extract_ecb_body`.**
   - **Cut the footer first** at the earliest line-start `ECB_END_MARKERS` hit: "the president of the ecb will comment" — "a terminator, not noise: it's always the last line of the release, and on some pages (e.g. 6 June 2002) it's broken mid-sentence across three lines … so cutting here also removes those dangling fragments" [stated] — plus `related topics`, `see also`, `contact` (alone on its line), `media contacts`, `disclaimer`, cookie/feedback banners. **Anchored to line starts "so a stray 'contact' inside prose can't clip the body"** [stated].
   - **Find the body start:** locate the **first** occurrence of "monetary policy decisions" — "anchoring on the FIRST … means a body that repeats the phrase (e.g. 'took the following monetary policy decisions:') is left untouched" [stated] — then skip past the first full dateline `\d{1,2}\s+<Month>\s+\d{4}` after it: "'in the first half of 2000' won't match (needs day+month+year), so the first hit is always the real dateline" [stated].
   - **Line-wise tidy:** drop exact navigation-chrome lines from a fixed whole-line set (skip-links, menu items, search-widget tokens, `press release`, `monetary policy decisions`, …) — "matched as WHOLE lines … so they can never clip real prose" [stated] — and `***` separator lines, which are "genuine mid-body markup" whose *following* paragraph ("The Governing Council stands ready …") is real content, hence deleted line-wise rather than used as a terminator [stated].
3. **Date parsing — `parse_ecb_date`.** First `d Month yyyy` after the title; `strptime("%d %B %Y")` → `YYYY-MM-DD` "matches the FOMC date format" [stated]; `None` on failure.
4. **Frame shaping + cross-checks — `clean_ecb_frame`.** Output shaped like the FOMC `docs` (`raw_text` + `doc_type`) so it "drops straight into the downstream you already built" [stated]. `ECB_DOC_TYPE = "statement"` kept deliberately for compatibility with the FOMC holdout sampler; the alternative `"ecb_statement"` is documented for the concat-both-banks case [stated]. Pages with no extractable body are dropped and each is printed ("flag them, never guess" [stated]); a **folder-year vs dateline-year cross-check** prints any mismatch ("a misfiled or misparsed document" [stated]). Frame sorted by date.
5. **Sentence stage — identical FOMC machinery** (verbatim copies of `clean_text`, the `(?<=[.!?])\s+` base splitter, Panels A1/B1/A2/B2/**C**, `USE_PREFIX_MATCH = False`, `_dict_regex` with letter-lookaround boundaries, `is_target_sentence`, `SPLIT_RE`, `split_on_connectives` — see `README_Text_Data_Cleaning_Pipeline.md` §8–9 for the full documentation, including the exact panel lists). One ECB-specific addition:
   - **`split_blocks` — newlines are hard sentence boundaries.** Split on `\n+` **first**, then run `clean_text` + the base splitter *within* each block. Reason, verbatim [stated]: "an ECB page puts each paragraph and each subsection header on its own line, so a newline is a real sentence boundary (unlike your FOMC PDF text, where a line break is a mid-sentence wrap …). Without it, modern releases glue a header onto the next sentence ('Refinancing operations As banks are repaying …'); with it, a bare header becomes its own fragment and the Panel A1/B1 filter drops it."
   - **No FOMC boilerplate guard** is included: `extract_ecb_body` already removed the chrome, so "there's nothing for it to catch" [stated].
6. **Documented coverage limitation — reproduce in the paper** [stated verbatim]: pre-2008 releases call the headline rate the "**minimum bid rate**"; Panel A1 "has interest/bank/fund rate but not 'bid rate', so the main-rate sentence in 1999–2007 releases won't pass the filter (the marginal-lending and deposit 'interest rate' sentences still do). Add 'bid rate' to PANEL_A1 if you want them — left as-is here since the dictionary is your call." **Decide, and state the decision.**
7. **Train/test split (cell 2 tail).** `N_TRAIN = 800`, `SEED = 42`, `OVERWRITE = False`. `make_train_test` is pure ("WITHOUT mutating df. Deterministic for a given seed" [stated]): `train = df.sample(n=min(800, len), random_state=42)`, `test` = the rest. `_prep` adds a stable `id` = the `ecb_sents` row index ("trace back to ecb_sents" [stated]) and an empty `label` column ("fill in by hand" [stated]). Overwrite guard, verbatim [stated]: "Regenerating would create a DIFFERENT split and discard any labels you've started. Set OVERWRITE=True to proceed." — same safeguard as the FOMC holdout.

## Recorded run numbers

314 raw files merged · **one year mismatch flagged: `2020/2020-03-12-ecb-mp.txt` folder=2020, dateline=2019** · 314 releases retained with an isolated body · filtration kept **1,033 / 1,935** sentences (Panel A1/B1) · connective splitting **1,033 → 1,058** segments · split **train 800 / test 258** [all recorded].

## Reproducibility

- **Seed:** `SEED = 42` via `DataFrame.sample(n=800, random_state=42)` — the only RNG. Deterministic given the same file tree (row order comes from the sorted `rglob` and the date sort) and pandas version.
- **Environment:** Python 3.13.12, kernel `base` [recorded]; `pandas`, stdlib (`re`, `datetime`, `pathlib`, `unicodedata`). No versions pinned.
- **Path hazard:** absolute macOS path with the trailing-space folder `BANQUE DE FRANCE `.
- Input tree drift: `txt/2026/` grows with new decisions; snapshot + hash the tree used for the published numbers.

## Audit notes / pre-submission checks

1. **Known-bad date:** the flagged `2020-03-12` release parsed a **2019** dateline — the date parser latched onto a 2019 date on that page, so this release (the 12 March 2020 decision) very likely carries a **wrong date** in `ecb_docs` / `ecb_sents` and any split rows derived from it [recorded + inferred consequence]. Verify the file, correct the date manually, and re-derive the affected downstream rows.
2. **Deprecate cell 0** (superseded by cell 2).
3. The dictionary is a manual copy of the FOMC one — if either side changes, **mirror it** [stated]; better: factor into a shared module so drift is impossible.
4. Do **not** regenerate the 800/258 split for the published results (the guard protecting started labels is intentional).
5. `minimum bid rate` coverage decision (point 6 above) — decide and document.
