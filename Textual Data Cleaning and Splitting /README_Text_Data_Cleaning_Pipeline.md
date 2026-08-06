# README — `Text_Data_Cleaning_Pipeline.ipynb`

**Role in the project:** the original FOMC sentence pipeline. Cleans FOMC statements & minutes, isolates the statement body, normalizes text, splits into sentences, applies the Table 1 dictionary filtration and connective splitting, and draws a seeded holdout for manual labeling, "kept separate from all the training of the model to avoid contaminating the data set" [stated]. The two matcher scripts (`FED_Fomc_announcements_Id_mathcher.py`, `Text_Matcher.py`) exist solely to replicate this pipeline's rules; `Annotator_Clean.ipynb` consumes the hand-labeled descendant of this notebook's holdout.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs.

> **Citation caveat.** The dictionary is attributed in-code to "Gorodnichenko et al." ("Table 1") — reproduced here as written; verify the exact reference before citing.

---

## Input

`fomc_minutes_statements.csv` — columns `Date, Release Date, Type, Text`; **ordered newest-first** (row 0 of the saved output is the 2026-06-17 statement) [recorded]; 465 documents after type filtering [recorded]. A living scrape — snapshot + hash the exact file used. The newest-first ordering matters: a re-scrape after new FOMC meetings *prepends* rows and shifts every downstream sentence index (see "Holdout" below and the matcher READMEs).

## Outputs

- `fomc_statements_test_500.csv` / `.jsonl` — the seeded holdout for labeling (columns `id, raw_text, doc_type, clean_text, sentence, label`, `label` empty). **Naming hazard: the saved configuration is `N_SAMPLE = 1000` and the recorded run wrote 1,000 rows into these `_500`-named files** (docstring still says "500-sentence holdout"). See Audit notes.
- `fomc_sentences_remaining.csv` — everything not held out (17,019 rows [recorded]).

## Pipeline, decision by decision

1. **Type split (cell 7).** `Type` matched case-insensitively against `\bstatement\b` / `\bminute\b` — word boundaries "to avoid partial matches" [stated].
2. **Text-column auto-detection + stacking (cell 9).** Among non-numeric, non-datetime columns (excluding `Type`), pick the one with the greatest mean string length [inferred rationale: schema-robust detection of the body column]. Result: `Text` [recorded]. Statements then minutes are concatenated with a `doc_type` tag and reduced to `[raw_text, doc_type]`. **The date columns are dropped here** — the root cause of the later date re-attachment work (Pass 1/Pass 2 scripts): the holdout was born without dates. *Fix for future runs:* carry `Date`/`Release Date` through from this cell.
3. **Normalization — `clean_text` (cell 12 §1).** NFKC; curly quotes → straight; en/em dashes → `-`; NBSP → space; de-hyphenation across line breaks (`infla-\ntion` → `inflation` [stated]); newlines → spaces; whitespace collapse. Purpose: one canonical single-line form so splitting and keyword regexes behave uniformly [inferred].
4. **Leading-wrapper strip.** A repeated alternation removes any leading run of `Share / Print / PDF / For immediate release / For release at …<year> / Last update …<year> / <Month d, yyyy> / <bare 4-digit year>` plus separators. Reason, verbatim [stated]: `clean_text` collapses line breaks, which would otherwise glue this header block onto the first body sentence, and the per-sentence header skip would then delete real content; stripping at document level "keeps the first sentence intact. No-op when absent."
5. **Body isolation — `keep_statement_sentences` (cell 12 §2).**
   - **Stop** at the first sentence matching `END_MARKERS` (`voting for/against/in favor`, `voted for/against/to/in favor/unanimously`, `voting on this/the action/matter`): the voting/dissent paragraph onward is roster/administrative text — "the body is over" [stated]. **Consequence to confirm:** the rule was designed for statements but applies to minutes too, so **each minutes document is truncated at its first voting-language sentence** and everything after (including dissent explanations) is discarded. Confirm intent and state it in the paper.
   - **Skip** `HEADER_MARKERS` (release headers / scraped nav: `for immediate release`, `for release at`, `last update`, leading `share/print/pdf`).
   - **Skip** `NOISE` (media-inquiry lines, "please e-mail/call", `implementation note`, URLs, `federalreserve.gov`, phone numbers, e-mail addresses, the scraper artifact `[email protected]`, "board of governors of the federal reserve") — administrative furniture; a "backup for docs with no voting marker" [stated].
6. **Fragment merge — `clean_doc_sentences`.** Kept sentences **shorter than 6 words** are appended to the previous sentence within the document. [Inferred rationale]: the regex splitter over-splits at abbreviations/decimals; merging preserves grammatical units. (Contrast: `Fed_Speech_Cleaner.ipynb` *drops* <6-word fragments instead — a cross-corpus difference to note.)
7. **Final boilerplate guard — `is_boilerplate_sentence`** ("belt-and-suspenders" [stated]). Re-applies END/HEADER/NOISE plus a **name-roster heuristic**: ≥3 commas/semicolons AND >60% Title-case tokens AND a roster title (`chair(man)`, `vice chair(man)`, `governor`, `secretary`, `president`) ⇒ drop. Intent stated ("name-list heuristic"); the specific thresholds are unexplained in code [inferred: tuned by inspection].
8. **Data & Title Filtration.** Keep a sentence iff it contains a Panel A1/B1 *topic* term. Dictionary (attributed to Gorodnichenko et al., "Table 1" [stated]):

   ```python
   PANEL_A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate",
               "price", "economic activity", "inflation", "employment"]
   PANEL_B1 = ["unemployment", "growth", "exchange rate", "productivity",
               "deficit", "demand", "job market", "monetary policy"]
   PANEL_A2 = ["anchor", "cut", "subdue", "decline", "decrease", "reduce", "low",
               "drop", "fall", "fell", "decelerate", "slow", "pause", "pausing",
               "stable", "non-accelerating", "downward", "tighten"]
   PANEL_B2 = ["ease", "easing", "rise", "rising", "increase", "expand", "improve",
               "strong", "upward", "raise", "high", "rapid"]
   PANEL_C  = ["weren't", "were not", "wasn't", "was not", "did not", "didn't",
               "do not", "don't", "will not", "won't"]
   ```

   Matching (`_dict_regex`): multi-word phrases joined with `\s+`; alternatives sorted longest-first; letter-lookaround anchors `(?<![A-Za-z]) … (?![A-Za-z])`; case-insensitive. Word-level, not substring, "so 'low' never fires on 'below'/'flow'", consistent with the source table enumerating inflections by hand; accepted consequence: `expand` does **not** match `expanded`, `price` does **not** match `prices` [stated]. `USE_PREFIX_MATCH = False` (a documented switch to relax this "if recall is too low" [stated]) — **off in the recorded run**. Ordering rationale, verbatim [stated]: filtration runs *after* body isolation "so a voting line such as 'Voting for the monetary policy action were …' (which contains the B1 term 'monetary policy') is removed before the keyword filter ever sees it."
9. **Connective splitting.** Split at `;` and `but / however / even though / although / while` (connective consumed); **keep the split only if every segment still contains a Table 1 term (any panel)**, else return the sentence unsplit — isolates mixed-tone clauses without creating off-topic orphans [stated intent across the companion files; mechanism identical here].
10. **Base sentence splitter.** `re.split(r"(?<=[.!?])\s+", text)`. The cell prefers a `get_splitter()` result if one exists in the kernel; none exists in the saved notebook, so **the regex fallback is what actually ran** [inferred; consistent with both matcher scripts hard-coding the same regex].
11. **Holdout sampler (cell 15).** `N_SAMPLE = 1000`, `SEED = 42`, `DOC_TYPE = "statement"` (only statement sentences eligible; minutes stay in the main frame), `OVERWRITE = False`. Safeguards, all [stated]: pure split function (no mutation of `sents` until files are written); fixed seed; overwrite guard so a re-run "can't silently regenerate a *different* 500 and discard labels you've already added"; stable `id` = original row index in the **full combined** `sents` frame "so you can trace it back"; empty `label` column; a warning if `OUT_DIR` doesn't pre-exist ("catches typos/trailing spaces in the path"); `force_ascii=False` "so en-dashes / curly quotes stay human-readable".

## Recorded run numbers

465 documents → 41,248 sentences after explode → boilerplate guard removed **370** → 40,878 → filtration kept **16,799 / 40,878** → connective splitting **16,799 → 18,019** segments. Holdout: "Held out **1000** sentences for labeling → fomc_statements_test_500.csv / .jsonl. Main frame now **17,019** rows (**1,019** of doc_type='statement')" [all recorded] — i.e., 2,019 statement segments existed in total, 1,000 were held out.

## Reproducibility

- **Seed:** `SEED = 42` via `DataFrame.sample(n, random_state=SEED)` — the only RNG in the notebook. Deterministic for a given pandas version **and** a byte-identical input file + identical rules upstream (the sample is drawn from a frame whose row order the whole pipeline determines).
- **Environment:** Python 3.13.12, kernel `base` [recorded]. Imports include `bs4`, `datasketch`, `ftfy`, `argparse`, `dataclasses`, `typing` that are **never used in the executed cells** (they belong to a larger pipeline design [inferred from the import comments]) — but the import cell fails if they're missing, so they are de facto requirements as written. Trim or install. No versions pinned.
- **Path hazards:** absolute macOS paths under `…/BANQUE DE FRANCE /Data Folders/FED BRANCH/POLICY DECISIONS/FOMC ANNOUNCMENTS` — note the **trailing space** in `BANQUE DE FRANCE ` and the misspelled `ANNOUNCMENTS`, both of which must be reproduced verbatim if the tree is mirrored.

## Audit notes / pre-submission checks

1. **`_500` filename vs. 1,000-row content.** Docstring says 500; saved config says 1,000; recorded output wrote 1,000 rows into the `_500`-named files. Most plausible history [inferred]: an initial N=500 run created the test set; N was later raised to 1,000 (guard bypassed or file moved) to produce the batch that became the hand-labeled `fomc_statements_train_1000.xls` — the column sets match exactly. **Reconstruct the actual history, rename files to match contents, and state the final train/test provenance in the paper.**
2. **Dates dropped in cell 9** — root cause of the entire date re-attachment track; carry dates through in any future run.
3. **Minutes truncation** at the first voting-language sentence — confirm intended.
4. Do **not** regenerate the holdout for the published results — the overwrite guard protecting the labeled data is there on purpose.
