# README — `Fed_Speech_Cleaner.ipynb`

**Role in the project:** builds the Fed *speeches* sentence sample. Header, verbatim [stated]: "Fed Speech Cleaner — full method from **Shah, Paturi & Chava (2023)**, for a 2-column CSV … Pipeline (paper section 3.1): (1) clean boilerplate around each speech … (2) split speeches into sentences (NLTK if available, regex fallback otherwise) (3) KEYWORD MATCH: keep 'target' sentences containing a Panel A1/B1 term (**Table 1, Gorodnichenko et al. 2021 dictionary**) (4) sample ~TARGET_N sentences, balanced across speeches (paper: 5 per file → 994) (5) SENTENCE SPLIT: break mixed-tone sentences at but/however/even though/although/while/; keeping the split only if every segment still contains a Table 1 word (6) save everything to one Excel file with 3 sheets."

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs.

> **Citation caveat.** The Shah, Paturi & Chava (2023) and Gorodnichenko et al. (2021) attributions above are reproduced exactly from the code — verify the exact references, section, and table numbers before citing.

---

## Input / output

- **Input:** `…/Data Folders/filtered_speeches/fed_speeches.csv` (produced by `Central_Banker_Speeches_extraction_from_BIS_Frame.ipynb`; 2,671 speeches at snapshot time). `TEXT_COL = "text"`, `DATE_COL = "date"`.
- **Output:** `fed_speeches_sentences_1000.xlsx`, one workbook, three sheets [stated design; rationale inferred: archives the sample, its split form, **and the sampling frame** together for auditability]:
  - `sampled` — the 1,000 sampled sentences (`date`, `sentence`);
  - `sampled_split` — after connective splitting (`date`, `sentence`, `was_split`);
  - `all_target_sentences` — the full candidate pool (81,551 rows).

## Configuration

| Constant | Value | Stated reason |
|---|---|---|
| `SEED` | **5768** | "one of the paper's seeds" [stated] — i.e., a seed used by Shah, Paturi & Chava (2023); verify against the paper |
| `TARGET_N` | 1000 | size of the final sentence sample |
| `MIN_WORDS` | 6 | "drop fragments shorter than this (paper's Fed sentences average ~30 words)" [stated] |

## Method, decision by decision

1. **Dictionary regex — plural-tolerant (differs from the FOMC/ECB pipelines).** `term_to_regex` appends `s?` to **every word** of every term and anchors with `\b…\b` on lowercased text — "word boundaries + optional plural, so 'fund rate' also matches 'federal funds rate'" [stated]. So here `price` **does** match `prices` and `fund rate` matches `funds rate`. Panels A1/B1 (topics) build `TARGET_PATTERN` (the keyword filter); A1∪B1∪A2∪B2 build `ALL_TABLE1_PATTERN` (split validation). **No Panel C negations in this file** (the FOMC/ECB variant includes them in its split-validation pattern). **Cross-corpus comparability caveat:** the FOMC/ECB `_dict_regex` matches exact forms only (`USE_PREFIX_MATCH = False`, letter-lookaround boundaries) — harmonize or justify the difference in the paper.
2. **Boilerplate cleaning — `clean_speech_text`.**
   - Truncate at a `Footnotes / References / Bibliography` section header **only if it occurs in the second half of the document** (`m.start() > len·0.5`). [Inferred rationale]: an early in-text mention of "references" must not amputate the speech; true footnote sections live at the end.
   - Remove "return to text" links, URLs, `[n]` bracketed footnote markers, and digits glued to sentence-ending punctuation (`…rates.12` → `…rates. `).
   - Strip control characters `openpyxl` cannot write (`ILLEGAL_XLSX`) — a practical Excel-compatibility decision [inferred from the name]; collapse whitespace.
3. **Per-sentence keep rule — `keep_sentence`.** Require ≥6 words (**fragments dropped**, not merged — the opposite mitigation to the FOMC pipeline's <6-word *merge*; a difference to note), alphabetic-character ratio ≥0.5 (filters table/number debris [inferred]), and no hit on a speech-specific boilerplate regex: thanks/greetings/congratulations, the "views are my own / not necessarily those of…" disclaimer family, figure/table/slide/box captions, `sources:/notes:`, `references`, "as prepared for delivery", "before I begin", leading "welcome". *Why these patterns* [inferred]: speeches open and close with ceremony and disclaimers carrying no policy signal.
4. **Sentence tokenizer — environment-dependent branch.** NLTK Punkt preferred ("paper uses NLTK" [stated]), with runtime download of `punkt`/`punkt_tab` if missing (**hidden network dependency** — pre-download/vendor for offline reproduction); otherwise a regex fallback `(?<=[.!?])\s+(?=[A-Z0-9"'(])`. **The two branches produce different sentence boundaries ⇒ a different candidate pool ⇒ a different seeded sample.** The recorded run printed "**using NLTK sentence tokenizer**" [recorded] — reproduction must take the same branch (and, strictly, the same Punkt model files).
5. **Pool construction.** Iterate speeches in CSV row order; clean; tokenize; keep sentences passing `keep_sentence` **and** containing a target term; record `file_id` (row index) and `date`; **drop exact duplicate sentences** (keep first).
6. **Sampling to exactly 1,000, balanced across speeches.** Per-speech quota `per_file = ceil(TARGET_N / n_files)`; shuffle the pool (`sample(frac=1, random_state=5768)`); `groupby("file_id").head(per_file)`; a seeded top-up fills any shortfall from the remainder; a seeded down-sample trims any excess to exactly `TARGET_N`; sort by `file_id`. *Why balanced* [stated/inferred]: mirrors the paper's per-file sampling ("paper uses 5" per file [stated]) and prevents long speeches from dominating [inferred]. With 2,527 contributing speeches the quota was **1/speech**, and the 2,527 quota picks were down-sampled to 1,000 [recorded: "quota 1/speech"].
7. **Connective splitting — speech variant.** Split at `; , but , however / even though / although / while` — this regex additionally absorbs an optional comma before `but`/`however` and strips ` ,.;` from segment edges (slightly different mechanics from the FOMC/ECB `SPLIT_RE`). Validity: **every** segment must match the (plural-tolerant) all-panels pattern, else the sentence stays unsplit. `was_split` and `segment` are recorded per row (the `segment` column is not written to the sheet).
8. **Excel write** with an explicit, helpful failure if `openpyxl` is missing [stated: `SystemExit("pandas needs openpyxl … pip install openpyxl")`].

## Recorded run numbers

"using NLTK sentence tokenizer" · **81,551 target sentences from 2,527 speeches (32.3 per speech)** — note 2,527 < the 2,671 speeches in the input, i.e. **144 speeches yielded no target sentence** after cleaning [recorded + inferred arithmetic] · "sampled **1000** sentences (quota 1/speech)" · "after splitting: 1000 → **1019** sentences" · saved to `filtered_speeches/fed_speeches_sentences_1000.xlsx` [all recorded].

## Reproducibility

- **Seed 5768** consumed by three `DataFrame.sample` calls (shuffle, top-up, down-sample). Deterministic given the same pandas version, the same input CSV (row order matters — `file_id` is the row index), **and the same tokenizer branch**.
- **Dates pass through as raw CSV strings** (e.g., `1997-12-15 00:00:00`) — normalize to `YYYY-MM-DD` before merging with the FOMC/ECB frames, which use that format.
- **Environment:** Python 3.13.12, kernel `base` [recorded]; `pandas`, `nltk` (+ Punkt models), `openpyxl`, stdlib (`math`, `re`, `pathlib`). No versions pinned.
- **Path hazard:** absolute macOS path with the trailing-space folder `BANQUE DE FRANCE `.

## Pre-submission checks

1. Verify the Shah/Paturi/Chava seed claim (5768) and the §3.1 method description against the actual paper.
2. Decide whether the plural-tolerant matching here vs. exact-form matching in the FOMC/ECB pipelines is intended; harmonize or justify.
3. Vendor the NLTK Punkt models and record their versions; state in the appendix that the NLTK branch (not the regex fallback) produced the published sample.
4. Note the fragment policy difference (drop-here vs. merge-in-FOMC) wherever cross-corpus statistics are compared.
