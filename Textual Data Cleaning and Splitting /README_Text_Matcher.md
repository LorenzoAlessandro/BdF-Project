# README — `Text_Matcher.py`

**Role in the project:** date re-attachment, **Pass 2 (text-based)**. Run after the id-based Pass 1 (`FED_Fomc_announcements_Id_mathcher.py`) "left mismatches — the pipeline no longer reproduces the exact indices from the original run" [stated]. This script ignores ids entirely: it rebuilds the same full sentence frame (self-contained, identical rules), then assigns dates by **word-for-word** sentence-text matching, with a deliberately conservative policy for the cases where text matching cannot be trusted.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify.

---

## Inputs / outputs

| | Path (under `…/FED BRANCH/POLICY DECISIONS/FOMC ANNOUNCMENTS`) |
|---|---|
| Input (raw) | `fomc_minutes_statements.csv` |
| Input (labeled, **read-only** [stated]) | `fomc_statements_test_500.csv` |
| Output | `fomc_statements_test_500_dates_textmatch.csv` (leads with `id, date, match_method`) |
| Output | `text_match_report.csv` |

## Part 1 — pipeline rebuild

Identical to Pass 1 / the original notebook: type split, text/date column auto-detection (resolving to `Text` and `Date` on the current schema — see the Pass 1 README for the meeting-date-vs-release-date consequence), `clean_text` + leading-wrapper strip, body isolation (voting-record cutoff, header/noise skip, <6-word fragment merge), boilerplate guard, Panel A1/B1 topic filter, connective splitting. See `README_Text_Data_Cleaning_Pipeline.md` for the full decision-by-decision documentation of those shared rules.

## Part 2 — the text lookup

- Normalization for matching: `_norm` = **whitespace collapsed, case preserved** — matching is therefore **case-sensitive** by design; a labeling tool that altered capitalization would surface as `unmatched`.
- A lookup is built from normalized sentence text → the **sorted set of distinct dates** it appears under.
- The script prints how many rows in the rebuilt frame share their exact text with at least one other row — quantifying the core problem: "recurring boilerplate across meetings" [stated].

## Part 3 — matching policy (the heart of the script)

Per annotated sentence, all rationales [stated] in the header/comments:

| Case | `match_method` | `date` | Stated reason |
|---|---|---|---|
| Exactly one candidate date | `text_exact` | assigned | unique word-for-word hit |
| Same text under **multiple different dates** | `ambiguous` | `NaN`, all candidates listed in `candidate_dates` | "FOMC boilerplate recurs across meetings, so guessing one date would be wrong" — "no single date can be assigned from text alone" |
| No hit | `unmatched` | `NaN` | "usually mean a cleaning rule changed the sentence text itself (e.g. different splitter, merged fragments). They need manual dating or a fuzzy pass." |

## Part 4 — report

Prints a match-method summary and the **full** `unmatched` and `ambiguous` listings (id + sentence [+ candidate dates]) for manual review — these listings are the required follow-up work, not an error to ignore.

## Part 5 — save, with guards

- `OUTPUT_CSV` is **overwrite-guarded** (`FileExistsError`); the annotated input "is NOT touched" [stated].
- `MATCH_REPORT` is written **unconditionally** (no guard).

## Reproducibility

- **No RNG** — deterministic given identical inputs and library versions.
- **Environment:** `pandas` + stdlib. No versions pinned. Same path hazards as Pass 1 (trailing-space `BANQUE DE FRANCE `, misspelled `FOMC ANNOUNCMENTS`); config marked "change for your machine!!" [stated].

## Audit notes — code/comment discrepancies to fix or footnote

1. **`text_exact_dup` is promised but never emitted.** The header describes a fourth label — duplicate rows that all share ONE date → "safe and assigned with method = 'text_exact_dup'" — but the implementation aggregates *unique* dates per text key, so that case is indistinguishable from a single occurrence and is labeled `text_exact`. **Date assignments are unaffected** (the semantics are equivalent for correctness); only the audit granularity described in the comment is missing. Either implement the label or delete it from the header before the code ships with the paper.
2. `OUTPUT_CSV` and `MATCH_REPORT` contain the same rows (the report is simply not column-reordered) — near-duplicate outputs; harmless, but worth knowing.
3. Consider adding the missing overwrite guard on `MATCH_REPORT`.

## Pre-submission checks

- Record, for the run used in the paper: the `text_exact` / `ambiguous` / `unmatched` counts, and how every `ambiguous`/`unmatched` row was ultimately dated (manual resolution, exclusion, or fuzzy pass).
- If any rows remain undated, state how they are handled in the analysis.
