# README — `Tokenizer-_FED_Branch.ipynb`

**Role in the project:** Track C — turns cleaned long-form Fed/ECB documents into the **unlabeled continued-pre-training corpus** (chunked JSONL), counts tokens per corpus file, and runs an automated **audit gate** that decides whether a corpus is "adequate for pre-training". The opening markdown scopes it as "Textual Cleaning of ECB and FED Stability Reports and Monetary policy reports, effectively all the main text that I will not be labelling" [stated]; the same notebook served multiple corpora across runs by repointing `INPUT_DIRS` / `OUT_PATH`.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## ⚠ Two things to know before running

1. **Proven bug — fresh runs crash at the audit cell.** Cell 23 uses `defaultdict` (`issue_examples = defaultdict(list)`) but **the notebook never imports it**; the saved run succeeded only because the Jupyter kernel happened to have it in scope from earlier session state. A clean top-to-bottom execution raises `NameError: name 'defaultdict' is not defined`. **Fix: add `from collections import defaultdict`.**
2. **Duplicated cleaner.** Cell 11 embeds a copy of the `clean_fed_text.py` cleaner; the functional code is identical to the module (verified by normalized diff), but the **corpus builder uses the notebook copy** while the **audit imports the module's patterns** ("so the audit and the cleaner can never drift out of sync" [stated] — currently true only for the patterns, not the cleaning behavior). Consolidate: have the notebook import `clean_text` from the module.

## Configuration (cell 3), with stated reasons

| Constant | Value | Stated reason |
|---|---|---|
| `TARGET_TOKENS` | 2048 | "inline with the OLMO 2 models … the max number of tokens that the model can process; anything over is handled by splitting" |
| `MIN_TOKENS` | 64 | discard chunks likely to be "boiler plates from a failed scrape" |
| `MIN_CHARS` | 200 | discard too-thin chunks |
| `MIN_ALPHA` | 0.60 | discard low-letter-share chunks ("tables / number dumps") |
| `DROP_NOTES` | True | drop standalone Sources:/Notes:/Chart furniture |

Cells 5/7 preserve, commented out, the CSV→txt pre-steps used for the speeches and FOMC-remainder corpora [stated by the cell headers].

## Tokenizer choice (cell 9) — and a caveat

`tiktoken.get_encoding("cl100k_base")`, justified as sharing the base vocabulary with the **dolma2 tokenizer** (OLMo 2's), so "the token counts will match" [stated]. **Treat this as an author-stated approximation and verify it**: special-token handling and any vocab deltas between cl100k_base and dolma2 could make counts differ slightly — spot-check a sample against the actual dolma2 tokenizer before relying on the 2048 budget exactly. Related observation: the recorded max chunk is **2,054 > 2,048** — the packing loop sums *per-paragraph/per-sentence* token counts, but the `"\n\n"` / `" "` join separators add a few uncounted tokens [inferred from the algorithm]. The stored per-chunk `n_tokens` values are recomputed on the final chunk text and are accurate; only the packing budget slightly overshoots. State the budget as "≈2048" or subtract a separator allowance.

## Chunking (cell 13)

Greedy hierarchical packing: paragraphs (split on blank lines — which is why the cleaner preserves them) are packed into a buffer up to `TARGET_TOKENS`; an **oversize paragraph** is flushed and packed sentence-by-sentence (same `(?<=[.!?])\s+` splitter used project-wide); a single **oversize sentence** is hard-cut on token-id windows (`enc.decode(ids[i:i+target])` — may split mid-word; acceptable for pre-training [inferred]); chunks under `MIN_TOKENS` are dropped at the end.

## Corpus assembly (cell 16 — `build_corpus` + helpers)

Every design reason is written in the comments; in brief:

- Recursive `*.txt` glob per input root, **sorted** (deterministic file order); early loud failure when no files are found — "this has saved me a few times when INPUT_DIRS … was pointed at the wrong place" [stated]; `errors="replace"` reads so "a stray bad byte in one scraped file doesn't kill the whole run" [stated].
- `clean_text` per document (see `README_clean_fed_text.md` for the full five-stage documentation), then chunking.
- **Chunk-level quality gate** `quality_ok` re-checks `MIN_CHARS` / `MIN_ALPHA` because a fragment "can still slip through that's just a fragment of a table or a stray line of chart data" [stated].
- **Cross-file exact dedup**: `dedup_key` = MD5 of whitespace-collapsed, lowercased text — "md5 is just a fingerprint here, not doing anything cryptographic" [stated]; whitespace/case normalization "so two chunks that are identical apart from a line break or capitalization still hash to the same key" [stated]; the `seen` set deliberately spans **all files in the run** "since the repeated boilerplate problem shows up across different reports, not just within one" [stated].
- **Source tagging**: `source_of` = first folder under the input root — "that info is encoded in the folder structure rather than the filename itself" [stated] (for the annual-report run, the year folders became the `source` values).
- **Records**: `{text, source, path, chunk, n_tokens}`, one JSON object per line — JSONL rather than one array "so I can stream through this later without loading the whole corpus into memory" [stated].
- Per-file progress lines and end-of-run `dups_skipped` / `low_quality_skipped` counters "so I can eyeball whether the dedup/quality thresholds are too aggressive or too lax" [stated].
- **No overwrite guard** — `OUT_PATH` is overwritten unconditionally (unlike the labeled-data samplers). Acceptable for derived corpora, but note it.

## Recorded runs

**Annual-report build (cells 17–20):** annual reports 1995–2024 (30 files; year folders → `source`) → **4,002 chunks**; tokens/chunk min 158 / median 1,646 / max 2,054; **6,438,258 tokens**; → `FED_annualreport_unlablled.jsonl` [recorded].

**FED unlabeled-corpus census (cell 22)** [recorded]:

| JSONL | Chunks | Tokens |
|---|---:|---:|
| FED_annualreport_unlablled | 4,002 | 6,438,258 |
| FED_speeches_unlablled | 5,364 | 10,384,997 |
| FED_testimony_unlablled | 92 | 128,595 |
| Financial_Stability_Reports_unlabeled | 297 | 480,600 |
| fed_Supervison_Regulation_unlabeled | 151 | 230,343 |
| fed_minutes_unlabeled | 1,060 | 2,076,029 |
| fed_monetarypolicyrepos_unlabeled | 1,138 | 1,813,700 |
| fed_remaining_statements_unlabeled | 274 | 552,138 |
| fed_unlabeled_beigebooks | 6,441 | 7,322,027 |
| **Total** | **18,819** | **29,426,687** |

## The audit gate (cell 23)

A self-contained QA program with dual entry: CLI via argparse, or in-notebook via `main([])` — "inside a Jupyter notebook the command line belongs to the kernel … sys.exit is also avoided here because in a notebook it only produces the unhelpful 'use %tb' message" [stated]. Design decisions, all [stated]:

- **Patterns imported from `clean_fed_text.py`**, located by a documented search order (script dir → CWD → data dir → whole project tree, with a helpful failure message) — sync-by-import.
- **Schema tolerance:** the text field is auto-detected per file from a candidate list (`text, cleaned_text, clean_text, content, body, document`), falling back to the longest string value, with **per-record re-detection** because "mixed schemas inside one file do happen across my scraping runs."
- **Checks and severities** ("FAIL keeps a document out of the training mix, WARN means I should eyeball it"): FAIL — parse errors; missing text; near-empty docs (`--min-alpha`, default 200 alphabetic chars); mojibake signatures ftfy should have repaired; Unicode leftovers the normalizer should have eliminated; residual boilerplate (the cleaner's own line patterns re-applied — "direct evidence that the cleaning step failed or was skipped" — plus substring boiler-phrases); residual NOTE blocks; **cross-file duplicate documents** (SHA-1 of whitespace/case-normalized text, spanning every audited file — a stronger hash than the builder's MD5). WARN — residual junk lines; an intra-document line ≥10 chars repeated ≥3× ("almost certainly a running header that survived both the hardcoded lists and the frequency filter"); un-joined line-end hyphens ("de-hyphenation did not run"); mid-line split-word leftovers (suspended-hyphen words excluded); ≥25-letter glued tokens; residual URLs ("only a WARN because the body text can very occasionally cite an address legitimately"); **poor reflow** (>40% of lines short non-sentences ⇒ "still in chopped-up PDF form" — ratio-based because "a real document only has a handful" of headings); overall letter density <0.45 ("FED prose sits around 80%").
- **Verdict** = worst severity per document; **corpus gate**: FAIL fraction above `--max-fail-rate` (default **0.0** — "strict by default") → exit code 1, "NOT adequate for pre-training"; optional `--report` writes one JSON row per flagged document ("so I can … feed the FAIL ids into a filtering step that excludes them from the training mix") and the report path is **excluded from the audit set** so it is never mistaken for corpus data on a later run.

**Recorded audit (ECB corpus, `data/ECB/Token`, 14 files)** [recorded]: 16,952 documents; 25,319,172 words / 159,904,440 characters; **PASS 14,471 / WARN 2,435 / FAIL 46 (0.3% fail)** — at the default gate the corpus is *not* adequate until the 46 FAILs are excluded or the rate explicitly tolerated. Top WARNs: residual URLs (1,481 docs), repeated lines (428), residual junk lines (421).

## Reproducibility

- **No RNG** — chunking, dedup, and audit are deterministic given identical input trees and library versions (`tiktoken` vocab included).
- **Environment:** Python 3.13.12, kernel `base` [recorded]; `tiktoken`, `ftfy`, `pandas`, stdlib (`hashlib`, `glob`, `argparse`, `json`, `statistics`, `collections`). No versions pinned.
- **Path hazards:** absolute macOS paths with the trailing-space folder `BANQUE DE FRANCE ` and the misspelled `RESEARCH and SURVAILENCE`.
- **Provenance gap:** because one notebook produced many corpus files across runs, record per emitted JSONL which `INPUT_DIRS`/config produced it (a small provenance JSON next to each corpus file would do).

## Pre-submission checks

1. Add the `defaultdict` import (bug above), then re-run the audit cleanly.
2. Consolidate the duplicated cleaner (import from `clean_fed_text.py`).
3. Exclude the 46 FAIL documents (or an updated FAIL set) from the training mix via the `--report` workflow, and state the final corpus sizes after exclusion.
4. Verify the cl100k_base ≈ dolma2 token-count claim on a sample; describe the chunk budget as ≈2048 (observed max 2,054).
5. Snapshot + hash the input `.txt` trees behind each corpus JSONL.
