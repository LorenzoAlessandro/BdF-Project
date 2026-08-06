# README — `Annotator_Clean.ipynb`

**Role in the project:** converts the hand-labeled FOMC Excel file into instruction-tuning data (TRL-native prompt/completion JSONL) for supervised fine-tuning, including label normalization, cleaning, deduplication, conflict removal, and a seeded stratified train/validation split.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Input

`…/Github Repos/Project/BdF-Project/data/FED/Annotated/fomc_statements_train_1000.xls` — 1,000 rows; columns `id, raw_text, doc_type, clean_text, sentence, label` [recorded]. Notes:

- Legacy `.xls` format → `pd.read_excel` requires **`xlrd`** (the cell's own install note says `pip install pandas ftfy xlrd` [stated]).
- **Label scheme:** raw labels include `D, MD, N, MH, H` — a 5-point scale with "mildly dovish/hawkish" grades — collapsed to 3 classes ("Collapse mildly-dovish/hawkish into main classes" [stated]: `MD→Dovish`, `MH→Hawkish`). Document the original 5-point scheme and the collapse rule in the paper.
- Provenance [inferred]: the column set exactly matches the holdout sampler's output in `Text_Data_Cleaning_Pipeline.ipynb` (which wrote 1,000 rows under `N_SAMPLE = 1000`), so this file is almost certainly that sampler's output after hand-labeling and export to `.xls`. Reconstruct and state the actual lineage (see that notebook's README, audit note 1).

## Outputs

| File | What | Status |
|---|---|---|
| `fomc_statements_train_1000_3_vals.jsonl` | Cell 3: Alpaca-style (`instruction/input/response/text`) with single-letter labels | **Superseded — do not train on this** (see Audit notes) |
| `fomc_train.jsonl` | Cell 4: TRL conversational prompt/completion, 432 examples [recorded] | authoritative |
| `fomc_val.jsonl` | Cell 4: same format, 76 examples [recorded] | authoritative |

## Cell 4 — the authoritative pipeline, decision by decision

Configuration: `VAL_FRACTION = 0.15`, `SEED = 42`, `MIN_SENTENCE_CHARS = 15`. Instruction (fixed for every example): *"Classify the following sentence from a Federal Reserve FOMC statement as Hawkish, Dovish, or Neutral. Answer with exactly one word."*

1. **Mojibake repair.** `ftfy.fix_text` if installed; otherwise a hand-built table of common UTF-8/latin-1 corruption sequences (graceful degradation, with a printed WARNING [stated]); then whitespace collapse.
2. **Label normalization** via `LABEL_MAP` to the full words `Dovish / Hawkish / Neutral` — deliberately "so the model's answer matches the words offered in the instruction" [stated]. Unrecognized labels dropped: **1 row** [recorded].
3. **Minimum length:** drop sentences <15 chars: **3 rows** [recorded] — fragments too short to classify [inferred].
4. **Exact deduplication** on `(sentence, label)`, keep-first: **386 rows removed** [recorded]. *Why so many* [inferred]: FOMC statements recycle boilerplate across meetings, so the same sentence was sampled and labeled at multiple occurrences (consistent with the recurrence counts printed by `Text_Matcher.py`).
5. **Conflict handling:** sentences still duplicated *across different labels* are **dropped entirely** — **50 distinct sentences / 102 rows** [recorded], each conflict printed (e.g., `['Dovish','Hawkish'] :: "Although economic activity is likely to remain weak for a time, …"`). *Why drop rather than adjudicate* [inferred]: conflicting gold labels on identical text are annotation noise; excluding them keeps the training signal clean. **These printed conflicts are raw material for the paper's annotation-consistency discussion — keep the log.**
6. **Stratified 85/15 split.** Per label class: indices shuffled with `random.shuffle` under `random.seed(42)`; `n_val = max(1, round(n · 0.15))` go to validation. Both frames then row-shuffled with `sample(frac=1, random_state=42)`. *Why stratified* [inferred]: preserve label balance in both splits — borne out by the recorded distributions.
7. **Output format:** TRL conversational prompt-completion —
   `prompt = [{role: "user", content: INSTRUCTION + "\n\nSentence: …"}]`, `completion = [{role: "assistant", content: <label>}]` — written with `ensure_ascii=False`.

## Recorded results

1,000 → 996 (label + length drops) → 610 (dedup) → **508** clean rows → **train 432 / val 76** [recorded].

| Label | Train | Val |
|---|---:|---:|
| Neutral | 146 (33.8%) | 26 (34.2%) |
| Dovish | 143 (33.1%) | 25 (32.9%) |
| Hawkish | 143 (33.1%) | 25 (32.9%) |

The near-perfect thirds indicate the labeled batch was balanced. **Report the full attrition chain (1,000 → 508 → 432/76) with the dedup (386) and conflict (50 sentences / 102 rows) figures in the paper's data section.**

## Reproducibility

- **Seed:** `SEED = 42`, consumed by both `random` (per-class shuffles) and pandas (`sample(frac=1, random_state=42)` × 2). Deterministic given the same input file and pandas/Python versions.
- **No LLM inference / no temperature anywhere** — this notebook *produces* SFT data; decoding settings belong to the training/eval code that consumes `fomc_train.jsonl` / `fomc_val.jsonl` (recommend greedy / temperature 0 for the one-word classification evaluation, and say so in the paper).
- **Environment:** Python 3.13.12, kernel `base` [recorded]; needs `pandas`, `ftfy` (optional but recommended), `xlrd`, stdlib. No versions pinned.
- **Path hazard:** absolute macOS path with the trailing-space folder `BANQUE DE FRANCE `.

## Audit notes / pre-submission checks

1. **Cell-order bug:** cell 2 (`Fomc_sents.columns`) executes *before* the data is loaded in cell 3 on a clean run — its saved output (showing the `Instruction_prompt` column) proves it was last executed *after* cell 3. Reorder so the notebook runs top-to-bottom.
2. **Quarantine the cell-3 artifact** (`…_3_vals.jsonl`): it performs **no dedup and no conflict removal** (the 386 duplicates and 102 conflicting rows flow straight into it), and its single-letter responses (`H/D/N`) contradict the instruction's request for full words. Cell 4 supersedes it on both counts.
3. The `id` column is not propagated into the TRL files — fine for training; keep the intermediate files for traceability.
