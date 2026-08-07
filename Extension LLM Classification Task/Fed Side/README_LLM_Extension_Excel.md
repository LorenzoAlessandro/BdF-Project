# README — `LLM_Extension_Excel.ipynb`

**Role in the project:** the LLM benchmark, **Excel edition** — the same 8-model × 5-prompt engine as `LLM_Extension.ipynb`, run directly over the **annotation spreadsheets**, with predictions written back into a copy of each workbook "next to your labels with the timeline preserved" [stated]. This is the notebook that produced `fed_statements_presconf_WITH_LLM_LABELS.xlsx` and `fed_speeches_WITH_LLM_LABELS.xlsx`, the inputs to `LLM_Extension_Plots.ipynb`. "Pipeline (prompts, sampling settings, retries, resume, cost tracking) is identical to the verified JSONL version" [stated] — with the deltas documented below.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## The two configured runs (cell 26) — recorded inputs

| Run tag | Sources | Merge rule | Few-shot demos from | n classified [recorded] | Gold distribution [recorded] |
|---|---|---|---|---|---|
| `fed_statements_presconf` | `fomc_statements_train_1000.xls` + `presconf_to_annotate.xlsx` | concatenated, **stable-sorted by date** (ties: statements-then-presconf, original order within each) [stated] | `SPEECHES_PATH` | **1,998** | H 491 / N 877 / D 628 / **unscored 2** |
| `fed_speeches` | `lab-manual-sp-train-5768-augmented.xlsx` | year-only dates → never merged, original order [stated] | `STATEMENTS_PATH` | **951** | D 201 / N 520 / H 230 |

Full run plan: **8 × 5 × {1,998; 951} = 117,960 calls** [recorded]. (The speeches filename — "Annotated with Shah Merge / lab-manual-sp-train-5768-augmented" — indicates the speeches annotations were merged/augmented with the Shah et al. lab-manual speech data under the 5768 sample; document that construction in the data appendix [inferred from folder/file names].)

## Label handling — two layers (cells 5–6)

- `LABEL_MAPPING` (digits 0/1/2) drives the prompts, **unchanged** from the JSONL edition [stated].
- `GOLD_TO_DIGIT` translates the human letters **for scoring only**: `D/MD → 0`, `H/MH → 1`, `N → 2` ("standard 3-class evaluation" [stated]). Originals are never modified; the output adds a `gold_collapsed` column "so the collapse is fully transparent" [stated]. **Unknown labels (e.g. the single `M` in statements and in presconf) are still classified but excluded from accuracy/F1** [stated] — the 2 "unscored" rows above.

## Annotation-privacy guarantee (cells 12, 19–20)

`build_items` extracts, per row, **only** the sentence text plus bookkeeping (`source`, `orig_row`, sort date); "the prompt renderer receives `item['text']` and nothing else — your `label` column physically cannot reach a request" [stated]. Few-shot demos are drawn **cross-source** (statements+presconf run ← speeches file; speeches run ← statements file), "never from the file being evaluated (sentence overlap across your three files was verified to be zero)" [stated — record how that overlap check was done]; demos use only pure `D/H/N` rows, "skip collapsed MD/MH to keep examples unambiguous" [stated]; evaluated-set texts are additionally excluded via `exclude_texts`.

## Deltas vs. the JSONL edition

1. **Output caps raised** (the gpt-5-mini fix): claude-sonnet `max_out = 4000`, claude-haiku `2000`, **gpt-5-mini `2000` ("← was 16; headroom for minimal reasoning + the answer" [stated])**, deepseek-reasoner `4000`; gpt-5.1 stays 16. ⚠ **The "Reproducibility settings" markdown table (cell 15) was not updated** — it still says "8 (16 for GPT-5.x, 32 for DeepSeek-reasoner)". Fix the table before it goes in the paper; the `MODELS` dict is the truth.
2. **Mistral seed passed correctly:** `extra_body={"random_seed": RANDOM_SEED}` (the JSONL edition's direct kwarg likely never reached the API) [inferred; see that README].
3. **DeepSeek env var:** `DEEPSEEK_API_KEY` (all caps) — differs from the JSONL edition's `Deepseek_API_KEY`; unify.
4. **Key-file loader:** optionally reads `~/llm_keys.env` (KEY=VALUE lines) into the environment — keys never in the notebook [stated].
5. **Mistral throttling:** `mistral_slot = threading.Semaphore(4)` — ⚠ the comment says "cap Mistral to 1 request in flight (free-tier concurrency limit)" but the code allows **4**; and `RPM = 200` while a comment on the limiter line records the "real cap: 0.83 req/s (49.8/min) AND 25,000 tok/min". As written, the run leans on the retry/backoff loop to absorb Mistral 429s — visible in the recorded output as mistral-large being the only model with leftover failures. Align semaphore/comment/RPM, or accept and note the retry-based behavior.
6. **Per-run result files** `results__{run}__{model}__{prompt}.jsonl`; every row carries `source` + `orig_row` "so predictions can be joined back onto your exact spreadsheet rows at any time, even mid-run" [stated].

## Preflight — alignment-based resume guard (cells 27–28)

Beyond paths/versions/checksums, the guard is **alignment-based** [stated]: every already-written result row must still map to the same `(source, orig_row)` under the current configuration — so changing `LIMIT` "passes automatically — pilot → full run just resumes"; a **hard stop** occurs only when existing results no longer line up (rows added/removed/reordered) or an input file's SHA-256 changed since results were written. For deliberate, harmless edits (labels only; sentences and row order intact) set `ALLOW_INPUT_CHANGE = True` for one run [stated]. The manifest records, per run: `n_items`, an `items_sha256` fingerprint, input-file hashes, few-shot source, and the gold distribution.

**Recorded preflight:** "fed_statements_presconf: 1998 sentences … verified **75,361** existing result rows aligned"; "fed_speeches: 951 … verified **14,265** existing result rows aligned"; packages `{anthropic 0.120.2, openai 2.53.0, pandas 3.0.5, openpyxl 3.1.5, xlrd 2.0.2}` [recorded] — these are the **actual environment versions** for the paper (Anaconda, Python 3.13).

## Run + self-healing (cells 30–32) — recorded

The saved run resumed a nearly complete state: all combinations "already complete (1998 rows)" except **mistral-large**, which had 2/14/23/65/64 rows left across p1–p5 (3 new errors total) [recorded]. Cell 31 then **rewrites each results file dropping rows with `error != None`** — recorded removals: mistral-large p1 19, p2 14, p3 23, p4 65, p5 64 failed rows "for retry" [recorded]; cell 32 similarly drops rows with `pred is None` (unparsed) and re-runs `run_one` for every affected (run, model, prompt), then re-summarizes and re-writes the Excel. ⚠ Both cells rewrite the JSONLs **in place with no backup** — safe as designed (only failed/unparsed rows are dropped and immediately re-queued), but keep a copy of `results/` before running them, and re-run them until clean so the final files contain no gaps.

## Excel output (cells 23–24)

Per run, `results/{run}_WITH_LLM_LABELS.xlsx`, rebuilt from the logs at any time ("safe to re-run … even mid-run" [stated]):

- **Sheet `data`:** original columns in original/merged-timeline order (including stray `Unnamed:` legend columns; empty rows kept with blank predictions [stated]) — with the human `label` column **normalized MD→D / MH→H in the output only** ("source file untouched" [stated]) — then, right after `label`: `gold_collapsed`, `llm_majority`, and one **letter** column (`D/H/N`) per model×prompt; `?` marks an unparsed reply, `ERR` a failed call. ⚠ `llm_majority` is a plain majority vote across all 40 combos; **ties are broken arbitrarily** (`max(set, key=count)` over an unordered set) — if `llm_majority` is used analytically, define a deterministic tie rule [inferred].
- **Sheet `summary`:** accuracy & macro-F1 per combination (scored on rows with a valid gold), token usage, realized cost.
- Cells sanitized via openpyxl's `ILLEGAL_CHARACTERS_RE`; bold Arial headers, frozen top row, widened sentence columns.

## Reproducibility

Same core as the JSONL edition: `RANDOM_SEED = 42`; temperature 0 where supported; GPT-5.x via `seed` + `reasoning_effort`; deterministic few-shot; every raw reply stored; manifest with prompt hash + file hashes; the honest caveat that hosted-model outputs are highly stable but not bit-guaranteed [stated]. Pin dated snapshots (the two `mistral-*-latest` aliases are still unpinned in the saved config). **Environment (recorded, above)** — turn it into the shipped `requirements.txt`. Path hazards: all three input paths sit under the trailing-space folder `BANQUE DE FRANCE `.

## Pre-submission checks

1. Fix the stale max-output row in the cell-15 reproducibility table (code is authoritative: 4000/2000/16/2000/…/4000).
2. Align the Mistral semaphore/comment/RPM story; re-run cells 31–32 until no `ERR`/`?` rows remain, then regenerate both workbooks and `summary__*.csv`.
3. Pin dated Mistral (and any other alias) snapshots; unify the DeepSeek env-var name with the JSONL edition.
4. Record how the zero-overlap check across the three annotation files was performed; document the construction of `lab-manual-sp-train-5768-augmented.xlsx`.
5. Define a deterministic tie-break for `llm_majority` if it is used in the paper.
6. Attach `run_manifest.json` + `results/*.jsonl` to the supplementary material; state the recorded package versions and the 117,960-call design.
