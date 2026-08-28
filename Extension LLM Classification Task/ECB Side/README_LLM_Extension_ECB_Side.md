# README — `LLM_Extension_ECB_Side.ipynb`

**Role in the project:** the **ECB half of Part 1** — the same 8-model × 5-prompt engine as the Fed side, run over the ECB annotation spreadsheets. This is the Excel edition (predictions written back beside the human labels, timeline preserved), and it is the notebook that produced `ECB_statements_presconf_WITH_LLM_LABELS.xlsx` and `ECB_speeches_WITH_LLM_LABELS.xlsx`, the inputs to `LLM Extension Plots ECB.ipynb`.

The engine is shared with `../Fed Side/LLM_Extension_Excel.ipynb` — see that README for the parts not repeated here (rate limiting, retry/backoff, alignment-based resume guard, Excel writer). **What is genuinely ECB-specific is the prompt wording, the input files and the two configured runs**, documented below.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## The two configured runs (cell 25)

| Run tag | Sources | Merge rule | Few-shot demos from |
|---|---|---|---|
| `ECB_statements_presconf` | `ecb_train-2.xls` (train) + `ecb_test.csv` (test) + `ecb_Press Conference Statements + QA_to_annotate.xlsx` | concatenated, **stable-sorted by date**; ties keep train → test → presconf, original order within each file [stated] | `SPEECHES_PATH` |
| `ECB_speeches` | `speeches_annotated_v3.xlsx` | year-only dates → **never merged**, original order [stated] | `STATEMENTS_PATH` |

All four files live under `data/ECB/Annotated/`.

**Recorded preflight** [recorded]:

```
ECB_statements_presconf: 2044 sentences, gold {N: 951, D: 633, H: 459, unscored: 1}
                         verified 71,540 existing result rows aligned
ECB_speeches:            1000 sentences, gold {N: 467, H: 216, D: 317}
                         verified 35,000 existing result rows aligned
```

Full plan = 8 × 5 × (2,044 + 1,000) = **121,760 calls**.

## What is ECB-specific in the prompts (cell 12)

`PROMPTS` is described in the markdown as "byte-identical to the verified pipeline" [stated] — **it is not, and the differences are deliberate and correct.** The p1 rules substitute the euro for the dollar:

- Dovish: "**euro appreciation**" (Fed side: "dollar appreciation")
- Hawkish: "**euro depreciation**" (Fed side: "dollar depreciation")

⚠ The hawkish rule still reads "increasing yields/**TIPS**" — a US Treasury instrument with no ECB equivalent. This is a leftover from the Fed prompt. It is a small thing, but it is a Fed-specific cue sitting in the ECB prompt, so it belongs either in the paper's prompt appendix as a known artefact or in a corrected re-run. The "maximum employment" phrase in the dovish rule is likewise Fed mandate language, not ECB.

Because the manifest hashes the prompt dict, the Fed and ECB runs will have **different `prompts_sha256` values** — expected, but state why in the appendix so it doesn't read as an accidental edit.

## Label handling — two layers (cells 5, 13)

- `LABEL_MAPPING` (digits 0/1/2) drives the prompts, unchanged from the Fed pipeline [stated].
- `GOLD_TO_DIGIT` translates human annotations **for scoring only**, accepting both letters and full words, case-insensitively: `D/MD/DOVISH → 0`, `H/MH/HAWKISH → 1`, `N/NEUTRAL → 2`. The 5-point scale is collapsed to 3 classes; originals are never modified and a `gold_collapsed` column makes the collapse visible in the output [stated].
- `DIGIT_TO_LETTER` (`0→D, 1→H, 2→N`) is how predictions are written back to Excel.
- Unknown labels are still classified but **excluded from accuracy/F1** — the single `unscored` row in the statements run [recorded].

## Annotation-privacy guarantee (cells 13, 18)

`build_items` extracts per row **only** the sentence text plus bookkeeping (`source`, `orig_row`, sort date); "the prompt renderer receives `item["text"]` and nothing else — your `label` column physically cannot reach a request" [stated].

Few-shot demos are drawn **cross-source** — the statements+presconf run takes them from the speeches file and vice versa, "never from the file being evaluated (sentence overlap across your three files was verified to be zero)" [stated]. `build_fewshot_examples` additionally accepts only pure `D/H/N` rows, skipping collapsed `MD/MH` "to keep examples unambiguous", and excludes any text present in the evaluated set via `exclude_texts` [stated].

## Format-tolerant loading (cell 19)

`load_source` handles the fact that these are real, messy annotation files [stated]:

- `.csv` (the ECB test split) and `.xls`/`.xlsx` all go through one entry point; a clear error tells you to `pip install xlrd` if the legacy `.xls` engine is missing.
- **`_promote_header`** detects a title row above the real header (the `ecb_train` file has `ecb_train` in A1) and promotes the header row from within the first 10 rows. Well-formed files pass through untouched.
- Column names are auto-detected (`sentence`/`text`, `label`, `date`/`year`).
- Dates parse **day-first** (`28/10/2021` = 28 October) — the European convention, and a real hazard on ECB files that the Fed side does not face.
- `Unnamed:` legend columns and empty rows are preserved so the output workbook matches the input layout.

## Outputs

| File | Contents |
|---|---|
| `results/results__{run}__{model}__{prompt}.jsonl` | incremental log; every row carries `source` + `orig_row` for joining back to the exact spreadsheet row |
| `results/{run}_WITH_LLM_LABELS.xlsx` | sheet **data** = original columns in order, then `gold_collapsed`, `llm_majority`, and one D/H/N column per model×prompt; sheet **summary** = accuracy and macro-F1 per combination |
| `results/summary__{run}.csv` | the same scores as a flat CSV |
| `results/run_manifest.json` | seed, versions, model IDs, prompt hash, file checksums, gold distributions |

Excel cells are sanitised with `openpyxl`'s `ILLEGAL_CHARACTERS_RE`; nothing in the original data is altered [stated].

**Recorded:** `ECB_speeches_WITH_LLM_LABELS.xlsx` — 1,000 rows × 54 columns, **40 prediction columns** (8 models × 5 prompts) [recorded].

## Recorded results — ECB speeches, macro-F1 mean ± SD across the 5 prompts

| Model | macro-F1 | SD |
|---|---|---|
| claude-sonnet | **0.678** | 0.017 |
| mistral-large | 0.655 | 0.024 |
| claude-haiku | 0.651 | 0.028 |
| gpt-5.1 | 0.636 | 0.036 |
| deepseek-reasoner | 0.604 | 0.045 |
| gpt-5-mini | 0.604 | 0.037 |
| mistral-small | 0.600 | 0.028 |
| deepseek-chat | 0.563 | 0.074 |

k = 5 prompts per model. **Total realised cost recorded: $13.70** [recorded].

Note that `deepseek-chat` again shows by far the widest prompt sensitivity (SD 0.074) — the same pattern the root README reports on the Fed side, which strengthens it as a model property rather than a corpus artefact.

## Self-healing cells (30–32)

Three maintenance passes, all rewriting the JSONLs **in place with no backup** [stated]:

- **Cell 30** drops rows with a non-null `error` so they are re-queued on the next run.
- **Cell 31** drops rows where `pred is None` *and* there was no error — i.e. replies that came back but could not be parsed — then re-runs those combinations. **Recorded: 1 unparsed row removed from `deepseek-reasoner__p4_analyst_json`, then "No unparsed rows found"** [recorded].
- **Cell 32** `dedupe_results` removes duplicate `idx` rows keeping the last occurrence. The docstring gives the cause: "interrupting the kernel without a full restart can leave a background worker thread writing a row after the next run already resumed past it" [stated]. **Recorded: no duplicates found** [recorded].

Keep a copy of `results/` before running these; re-run until clean.

## Reproducibility

- **`RANDOM_SEED = 42`** — Python RNG, OpenAI `seed`, Mistral `random_seed`. Anthropic and DeepSeek expose no seed.
- **Recorded environment: `anthropic 1.0.0`, `openai 3.3.1`, `pandas 3.0.5`, `openpyxl 3.1.5`, `xlrd 2.0.2`** [recorded], Anaconda Python 3.13.
- Keys come from environment variables, an optional `~/llm_keys.env` loader, or a hidden `getpass` prompt — never from the notebook [stated]. This side uses `DEEPSEEK_API_KEY` (all caps).
- **Path hazard:** absolute macOS paths with the trailing-space folder `BANQUE DE FRANCE `.

## Audit notes / pre-submission checks

1. **The SDK versions differ from the Fed side** — `anthropic 1.0.0` / `openai 3.3.1` here versus `anthropic 0.120.2` / `openai 2.53.0` recorded on the Fed run. A major-version jump in both SDKs between the two halves of a paired comparison is worth either re-running one side or stating explicitly. Report both manifests.
2. **`max_out` is 10 for gpt-5.1 and gpt-5-mini here**, with the comment "← was 16", whereas the Fed Excel edition raised gpt-5-mini to **2000** for exactly this reason. Ten tokens is *lower* than the setting that already failed on the Fed side. Check the gpt-5-mini error rate in this run's JSONLs before trusting its 0.604.
3. **Fix or document the "TIPS" and "maximum employment" leftovers** in the ECB p1 prompt.
4. **The cell-14 reproducibility table is stale** — it still states "max output tokens 8 (16 for GPT-5.x, 32 for DeepSeek-reasoner)", contradicting the `MODELS` dict above it. The dict is the truth; fix the table before it reaches the paper.
5. **`mistral_slot = threading.Semaphore(4)`** is commented "cap Mistral to 1 request in flight (free-tier concurrency limit)" — the code allows 4. Same mismatch as the Fed edition; align the comment, the semaphore and `RPM` (the limiter line records a real cap of 0.83 req/s and 25,000 tok/min).
6. The recorded speeches numbers above are **ECB speeches**; the root README's headline LLM table is statements & press conferences. Keep the two straight when citing.
