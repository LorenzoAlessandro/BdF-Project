# README — `LLM_Extension.ipynb`

**Role in the project:** the **reference implementation of Part 1** — the generative-LLM benchmark. 8 hosted models × 5 prompt formulations over the Fed **JSONL splits**, one API call per (model, prompt, sentence). This is the "verified JSONL version" that `LLM_Extension_Excel.ipynb` describes itself as a variant of: the prompts, sampling settings, rate limiting, retry policy, resume logic and cost tracking all originate here. Use this notebook to understand the engine; use the Excel edition for the runs that produced the annotated workbooks.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Input

Three JSONL files, each line `{"text": ..., "label": 0|1|2}`, declared as **explicit absolute paths** in the Configuration cell (cell 21) [stated]:

```python
TRAIN_PATH / VAL_PATH / TEST_PATH   →  .../BdF-Project/data/{train,val,test}.jsonl
FILES        = [TEST_PATH]          # what gets classified
FEWSHOT_FILE = TRAIN_PATH           # source of the p5 demonstrations
```

Only the **test** file is classified in the saved configuration. Drawing few-shot demos from `train.jsonl` while evaluating `test.jsonl` is what keeps p5 honest — no demonstration can be a test sentence [inferred from the config, consistent with the Excel edition's explicit cross-source rule].

**Recorded preflight:** `test.jsonl` → **407 sentences, labels {0: 104, 1: 112, 2: 191}** [recorded] — matching the FED test split (n = 407) in the root README.

## Label semantics (cell 5)

`LABEL_MAPPING` is declared the "single source of truth" [stated] and is used to render the prompts, parse replies and score:

```
0 = Dovish  --> leans towards lower rates / more accommodative policy
1 = Hawkish --> leans towards higher rates / tighter policy
2 = Neutral --> neither strongly dovish nor hawkish
```

Same 0/1/2 convention as the encoder pipeline, so Part 1 and Part 2 predictions are directly comparable.

## The 8 models (cell 7)

Each entry carries `provider`, `model_id`, per-1M-token `price_in`/`price_out` for realised-cost tracking, and per-model quirk flags. The quirks are the interesting part [stated]:

| Model | Notable configuration |
|---|---|
| `claude-sonnet` / `claude-haiku` | `temperature=0`; Anthropic exposes no seed |
| `gpt-5.1` | `reasoning_effort: "none"`, `no_temperature: True` ("gpt-5 family rejects temperature != 1"), `use_max_completion_tokens: True`, `max_out: 16` |
| `gpt-5-mini` | `reasoning_effort: "minimal"` ("lowest available for mini"), otherwise as above |
| `mistral-large` / `mistral-small` | `temperature=0`, `random_seed=42` |
| `deepseek-chat` | `temperature=0` |
| `deepseek-reasoner` | sampling parameters documented as ignored by the API |

A comment flags that **`-latest` aliases move silently** and should be replaced with dated snapshots before the run that goes in the paper [stated]. The `reasoning_effort` setting is called out as "the difference between ~$0.72 and ~$7.60 on the test run" [stated].

## Prompts (cell 11)

Five frozen system+user pairs — `p1_analyst_direct`, `p2_analyst_rubric`, `p3_analyst_rules`, `p4_analyst_json`, `p5_analyst_fewshot` — matching P1–P5 in the paper. `render_prompt` (cell 12) substitutes `{labels}` from `LABEL_MAPPING`, `{examples}` from the few-shot builder, and `{sentence}`, **replacing any `"` in the sentence with `'`** so the quoting in the template cannot be broken by the data [stated].

`build_fewshot_examples` picks the **first occurrence of each class** in `FEWSHOT_FILE`, in label order — deterministic, no sampling [stated].

The Preflight cell hashes the prompt dict (SHA-256) into the manifest so that any later edit is detectable [stated].

## API plumbing (cell 15)

- **`RateLimiter`** — a per-provider global throttle at `RPM` requests/minute, implemented as a lock-protected next-allowed timestamp.
- **`get_client`** — lazily constructs and caches one client per provider; Anthropic uses the `anthropic` SDK, everything else uses `openai.OpenAI` pointed at an OpenAI-compatible `base_url` (`https://api.mistral.ai/v1`, `https://api.deepseek.com`). This is why only two SDKs are needed [stated].
- **`call_model`** — returns `(raw_text, in_tokens, out_tokens)`, raises on final failure. Its `dry_run` path returns a **deterministic fake label** derived from `zlib.crc32(user)` — chosen over `hash()` because Python's string hash is randomised per process [stated]. That makes `DRY_RUN=True` a genuine no-keys pipeline test.

**Keys** are read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY` and `Deepseek_API_KEY` (cell 9); `_ensure_keys` prompts via `getpass` for any that are missing, "kept in memory only" [stated]. Nothing is written to disk.

## Parsing and scoring (cell 17)

`parse_label` tries three strategies in order, so no reply format is silently lost:
1. JSON form — `"label"\s*:\s*([012])` (needed for p4)
2. bare digit — `\b([012])\b`
3. word fallback — the first word of each `LABEL_MAPPING` value found case-insensitively in the reply

Returns `None` if all three fail; such rows are recorded and excluded from scoring. `macro_f1` is implemented by hand, per class, deliberately avoiding a scikit-learn dependency [stated].

## Runner and resume (cell 19)

One JSONL per (model, prompt): `results__{model}__{prompt}.jsonl`. On start it reads the file and collects finished `idx` values, then only queues the remainder — so an interrupted run resumes by simply re-running the cell [stated]. Work is spread over `WORKERS` threads with a shared lock around the append. Every row records `idx, model, prompt, gold, pred, raw, in_tok, out_tok, cost, error` — including the **raw model reply**, which is what makes the error analysis notebook possible.

Cost is computed per row as `(in_tok·price_in + out_tok·price_out) / 1e6`.

## Configuration (cell 21) — the only cell meant to be edited

```
LIMIT   = None    # pilot with 25 first; None = full run
WORKERS = 4
RPM     = 50      # per provider
OUTDIR  = Path("results")
DRY_RUN = False
```

## Preflight and manifest (cell 23)

Checks every configured file exists (raising a message that names the missing file), prints label distributions, and writes `results/run_manifest.json` containing: UTC timestamp, `random_seed`, Python version and platform, package versions, resolved model IDs, the **prompt-dict SHA-256**, the label mapping, the run config, and per-file **SHA-256 checksums** and label distributions. The markdown instructs attaching this file to the paper's supplementary material [stated].

**Recorded environment: `anthropic 0.120.2`, `openai 2.53.0`** [recorded].

## Outputs

| File | Contents |
|---|---|
| `results/results__{model}__{prompt}.jsonl` | one row per sentence, with the raw reply |
| `results/run_manifest.json` | the reproducibility record above |
| `results/predictions_detailed.csv` | long form: idx, sentence, gold + name, model, prompt, prediction + name, `correct`, raw reply |
| `results/predictions_wide.csv` | one row per sentence, one column per `model__prompt` combination |

The wide CSV is the natural input to the agreement/error analysis; the detailed CSV is what a human reads when checking a disputed case.

## Recorded run — a pilot, not the full run

The saved outputs are from a **`LIMIT = 25` pilot**, not the reported run: every combination reads "25 to do (0 resumed)" [recorded]. Two things in that log matter:

1. **`gpt-5-mini x p1_analyst_direct: 25/25 (25 errors)`** — a 100% failure rate [recorded]. This is the `max_out: 16` problem: with `reasoning_effort: "minimal"` the model spends its entire 16-token budget on reasoning and never emits the label. The Excel edition raises gpt-5-mini's cap to 2000 with the comment "← was 16; headroom for minimal reasoning + the answer". **The fix is not applied in this notebook.**
2. Claude and gpt-5.1 combinations complete with 0 errors at the same cap, confirming the failure is specific to the mini model's reasoning budget.

## Reproducibility

- **`RANDOM_SEED = 42`** seeds Python's `random`, the dry-run fakes, OpenAI's `seed` and Mistral's `random_seed`. Anthropic and DeepSeek expose no seed.
- The reproducibility table in cell 13 carries an explicit **honest caveat** that `temperature=0` plus a seed makes outputs highly stable but not formally deterministic [stated] — keep that sentence in the methods section.
- **Path hazard:** absolute macOS paths with the trailing-space folder `BANQUE DE FRANCE `. `OUTDIR` is relative, so `results/` lands next to wherever the kernel started.

## Audit notes / pre-submission checks

1. **Port the `max_out` fix.** As committed this notebook cannot produce usable gpt-5-mini predictions; the Excel edition's caps (haiku 2000, sonnet 4000, gpt-5-mini 2000, deepseek-reasoner 4000) should be mirrored here, and cell 13's settings table updated with them — it currently states "8 (16 for GPT-5.x, 32 for DeepSeek-reasoner)".
2. **`Deepseek_API_KEY` is mixed case here** but `DEEPSEEK_API_KEY` in the Excel edition. Unify, or one of the two notebooks will silently prompt for a key the user already set.
3. **Pin dated model snapshots** before the reported run — the notebook's own comment asks for this, and four of the eight entries still use `-latest` or undated ids.
4. **Only `test.jsonl` is configured.** If the paper reports results over statements/press conferences *and* speeches, that came from the Excel edition; be explicit about which artefact backs which table.
5. **The committed outputs are a 25-sentence pilot.** Re-run at `LIMIT = None` and commit the manifest for the run actually reported, or point the paper at the Excel edition's results.
