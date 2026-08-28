# README — `LLM_Extension_Excel.ipynb` (repository-root copy)

**Role in the project: this is a second copy of `Fed Side/LLM_Extension_Excel.ipynb`, and the two have diverged.** It is the same Excel-edition engine — 8 models × 5 prompts over the Fed annotation spreadsheets, predictions written back beside the human labels — and it writes to the **same output directories** as the `Fed Side/` copy. For the pipeline itself, read `Fed Side/README_LLM_Extension_Excel.md`; this file exists to document the divergence, because running the wrong copy silently overwrites the other's results.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Both copies write to the same place

```python
OUTDIR            = .../Extension LLM Classification Task/Fed Side/Results
Outdir_excel_file = .../Extension LLM Classification Task/Fed Side/Excel Sentences Results
```

Identical in both notebooks [stated]. The result JSONLs use the same `results__{run}__{model}__{prompt}.jsonl` names, and the resume logic keys on `idx`, so whichever copy runs last extends and rewrites the other's files. **There is no way to tell from `Fed Side/Results/` which notebook produced it.**

## Where the two copies differ

**1. Output token caps — the substantive difference.**

| Model | This (root) copy | `Fed Side/` copy |
|---|---|---|
| claude-sonnet | 4000 | 4000 |
| claude-haiku | 2000 | 2000 |
| **gpt-5.1** | **16** | **10** |
| **gpt-5-mini** | **2000** | **10** |
| **deepseek-reasoner** | **4000** | **15** |

This matters. `gpt-5-mini` runs with `reasoning_effort: "minimal"`, and the JSONL edition (`Fed Side/LLM_Extension.ipynb`) recorded a **100% error rate at a cap of 16** — the model spends the budget on reasoning and never emits a label. The root copy's `2000` carries the code comment fixing exactly that: *"← was 16; headroom for minimal reasoning + the answer"* [stated]. The `Fed Side/` copy has since been lowered to **10**, below the value that already failed.

Note also that `Fed Side/README_LLM_Extension_Excel.md` documents the caps as **sonnet 4000 / haiku 2000 / gpt-5-mini 2000 / deepseek-reasoner 4000, gpt-5.1 16** — i.e. that README describes **this root copy**, not the `Fed Side/` notebook it sits next to. Whichever notebook produced the reported Fed numbers, the README and the file it names are currently out of sync.

**2. Cells only in this copy** [stated]:
- `dedupe_results(OUTDIR)` — removes duplicate `idx` rows keeping the last occurrence, for when an interrupted kernel leaves a worker thread writing past a resume point
- an explicit `summarize(OUTDIR, "fed_statements_presconf")` / `summarize(OUTDIR, "fed_speeches")` call

**3. Cells only in the `Fed Side/` copy** [stated]:
- an **audit cell** that reports, per run × model × prompt and with no API calls, how many rows are done, errored or unparsed
- a **staleness-repair cell** that ranks rows (2 = parsed and clean, 1 = clean but unparsed, 0 = error) and drops rows whose `(source, orig_row)` no longer matches the current item order

So neither copy strictly supersedes the other: the root copy has the working token caps, the `Fed Side/` copy has the better repair tooling.

## Recommendation

Reconcile them into **one** notebook before anything else in this folder is re-run:

1. Decide which token caps produced the reported Fed results — check the `gpt-5-mini` error counts inside `Fed Side/Results/results__*__gpt-5-mini__*.jsonl`. A file full of `error` rows means the low cap was in force.
2. Merge the audit and staleness-repair cells from `Fed Side/` into the copy with the working caps.
3. Delete the loser, and update `Fed Side/README_LLM_Extension_Excel.md` to describe the survivor.
4. Until then, **do not run either copy** — a run appends to shared output files under a configuration you cannot reconstruct afterwards.

## Everything else

Inputs, the two configured runs (`fed_statements_presconf`, `fed_speeches`), label handling and the `GOLD_TO_DIGIT` collapse, the annotation-privacy guarantee, the cross-source few-shot rule, the alignment-based preflight guard, and the Excel writer are all as documented in `Fed Side/README_LLM_Extension_Excel.md`. The known issues recorded there apply here too: the stale "Reproducibility settings" table (cell 15) that still says "8 (16 for GPT-5.x, 32 for DeepSeek-reasoner)", the `mistral_slot = threading.Semaphore(4)` whose comment claims a cap of 1, and the `DEEPSEEK_API_KEY` / `Deepseek_API_KEY` naming split against the JSONL edition.
