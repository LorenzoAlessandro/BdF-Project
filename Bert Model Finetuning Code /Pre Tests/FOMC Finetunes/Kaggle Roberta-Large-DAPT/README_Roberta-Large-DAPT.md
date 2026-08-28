# README — `Roberta-Large-DAPT.ipynb`

**Role in the project:** produces the **DAPT'd RoBERTa-large checkpoint** — the "roberta-large DAPT" row of the transfer table, and the model published as `LorenzoAleCon29/roberta-large-fomc-dapt`. Continued masked-LM pre-training only: no labels, no classifier head. Written for **Kaggle** (2× T4).

Cells 0–10 are **byte-for-byte the same pipeline** as `../Kaggle Roberta DAPT Finetune/dapt-finetune-bert-models.ipynb` — see that README for the sentence segmentation and greedy block-packing logic, which is not repeated here. Everything below is what differs.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## What differs from the base run

| Setting | base (`roberta-base`) | **this notebook (`roberta-large`)** | Reason given in the code |
|---|---|---|---|
| `Checkpoint` | `roberta-base` | `roberta-large` | — |
| `Out_directory` | `/kaggle/working/dapt/roberta` | `/kaggle/working/dapt/roberta-large` | — |
| `learning_rate` | 5e-5 | **3e-5** | "large models want a gentler rate (same lesson as the fine-tune)" |
| `per_device_train_batch_size` | 8 | **4** | "large at block 512 won't fit 8/GPU on a 15GB T4" |
| `gradient_accumulation_steps` | 8 | **16** | "4 × 2 GPUs × 16 = effective 128, identical to the base run" |
| `num_train_epochs` | 5 | **3** | timing — a `max_steps=950` cap (~9h) is left commented out as a harder safety net |
| `eval_steps` = `save_steps` | 250 | **150** | "only ~1080 total steps now, so eval finer" |
| Hub push | none | **yes** (cell 12) | — |

`Block_size` (512), `MLM_PROB` (0.15), `Seed` (42), `warmup_ratio` (0.06), `weight_decay` (0.01), `save_total_limit` (1), `load_best_model_at_end` on `loss`, and `fp16` are unchanged.

⚠ The effective-batch comment assumes **2 GPUs**. On a single-GPU session the effective batch is 4 × 16 = 64, not 128 — half the base run's, not equal to it. Confirm the accelerator the reported checkpoint was trained on, because the claim of an identical optimisation regime across scales depends on it.

## Input

Same as the base run: `/kaggle/input/datasets/lorenzouberti/unlabbled-fomc-data/*.jsonl`, filtered for non-empty `text`, with a 1% perplexity holdout at `seed=42`.

## Output (cell 12)

```python
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
model.push_to_hub("LorenzoAleCon29/roberta-large-fomc-dapt", private=True)
tokenizer.push_to_hub("LorenzoAleCon29/roberta-large-fomc-dapt", private=True)
```

The HF token is read from a **Kaggle secret** named `HF_TOKEN`, never written in the notebook [stated]; the comment *"attach the secret to THIS notebook too"* records that Kaggle secrets are per-notebook. The repo is pushed **private** — anyone reproducing the study needs it made public or needs to re-run DAPT.

`{Out_directory}/final` also holds the local copy.

## Reproducibility

- **Seed 42**, via the holdout split and `TrainingArguments(seed=...)`.
- **Environment:** Kaggle; `pysbd` installed in cell 0; `kaggle_secrets` is Kaggle-only and is not pip-installable — off Kaggle, replace cell 12's login with `os.environ["HF_TOKEN"]`.
- **Not runnable off Kaggle without path edits** (`/kaggle/input`, `/kaggle/working`).
- The committed notebook has **no saved outputs at all** [recorded] — no dataset sizes, no loss, no perplexity, no push confirmation.

## Audit notes / pre-submission checks

1. **Capture the final eval loss / perplexity.** With no saved outputs there is currently no evidence in the repo that this run converged, or how it compares with the base DAPT run.
2. **Resolve the effective-batch question** above (1 vs 2 GPUs) before claiming base and large were pre-trained under matched optimisation.
3. **Make `LorenzoAleCon29/roberta-large-fomc-dapt` public**, or state in the paper that the DAPT checkpoints are available on request.
4. Same `Block_size = 512` vs. the root README's "≤2,048-token chunks" discrepancy as the base notebook — settle it once for both.
5. Cell 5's `.copy()` → plain-dict quirk and the cell-6 workaround are inherited verbatim; fix in both notebooks or neither.
