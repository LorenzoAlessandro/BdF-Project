# FED + ECB stance classification — full pipeline (Kaggle)

One notebook, three stages, restart-safe end to end: fine-tune 7 encoders on Federal Reserve data, fine-tune the same 7 on ECB data, then evaluate every run against **both** test sets to measure cross-institution transfer in hawkish/dovish/neutral classification.

```
Stage 1   FED: 7 models × 5 seeds  →  HF branch FED-v6      35 fine-tunes
Stage 2   ECB: 7 models × 5 seeds  →  HF branch ECB-v6      35 fine-tunes
Stage 3   cross-eval: 70 runs × 2 test sets                140 evaluations
End       lightweight artefacts (CSVs, logs, predictions) → GitHub
```

## Verified run

The committed notebook is a complete successful execution, not a template.

| | |
|---|---|
| Executed | 2026-08-20, 08:58 → 13:46 UTC |
| Wall time | 17,296 s (**4 h 48 m**) |
| Papermill exception | `null` |
| Hardware | Kaggle **Tesla T4**, 20.9 GB free on `/kaggle/working` |
| Fine-tunes | **35/35 FED**, **35/35 ECB** — every one uploaded and verified |
| Evaluations | **140/140** recorded |
| Predictions | 56,840 rows written to `cross_eval_predictions.csv.gz` |
| GitHub push | succeeded |

Two checks make this auditable rather than asserted:

- **Every upload was verified file-by-file** before the local copy was deleted. `upload_and_verify` re-lists the run on the Hub and compares each filename *and byte size*; it raises otherwise. All 70 runs logged `verified 8 files at …`.
- **Every in-domain cross-eval reproduced its training-time metric exactly.** Stage 3 re-evaluates each checkpoint from scratch and compares against the `test_metrics.json` saved during training: all 70 report `|Δacc| = 0.0000`.

## Why Kaggle

**This runs on Kaggle because of the GPU.** 70 fine-tunes — including ten `roberta-large` runs — is not CPU-feasible; the pre-flight cell warns explicitly that without an accelerator this takes days. Kaggle's free T4 and 30 h weekly GPU quota are what the notebook is built around, and every design decision below follows from that constraint:

- **12-hour session limit** → the pipeline is restart-safe. Just *Run All* again (or *Save Version → Save & Run All (Commit)* for unattended runs). Finished runs are checked against the local CSV first, then against the Hub, and skipped with their metrics pulled back down. Nothing on the Hub is ever overwritten.
- **~20 GB of working disk** → a `roberta-large` run peaks at 10–13 GB, so each run's folder is deleted the moment its upload is verified, and each cross-eval download (≤1.5 GB) is deleted right after use with the HF cache cleared. The disk never holds more than one model.
- **Weights don't belong in git** → they live on the Hugging Face Hub, one private repo per model, one branch per institution, one folder per run. Only CSVs, trainer states, splits, config and predictions are pushed to GitHub.

If both institutions are trained from scratch this spans several sessions and most of the weekly quota. That is exactly what the resume logic is for.

## Before running

- GPU on, Internet on
- Kaggle secrets attached **to this notebook**: `HF_TOKEN`, `GITHUB_PAT`
- Kaggle inputs mounted: FED annotations (`annotated-fomc`, `annotated`), ECB annotations (`ecb-labblled-dataset`, `trained`, `speeches-annotated-v3`), FED DAPT checkpoints (`roberta-base-dapt-finetune` dataset + `dapt-finetune-bert-models` notebook output). ECB DAPT checkpoints come from the Hub.
- Set `HF_USERNAME` in the CONFIG cell to your own account.

The pre-flight cell clears leftovers from dead sessions, prints free space and names the GPU. The plan cell then reports how many runs are already on the Hub and asserts that the DAPT checkpoints are reachable — but only for institutions that still have work to do.

## Configuration

`EXPERIMENT` selects a *pair* of (HF branches, learning rates), so FED and ECB always share hyperparameters and the comparison stays clean. The committed run is **`v6`**: lr `3e-5` for all seven checkpoints, branches `FED-v6` / `ECB-v6`, best epoch by **lowest eval loss**.

Reusing a branch name resumes; a new branch name trains from scratch.

| Model | FED checkpoint | ECB checkpoint |
|---|---|---|
| roberta-base | `roberta-base` | `roberta-base` |
| roberta-dapt | FED-domain DAPT (Kaggle input) | `LorenzoAleCon29/roberta-base-ECB-dapt` |
| roberta-large | `roberta-large` | `roberta-large` |
| roberta-large-dapt | FED-domain DAPT (Kaggle input) | `LorenzoAleCon29/roberta-large-ECB-dapt` |
| deberta-v3 | `microsoft/deberta-v3-base` | same |
| finbert | `ProsusAI/finbert` | same |
| finbert-tone | `yiyanghkust/finbert-tone` | same |

**Training** (identical across all models and both institutions): 5 epochs · train batch 16 / eval 32 · `max_length=256` · weight decay 0.01 · warmup 0.10 · fp16 · per-epoch eval and save · `save_total_limit=1` · seeds 42–46. Weights load in fp32 even under fp16 AMP, because DeBERTa-v3 ships reduced-precision weights that otherwise break AMP.

## Data

Both preparations sort labels alphabetically to `{dovish: 0, hawkish: 1, neutral: 2}`, and the notebook **asserts** this mapping on both sides, again on every checkpoint at eval time. It also asserts that the ECB test set differs from the FED one, catching a stale `datasets` cache that would silently hand back the same files.

| | Split | N | Cutoff |
|---|---|---|---|
| **FED** | train / val / test | 2367 / 200 / 407 | fixed `2021-01-01` |
| **ECB** | train / val / test | 2381 / 198 / 405 | 80th percentile of dates → `2022-11-18` |

Post-cutoff data is split 33/67 into validation and test, stratified on label, `random_state=42`. FED labels: 1451 neutral, 815 dovish, 709 hawkish. ECB: 1400 / 941 / 665 after dropping 38 unlabelled rows.

## Results (v6, mean ± sd over 5 seeds)

**In-domain accuracy**

| Model | FED | ECB |
|---|---|---|
| roberta-base | 0.6909 ± 0.0168 | 0.6370 ± 0.0248 |
| roberta-dapt | 0.6870 ± 0.0230 | 0.6286 ± 0.0288 |
| **roberta-large** | **0.7160 ± 0.0073** | 0.6568 ± 0.0401 |
| **roberta-large-dapt** | 0.6948 ± 0.0123 | **0.6686 ± 0.0319** |
| deberta-v3 | 0.5509 ± 0.0515 | 0.4711 ± 0.0629 |
| finbert | 0.6290 ± 0.0288 | 0.5960 ± 0.0293 |
| finbert-tone | 0.6275 ± 0.0117 | 0.5714 ± 0.0346 |

**Transfer (accuracy)** — the headline result is the asymmetry:

| Model | FED→FED | FED→ECB | ECB→ECB | ECB→FED |
|---|---|---|---|---|
| roberta-base | 0.6909 | 0.5417 | 0.6370 | 0.6452 |
| roberta-dapt | 0.6870 | 0.5763 | 0.6286 | 0.6609 |
| roberta-large | 0.7160 | 0.6158 | 0.6568 | 0.6536 |
| roberta-large-dapt | 0.6948 | 0.6099 | 0.6686 | 0.6511 |

FED-trained models lose 8–15 accuracy points on ECB text. ECB-trained models lose roughly nothing on FED text — for `roberta-base` and `roberta-dapt` the transfer gap is *negative* (−0.008, −0.032), i.e. they score slightly higher out-of-domain than in. DAPT narrows the FED gap substantially at base scale (0.149 → 0.111).

`deberta-v3` is unstable at this learning rate (macro-F1 sd up to 0.137) and should be read with caution.

## Outputs

```
results/results_per_run_{FED,ECB}-v6.csv     per-run metrics
results/results_summary_{FED,ECB}-v6.csv     mean ± sd (ddof=1)
results/cross_eval/FED-v6_x_ECB-v6/
    cross_eval_per_run.csv                   140 evaluations
    cross_eval_summary.csv
    cross_eval_matrix_{accuracy,f1_macro}.csv
    cross_eval_accuracy_gap.csv
    cross_eval_predictions.csv.gz            56,840 per-sentence predictions
    cross_eval_test_sets.csv.gz              text + gold label
    cross_eval_config.json                   versions, seeds, tokenisation
data/{FED,ECB}/                              frozen splits + label mapping
logs/{FED,ECB}-v6/                           140 trainer states + test metrics
```

Per-sentence predictions are kept for every evaluation, so error analysis and significance tests never need a re-download. Sample sd uses Bessel's correction: the five runs are a sample from the distribution of fine-tuning outcomes, not a population.

## Known warnings

- **Early stopping did not fire.** Every run logs `early stopping required metric_for_best_model, but did not find eval_loss so early stopping is disabled`, so all runs completed the full 5 epochs. This does not invalidate the results — training, checkpointing and evaluation all ran correctly — but the `EarlyStoppingCallback(patience=2)` in the config was inactive for this run.
- HF `LOAD REPORT` blocks showing `lm_head.*` as UNEXPECTED and `classifier.*` as MISSING are expected: a fresh classification head is being initialised on top of an MLM checkpoint.
- `warmup_ratio is deprecated` and the `torch.autograd` gather warning are benign.
- The GitHub cell deletes `/kaggle/working/repo` at the end — **don't skip it**, the PAT sits in `.git/config`.

## Loading a run

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "LorenzoAleCon29/roberta-large-Multi_Model_Full_Pipeline_Run"
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="run-1", revision="FED-v6")
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="run-1", revision="FED-v6")
```

Each `run-i/` folder holds weights, tokenizer, `trainer_state.json`, `test_metrics.json` and `run_info.json`. Every model repo also carries a generated `README.md` and `summary.json` per branch, written only once all 5 runs are recorded.
