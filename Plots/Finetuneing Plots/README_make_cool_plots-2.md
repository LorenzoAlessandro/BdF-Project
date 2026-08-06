# README — `make_cool_plots-2.py`

**Role in the project:** publication figures for the **fine-tuning** track. A zero-configuration CLI script ("same spirit as make_plots.py" [stated]) that reads `results_per_run.csv` — per-seed test metrics from the encoder fine-tuning experiments — plus one `trainer_state.json` per model, and renders nine 300-dpi PNG figures comparing eight encoder models on the FOMC hawkish/dovish/neutral test set. **This file is also the written record of the fine-tuning track's reproducibility frame: 5 runs per model with seeds 42–46, evaluated on the FOMC test set, n = 407 sentences** [stated in the shared footnote] — the same 407-sentence `test.jsonl` used by the LLM benchmark, tying the two tracks to one evaluation set [inferred link]. The fine-tuning code itself is not among the uploaded files; its settings must be documented in that repo.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = n/a (a script; no saved outputs).

---

## Inputs

- **`results_per_run.csv`** — one row per (model, seed) with at least: `model`, `seed`, `test_f1_macro`, `test_accuracy`, `test_f1_weighted`, `test_loss` [inferred from the aggregations]. Default location: `…/BdF-Project/Bert Model Finetuning Code /Multi Model Finetune - Bert /results_per_run.csv` — ⚠ note **two more trailing-space folder names** (`Bert Model Finetuning Code ` and `Multi Model Finetune - Bert `) on top of `BANQUE DE FRANCE `. If the path is absent, the script **auto-discovers** the CSV via `rglob` under the CWD [stated].
- **`trainer_state.json`** (HuggingFace Trainer state; `log_history` is parsed) — one per model for the learning-curve figures f8/f9, default **run 3** of each model, from the `TRAINER_STATES` dict; "anything missing is skipped with a warning"; overridable via `--logs` / `--run` [stated].

## The eight models

`roberta-base`, `roberta-base +DAPT`, `roberta-large`, `roberta-large +DAPT`, `deberta-v3-base`, `FinBERT`, `FinBERT-tone`, `ModernBERT-base` (fixed display order, names, and colors; unknown models get a fallback grey and are appended) [stated]. "DAPT" = domain-adaptive pre-training on a **"30M-token FOMC corpus"** [stated in figure f2's labels] — this matches the 29,426,687-token FED unlabeled corpus assembled by `Tokenizer-_FED_Branch.ipynb` [inferred link; confirm and cite the exact corpus + token count consistently].

## The nine figures, and the statistics inside them

| File | Content | Statistical choices [stated in code] |
|---|---|---|
| `f1_performance_overview.png` | forest-style comparison, macro-F1 + accuracy | per-model mean, **95% CI = Student-t on the 5 seeds** (`t.ppf(0.975, n−1)·sd/√n`), thin bar = min–max seed range, dots = individual seeds |
| `f2_dapt_interaction.png` | paired slopegraphs, base vs +DAPT (125M and 355M) | seed-paired lines; mean Δ in pp; **paired t-test** p-value per pair |
| `f3_stability_map.png` | mean macro-F1 vs seed-sd quadrants | medians as quadrant lines; quadrant captions ("strong & seed-robust", …) |
| `f4_metric_heatmap.png` | models × {macro-F1, weighted-F1, accuracy, CE loss} | column-min-max normalized, loss inverted; ★ = best in column |
| `f5_rank_bump.png` | rank of each model under the four metrics | `rank(method="min")`, loss ascending |
| `f6_loss_vs_f1.png` | per-run test loss vs macro-F1 ("loss ≠ quality") | scatter of raw runs |
| `f7_seed_trajectories.png` | seed-level spaghetti across models | raw per-seed lines |
| `f8_learning_curves.png` | per-epoch validation loss + macro-F1, all models | from `log_history` eval entries (run 3) |
| `f9_train_vs_val.png` | train vs validation loss grid | shaded generalization gap (train interpolated onto eval epochs); ★ best validation epoch |

Styling is fully specified in-file (rcParams, palette, halo path-effects, dodge algorithm for overlapping labels) — figures are deterministic pixel-for-pixel given the same inputs and matplotlib version [inferred].

## ⚠ Hard-coded narrative annotations — regenerate-and-verify

Several figure texts encode **results, not computations**, and will silently become false if the CSV changes:

- f2's title/labels: "DAPT drags the 125M model down on every seed, but lifts the 355M model on 4 of 5" and the in-plot note "only seed 45 regresses" (attached to any regressing roberta-large seed, captioned as seed 45);
- f3's callout: "seed 43 collapses to 0.43 macro-F1 — classic DeBERTa fine-tuning instability" and the subtitle's "RoBERTa-base +DAPT: mediocre but almost seed-proof";
- f5's title: "Rankings barely move — until you look at loss".

Before submission, re-run on the final CSV and **verify every one of these sentences against the data** (or replace them with computed text). The footnote text ("5 fine-tuning runs per model (seeds 42–46) · FOMC test set, n = 407 sentences · …") is likewise a constant — keep it in sync with the actual design.

## Usage

```
python make_cool_plots-2.py                       # auto-finds results_per_run.csv under cwd
python make_cool_plots-2.py path/to/results_per_run.csv --out figures
python make_cool_plots-2.py --logs "path/to/logs" --run 3
```

Outputs land in `--out` (default `figures/`), 300 dpi, tight bbox, white background.

## Reproducibility & environment

- **No RNG in this script** — deterministic given `results_per_run.csv`, the trainer states, and library versions. The *randomness it visualizes* (seeds 42–46) lives in the fine-tuning repo: document there, per model, the full training configuration (LR, epochs, batch size, max length, scheduler, hardware, library versions) and confirm the seed set.
- **Dependencies:** `numpy`, `pandas`, `matplotlib`, `scipy` (only `scipy.stats` for t critical values and the paired t-test). None pinned — pin, and note the matplotlib version if pixel-identical figures matter.
- With n = 5 seeds, the t-based CIs are wide and the paired t-tests are low-powered — fine to report, but phrase them as descriptive; a referee may ask.

## Pre-submission checks

1. Re-run on the final `results_per_run.csv` and audit every hard-coded annotation (above).
2. Ship `results_per_run.csv` and the seven `trainer_state.json` files with the replication package (they are the figures' inputs), plus the fine-tuning configs behind them.
3. Replace/relativize the hard-coded `BASE` path (three trailing-space folder names) or rely on the auto-discovery from the repo root.
4. Confirm the "30M-token FOMC corpus" wording against the exact DAPT corpus (29,426,687 tokens per the Tokenizer notebook) and cite it consistently.
5. Pin the plotting environment; commit the generated `figures/`.
