# README — `Training Curves.ipynb`

**Role in the project:** the diagnostic that shows *how* the reported fine-tunes trained, rather than what they scored. It reads the `trainer_state.json` written by every Hugging Face `Trainer` run and plots train-loss vs. eval-loss per epoch as a grid — one panel per (model, seed) run, one figure per institution. This is the evidence behind the paper's claims about early stopping and `load_best_model_at_end`: it makes visible, for each of the five seeds, where the eval loss turned and which checkpoint was actually kept.

**Evidence tags:** **[stated]** = written in code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = from the saved notebook outputs (last executed run).

---

## Input

Glob pairs over the **v5 run** (the different-data-split robustness trial), one per institution [stated]:

```
.../Trail of Different Data Split/Learning rate - 3e-5/logs/FED-v5/*trainer_state.json
.../Trail of Different Data Split/Learning rate - 3e-5/logs/ECB-v5/*trainer_state.json
```

Each `logs/<INST>-v5/` directory holds `<model>_run<N>_trainer_state.json` plus a matching `<model>_run<N>_test_metrics.json` — 7 models × 5 seeds. Only the `trainer_state` files are read; `log_history` supplies `loss` (per logging step) and `eval_loss` (per epoch), and the top-level `epoch`, `best_global_step` / `best_model_checkpoint` and `num_train_epochs` supply the annotation lines.

⚠ **Broken paths [stated].** The hard-coded paths omit the `Full Cross Eval MultiModel Runs/` level — they read
`.../Bert Model Finetuning Code /Trail of Different Data Split/...`, but the directory actually lives at
`.../Bert Model Finetuning Code /Full Cross Eval MultiModel Runs/Trail of Different Data Split/...`.
The glob silently returns an empty list rather than raising, so the notebook would render empty axes on a fresh run. Fix the path (and repoint the `/Users/lorenzouberti/...` prefix) before re-running.

## Output

Four figures, displayed inline only — nothing is written to disk [stated]. Grid layout is 5 columns × ⌈n/5⌉ rows, i.e. **one row per model, one column per seed** [stated]. Unused axes are switched off; the legend is drawn on the first panel only; `suptitle` is the institution tag.

## The three plotting cells

Cells 2, 3 and 4 are near-identical redraws of the same grid. The differences are the whole point:

| Cell | Annotation lines | Use |
|---|---|---|
| 2 | black dashed = `s["epoch"]`, the epoch training actually **stopped** at | baseline view |
| 3 | adds a **green dotted line at the best checkpoint** — the epoch `load_best_model_at_end` restored and that was pushed to the Hub | **the authoritative version** |
| 4 | identical to cell 2 | leftover duplicate |

Cell 3 resolves the best checkpoint defensively: `best_global_step` if the key exists, otherwise the step parsed out of `best_model_checkpoint` with `re.search(r"checkpoint-(\d+)")`, then maps that step back to its epoch via the first `log_history` entry carrying both that `step` and an `eval_loss` [stated]. **Use cell 3 for anything that goes in the paper** — the gap between the green (kept) and black (stopped) lines is exactly the early-stopping patience of 2 described in the methods, and cells 2 and 4 hide it.

Cell 0 is imports; cell 1 only re-declares the two path variables and produces no output.

## How to read the figures

- Train loss (blue) falling while eval loss (red) flattens or turns up = the overfitting the patience-2 early stop is there to catch.
- Green line well left of black = early stopping fired and an earlier checkpoint was kept; green ≈ black = the run improved until it ran out of epochs, so the 5-epoch budget may be binding for that model.
- Comparing the FED and ECB figures side by side is the intended use: the ECB v5 split is the robustness variant that replaces the chronological cut with 80/20, so its curves should be checked for the smoother, easier-looking convergence that a random split produces.

## Reproducibility

- **No randomness in this notebook** — it is pure plotting over saved JSON. Determinism belongs to the training runs that produced `logs/`.
- **Environment:** Anaconda Python 3.13, kernel `base` [recorded]; needs `pandas` (imported but effectively unused), `matplotlib`, stdlib `json`/`glob`/`re`/`pathlib`. A `numexpr` version `UserWarning` from pandas appears in the saved output and is harmless [recorded].
- **Path hazards:** absolute macOS paths, the trailing-space folders `BANQUE DE FRANCE ` and `Bert Model Finetuning Code `, and the missing directory level noted above.

## Audit notes / pre-submission checks

1. **Fix the glob path** (missing `Full Cross Eval MultiModel Runs/`) — as committed, this notebook cannot reproduce its own saved figures.
2. **Delete cells 2 and 4, keep cell 3.** Three copies of one plot with two of them missing the best-checkpoint marker is a trap for anyone re-running this.
3. **The notebook only covers v5.** The reported grid is `Learning Rate 3e-5` (config v4); v5 is the alternate-split robustness run. If a training-curve figure goes in the paper, either point the globs at the reported run or state explicitly in the caption that the curves are from the robustness split.
4. **No figure is saved.** Add `fig.savefig(...)` into `figures/` if these are cited, so the PNG in the paper is traceable to a run.
5. Cell 1 is dead code (re-declares variables that cells 2–4 each re-declare anyway) — drop it.
