"""
Build confusion matrices from cross_eval_predictions.csv.gz.

Works on either source of predictions:
  - the file already on GitHub:  results/cross_eval/FED-v5_x_ECB-v5/cross_eval_predictions.csv.gz
  - the file predict_v5.py writes: out_v5/cross_eval_predictions.csv.gz

Outputs, into --out:
  cm_counts_long.csv            every cell of every matrix, tidy format
  cm_per_run_long.csv           same but not pooled over runs (for per-run analysis)
  per_class_metrics.csv         precision/recall/F1 per class, mean +/- std over the 5 runs
  summary_metrics.csv           macro-F1 and accuracy per (train, eval, model), mean +/- std
  figs/cm_<model>.png           2x2 panel per model: train institution x eval institution
  figs/cm_pooled_<train>_on_<eval>.png   all models pooled, one per quadrant

Pooling: counts are SUMMED over the 5 runs (so a 407-row test set gives 2035 predictions
per matrix). Percentages are row-normalised, i.e. each row is recall for that true class.

Panel titles show macro-F1 as mean +/- sd across runs: the score is computed separately
for each run (each model x run fit in the pooled figures), then averaged, with sd using
ddof=1. This is the same convention as the repo's mean/std summaries and is NOT the F1
of the pooled matrix, which can differ slightly. --stat accuracy shows accuracy instead,
also as mean +/- sd across runs.

    python confusion_matrices.py --preds out_v5/cross_eval_predictions.csv.gz --out figs_v5
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

LABELS = ["dovish", "hawkish", "neutral"]   # ids 0, 1, 2
IDS = [0, 1, 2]
INSTITUTIONS = ["FED", "ECB"]
STAT_LABEL = {"f1_macro": "macro-F1", "accuracy": "acc"}


def cm_for(df, normalize=None):
    return confusion_matrix(df["y_true"], df["y_pred"], labels=IDS, normalize=normalize)


def score(y_true, y_pred, stat):
    if stat == "f1_macro":
        return f1_score(y_true, y_pred, labels=IDS, average="macro", zero_division=0)
    return accuracy_score(y_true, y_pred)


def run_stats(g, stat):
    """mean, sd (ddof=1) and n of `stat`, computed once per (model, run) fit in g."""
    vals = np.array([score(gr["y_true"], gr["y_pred"], stat)
                     for _, gr in g.groupby(["model", "run"])], dtype=float)
    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
    return float(vals.mean()), float(sd), len(vals)


def stat_line(g, stat, unit="runs"):
    m, sd, n = run_stats(g, stat)
    return f"{STAT_LABEL[stat]} {m:.3f} ± {sd:.3f} ({n} {unit})"


def draw(ax, cm, title, show_pct=True):
    """cm is a count matrix; shading is row-normalised."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = cm / cm.sum(axis=1, keepdims=True)
    pct = np.nan_to_num(pct)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3), LABELS, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(3):
        for j in range(3):
            txt = f"{cm[i, j]:d}\n{pct[i, j]:.0%}" if show_pct else f"{cm[i, j]:d}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 0.5 else "black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True,
                    help="path to cross_eval_predictions.csv.gz")
    ap.add_argument("--out", default="cm_v5")
    ap.add_argument("--tag", default="v5", help="label used in figure titles")
    ap.add_argument("--stat", choices=list(STAT_LABEL), default="f1_macro",
                    help="score shown in panel titles as mean ± sd across runs "
                         "(default: f1_macro)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preds)
    need = {"train_institution", "model", "run", "eval_institution", "y_true", "y_pred"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"predictions file is missing columns: {sorted(missing)}")

    models = sorted(df["model"].unique())
    print(f"{len(df):,} predictions | {len(models)} models | "
          f"runs {sorted(int(r) for r in df['run'].unique())}")
    # the two test sets legitimately differ in size, so compare within eval_institution
    for ev, g in df.groupby("eval_institution"):
        c = g.groupby(["train_institution", "model", "run"]).size()
        if c.nunique() > 1:
            print(f"note: uneven prediction counts on the {ev} test set — "
                  f"sizes seen: {sorted(c.unique())}")
        n_cells = len(c)
        print(f"  {ev} test: {n_cells} (train,model,run) cells, "
              f"{c.iloc[0]} sentences each")

    # ---- tidy long tables -------------------------------------------------------
    long_rows, per_run_rows, pcm_rows, summary_rows = [], [], [], []

    for (tr, ev, m), g in df.groupby(["train_institution", "eval_institution", "model"]):
        cm = cm_for(g)
        for i, ti in enumerate(LABELS):
            for j, pj in enumerate(LABELS):
                long_rows.append({"train_institution": tr, "eval_institution": ev,
                                  "model": m, "true": ti, "pred": pj,
                                  "count": int(cm[i, j]),
                                  "row_pct": float(cm[i, j] / max(cm[i].sum(), 1))})

        # per-class and overall metrics, mean +/- std across runs
        stats, overall = [], {k: [] for k in STAT_LABEL}
        for run, gr in g.groupby("run"):
            p, r, f, _ = precision_recall_fscore_support(
                gr["y_true"], gr["y_pred"], labels=IDS, zero_division=0)
            stats.append(np.vstack([p, r, f]))
            for k in STAT_LABEL:
                overall[k].append(score(gr["y_true"], gr["y_pred"], k))
            cmr = cm_for(gr)
            for i, ti in enumerate(LABELS):
                for j, pj in enumerate(LABELS):
                    per_run_rows.append({"train_institution": tr, "eval_institution": ev,
                                         "model": m, "run": int(run), "true": ti,
                                         "pred": pj, "count": int(cmr[i, j])})
        arr = np.stack(stats)                     # (n_runs, 3 metrics, 3 classes)
        for k, mname in enumerate(["precision", "recall", "f1"]):
            for c, cname in enumerate(LABELS):
                pcm_rows.append({"train_institution": tr, "eval_institution": ev,
                                 "model": m, "metric": mname, "class": cname,
                                 "mean": float(arr[:, k, c].mean()),
                                 "std": float(arr[:, k, c].std(ddof=1))
                                 if arr.shape[0] > 1 else 0.0,
                                 "n_runs": int(arr.shape[0])})
        for k, vals in overall.items():
            vals = np.asarray(vals, dtype=float)
            summary_rows.append({"train_institution": tr, "eval_institution": ev,
                                 "model": m, "in_domain": tr == ev, "metric": k,
                                 "mean": float(vals.mean()),
                                 "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                                 "n_runs": int(len(vals))})

    pd.DataFrame(long_rows).to_csv(out / "cm_counts_long.csv", index=False)
    pd.DataFrame(per_run_rows).to_csv(out / "cm_per_run_long.csv", index=False)
    pd.DataFrame(pcm_rows).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "summary_metrics.csv", index=False)

    # ---- one 2x2 figure per model ----------------------------------------------
    for m in models:
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
        for a, tr in enumerate(INSTITUTIONS):
            for b, ev in enumerate(INSTITUTIONS):
                g = df[(df.model == m) & (df.train_institution == tr)
                       & (df.eval_institution == ev)]
                ax = axes[a][b]
                if g.empty:
                    ax.axis("off")
                    ax.set_title(f"{tr} -> {ev}: no data", fontsize=9)
                    continue
                tail = " (in-domain)" if tr == ev else ""
                draw(ax, cm_for(g),
                     f"{tr} model -> {ev} test{tail}\n{stat_line(g, args.stat)}")
        fig.suptitle(f"{m} — {args.tag}, counts pooled over {df['run'].nunique()} runs, "
                     f"{STAT_LABEL[args.stat]} mean ± sd across runs", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figs" / f"cm_{m}.png", dpi=160)
        plt.close(fig)

    # ---- all models pooled, one figure per quadrant ------------------------------
    for tr in INSTITUTIONS:
        for ev in INSTITUTIONS:
            g = df[(df.train_institution == tr) & (df.eval_institution == ev)]
            if g.empty:
                continue
            fig, ax = plt.subplots(figsize=(4.2, 4))
            draw(ax, cm_for(g), f"all models, {tr} -> {ev}\n"
                                f"{stat_line(g, args.stat, 'model×run fits')}")
            fig.tight_layout()
            fig.savefig(out / "figs" / f"cm_pooled_{tr}_on_{ev}.png", dpi=160)
            plt.close(fig)

    print(f"wrote tables + {len(list((out / 'figs').glob('*.png')))} figures to {out}/")


if __name__ == "__main__":
    main()
