"""
Advanced dissertation figures from results_per_run.csv (per-seed test metrics).

Zero configuration, same spirit as make_plots.py:
    python make_cool_plots.py            # auto-finds results_per_run.csv under cwd
    python make_cool_plots.py path/to/results_per_run.csv --out figures

Figures produced (all 300 dpi PNG):
  f1_performance_overview.png   forest-style seed/CI comparison (macro-F1 + accuracy)
  f2_dapt_interaction.png       paired slopegraphs: the DAPT x scale interaction
  f3_stability_map.png          mean performance vs seed sensitivity quadrants
  f4_metric_heatmap.png         models x metrics, column-normalised + annotated
  f5_rank_bump.png              bump chart of ranks across all four metrics
  f6_loss_vs_f1.png             per-run loss vs macro-F1 (loss != quality)
  f7_seed_trajectories.png      seed-level spaghetti across models
  f8_learning_curves.png        per-epoch validation loss + macro-F1, all models
  f9_train_vs_val.png           train vs validation loss grid, gap + best epoch

f8/f9 read ONE trainer_state.json per model from the hard-coded
TRAINER_STATES dict below. Edit those paths for your machine, or trim the
dict to a single line (e.g. just roberta-base) for a baseline-only version.
Anything missing is skipped with a warning. You can also override from the
command line without touching the file:
    python make_cool_plots.py --logs "path/to/logs" --run 3
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from scipy import stats

# ==================== HARD-CODED INPUT PATHS (edit me) ======================
BASE = Path("/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/"
            "Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/"
            "Multi Model Finetune - Bert")
RESULTS_CSV = BASE / "results_per_run.csv"     # falls back to auto-discovery
LOGS = BASE / "logs"

# One trainer_state per model for the learning-curve figures (f8 + f9).
# Keep all seven, or delete every line except roberta-base for baseline-only.
TRAINER_STATES = {
    "roberta-base":       LOGS / "roberta-base_run3_trainer_state.json",
    "roberta-dapt":       LOGS / "roberta-dapt_run3_trainer_state.json",
    "roberta-large":      LOGS / "roberta-large_run3_trainer_state.json",
    "roberta-large-dapt": LOGS / "roberta-large-dapt_run3_trainer_state.json",
    "deberta-v3":         LOGS / "deberta-v3_run3_trainer_state.json",
    "finbert":            LOGS / "finbert_run3_trainer_state.json",
    "finbert-tone":       LOGS / "finbert-tone_run3_trainer_state.json",
}
# ============================================================================

# ----------------------------------------------------------------- style ----
INK      = "#22303C"   # near-black text
SOFT     = "#6B7785"   # secondary text
GRID     = "#DCE2E8"
UP_GREEN = "#2E9E77"
DN_RED   = "#D65F5F"

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#B9C2CB", "axes.labelcolor": INK, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "grid.alpha": 0.6,
    "xtick.color": SOFT, "ytick.color": SOFT,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.frameon": False,
})

ORDER = ["roberta-base", "roberta-dapt", "roberta-large", "roberta-large-dapt",
         "deberta-v3", "finbert", "finbert-tone", "modernbert"]

DISPLAY = {
    "roberta-base":       "RoBERTa-base",
    "roberta-dapt":       "RoBERTa-base +DAPT",
    "roberta-large":      "RoBERTa-large",
    "roberta-large-dapt": "RoBERTa-large +DAPT",
    "deberta-v3":         "DeBERTa-v3-base",
    "finbert":            "FinBERT",
    "finbert-tone":       "FinBERT-tone",
    "modernbert":         "ModernBERT-base",
}

COLORS = {
    "roberta-base":       "#4E97D1",
    "roberta-dapt":       "#1F5F9E",
    "roberta-large":      "#8C7AE6",
    "roberta-large-dapt": "#5B3FBF",
    "deberta-v3":         "#D65C8B",
    "finbert":            "#E58A3A",
    "finbert-tone":       "#EFB84C",
    "modernbert":         "#3AA6A6",
}
FALLBACK = "#7F8C99"

HALO = [withStroke(linewidth=3, foreground="white")]


def col(m):
    return COLORS.get(m, FALLBACK)


def disp(m):
    return DISPLAY.get(m, m)


def footnote(fig, extra="", dataset="FOMC test set, n = 407 sentences"):
    txt = ("5 fine-tuning runs per model (seeds 42\u201346) \u00b7 "
           + dataset + (" \u00b7 " + extra if extra else ""))
    fig.text(0.01, -0.015, txt, ha="left", va="top", fontsize=8.8, color=SOFT)


def titles(fig, title, subtitle, x=0.01):
    fig.text(x, 1.035, title, ha="left", va="bottom",
             fontsize=16.5, fontweight="bold", color=INK)
    fig.text(x, 1.005, subtitle, ha="left", va="bottom",
             fontsize=11, color=SOFT)


# ------------------------------------------------------------------ data ----
def load(csv_path):
    df = pd.read_csv(csv_path)
    present = [m for m in ORDER if m in df.model.unique()]
    present += [m for m in df.model.unique() if m not in present]
    df["model"] = pd.Categorical(df.model, categories=present, ordered=True)
    return df.sort_values(["model", "seed"]).reset_index(drop=True), present


def summarise(df):
    g = df.groupby("model", observed=True)
    s = g.agg(n=("test_f1_macro", "size"),
              f1=("test_f1_macro", "mean"),  f1_sd=("test_f1_macro", "std"),
              f1_lo=("test_f1_macro", "min"), f1_hi=("test_f1_macro", "max"),
              acc=("test_accuracy", "mean"), acc_sd=("test_accuracy", "std"),
              acc_lo=("test_accuracy", "min"), acc_hi=("test_accuracy", "max"),
              f1w=("test_f1_weighted", "mean"),
              loss=("test_loss", "mean"), loss_sd=("test_loss", "std"))
    tcrit = stats.t.ppf(0.975, s.n - 1)
    s["f1_ci"] = tcrit * s.f1_sd / np.sqrt(s.n)
    s["acc_ci"] = tcrit * s.acc_sd / np.sqrt(s.n)
    return s.reset_index()


# ---------------------------------------------------- fig 1: forest plot ----
def fig_overview(df, s, out):
    s = s.sort_values("f1", ascending=True).reset_index(drop=True)  # best on top
    order = list(s.model)
    ypos = {m: i for i, m in enumerate(order)}
    best = order[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.0), sharey=True)
    panels = [("test_f1_macro", "f1", "f1_ci", "f1_lo", "f1_hi", "Macro-F1"),
              ("test_accuracy", "acc", "acc_ci", "acc_lo", "acc_hi", "Accuracy")]
    jit = np.linspace(-0.16, 0.16, 5)

    for ax, (runcol, m_, ci_, lo_, hi_, label) in zip(axes, panels):
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        ax.grid(False, axis="y")
        for i in range(len(order)):                      # zebra rows
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#F3F6F9", zorder=0)
        ax.axhspan(ypos[best] - 0.5, ypos[best] + 0.5,   # gold band for winner
                   color="#F5B301", alpha=0.10, zorder=0)

        for _, r in s.iterrows():
            y, c = ypos[r.model], col(r.model)
            ax.plot([r[lo_], r[hi_]], [y, y], color=c, alpha=0.30, lw=2.6,
                    solid_capstyle="round", zorder=2)                 # seed range
            ax.plot([r[m_] - r[ci_], r[m_] + r[ci_]], [y, y], color=c,
                    lw=6.5, alpha=0.95, solid_capstyle="round", zorder=3)  # 95% CI
            runs = df.loc[df.model == r.model, runcol].to_numpy()
            ax.scatter(runs, y + jit[: len(runs)], s=26, color=c, alpha=0.95,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.scatter([r[m_]], [y], s=115, color=c, edgecolor="white",
                       linewidth=1.6, zorder=5)
            anchor = max(r[hi_], r[m_] + r[ci_])
            ax.annotate(f"{r[m_]:.3f}", (anchor, y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9.6,
                        fontweight="bold" if r.model == best else "normal",
                        color=INK if r.model == best else SOFT, zorder=6)

        lo = min(s[lo_].min(), (s[m_] - s[ci_]).min())
        hi = max(s[hi_].max(), (s[m_] + s[ci_]).max())
        pad = (hi - lo) * 0.10
        ax.set_xlim(lo - pad, hi + pad * 3.2)
        ax.set_ylim(-0.55, len(order) - 0.45)
        ax.set_title(label, fontsize=13, fontweight="bold", color=INK, pad=10)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([disp(m) for m in order], fontsize=11)
    for tl in axes[0].get_yticklabels():
        key = [m for m in order if disp(m) == tl.get_text()][0]
        tl.set_color(col(key))
        tl.set_fontweight("bold" if key == best else "normal")

    titles(fig, "Who reads the Fed best?",
           "Test performance on FOMC hawkish/dovish/neutral classification \u2014 "
           "dots: individual seeds \u00b7 thick bar: 95% CI \u00b7 thin bar: seed range")
    footnote(fig, "CI: Student-t on 5 seeds")
    fig.tight_layout()
    fig.savefig(out / "f1_performance_overview.png")
    plt.close(fig)
    print("[ok] f1_performance_overview.png")


def dodge(vals, min_gap):
    """Nudge overlapping label positions apart while preserving order."""
    idx = np.argsort(vals)
    pos = np.array(vals, dtype=float)[idx]
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < min_gap:
            pos[i] = pos[i - 1] + min_gap
    outp = np.empty_like(pos)
    outp[idx] = pos
    return outp


# ------------------------------------------- fig 2: DAPT paired slopes ------
def fig_dapt(df, out):
    pairs = [("roberta-base", "roberta-dapt", "RoBERTa-base (125M)"),
             ("roberta-large", "roberta-large-dapt", "RoBERTa-large (355M)")]
    pairs = [p for p in pairs
             if {p[0], p[1]} <= set(df.model.astype(str).unique())]
    if not pairs:
        print("[skip] f2: no base/DAPT pairs present")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(11.2, 6.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (a, b, ttl) in zip(axes, pairs):
        A = df[df.model == a].set_index("seed").test_f1_macro
        B = df[df.model == b].set_index("seed").test_f1_macro
        seeds = sorted(set(A.index) & set(B.index))
        A, B = A.loc[seeds], B.loc[seeds]
        d = (B - A).mean() * 100
        t, p = stats.ttest_rel(B, A)
        cdir = UP_GREEN if d > 0 else DN_RED

        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.grid(False, axis="x")
        span = (max(A.max(), B.max()) - min(A.min(), B.min())) or 1.0
        lab_y = dict(zip(seeds, dodge([A[sd] for sd in seeds], span * 0.045)))
        for sd in seeds:                                           # seed lines
            good = B[sd] >= A[sd]
            ax.plot([0, 1], [A[sd], B[sd]], color="#9AA6B2", lw=1.4,
                    alpha=0.85, zorder=2)
            ax.scatter([0, 1], [A[sd], B[sd]], s=34, color="#9AA6B2",
                       edgecolor="white", lw=0.8, zorder=3)
            ax.annotate(str(sd), (0, lab_y[sd]), xytext=(-14, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8.4, color=SOFT)
            if not good and a == "roberta-large":
                ax.annotate("only seed 45\nregresses", (1, B[sd]),
                            xytext=(14, -6), textcoords="offset points",
                            fontsize=8.6, color=SOFT, va="top")
        ax.plot([0, 1], [A.mean(), B.mean()], color=cdir, lw=4.2,
                solid_capstyle="round", zorder=4)                  # mean slope
        ax.scatter([0, 1], [A.mean(), B.mean()], s=170, color=cdir,
                   edgecolor="white", lw=1.8, zorder=5)

        sign = "+" if d > 0 else "\u2212"
        cx, ha = (0.04, "left") if d > 0 else (0.96, "right")
        ax.text(cx, 0.97, f"{sign}{abs(d):.1f} pp", transform=ax.transAxes,
                ha=ha, va="top", fontsize=16, fontweight="bold", color=cdir,
                path_effects=HALO)
        ax.text(cx, 0.885, f"paired t: p = {p:.2f}", transform=ax.transAxes,
                ha=ha, va="top", fontsize=9.2, color=SOFT, path_effects=HALO)

        ax.set_xlim(-0.42, 1.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["off-the-shelf", "+ DAPT\n(30M-token FOMC corpus)"],
                           fontsize=10.5, color=INK)
        ax.set_title(ttl, fontsize=13, fontweight="bold", pad=10)

    axes[0].set_ylabel("test macro-F1")
    titles(fig, "Domain-adaptive pre-training only pays off at scale",
           "Same 30M-token FOMC corpus, same recipe \u2014 DAPT drags the 125M model "
           "down on every seed, but lifts the 355M model on 4 of 5")
    footnote(fig, "lines connect identical seeds")
    fig.tight_layout()
    fig.savefig(out / "f2_dapt_interaction.png")
    plt.close(fig)
    print("[ok] f2_dapt_interaction.png")


# --------------------------------------------- fig 3: stability quadrant ----
def fig_stability(s, out):
    x = s.f1_sd * 100          # seed std in pp
    y = s.f1
    xm, ym = x.median(), y.median()

    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.set_axisbelow(True)
    ax.axvline(xm, color="#B9C2CB", lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(ym, color="#B9C2CB", lw=1.1, ls=(0, (4, 3)), zorder=1)

    for _, r in s.iterrows():
        ax.scatter(r.f1_sd * 100, r.f1, s=430, color=col(r.model), alpha=0.95,
                   edgecolor="white", linewidth=1.8, zorder=3)

    offsets = {
        "roberta-base":       (0, 16, "center", "bottom"),
        "roberta-dapt":       (0, 16, "center", "bottom"),
        "roberta-large":      (0, -20, "center", "top"),
        "roberta-large-dapt": (0, 16, "center", "bottom"),
        "deberta-v3":         (0, 16, "center", "bottom"),
        "finbert":            (12, 10, "left", "bottom"),
        "finbert-tone":       (12, -10, "left", "top"),
    }
    for _, r in s.iterrows():
        dx, dy, ha, va = offsets.get(r.model, (0, 16, "center", "bottom"))
        ax.annotate(disp(r.model), (r.f1_sd * 100, r.f1), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=10.5,
                    fontweight="bold", color=col(r.model), path_effects=HALO)

    if "deberta-v3" in set(s.model):
        r = s[s.model == "deberta-v3"].iloc[0]
        ax.annotate("seed 43 collapses to 0.43 macro-F1 \u2014\n"
                    "classic DeBERTa fine-tuning instability",
                    (r.f1_sd * 100, r.f1), xytext=(-46, 52),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=9, color=SOFT,
                    arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.1,
                                    connectionstyle="arc3,rad=-0.25"))

    qkw = dict(fontsize=10, color="#A6AFB9", style="italic",
               path_effects=HALO, zorder=2)
    ax.text(0.02, 0.98, "strong & seed-robust", transform=ax.transAxes,
            ha="left", va="top", **qkw)
    ax.text(0.98, 0.98, "strong but seed-sensitive", transform=ax.transAxes,
            ha="right", va="top", **qkw)
    ax.text(0.02, 0.02, "weak & seed-robust", transform=ax.transAxes,
            ha="left", va="bottom", **qkw)
    ax.text(0.98, 0.02, "weak & seed-sensitive", transform=ax.transAxes,
            ha="right", va="bottom", **qkw)

    ax.set_xlabel("seed sensitivity \u2014 std of macro-F1 across 5 seeds (pp)")
    ax.set_ylabel("mean test macro-F1")
    ax.set_xlim(0, x.max() * 1.14)
    ax.set_ylim(y.min() - 0.022, y.max() + 0.022)
    titles(fig, "The performance\u2013stability map",
           "Where you want to live is the top-left \u2014 dashed lines mark medians; "
           "note RoBERTa-base +DAPT: mediocre but almost seed-proof")
    footnote(fig)
    fig.tight_layout()
    fig.savefig(out / "f3_stability_map.png")
    plt.close(fig)
    print("[ok] f3_stability_map.png")


# ------------------------------------------------- fig 4: metric heatmap ----
def fig_heatmap(s, out):
    s = s.sort_values("f1", ascending=False).reset_index(drop=True)
    cols = [("f1", "Macro-F1", False), ("f1w", "Weighted-F1", False),
            ("acc", "Accuracy", False), ("loss", "CE loss  \u2193", True)]
    raw = np.array([[r[k] for k, _, _ in cols] for _, r in s.iterrows()])
    norm = np.zeros_like(raw)
    for j, (_, _, inv) in enumerate(cols):
        v = raw[:, j]
        n = (v - v.min()) / (v.max() - v.min() + 1e-12)
        norm[:, j] = 1 - n if inv else n

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "bdf", ["#F5F8FA", "#BFD8EC", "#5C9BD1", "#1F5F9E"])

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.grid(False)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v, nrm = raw[i, j], norm[i, j]
            best = nrm >= norm[:, j].max() - 1e-12
            txt = f"{v:.3f}" + (" \u2605" if best else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if best else "normal",
                    color="white" if nrm > 0.55 else INK)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lbl for _, lbl, _ in cols], fontsize=11.5,
                       fontweight="bold", color=INK)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([disp(m) for m in s.model], fontsize=11)
    for tl, m in zip(ax.get_yticklabels(), s.model):
        tl.set_color(col(m))
        tl.set_fontweight("bold")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)   # white grid
    ax.set_yticks(np.arange(-0.5, len(s)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    titles(fig, "The scoreboard",
           "Mean over 5 seeds \u00b7 colour = column-normalised (darker is better; "
           "loss inverted) \u00b7 \u2605 = best in column")
    footnote(fig)
    fig.tight_layout()
    fig.savefig(out / "f4_metric_heatmap.png")
    plt.close(fig)
    print("[ok] f4_metric_heatmap.png")


# ----------------------------------------------------- fig 5: bump chart ----
def fig_bump(s, out):
    metrics = [("f1", "Macro-F1", False), ("f1w", "Weighted-F1", False),
               ("acc", "Accuracy", False), ("loss", "CE loss", True)]
    ranks = pd.DataFrame({"model": s.model})
    for k, _, inv in metrics:
        ranks[k] = s[k].rank(ascending=inv, method="min").astype(int)

    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    ax.set_axisbelow(True)
    ax.grid(False)
    n = len(s)
    xs = np.arange(len(metrics))
    for yy in range(1, n + 1):
        ax.axhline(yy, color="#EEF2F5", lw=1.4, zorder=0)

    champ = s.sort_values("f1", ascending=False).model.iloc[0]
    for _, r in ranks.iterrows():
        ys = [r[k] for k, _, _ in metrics]
        c = col(r.model)
        lw = 4.0 if r.model == champ else 2.2
        ax.plot(xs, ys, color=c, lw=lw, alpha=0.95, zorder=2,
                solid_capstyle="round")
        ax.scatter(xs, ys, s=300, color=c, edgecolor="white", lw=1.6, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y, str(y), ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white", zorder=4)
        kw = dict(va="center", fontsize=10.5, fontweight="bold", color=c,
                  path_effects=HALO)
        ax.text(-0.14, ys[0], disp(r.model), ha="right", **kw)
        ax.text(len(metrics) - 1 + 0.14, ys[-1], disp(r.model), ha="left", **kw)

    ax.set_xlim(-1.55, len(metrics) - 1 + 1.55)
    ax.set_ylim(n + 0.6, 0.4)                      # rank 1 on top
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lbl}{'  \u2193' if inv else ''}"
                        for _, lbl, inv in metrics],
                       fontsize=11.5, fontweight="bold", color=INK)
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, n + 1)], fontsize=9.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    titles(fig, "Rankings barely move \u2014 until you look at loss",
           "Rank of each model (1 = best) under four evaluation metrics, "
           "means over 5 seeds \u00b7 loss ranks reshuffle the podium")
    footnote(fig)
    fig.tight_layout()
    fig.savefig(out / "f5_rank_bump.png")
    plt.close(fig)
    print("[ok] f5_rank_bump.png")


# ---------------------------------------------------- fig 6: loss vs F1 -----
def fig_loss(df, s, out):
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.set_axisbelow(True)

    for m in s.model:
        d = df[df.model == m]
        ax.scatter(d.test_loss, d.test_f1_macro, s=105, color=col(m),
                   alpha=0.92, edgecolor="white", linewidth=1.0, zorder=3,
                   label=disp(m))
    for _, r in s.iterrows():
        ax.scatter([r.loss], [r.f1], s=300, marker="D", color=col(r.model),
                   edgecolor=INK, linewidth=1.3, alpha=0.95, zorder=4)

    rho, pval = stats.spearmanr(df.test_loss, df.test_f1_macro)
    ptxt = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
    ax.text(0.985, 0.975,
            f"pooled Spearman \u03c1 = {rho:.2f} ({ptxt}), all "
            f"{len(df)} runs",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.6,
            color=SOFT, path_effects=HALO)

    rb = df[(df.model == "roberta-base") & (df.test_loss < 1.0)]
    if len(rb):
        cx, cy = rb.test_loss.mean(), rb.test_f1_macro.mean()
        ax.annotate("RoBERTa-base seeds 42\u201344: same F1\n"
                    "as seeds 45\u201346, at half the loss",
                    (rb.test_loss.max(), cy), xytext=(28, -46),
                    textcoords="offset points", fontsize=9.2, color=SOFT,
                    arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.1,
                                    connectionstyle="arc3,rad=-0.25"))
    dv = df[(df.model == "deberta-v3") & (df.test_f1_macro < 0.5)]
    for _, r in dv.iterrows():
        ax.annotate("DeBERTa seed 43\ncollapse", (r.test_loss, r.test_f1_macro),
                    xytext=(-64, 10), textcoords="offset points", fontsize=9.2,
                    color=SOFT,
                    arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.1,
                                    connectionstyle="arc3,rad=0.2"))

    ax.set_xlabel("test cross-entropy loss")
    ax.set_ylabel("test macro-F1")
    ax.legend(loc="lower left", fontsize=9.3, ncols=2, handletextpad=0.15,
              columnspacing=0.9, borderaxespad=0.2)
    titles(fig, "Loss separates models \u2014 not runs",
           "Dots: single runs, diamonds: model means. Pooled, loss tracks F1; "
           "within a model it mostly measures confidence \u2014 RoBERTa-base's "
           "loss doubles at identical F1")
    footnote(fig)
    fig.tight_layout()
    fig.savefig(out / "f6_loss_vs_f1.png")
    plt.close(fig)
    print("[ok] f6_loss_vs_f1.png")


# ---------------------------------------------- fig 7: seed spaghetti -------
def fig_seeds(df, s, out):
    order = list(s.sort_values("f1", ascending=False).model)
    xs = np.arange(len(order))
    seeds = sorted(df.seed.unique())
    shades = ["#1C2B3A", "#3E5871", "#6784A0", "#93AABF", "#C0CFDC"]

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.grid(False, axis="x")

    piv = df.pivot_table(index="seed", columns="model", values="test_f1_macro",
                         observed=True)
    for sd, c in zip(seeds, shades):
        ys = [piv.loc[sd, m] for m in order]
        ax.plot(xs, ys, color=c, lw=1.7, alpha=0.95, zorder=2,
                marker="o", ms=5.5, mec="white", mew=0.7, label=f"seed {sd}")
    ax.plot(xs, [s.set_index("model").f1[m] for m in order], color="#111111",
            lw=3.4, zorder=3, marker="o", ms=8, mec="white", mew=1.2,
            solid_capstyle="round", label="mean")

    hi = piv.loc[45, "roberta-large"] if "roberta-large" in piv else None
    if hi is not None:
        ax.annotate("seed 45 alone makes\nRoBERTa-large look great",
                    (order.index("roberta-large"), hi), xytext=(6, 26),
                    textcoords="offset points", fontsize=9.2, color=SOFT,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.1,
                                    connectionstyle="arc3,rad=-0.25"))
    lo = piv.loc[43, "deberta-v3"] if "deberta-v3" in piv else None
    if lo is not None:
        ax.annotate("seed 43 wipes out\nDeBERTa-v3",
                    (order.index("deberta-v3"), lo), xytext=(-120, 26),
                    textcoords="offset points", ha="right", fontsize=9.2,
                    color=SOFT,
                    arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.1,
                                    connectionstyle="arc3,rad=0.25"))

    ax.set_xticks(xs)
    ax.set_xticklabels([disp(m) for m in order], fontsize=10.2, rotation=14)
    for tl, m in zip(ax.get_xticklabels(), order):
        tl.set_color(col(m))
        tl.set_fontweight("bold")
    ax.set_ylabel("test macro-F1")
    ax.legend(loc="lower left", fontsize=9.3, ncols=6, columnspacing=1.1,
              handletextpad=0.4)
    titles(fig, "Same seed, different story",
           "One line per random seed across all models (sorted by mean) \u2014 "
           "a single-run comparison could crown almost anyone")
    footnote(fig)
    fig.tight_layout()
    fig.savefig(out / "f7_seed_trajectories.png")
    plt.close(fig)
    print("[ok] f7_seed_trajectories.png")


# ------------------------------------------- trainer_state ingestion --------
def load_states(paths, df):
    """Read hard-coded trainer_state.json files -> {model: (train, eval)}."""
    states, runs = {}, set()
    for m, p in paths.items():
        p = Path(p)
        if not p.exists():
            print(f"[warn] trainer_state missing for {m}: {p}")
            continue
        try:
            hist = json.loads(p.read_text())["log_history"]
        except Exception as e:
            print(f"[warn] could not parse {p.name}: {e}")
            continue
        tr = pd.DataFrame([e for e in hist
                           if "loss" in e and "eval_loss" not in e])
        ev = pd.DataFrame([e for e in hist if "eval_loss" in e])
        states[m] = (tr, ev)
        mm = re.search(r"_run(\d+)_", p.name)
        if mm:
            runs.add(int(mm.group(1)))
    note = "one hand-picked run per model"
    if len(runs) == 1:
        rn = runs.pop()
        sd = df.loc[df.run == rn, "seed"]
        note = (f"run {rn} (seed {int(sd.iloc[0])}) for every model"
                if len(sd) else f"run {rn} for every model")
    return states, note


# --------------------------------------------- fig 8: learning curves -------
def fig_learning(states, note, out):
    if not states:
        print("[skip] f8: no trainer_state files found")
        return
    has_f1 = any((not ev.empty) and "eval_f1_macro" in ev.columns
                 for _, ev in states.values())
    ncols = 2 if has_f1 else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.3 * ncols, 5.6))
    axes = np.atleast_1d(axes)

    for m, (_, ev) in states.items():
        if ev.empty:
            continue
        axes[0].plot(ev.epoch, ev.eval_loss, color=col(m), lw=2.3, marker="o",
                     ms=4.8, mec="white", mew=0.7, label=disp(m))
        if has_f1 and "eval_f1_macro" in ev.columns:
            axes[1].plot(ev.epoch, ev.eval_f1_macro, color=col(m), lw=2.3,
                         marker="o", ms=4.8, mec="white", mew=0.7,
                         label=disp(m))

    axes[0].set_title("Validation loss", fontsize=13, fontweight="bold",
                      color=INK, pad=10)
    if has_f1:
        axes[1].set_title("Validation macro-F1", fontsize=13,
                          fontweight="bold", color=INK, pad=10)
        axes[1].legend(loc="lower right", fontsize=9, ncols=2,
                       handletextpad=0.4, columnspacing=1.0)
    else:
        axes[0].legend(loc="best", fontsize=9)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    titles(fig, "How they learn",
           f"Per-epoch validation metrics during fine-tuning \u2014 {note}")
    footnote(fig, dataset="metrics on the FOMC validation split")
    fig.tight_layout()
    fig.savefig(out / "f8_learning_curves.png")
    plt.close(fig)
    print("[ok] f8_learning_curves.png")


# ------------------------------------ fig 9: train vs validation grid -------
def fig_train_val(states, note, out):
    if not states:
        print("[skip] f9: no trainer_state files found")
        return
    items = list(states.items())
    n = len(items)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.0 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, (m, (tr, ev)) in zip(axes, items):
        c = col(m)
        ax.set_axisbelow(True)
        if not tr.empty:
            ax.plot(tr.epoch, tr.loss, color=c, lw=1.5, alpha=0.45,
                    label="training")
        if not ev.empty:
            ax.plot(ev.epoch, ev.eval_loss, color=c, lw=2.4, marker="o",
                    ms=4.8, mec="white", mew=0.8, label="validation")
            if not tr.empty:
                gap = np.interp(ev.epoch, tr.epoch, tr.loss)
                ax.fill_between(ev.epoch, gap, ev.eval_loss, color=c,
                                alpha=0.10, lw=0)
            b = ev.loc[ev.eval_loss.idxmin()]
            ax.scatter([b.epoch], [b.eval_loss], marker="*", s=230, color=c,
                       edgecolor="white", lw=0.9, zorder=5)
            ax.annotate(f"best ep {int(round(b.epoch))}",
                        (b.epoch, b.eval_loss), xytext=(0, -15),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8.6, color=SOFT, path_effects=HALO)
        ax.set_title(disp(m), fontsize=12, fontweight="bold", color=c, pad=8)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    for ax in axes[n:]:
        ax.axis("off")
    for j in range(n):
        if j % ncols == 0:
            axes[j].set_ylabel("loss")
        if j + ncols >= n:                    # nothing visible underneath
            axes[j].tick_params(labelbottom=True)
            axes[j].set_xlabel("epoch")
    axes[0].legend(loc="upper right", fontsize=9)

    titles(fig, "Where the generalisation gap opens",
           f"Training vs validation loss, {note} \u2014 shaded band: gap "
           "between the curves \u00b7 \u2605 best validation epoch")
    footnote(fig, dataset="losses on the FOMC train/validation splits")
    fig.tight_layout()
    fig.savefig(out / "f9_train_vs_val.png")
    plt.close(fig)
    print("[ok] f9_train_vs_val.png")


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--logs", default=None,
                    help="override TRAINER_STATES: read "
                         "<logs>/<model>_run<RUN>_trainer_state.json")
    ap.add_argument("--run", type=int, default=3,
                    help="run number used with --logs (default 3)")
    a = ap.parse_args()

    csv = Path(a.csv) if a.csv else None
    if csv is None and RESULTS_CSV.exists():
        csv = RESULTS_CSV
    if csv is None or not csv.exists():
        hits = sorted(Path(".").rglob("results_per_run.csv"))
        if not hits:
            sys.exit("results_per_run.csv not found - pass its path explicitly")
        csv = hits[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("data:", csv)

    df, _ = load(csv)
    s = summarise(df)

    fig_overview(df, s, out)
    fig_dapt(df, out)
    fig_stability(s, out)
    fig_heatmap(s, out)
    fig_bump(s, out)
    fig_loss(df, s, out)
    fig_seeds(df, s, out)

    state_paths = ({m: Path(a.logs) / f"{m}_run{a.run}_trainer_state.json"
                    for m in TRAINER_STATES} if a.logs else TRAINER_STATES)
    states, note = load_states(state_paths, df)
    fig_learning(states, note, out)
    fig_train_val(states, note, out)
    print("\ndone ->", out.resolve())


if __name__ == "__main__":
    main()
