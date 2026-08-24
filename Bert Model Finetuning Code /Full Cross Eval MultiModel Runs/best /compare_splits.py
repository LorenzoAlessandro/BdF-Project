#!/usr/bin/env python3
"""
Compare two results directories that share a learning rate, isolating whatever
else differs between them (here: the data split, v3 vs v5).

Usage:
    python compare_splits.py "Learning Rate 3e-5/results" \
                             "Trail of Different Data Split/Learning rate - 3e-5/results" \
                             --labels v3 v5

Reports, per model and institution:
  - mean macro F1 under each split and the delta
  - a paired-by-seed t-test if results_per_run_*.csv are present
  - whether the delta exceeds the seed-level noise in either split
  - whether the MODEL RANKING is preserved (Spearman + exact order)

The ranking check is the important one: if rankings agree, split choice is not
driving your conclusions, and you can report results at a single split with a
robustness footnote. If they disagree, the split is doing more work than the
models are.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

INST_PATTERN = re.compile(r"results_(?:summary|per_run)_([A-Za-z]+)", re.I)


def load(results_dir: Path, kind: str, label: str) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob(f"results_{kind}_*.csv")):
        if "cross_eval" in {p.lower() for p in path.parts}:
            continue
        df = pd.read_csv(path)
        m = INST_PATTERN.search(path.stem)
        df["institution"] = m.group(1).upper() if m else "UNKNOWN"
        df["split"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--metric", default="test_f1_macro_mean")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    la, lb = args.labels
    da, db = Path(args.dir_a).expanduser(), Path(args.dir_b).expanduser()
    metric = args.metric
    run_metric = metric.replace("_mean", "")  # per-run column name
    std_col = metric.replace("_mean", "_std")

    summ = pd.concat([load(da, "summary", la), load(db, "summary", lb)], ignore_index=True)
    if summ.empty:
        raise SystemExit("No results_summary_*.csv found in one or both directories.")

    piv = summ.pivot_table(index=["institution", "model"], columns="split", values=metric)
    sd = summ.pivot_table(index=["institution", "model"], columns="split", values=std_col) \
        if std_col in summ.columns else None

    out = piv.copy()
    out["delta"] = out[lb] - out[la]
    if sd is not None:
        pooled = ((sd[la] ** 2 + sd[lb] ** 2) / 2) ** 0.5
        out["pooled_seed_sd"] = pooled
        out["delta_in_sd"] = out["delta"] / pooled
        out["beyond_noise"] = out["delta"].abs() > pooled

    # paired seed-level test where per-run files exist
    runs = pd.concat([load(da, "per_run", la), load(db, "per_run", lb)], ignore_index=True)
    if HAVE_SCIPY and not runs.empty and run_metric in runs.columns and "seed" in runs.columns:
        pvals = {}
        for (inst, model), g in runs.groupby(["institution", "model"]):
            a = g[g.split == la].sort_values("seed")[run_metric].values
            b = g[g.split == lb].sort_values("seed")[run_metric].values
            if len(a) == len(b) and len(a) > 1:
                pvals[(inst, model)] = stats.ttest_rel(b, a).pvalue
        if pvals:
            out["p_paired"] = pd.Series(pvals)
            out.loc[out["p_paired"].notna(), "note"] = \
                out["p_paired"].map(lambda p: "" if pd.isna(p) else
                                    ("sig" if p < 0.05 else "n.s."))
    elif runs.empty:
        print("  (no results_per_run_*.csv found — summary-level comparison only)\n")

    print(f"=== {metric}: {la} vs {lb} (same LR, different split) ===")
    print(out.round(4).to_string())

    if "beyond_noise" in out.columns:
        n = int(out["beyond_noise"].sum())
        print(f"\n  {n}/{len(out)} cells move by more than one pooled seed SD.")
        print(f"  Mean |delta|: {out['delta'].abs().mean():.4f}   "
              f"Max |delta|: {out['delta'].abs().max():.4f} "
              f"({out['delta'].abs().idxmax()})")
        print(f"  Mean signed delta: {out['delta'].mean():+.4f}  "
              f"(a consistent sign means one split is simply easier)")

    # ---- ranking stability, the thing that actually matters ----
    print(f"\n=== Model ranking stability ===")
    for inst in sorted(piv.index.get_level_values("institution").unique()):
        sub = piv.xs(inst, level="institution").dropna()
        ra, rb = sub[la].rank(ascending=False), sub[lb].rank(ascending=False)
        order_a = list(sub[la].sort_values(ascending=False).index)
        order_b = list(sub[lb].sort_values(ascending=False).index)
        line = f"  {inst}: "
        if HAVE_SCIPY and len(sub) > 2:
            rho, p = stats.spearmanr(ra, rb)
            line += f"Spearman rho={rho:.3f} (p={p:.3f})  "
        line += "IDENTICAL order" if order_a == order_b else "order CHANGED"
        print(line)
        if order_a != order_b:
            print(f"    {la}: {' > '.join(order_a)}")
            print(f"    {lb}: {' > '.join(order_b)}")

    outpath = Path(args.outdir) / f"split_comparison_{la}_vs_{lb}.csv"
    out.to_csv(outpath)
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    main()
