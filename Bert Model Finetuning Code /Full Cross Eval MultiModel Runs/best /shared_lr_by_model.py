#!/usr/bin/env python3
"""
Pick ONE learning rate per model, shared across both training institutions.

Unlike best_lr_by_model.py (which lets ECB and FED each pick their own winner),
this selects a single LR per model using a joint criterion over both
institutions, and reports what that shared choice costs each side relative to
its own individual optimum.

Criteria:
    mean  (default)  maximise the average macro F1 across institutions
    min              maximin — maximise the worst institution's score
    rank             lowest summed rank across institutions (robust to scale)

Also reports a single GLOBAL LR across all models, which is usually the
defensible protocol when per-model differences are within noise.

Usage:
    python shared_lr_by_model.py "Full Cross Eval MultiModel Runs"
    python shared_lr_by_model.py . --version v4 --criterion min
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

LR_PATTERN = re.compile(r"(?:learning[\s_-]*rate|lr)[\s_-]*(\d+)\s*-?\s*e\s*-?\s*(\d+)", re.I)
INST_PATTERN = re.compile(r"results_(?:summary|per_run)_([A-Za-z]+)", re.I)
VERSION_PATTERN = re.compile(r"-(v\d+)", re.I)


def parse_lr(path: Path):
    for part in reversed(path.parts):
        m = LR_PATTERN.search(part)
        if m:
            mant, exp = m.groups()
            return float(f"{mant}e-{exp}"), f"{mant}e-{exp}"
    return None, None


def collect(root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(root.rglob("results_summary_*.csv")):
        if "cross_eval" in {p.lower() for p in csv_path.parts}:
            continue
        lr_value, lr_label = parse_lr(csv_path)
        if lr_value is None:
            print(f"  [skip] no LR in path: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        inst = INST_PATTERN.search(csv_path.stem)
        ver = VERSION_PATTERN.search(csv_path.stem)
        df["lr"] = lr_value
        df["lr_label"] = lr_label
        df["institution"] = inst.group(1).upper() if inst else "UNKNOWN"
        df["version"] = ver.group(1).lower() if ver else "none"
        frames.append(df)
    if not frames:
        raise SystemExit(f"No results_summary_*.csv found under {root}")
    return pd.concat(frames, ignore_index=True)


def score_table(df, metric, criterion):
    """model x lr -> joint score, plus per-institution columns."""
    wide = df.pivot_table(index=["model", "lr", "lr_label"],
                          columns="institution", values=metric, aggfunc="mean")
    insts = list(wide.columns)
    wide = wide.dropna(subset=insts)  # drop LRs missing an institution

    if criterion == "mean":
        wide["joint"] = wide[insts].mean(axis=1)
    elif criterion == "min":
        wide["joint"] = wide[insts].min(axis=1)
    elif criterion == "rank":
        ranks = wide.reset_index().groupby("model")[insts].rank(ascending=False)
        wide["joint"] = -ranks.sum(axis=1).values
    else:
        raise SystemExit(f"unknown criterion {criterion}")
    return wide.reset_index(), insts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--metric", default="test_f1_macro_mean")
    ap.add_argument("--version", default=None, help="Keep only this data version, e.g. v4")
    ap.add_argument("--criterion", default="mean", choices=["mean", "min", "rank"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    df = collect(root)
    std_col = args.metric.replace("_mean", "_std")
    versions = sorted(df["version"].unique())
    print(f"Searching {root}")
    print(f"{len(df)} rows | LRs {sorted(df.lr_label.unique())} | "
          f"institutions {sorted(df.institution.unique())} | versions {versions}")

    if args.version:
        df = df[df["version"] == args.version.lower()]
        print(f"Filtered to {args.version}: {len(df)} rows")
    elif len(versions) > 1:
        print(f"  !! Mixed data versions {versions} — LR is confounded with dataset. "
              f"Consider --version {versions[-1]}")

    tbl, insts = score_table(df, args.metric, args.criterion)
    if len(insts) < 2:
        raise SystemExit(f"Need two institutions to share an LR; found {insts}")

    # per-institution individual optima, for the cost-of-sharing calculation
    indiv = (df.groupby(["model", "institution"])[args.metric].max()
               .unstack("institution"))

    # dispersion per (model, lr): mean SD across institutions, for the noise check
    disp = (df.groupby(["model", "lr_label"])[std_col].mean()
              if std_col in df.columns else None)

    rows = []
    for model, g in tbl.groupby("model"):
        g = g.sort_values("joint", ascending=False).reset_index(drop=True)
        best, runner = g.loc[0], (g.loc[1] if len(g) > 1 else None)
        rec = {"model": model, "shared_lr": best["lr_label"]}
        for i in insts:
            rec[f"{i}_f1"] = best[i]
            rec[f"{i}_cost_vs_own_best"] = best[i] - indiv.loc[model, i]
        rec["joint"] = best["joint"]
        if runner is not None:
            margin = best["joint"] - runner["joint"]
            rec["runner_up_lr"] = runner["lr_label"]
            rec["margin"] = margin
            if disp is not None:
                pooled = ((disp.loc[model, best["lr_label"]] ** 2
                           + disp.loc[model, runner["lr_label"]] ** 2) / 2) ** 0.5
                rec["margin_in_pooled_sd"] = margin / pooled if pooled > 0 else float("nan")
                rec["separable"] = "yes" if margin > pooled else "within noise"
        rows.append(rec)

    shared = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    print(f"\n=== Joint {args.metric} by model x LR (criterion: {args.criterion}) ===")
    print(tbl.pivot_table(index="model", columns="lr_label", values="joint")
             .round(4).to_string())

    print(f"\n=== Shared LR per model (both institutions use the same LR) ===")
    print(shared.round(4).to_string(index=False))

    if "separable" in shared.columns:
        n = (shared["separable"] == "within noise").sum()
        if n:
            print(f"\n  {n}/{len(shared)} shared choices are within one pooled SD of the "
                  f"runner-up — those are ties, and the LR is effectively arbitrary.")

    # ---- single global LR across all models ----
    glob = tbl.groupby("lr_label")["joint"].mean().sort_values(ascending=False)
    print(f"\n=== Single global LR across all models (mean joint score) ===")
    print(glob.round(4).to_string())
    gbest = glob.index[0]
    print(f"\n  Global winner: {gbest}")
    cost = []
    for model, g in tbl.groupby("model"):
        at_g = g.loc[g.lr_label == gbest, "joint"]
        cost.append({"model": model,
                     "joint_at_global": at_g.iloc[0] if len(at_g) else float("nan"),
                     "joint_at_own_shared": g["joint"].max(),
                     "cost": (at_g.iloc[0] if len(at_g) else float("nan")) - g["joint"].max()})
    cost = pd.DataFrame(cost)
    print(f"  Cost of forcing {gbest} on every model:")
    print(cost.round(4).to_string(index=False))
    print(f"  Worst-case loss: {cost['cost'].min():.4f} | mean loss: {cost['cost'].mean():.4f}")

    shared.to_csv(outdir / "shared_lr_per_model.csv", index=False)
    tbl.to_csv(outdir / "shared_lr_joint_scores.csv", index=False)
    print(f"\nWrote:\n  {outdir/'shared_lr_per_model.csv'}\n  {outdir/'shared_lr_joint_scores.csv'}")


if __name__ == "__main__":
    main()
