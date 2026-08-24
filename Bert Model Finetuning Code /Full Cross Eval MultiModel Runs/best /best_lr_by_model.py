#!/usr/bin/env python3
"""
Find the best-performing learning rate per model, ranked by macro F1.

Walks a directory tree like:

    Full Cross Eval MultiModel Runs/
        Learning Rate 1e-5/
            results/
                results_summary_ECB-v4.csv
                results_summary_FED-v4.csv
                results_per_run_ECB-v4.csv
                ...
        Learning Rate 2e-5/
        Learning Rate 3e-5/
        Learning Rate 5-e5/

and collects every results_summary_*.csv it finds, tagging each row with the
learning rate (parsed from the ancestor folder name), the training institution
(ECB / FED, from the filename), and the data version (v3 / v4, if present).

Usage:
    python best_lr_by_model.py "Full Cross Eval MultiModel Runs"
    python best_lr_by_model.py . --metric test_f1_macro_mean --version v4
    python best_lr_by_model.py . --pooled          # ignore institution split
    python best_lr_by_model.py . --per-run         # also use per-run files for a paired check
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# "Learning Rate 1e-5", "Learning Rate 5-e5", "LR 3e-5", "lr_2e-5" ...
LR_PATTERN = re.compile(r"(?:learning[\s_-]*rate|lr)[\s_-]*(\d+)\s*-?\s*e\s*-?\s*(\d+)", re.I)
INST_PATTERN = re.compile(r"results_(?:summary|per_run)_([A-Za-z]+)", re.I)
VERSION_PATTERN = re.compile(r"-(v\d+)", re.I)


def parse_lr(path: Path):
    """Pull the learning rate out of any ancestor folder name. Returns (float, label)."""
    for part in reversed(path.parts):
        m = LR_PATTERN.search(part)
        if m:
            mantissa, exponent = m.groups()
            value = float(f"{mantissa}e-{exponent}")
            return value, f"{mantissa}e-{exponent}"
    return None, None


def collect(root: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(root.rglob(pattern)):
        # skip cross-eval outputs — different schema, handled separately
        if "cross_eval" in {p.lower() for p in csv_path.parts}:
            continue

        lr_value, lr_label = parse_lr(csv_path)
        if lr_value is None:
            print(f"  [skip] no learning rate in path: {csv_path}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)
        inst = INST_PATTERN.search(csv_path.stem)
        ver = VERSION_PATTERN.search(csv_path.stem)

        df["lr"] = lr_value
        df["lr_label"] = lr_label
        df["institution"] = inst.group(1).upper() if inst else "UNKNOWN"
        df["version"] = ver.group(1).lower() if ver else "none"
        df["source_file"] = str(csv_path.relative_to(root))
        frames.append(df)

    if not frames:
        raise SystemExit(f"No files matching {pattern!r} found under {root}")
    return pd.concat(frames, ignore_index=True)


def pick_best(df: pd.DataFrame, metric: str, group_cols: list) -> pd.DataFrame:
    """For each group, return the LR row with the highest metric, plus a
    noise check against the runner-up."""
    std_col = metric.replace("_mean", "_std")
    rows = []

    for keys, g in df.groupby(group_cols, sort=True):
        g = g.sort_values(metric, ascending=False).reset_index(drop=True)
        best = g.loc[0]
        runner_up = g.loc[1] if len(g) > 1 else None

        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec["best_lr"] = best["lr_label"]
        rec[metric] = best[metric]
        if std_col in g.columns:
            rec[std_col] = best[std_col]
        rec["test_accuracy_mean"] = best.get("test_accuracy_mean")
        rec["test_loss_mean"] = best.get("test_loss_mean")

        if runner_up is not None:
            margin = best[metric] - runner_up[metric]
            rec["runner_up_lr"] = runner_up["lr_label"]
            rec["margin_over_runner_up"] = margin
            if std_col in g.columns:
                # pooled SD of the two configs being compared
                pooled = ((best[std_col] ** 2 + runner_up[std_col] ** 2) / 2) ** 0.5
                rec["margin_in_pooled_sd"] = margin / pooled if pooled > 0 else float("nan")
                rec["separable"] = "yes" if margin > pooled else "within noise"
        rec["n_lrs_compared"] = len(g)
        rec["source_file"] = best["source_file"]
        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="Directory to search (default: cwd)")
    ap.add_argument("--metric", default="test_f1_macro_mean", help="Column to rank on")
    ap.add_argument("--version", default=None, help="Keep only this data version, e.g. v4")
    ap.add_argument("--pooled", action="store_true", help="Best LR per model, ignoring ECB/FED split")
    ap.add_argument("--outdir", default=None, help="Where to write CSVs (default: root)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Searching {root}\n")
    df = collect(root, "results_summary_*.csv")

    versions = sorted(df["version"].unique())
    print(f"Found {len(df)} rows | LRs: {sorted(df.lr_label.unique())} | "
          f"institutions: {sorted(df.institution.unique())} | versions: {versions}")

    if args.version:
        df = df[df["version"] == args.version.lower()]
        print(f"Filtered to version {args.version}: {len(df)} rows")
    elif len(versions) > 1:
        print(f"\n  !! Multiple data versions present {versions} — these are different "
              f"datasets, not different seeds.\n     Comparing across them confounds LR with "
              f"data version. Re-run with --version {versions[-1]} to isolate.\n")

    if args.metric not in df.columns:
        raise SystemExit(f"Metric {args.metric!r} not in columns: {list(df.columns)}")

    group_cols = ["model"] if args.pooled else ["institution", "model"]
    if args.pooled:
        # average across institutions before picking
        agg = {args.metric: "mean", "test_accuracy_mean": "mean", "test_loss_mean": "mean",
               args.metric.replace("_mean", "_std"): "mean"}
        agg = {k: v for k, v in agg.items() if k in df.columns}
        df = (df.groupby(["model", "lr", "lr_label"], as_index=False)
                .agg({**agg, "source_file": "first"}))

    # ---- full landscape: model x LR ----
    idx = ["model"] if args.pooled else ["institution", "model"]
    pivot = df.pivot_table(index=idx, columns="lr_label", values=args.metric, aggfunc="mean")
    pivot = pivot[sorted(pivot.columns, key=lambda c: float(c.replace("e-", "e-")))]
    print(f"\n=== {args.metric} by model x learning rate ===")
    print(pivot.round(4).to_string())

    # ---- winners ----
    best = pick_best(df, args.metric, group_cols)
    best = best.sort_values(group_cols).reset_index(drop=True)
    print(f"\n=== Best learning rate per model (by {args.metric}) ===")
    show = [c for c in best.columns if c not in ("source_file",)]
    print(best[show].round(4).to_string(index=False))

    if "separable" in best.columns:
        noisy = (best["separable"] == "within noise").sum()
        if noisy:
            print(f"\n  Note: {noisy}/{len(best)} winners beat the runner-up by less than one "
                  f"pooled SD — treat those as ties, not wins.")

    pivot_path = outdir / "lr_sweep_matrix.csv"
    best_path = outdir / "best_lr_per_model.csv"
    all_path = outdir / "lr_sweep_all_rows.csv"
    pivot.to_csv(pivot_path)
    best.to_csv(best_path, index=False)
    df.to_csv(all_path, index=False)
    print(f"\nWrote:\n  {pivot_path}\n  {best_path}\n  {all_path}")


if __name__ == "__main__":
    main()
