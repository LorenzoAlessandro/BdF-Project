#!/usr/bin/env python3
"""
Identity comparison: v3 vs v5, at full float precision.

Rounded tables can make two numbers look equal when they differ in the 6th
decimal, and can hide that two numbers are bit-identical — which is the
signature of a result that was copied rather than re-run. This compares every
(institution, model, metric) cell at full precision and classifies it:

    IDENTICAL   exact float equality  -> result was NOT regenerated
    ~equal      differs below 1e-9    -> same computation, float noise
    differs     genuinely different

Runs on results_summary_*.csv and, if present, results_per_run_*.csv, where
seed-level identity is the decisive test.

Usage:
    python identity_check.py "Learning Rate 3e-5/results" \
                             "Trail of Different Data Split/Learning rate - 3e-5/results" \
                             --labels v3 v5
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

INST_PATTERN = re.compile(r"results_(?:summary|per_run)_([A-Za-z]+)", re.I)
METRICS = ["test_f1_macro_mean", "test_accuracy_mean", "test_f1_weighted_mean",
           "test_loss_mean", "test_f1_macro", "test_accuracy", "test_loss"]


def load(d: Path, kind: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(d / f"results_{kind}_*.csv"))):
        if "cross_eval" in f.lower():
            continue
        df = pd.read_csv(f)
        m = INST_PATTERN.search(Path(f).stem)
        df["institution"] = m.group(1).upper() if m else "UNKNOWN"
        df["_file"] = Path(f).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def classify(x, y, tol=1e-9):
    if pd.isna(x) or pd.isna(y):
        return "missing"
    if x == y:
        return "IDENTICAL"
    if abs(x - y) < tol:
        return "~equal"
    return "differs"


def compare(a, b, la, lb, keys, metrics, title):
    metrics = [m for m in metrics if m in a.columns and m in b.columns]
    if not metrics:
        print(f"  (no shared metric columns for {title})")
        return None
    merged = a.merge(b, on=keys, suffixes=(f"_{la}", f"_{lb}"))
    if merged.empty:
        print(f"  (no overlapping rows for {title} on keys {keys})")
        return None

    out = merged[keys].copy()
    for m in metrics:
        xa, xb = merged[f"{m}_{la}"], merged[f"{m}_{lb}"]
        out[f"{m}_{la}"] = xa
        out[f"{m}_{lb}"] = xb
        out[f"{m}_delta"] = xb - xa
        out[f"{m}_status"] = [classify(p, q) for p, q in zip(xa, xb)]

    print(f"\n=== {title} ===")
    primary = metrics[0]
    cols = keys + [f"{primary}_{la}", f"{primary}_{lb}",
                   f"{primary}_delta", f"{primary}_status"]
    with pd.option_context("display.float_format", lambda v: f"{v:.8f}"):
        print(out[cols].to_string(index=False))

    counts = out[f"{primary}_status"].value_counts()
    print(f"\n  {dict(counts)}")
    ident = out[out[f"{primary}_status"].isin(["IDENTICAL", "~equal"])]
    if len(ident):
        who = ident[keys].astype(str).agg(" / ".join, axis=1).tolist()
        print(f"  !! Not regenerated: {', '.join(who)}")
        print(f"     Bit-identical results cannot come from a re-run. Either these "
              f"configs were skipped (results-exist guard) or the files were copied.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--labels", nargs=2, default=["v3", "v5"])
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    la, lb = args.labels
    da, db = Path(args.dir_a).expanduser(), Path(args.dir_b).expanduser()

    print(f"A = {da}   (label {la})")
    print(f"B = {db}   (label {lb})")

    sa, sb = load(da, "summary"), load(db, "summary")
    if sa.empty or sb.empty:
        raise SystemExit("Missing results_summary_*.csv in one or both directories.")
    summ = compare(sa, sb, la, lb, ["institution", "model"], METRICS,
                   "Summary means, full precision")

    ra, rb = load(da, "per_run"), load(db, "per_run")
    runs = None
    if not ra.empty and not rb.empty and "seed" in ra.columns and "seed" in rb.columns:
        runs = compare(ra, rb, la, lb, ["institution", "model", "seed"], METRICS,
                       "Per-seed values, full precision")
        if runs is not None:
            pm = [c for c in runs.columns if c.endswith("_status")][0]
            per_model = (runs.assign(ident=runs[pm].isin(["IDENTICAL", "~equal"]))
                             .groupby(["institution", "model"])["ident"]
                             .agg(["sum", "count"]))
            per_model.columns = ["identical_seeds", "total_seeds"]
            print(f"\n=== Identical seeds per model ===")
            print(per_model.to_string())
            full = per_model[per_model.identical_seeds == per_model.total_seeds]
            if len(full):
                print(f"\n  Fully unchanged (every seed identical): "
                      f"{', '.join('/'.join(map(str, i)) for i in full.index)}")
    else:
        print("\n  (no per-run files with a seed column — summary-level only)")

    out = Path(args.outdir)
    if summ is not None:
        summ.to_csv(out / f"identity_summary_{la}_vs_{lb}.csv", index=False)
    if runs is not None:
        runs.to_csv(out / f"identity_perseed_{la}_vs_{lb}.csv", index=False)
    print(f"\nWrote to {out}")


if __name__ == "__main__":
    main()
