#!/usr/bin/env python3
"""
Consolidated sweep analysis — corrected.

Fixes two bugs in the earlier scripts:
  1. rglob walked into "Trail of Different Data Split", whose LR also parses as
     3e-5, so pivot_table(aggfunc="mean") silently averaged v3 and v5 into one
     cell. Nested trees are now excluded by default, and (lr, version) is
     treated as a single condition identity that can never be merged.
  2. Selection ignored the fact that each LR sits on its own dataset version.
     Conditions are now labelled "3e-5/v3" etc. so the confound is visible.

Outputs three views:
  A. Full condition table          model x condition x institution
  B. Strongest performer per model per institution (unconstrained pick)
  C. Strongest shared condition per model (both institutions, one condition)
  D. Strongest single global condition across all models

Usage:
    python lr_analysis.py "Full Cross Eval MultiModel Runs"
    python lr_analysis.py . --metric test_accuracy_mean
    python lr_analysis.py . --include-nested        # analyse the split trial too
    python lr_analysis.py . --exclude "Mixed Highest Performers" "Eval Stop Loss"
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

LR_PATTERN = re.compile(r"(?:learning[\s_-]*rate|lr)[\s_-]*(\d+)\s*-?\s*e\s*-?\s*(\d+)", re.I)
INST_PATTERN = re.compile(r"results_(?:summary|per_run)_([A-Za-z]+)", re.I)
VERSION_PATTERN = re.compile(r"-(v\d+)", re.I)

DEFAULT_EXCLUDE = ["Trail of Different Data Split", "Mixed Highest Performers", "cross_eval"]


def parse_lr(path: Path):
    """Nearest ancestor folder naming a learning rate wins."""
    for part in reversed(path.parts):
        m = LR_PATTERN.search(part)
        if m:
            mant, exp = m.groups()
            return float(f"{mant}e-{exp}"), f"{mant}e-{exp}"
    return None, None


def collect(root: Path, exclude, kind="summary") -> pd.DataFrame:
    frames = []
    for csv_path in sorted(root.rglob(f"results_{kind}_*.csv")):
        rel = str(csv_path.relative_to(root))
        if any(x.lower() in rel.lower() for x in exclude):
            continue
        lr_value, lr_label = parse_lr(csv_path)
        if lr_value is None:
            print(f"  [skip] no LR in path: {rel}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        inst = INST_PATTERN.search(csv_path.stem)
        ver = VERSION_PATTERN.search(csv_path.stem)
        df["lr"] = lr_value
        df["lr_label"] = lr_label
        df["version"] = ver.group(1).lower() if ver else "none"
        # condition = the thing that actually varies between folders
        df["condition"] = df["lr_label"] + "/" + df["version"]
        df["institution"] = inst.group(1).upper() if inst else "UNKNOWN"
        df["source"] = rel
        frames.append(df)
    if not frames:
        raise SystemExit(f"No results_{kind}_*.csv found under {root} "
                         f"(excluding {exclude})")
    return pd.concat(frames, ignore_index=True)


def integrity_check(df):
    """Catch the exact bug that produced the averaged 3e-5 cell."""
    problems = []
    dupes = df.groupby(["condition", "institution", "model"]).size()
    dupes = dupes[dupes > 1]
    if len(dupes):
        problems.append(f"{len(dupes)} (condition, institution, model) cells appear "
                        f"more than once — they would be silently averaged:\n"
                        f"{dupes.to_string()}")
    per_lr_versions = df.groupby("lr_label")["version"].nunique()
    if (per_lr_versions > 1).any():
        bad = per_lr_versions[per_lr_versions > 1]
        problems.append(f"LR(s) {list(bad.index)} span multiple dataset versions — "
                        f"kept separate as distinct conditions.")
    lr_per_version = df.groupby("version")["lr_label"].nunique()
    if (lr_per_version == 1).all() and df["version"].nunique() > 1:
        problems.append("Each dataset version appears at exactly one learning rate: "
                        "LR and version are perfectly confounded. Differences between "
                        "conditions CANNOT be attributed to the learning rate.")
    return problems


def pick(tbl, group_cols, score_col, sd_lookup=None):
    """Argmax within each group, with a runner-up noise check."""
    rows = []
    for keys, g in tbl.groupby(group_cols, sort=True):
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        best = g.loc[0]
        runner = g.loc[1] if len(g) > 1 else None
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec["best_condition"] = best["condition"]
        rec[score_col] = best[score_col]
        if runner is not None:
            margin = best[score_col] - runner[score_col]
            rec["runner_up"] = runner["condition"]
            rec["margin"] = margin
            if sd_lookup is not None:
                try:
                    s1 = sd_lookup.loc[(best["model"], best["condition"])]
                    s2 = sd_lookup.loc[(runner["model"], runner["condition"])]
                    pooled = ((s1 ** 2 + s2 ** 2) / 2) ** 0.5
                    rec["margin_in_sd"] = margin / pooled if pooled > 0 else float("nan")
                    rec["verdict"] = "separable" if margin > pooled else "tie (within noise)"
                except KeyError:
                    pass
        rec["n_conditions"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--metric", default="test_f1_macro_mean")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help=f"Path substrings to skip (default: {DEFAULT_EXCLUDE})")
    ap.add_argument("--include-nested", action="store_true",
                    help="Do not exclude the nested split-trial folders")
    ap.add_argument("--criterion", default="mean", choices=["mean", "min", "rank"],
                    help="How to combine institutions for the shared pick")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    exclude = args.exclude if args.exclude is not None else list(DEFAULT_EXCLUDE)
    if args.include_nested:
        exclude = [x for x in exclude if x == "cross_eval"]

    metric = args.metric
    std_col = metric.replace("_mean", "_std")

    df = collect(root, exclude)
    print(f"Searching {root}")
    print(f"Excluding: {exclude}")
    print(f"{len(df)} rows | conditions {sorted(df.condition.unique())} | "
          f"institutions {sorted(df.institution.unique())}")
    print(f"Files read:")
    for s in sorted(df["source"].unique()):
        print(f"    {s}")

    problems = integrity_check(df)
    if problems:
        print("\n!! INTEGRITY WARNINGS")
        for p in problems:
            print(f"  - {p}")

    if metric not in df.columns:
        raise SystemExit(f"{metric!r} not found. Columns: {list(df.columns)}")

    sd_lookup = (df.set_index(["model", "condition"])[std_col]
                 if std_col in df.columns else None)
    if sd_lookup is not None:
        sd_lookup = sd_lookup[~sd_lookup.index.duplicated()]

    # ---------- A. full condition table ----------
    print(f"\n=== A. {metric} by model x condition ===")
    full = df.pivot_table(index="model", columns=["institution", "condition"],
                          values=metric, aggfunc="mean")
    print(full.round(4).to_string())

    # ---------- B. strongest per model per institution ----------
    long = df[["model", "institution", "condition", metric]].copy()
    best_each = pick(long, ["institution", "model"], metric, sd_lookup)
    print(f"\n=== B. Strongest performer per model, per institution ===")
    print(best_each.round(4).to_string(index=False))

    # ---------- C. strongest shared condition ----------
    wide = df.pivot_table(index=["model", "condition"], columns="institution",
                          values=metric, aggfunc="mean")
    insts = list(wide.columns)
    wide = wide.dropna(subset=insts).reset_index()
    if len(insts) >= 2:
        if args.criterion == "mean":
            wide["joint"] = wide[insts].mean(axis=1)
        elif args.criterion == "min":
            wide["joint"] = wide[insts].min(axis=1)
        else:
            r = wide.groupby("model")[insts].rank(ascending=False)
            wide["joint"] = -r.sum(axis=1)

        shared = pick(wide, ["model"], "joint", sd_lookup)
        for i in insts:
            shared[f"{i}_at_shared"] = [
                wide[(wide.model == r.model) & (wide.condition == r.best_condition)][i].iloc[0]
                for r in shared.itertuples()]
            own_best = wide.groupby("model")[i].max()
            shared[f"{i}_cost"] = shared[f"{i}_at_shared"] - shared["model"].map(own_best)
        print(f"\n=== C. Strongest SHARED condition per model (criterion: {args.criterion}) ===")
        print(shared.round(4).to_string(index=False))

        if "verdict" in shared.columns:
            n = (shared["verdict"] == "tie (within noise)").sum()
            if n:
                print(f"\n  {n}/{len(shared)} picks are ties — the condition is arbitrary "
                      f"for those models.")

        # ---------- D. single global condition ----------
        glob = wide.groupby("condition")["joint"].mean().sort_values(ascending=False)
        print(f"\n=== D. Single global condition across all models ===")
        print(glob.round(4).to_string())
        gbest = glob.index[0]
        cost = (wide[wide.condition == gbest].set_index("model")["joint"]
                - wide.groupby("model")["joint"].max())
        print(f"\n  Global winner: {gbest}")
        print(f"  Cost of forcing it on every model: "
              f"worst {cost.min():+.4f}, mean {cost.mean():+.4f}")
        shared.to_csv(outdir / "strongest_shared_condition.csv", index=False)

    best_each.to_csv(outdir / "strongest_per_model.csv", index=False)
    full.to_csv(outdir / "condition_matrix.csv")
    df.to_csv(outdir / "all_rows.csv", index=False)
    print(f"\nWrote to {outdir}:\n  strongest_per_model.csv\n  condition_matrix.csv\n"
          f"  all_rows.csv")


if __name__ == "__main__":
    main()
