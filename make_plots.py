"""
Build all dissertation figures. ZERO configuration:
- run from anywhere inside BdF-Project:  python make_plots.py
- test.jsonl, label_mapping.json and *_trainer_state.json are found
  automatically anywhere under the current folder
- predictions are generated from the Hugging Face models into results/
  for any model missing its preds_<name>.csv (the folder is created for you)
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, f1_score)

HF_USERNAME = "LorenzoAleCon29"
FINISHED_MODELS = [
    "roberta-base",
    "roberta-dapt",
    "roberta-large",
    "deberta-v3",
    "modernbert",
    "finbert",
    "finbert-tone",
    # "roberta-large-dapt",   # uncomment once that run is fine-tuned + pushed
]
# CAUTION: never add plain "roberta" — that HF repo is the STALE pre-fix run.

ROOT = Path(".")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
PREDS = ROOT / "results"
PREDS.mkdir(exist_ok=True)

# ---------------------- auto-discover input files ---------------------------
def find_one(filename, preferred):
    if preferred.exists():
        return preferred
    hits = sorted(ROOT.rglob(filename))
    return hits[0] if hits else None

TEST_JSONL = find_one("test.jsonl", ROOT / "data" / "test.jsonl")
LABEL_MAP  = find_one("label_mapping.json", ROOT / "data" / "label_mapping.json")

all_states = sorted({str(p) for p in ROOT.rglob("*_trainer_state.json")})
dapt_states = [p for p in all_states if Path(p).name.startswith("dapt_")]
ft_states   = [p for p in all_states if not Path(p).name.startswith("dapt_")]

print("test set:      ", TEST_JSONL)
print("label mapping: ", LABEL_MAP)
print("fine-tune logs:", len(ft_states), "| DAPT logs:", len(dapt_states))

# quarantine the stale pre-fix predictions if still around
stale = PREDS / "preds_roberta.csv"
if stale.exists():
    stale.rename(PREDS / "preds_roberta.csv.STALE")
    print("quarantined stale preds_roberta.csv (old pre-fix model)")

class_names = ["dovish", "hawkish", "neutral"]
if LABEL_MAP:
    m = json.loads(Path(LABEL_MAP).read_text())
    id2label = {int(k): v for k, v in m["id2label"].items()}
    class_names = [id2label[i] for i in range(len(id2label))]
print("classes:", class_names)

# ------------------- generate missing preds from HF -------------------------
missing = [n for n in FINISHED_MODELS
           if not (PREDS / f"preds_{n}.csv").exists()]
if missing and TEST_JSONL is None:
    print(f"[warn] cannot generate preds for {missing}: no test.jsonl found")
elif missing:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    test = pd.read_json(TEST_JSONL, lines=True)
    print(f"generating predictions for {missing} ({len(test)} test sentences)")

    for name in missing:
        repo = f"{HF_USERNAME}/{name}-fomc-hawkish-dovish"
        print(f"pulling {repo} ...")
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForSequenceClassification.from_pretrained(repo).eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(test), 32):
                b = tok(list(test.text[i:i + 32]), truncation=True,
                        max_length=256, padding=True, return_tensors="pt")
                preds.extend(mdl(**b).logits.argmax(-1).tolist())

        out = PREDS / f"preds_{name}.csv"
        pd.DataFrame({"text": test.text, "y_true": test.label,
                      "y_pred": preds}).to_csv(out, index=False)
        print("  wrote", out)

pred_files = sorted(PREDS.glob("preds_*.csv"))

# ----------------------- 1. confusion matrices ------------------------------
if pred_files:
    n = len(pred_files)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    report_lines = []
    for ax, f in zip(axes, pred_files):
        name = f.stem.replace("preds_", "")
        d = pd.read_csv(f)
        cm = confusion_matrix(d.y_true, d.y_pred,
                              labels=range(len(class_names)))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name)
        report_lines.append(
            f"===== {name} =====\n"
            + classification_report(d.y_true, d.y_pred,
                                    labels=range(len(class_names)),
                                    target_names=class_names, digits=3)
            + "\n")

    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrices.png", dpi=200)
    plt.close()
    (FIG / "per_class_reports.txt").write_text("\n".join(report_lines))
    print(f"[ok] confusion_matrices.png ({n} models) + per_class_reports.txt")
else:
    print("[skip] no predictions available")

# ------------------- 2. fine-tune eval curves -------------------------------
if ft_states:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for f in ft_states:
        name = Path(f).name.replace("_trainer_state.json", "")
        hist = json.loads(Path(f).read_text())["log_history"]
        ev = pd.DataFrame([e for e in hist if "eval_loss" in e])
        if ev.empty:
            continue
        ax1.plot(ev.epoch, ev.eval_loss, marker="o", label=name)
        if "eval_f1_macro" in ev:
            ax2.plot(ev.epoch, ev.eval_f1_macro, marker="o", label=name)

    for ax, ylab, fname, fg in [(ax1, "validation loss", "eval_loss.png", fig1),
                                 (ax2, "validation macro-F1", "f1_macro.png", fig2)]:
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend()
        fg.tight_layout()
        fg.savefig(FIG / fname, dpi=200)
        plt.close(fg)
    print(f"[ok] eval_loss.png + f1_macro.png ({len(ft_states)} models)")
else:
    print("[skip] no fine-tune trainer states found")

# ------------- 2b. per-model training vs validation loss --------------------
if ft_states:
    n = len(ft_states)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, f in zip(axes, ft_states):
        name = Path(f).name.replace("_trainer_state.json", "")
        hist = json.loads(Path(f).read_text())["log_history"]
        tr = pd.DataFrame([e for e in hist if "loss" in e and "eval_loss" not in e])
        ev = pd.DataFrame([e for e in hist if "eval_loss" in e])

        def draw(a):
            if not tr.empty:
                a.plot(tr.epoch, tr.loss, alpha=0.7, label="training loss")
            if not ev.empty:
                a.plot(ev.epoch, ev.eval_loss, marker="o",
                       label="validation loss")
            a.set_title(name)
            a.set_xlabel("epoch")
            a.set_ylabel("loss")
            a.grid(alpha=0.3)
            a.legend()

        draw(ax)
        sfig, sax = plt.subplots(figsize=(7, 4.5))
        draw(sax)
        sfig.tight_layout()
        sfig.savefig(FIG / f"loss_{name}.png", dpi=200)
        plt.close(sfig)

    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "loss_per_model.png", dpi=200)
    plt.close(fig)
    print(f"[ok] loss_per_model.png + {n} individual loss_<model>.png files")

# ------------------------- 3. DAPT curves -----------------------------------
for f in dapt_states:
    p = Path(f)
    name = p.name.replace("_trainer_state.json", "")      # e.g. dapt_roberta
    accum = 16 if "large" in name else 8                  # each run's grad accum
    hist = json.loads(p.read_text())["log_history"]
    tr = pd.DataFrame([e for e in hist if "loss" in e and "eval_loss" not in e])
    ev = pd.DataFrame([e for e in hist if "eval_loss" in e])

    fig, ax = plt.subplots(figsize=(8, 5))
    if not tr.empty:
        ax.plot(tr.step, tr.loss / accum, alpha=0.6, label="training loss")
    if not ev.empty:
        ax.plot(ev.step, ev.eval_loss, marker="o", label="validation loss")
        best = ev.loc[ev.eval_loss.idxmin()]
        ax.annotate(f"best: {best.eval_loss:.3f}\n(ppl {math.exp(best.eval_loss):.1f})",
                    xy=(best.step, best.eval_loss),
                    xytext=(best.step, best.eval_loss + 0.25),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("step")
    ax.set_ylabel("MLM loss")
    ax.set_title(name.replace("_", " ") + " (30M-token FOMC corpus)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"[ok] {name}.png")
if not dapt_states:
    print("[skip] no dapt_*_trainer_state.json found")

# -------------------- 4. model comparison chart -----------------------------
if pred_files:
    rng = np.random.default_rng(42)

    def bootstrap_ci(y_true, y_pred, n_boot=2000):
        idx = np.arange(len(y_true))
        scores = [f1_score(y_true[s], y_pred[s], average="macro")
                  for s in (rng.choice(idx, size=len(idx), replace=True)
                            for _ in range(n_boot))]
        return np.percentile(scores, [2.5, 97.5])

    rows = []
    for f in pred_files:
        name = f.stem.replace("preds_", "")
        d = pd.read_csv(f)
        yt, yp = d.y_true.to_numpy(), d.y_pred.to_numpy()
        lo, hi = bootstrap_ci(yt, yp)
        rows.append({"model": name,
                     "f1": f1_score(yt, yp, average="macro"),
                     "lo": lo, "hi": hi})
    r = pd.DataFrame(rows).sort_values("f1").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(r) + 1.5))
    ax.barh(r.model, r.f1, xerr=[r.f1 - r.lo, r.hi - r.f1], capsize=4)
    for i, row in r.iterrows():
        ax.text(row.hi + 0.008, i, f"{row.f1:.3f}", va="center")
    ax.set_xlabel("test macro-F1 (95% bootstrap CI)")
    ax.set_xlim(0, min(1.0, r.hi.max() + 0.09))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=200)
    plt.close(fig)
    print("[ok] model_comparison.png (recomputed from per-sentence predictions)")
else:
    print("[skip] comparison chart needs predictions")

print("\ndone ->", FIG.resolve())