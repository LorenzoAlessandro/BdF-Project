"""
Build all dissertation figures from wherever your run artifacts live.

1) Fill in the PATHS block below (absolute paths, keep the quotes).
   Leave a path as "" to skip that figure.
2) Run:  python make_plots.py
"""

from pathlib import Path
import glob
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# ======================= EDIT THIS BLOCK =====================================
TEST_JSONL     = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/test.jsonl"          # needed only to (re)generate predictions
LABEL_MAPPING  = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/label_mapping.json"  # "" -> defaults to dovish/hawkish/neutral
FT_STATES_GLOB = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/logs/*_trainer_state.json"   # fine-tune histories (glob ok)
DAPT_STATE     = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/logs/dapt_roberta_trainer_state.json"  # "" if you don't have it
RESULTS_CSV    = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/results/results_partial.csv"    
PREDS_DIR      = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/preds_folder"        # preds_<model>.csv live / get written here
FIG_DIR        = "figures"                           # output folder (created if missing)

REGENERATE_MISSING_PREDS = True     # pull models from HF if no preds found in PREDS_DIR
HF_USERNAME = "LorenzoAleCon29"
FINISHED_MODELS = ["roberta", "roberta-dapt", "roberta-large", "deberta-v3"]

DAPT_GRAD_ACCUM = 8   # DAPT logged training loss SUMMED over its 8 accumulation
                      # steps; dividing rescales it onto the eval-loss axis
# =============================================================================

FIG = Path(FIG_DIR)
FIG.mkdir(parents=True, exist_ok=True)
PREDS = Path(PREDS_DIR) if PREDS_DIR else Path(".")
PREDS.mkdir(parents=True, exist_ok=True)

# --------------------------- class names ------------------------------------
class_names = ["dovish", "hawkish", "neutral"]
if LABEL_MAPPING and Path(LABEL_MAPPING).exists():
    mapping = json.loads(Path(LABEL_MAPPING).read_text())
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    class_names = [id2label[i] for i in range(len(id2label))]
else:
    print("(no label_mapping.json found - using default class order)")
print("classes:", class_names)

# ------------------- optional: rebuild preds from HF ------------------------
def regenerate_preds():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not (TEST_JSONL and Path(TEST_JSONL).exists()):
        print(f"[skip] cannot regenerate: TEST_JSONL not found at {TEST_JSONL!r}")
        return
    test = pd.read_json(TEST_JSONL, lines=True)

    for name in FINISHED_MODELS:
        repo = f"{HF_USERNAME}/{name}-fomc-hawkish-dovish"
        print(f"pulling {repo} ...")
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForSequenceClassification.from_pretrained(repo).eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(test), 32):
                batch = tok(list(test.text[i:i + 32]), truncation=True,
                            max_length=256, padding=True, return_tensors="pt")
                preds.extend(mdl(**batch).logits.argmax(-1).tolist())

        out = PREDS / f"preds_{name}.csv"
        pd.DataFrame({"text": test.text, "y_true": test.label,
                      "y_pred": preds}).to_csv(out, index=False)
        print("  wrote", out)

# ----------------------- 1. confusion matrices ------------------------------
pred_files = sorted(glob.glob(str(PREDS / "preds_*.csv")))
if not pred_files and REGENERATE_MISSING_PREDS:
    regenerate_preds()
    pred_files = sorted(glob.glob(str(PREDS / "preds_*.csv")))

if pred_files:
    n = len(pred_files)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    report_lines = []
    for ax, f in zip(axes, pred_files):
        name = Path(f).stem.replace("preds_", "")
        d = pd.read_csv(f)
        cm = confusion_matrix(d.y_true, d.y_pred,
                              labels=range(len(class_names)))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name)
        rep = classification_report(d.y_true, d.y_pred,
                                    labels=range(len(class_names)),
                                    target_names=class_names, digits=3)
        report_lines.append(f"===== {name} =====\n{rep}\n")

    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrices.png", dpi=200)
    plt.close()
    (FIG / "per_class_reports.txt").write_text("\n".join(report_lines))
    print(f"[ok] confusion_matrices.png ({n} models) + per_class_reports.txt")
else:
    print("[skip] no preds_*.csv found in PREDS_DIR")

# ------------------- 2. fine-tune eval curves -------------------------------
dapt_name = Path(DAPT_STATE).name if DAPT_STATE else None
ft_files = ([f for f in sorted(glob.glob(FT_STATES_GLOB))
             if Path(f).name != dapt_name]
            if FT_STATES_GLOB else [])

if ft_files:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for f in ft_files:
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
    print(f"[ok] eval_loss.png + f1_macro.png ({len(ft_files)} models)")
else:
    print("[skip] no fine-tune trainer states matched FT_STATES_GLOB")

# ------------- 2b. per-model training vs validation loss --------------------
if ft_files:
    n = len(ft_files)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, f in zip(axes, ft_files):
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

        # standalone copy of the same plot for the write-up
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

# ------------------------- 3. DAPT curve ------------------------------------
if DAPT_STATE and Path(DAPT_STATE).exists():
    hist = json.loads(Path(DAPT_STATE).read_text())["log_history"]
    tr = pd.DataFrame([e for e in hist if "loss" in e and "eval_loss" not in e])
    ev = pd.DataFrame([e for e in hist if "eval_loss" in e])

    fig, ax = plt.subplots(figsize=(8, 5))
    if not tr.empty:
        ax.plot(tr.step, tr.loss / DAPT_GRAD_ACCUM, alpha=0.6,
                label="training loss")
    if not ev.empty:
        ax.plot(ev.step, ev.eval_loss, marker="o", label="validation loss")
        best = ev.loc[ev.eval_loss.idxmin()]
        ax.annotate(f"best: {best.eval_loss:.3f}\n(ppl {math.exp(best.eval_loss):.1f})",
                    xy=(best.step, best.eval_loss),
                    xytext=(best.step, best.eval_loss + 0.25),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("step")
    ax.set_ylabel("MLM loss")
    ax.set_title("DAPT on roberta-base (30M-token FOMC corpus)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "dapt_loss.png", dpi=200)
    plt.close(fig)
    print("[ok] dapt_loss.png")
else:
    print("[skip] DAPT_STATE not set / not found")

# -------------------- 4. model comparison chart -----------------------------
if RESULTS_CSV and Path(RESULTS_CSV).exists():
    r = pd.read_csv(RESULTS_CSV)
    if "model" in r.columns:
        r = r.groupby("model")[["test_f1_macro", "test_accuracy"]].mean()
    else:
        r = pd.read_csv(RESULTS_CSV, index_col=0)[["test_f1_macro", "test_accuracy"]]
    r = r.sort_values("test_f1_macro")

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(r) + 1.5))
    ax.barh(r.index, r.test_f1_macro)
    for i, v in enumerate(r.test_f1_macro):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center")
    ax.set_xlabel("test macro-F1")
    ax.set_xlim(0, min(1.0, r.test_f1_macro.max() + 0.08))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=200)
    plt.close(fig)
    print(f"[ok] model_comparison.png (from {RESULTS_CSV})")
else:
    print("[skip] RESULTS_CSV not set / not found")

print("\ndone ->", FIG.resolve())
