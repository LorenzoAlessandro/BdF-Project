"""
Regenerate per-sentence test predictions from the Hugging Face models.

- Only models whose preds_<name>.csv is MISSING are run (FORCE=True redoes all).
- Finds test.jsonl automatically anywhere under the current folder.

Run from the BdF-Project root:   python regen_preds.py
"""

from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_USERNAME = "LorenzoAleCon29"
MODELS = [
    "roberta-base",
    "roberta-dapt",
    "roberta-large",
    "deberta-v3",
    "modernbert",
    "finbert",
    "finbert-tone",
    # "roberta-large-dapt",   # uncomment once that run is fine-tuned + pushed
]
# CAUTION: never add plain "roberta" — that repo is the STALE pre-fix run.

PREDS_DIR = Path("/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/"
                 "BdF-Project/Bert Model Finetuning Code /Finetuned Models/"
                 "ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/preds_folder")
FORCE = False

hits = list(Path(".").rglob("test.jsonl"))
assert hits, "no test.jsonl found anywhere under this folder"
TEST_JSONL = hits[0]
test = pd.read_json(TEST_JSONL, lines=True)
print(f"test set: {TEST_JSONL} ({len(test)} sentences)")   # expect 407

PREDS_DIR.mkdir(parents=True, exist_ok=True)

for name in MODELS:
    out = PREDS_DIR / f"preds_{name}.csv"
    if out.exists() and not FORCE:
        print("skip (exists):", out.name)
        continue

    repo = f"{HF_USERNAME}/{name}-fomc-hawkish-dovish"
    print(f"pulling {repo} ...")
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(repo).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 32):
            b = tok(list(test.text[i:i + 32]), truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            preds.extend(mdl(**b).logits.argmax(-1).tolist())

    pd.DataFrame({"text": test.text, "y_true": test.label,
                  "y_pred": preds}).to_csv(out, index=False)
    print("wrote", out.name)

print("done")
