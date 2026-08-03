import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEST_JSONL = "/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/BdF-Project/Bert Model Finetuning Code /Finetuned Models/ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/test.jsonl"
OUT = ("/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Github Repos/Project/"
       "BdF-Project/Bert Model Finetuning Code /Finetuned Models/"
       "ROBERTA,ROBERTA-DAPT,ROBERTA-LARGE/preds_folder/preds_roberta-base.csv")

repo = "LorenzoAleCon29/roberta-base-fomc-hawkish-dovish"
test = pd.read_json(TEST_JSONL, lines=True)
tok = AutoTokenizer.from_pretrained(repo)
mdl = AutoModelForSequenceClassification.from_pretrained(repo).eval()

preds = []
with torch.no_grad():
    for i in range(0, len(test), 32):
        b = tok(list(test.text[i:i+32]), truncation=True, max_length=256,
                padding=True, return_tensors="pt")
        preds.extend(mdl(**b).logits.argmax(-1).tolist())

pd.DataFrame({"text": test.text, "y_true": test.label,
              "y_pred": preds}).to_csv(OUT, index=False)
print("wrote", OUT)