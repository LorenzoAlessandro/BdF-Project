"""
Regenerate the v5 per-sentence predictions locally.

Downloads each of the 70 fine-tuned runs from the Hub one at a time, evaluates it
on BOTH test sets, writes per-sentence predictions, deletes the weights, moves on.
Disk never holds more than one model (~1.5 GB peak).

Evaluation path is copied from stage 3 of the Kaggle notebook (same tokenisation
max_length=256, same collator, same compute_metrics, same Trainer.predict), so the
numbers reproduce.

Restart-safe: finished (train_inst, model, run, eval_inst) tuples are skipped.
Ctrl-C and rerun whenever you like.

    python predict_v5.py --repo-dir ../BdF-Project --out out_v5
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login, snapshot_download, HfApi
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

# ----------------------------------------------------------------------------- config
HF_USERNAME = "LorenzoAleCon29"
REPO_SUFFIX = "Multi_Model_Full_Pipeline_Run"
EXPERIMENT = "v5"
BRANCH_OF = {"FED": "FED-v5", "ECB": "ECB-v5"}
INSTITUTIONS = ["FED", "ECB"]
MODEL_NAMES = ["roberta-base", "roberta-dapt", "roberta-large", "roberta-large-dapt",
               "deberta-v3", "finbert", "finbert-tone"]
N_RUNS = 5
BASE_SEED = 42
METRICS = ["test_accuracy", "test_f1_macro", "test_f1_weighted", "test_loss"]
EXPECTED_LABEL2ID = {"dovish": 0, "hawkish": 1, "neutral": 2}
EXPECTED_TEST_SIZES = {"FED": 407, "ECB": 405}   # from the notebook's own output

api = HfApi()


def repo_for(name):
    return f"{HF_USERNAME}/{name}-{REPO_SUFFIX}"


# ----------------------------------------------------------------------------- data
def load_test_sets(repo_dir: Path, frozen: Path = None):
    """Preferred source: cross_eval_test_sets.csv.gz, the frozen v5 archive written by
    cell 20 of the notebook. It is tagged by experiment, so it cannot drift.

    Fallback: data/<INST>/test.jsonl on the repo's current HEAD. Fine for FED (fixed
    2021-01-01 cutoff, random_state=42) but NOT guaranteed for ECB, whose cutoff is
    a data-dependent quantile(0.80) and is overwritten by every experiment's push.
    """
    if frozen is None:
        default = repo_dir / "results" / "cross_eval" / f"FED-{EXPERIMENT}_x_ECB-{EXPERIMENT}" \
                  / "cross_eval_test_sets.csv.gz"
        frozen = default if default.exists() else None

    if frozen is not None:
        df = pd.read_csv(frozen)
        sets = {inst: Dataset.from_list(
            g.sort_values("row_idx")[["text", "label_id"]]
             .rename(columns={"label_id": "label"}).to_dict("records"))
            for inst, g in df.groupby("eval_institution")}
        print(f"test sets from frozen archive {frozen}")
    else:
        print("WARNING: no frozen cross_eval_test_sets.csv.gz found for "
              f"{EXPERIMENT}; falling back to data/<INST>/test.jsonl at repo HEAD.\n"
              "         The ECB split may belong to a different experiment — watch the "
              "|dacc| check below.")
        sets = {}
        for inst in INSTITUTIONS:
            p = repo_dir / "data" / inst / "test.jsonl"
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Clone https://github.com/LorenzoAlessandro/BdF-Project "
                    f"and pass its path with --repo-dir.")
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            sets[inst] = Dataset.from_list(rows)

    for inst in INSTITUTIONS:
        if inst not in sets:
            raise SystemExit(f"no test set found for {inst}")
        if len(sets[inst]) != EXPECTED_TEST_SIZES[inst]:
            print(f"  WARNING: {inst} test set has {len(sets[inst])} rows, "
                  f"expected {EXPECTED_TEST_SIZES[inst]}")
    print("test sets:", {k: len(v) for k, v in sets.items()})
    return sets


# ----------------------------------------------------------------------------- eval
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted")}


def run_done_at(repo_id, branch, run):
    try:
        files = {Path(e.path).name: getattr(e, "size", None)
                 for e in api.list_repo_tree(repo_id, revision=branch,
                                             path_in_repo=f"run-{run}", recursive=True)}
    except Exception:
        return False
    has_weights = any(n.endswith(".safetensors") or n == "pytorch_model.bin" for n in files)
    return has_weights and "test_metrics.json" in files


def download_run(repo_id, branch, run, tmp_dir: Path):
    shutil.rmtree(tmp_dir, ignore_errors=True)
    snapshot_download(repo_id, revision=branch, allow_patterns=[f"run-{run}/*"],
                      local_dir=str(tmp_dir), cache_dir=str(tmp_dir / ".hf_cache"))
    return tmp_dir / f"run-{run}"


def evaluate_run(model_dir, eval_sets, eval_args):
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)

    got = {k: int(v) for k, v in model.config.label2id.items()}
    assert got == EXPECTED_LABEL2ID, f"label mapping mismatch in {model_dir}: {got}"

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    trainer = Trainer(model=model, args=eval_args, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    out = {}
    for inst, ds in eval_sets.items():
        pred = trainer.predict(ds.map(tok, batched=True), metric_key_prefix="test")
        out[inst] = {"metrics": {k: float(v) for k, v in pred.metrics.items()},
                     "y_true": [int(v) for v in pred.label_ids],
                     "y_pred": [int(v) for v in np.argmax(pred.predictions, axis=-1)]}
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="../BdF-Project",
                    help="path to your cloned BdF-Project GitHub repo (for the test sets)")
    ap.add_argument("--out", default="out_v5", help="output directory")
    ap.add_argument("--token", default=None,
                    help="HF token; omit if you've already run `huggingface-cli login`")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names (default: all 7)")
    ap.add_argument("--institutions", nargs="*", default=None,
                    help="subset of train institutions (default: FED ECB)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--test-sets", default=None,
                    help="explicit path to a cross_eval_test_sets.csv.gz "
                         "(default: auto-find the v5 one under --repo-dir)")
    args = ap.parse_args()

    if args.token:
        login(token=args.token)

    out_dir = Path(args.out)
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_model"
    per_run_csv = out_dir / "cross_eval_per_run.csv"

    test_sets = load_test_sets(Path(args.repo_dir),
                               Path(args.test_sets) if args.test_sets else None)

    # archive the test sets so predictions stay interpretable (same file the notebook writes)
    pd.concat([pd.DataFrame({"eval_institution": inst, "row_idx": range(len(ds)),
                             "text": ds["text"], "label_id": ds["label"]})
               for inst, ds in test_sets.items()], ignore_index=True).to_csv(
        out_dir / "cross_eval_test_sets.csv.gz", index=False, compression="gzip")

    eval_args = TrainingArguments(output_dir=str(out_dir / "tmp_eval"),
                                  per_device_eval_batch_size=args.batch_size,
                                  fp16=torch.cuda.is_available(), report_to="none")

    model_names = args.models or MODEL_NAMES
    institutions = args.institutions or INSTITUTIONS

    rows = pd.read_csv(per_run_csv).to_dict("records") if per_run_csv.exists() else []
    done = {(r["train_institution"], r["model"], int(r["run"]), r["eval_institution"])
            for r in rows}
    missing_on_hub = []

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} | "
          f"{len(done)} evaluations already recorded")

    for train_inst in institutions:
        branch = BRANCH_OF[train_inst]
        for name in model_names:
            repo_id = repo_for(name)
            for run in range(1, N_RUNS + 1):
                seed = BASE_SEED + (run - 1)
                todo = [ev for ev in INSTITUTIONS
                        if (train_inst, name, run, ev) not in done]
                if not todo:
                    print(f"skip {train_inst}/{name} run {run} (already done)")
                    continue
                if not run_done_at(repo_id, branch, run):
                    print(f"!! {repo_id}@{branch}/run-{run} not on the Hub - skipped")
                    missing_on_hub.append((train_inst, name, run))
                    continue

                print(f"\n===== {train_inst} ({branch}) | {name} | run {run}/{N_RUNS} "
                      f"| seed {seed} | eval on {todo} =====")
                model_dir = download_run(repo_id, branch, run, tmp_dir)
                res = evaluate_run(model_dir, {ev: test_sets[ev] for ev in todo}, eval_args)

                # consistency check against the metrics stored at training time
                tm_path = Path(model_dir) / "test_metrics.json"
                if train_inst in res and tm_path.exists():
                    stored = json.loads(tm_path.read_text())
                    if "test_accuracy" in stored:
                        d = abs(float(stored["test_accuracy"])
                                - res[train_inst]["metrics"]["test_accuracy"])
                        flag = "  <-- CHECK THIS" if d > 0.01 else ""
                        print(f"    in-domain |dacc| vs training time = {d:.4f}{flag}")

                for ev, r in res.items():
                    rows.append({"train_institution": train_inst, "train_branch": branch,
                                 "model": name, "run": run, "seed": seed,
                                 "eval_institution": ev, "in_domain": ev == train_inst,
                                 "n_test": len(r["y_true"]),
                                 **{k: r["metrics"].get(k) for k in METRICS}})
                    pd.DataFrame({"train_institution": train_inst, "train_branch": branch,
                                  "model": name, "run": run, "seed": seed,
                                  "eval_institution": ev,
                                  "row_idx": range(len(r["y_true"])),
                                  "y_true": r["y_true"], "y_pred": r["y_pred"]}).to_csv(
                        preds_dir / f"{train_inst}_{name}_run{run}_on_{ev}.csv", index=False)
                    done.add((train_inst, name, run, ev))

                pd.DataFrame(rows).to_csv(per_run_csv, index=False)
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # bundle every prediction into one file, same name/shape as the notebook's
    pred_files = sorted(preds_dir.glob("*.csv"))
    if pred_files:
        allp = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
        bundle = out_dir / "cross_eval_predictions.csv.gz"
        allp.to_csv(bundle, index=False, compression="gzip")
        print(f"\npredictions: {allp.shape} -> {bundle}")

    expected = len(institutions) * len(model_names) * N_RUNS * len(INSTITUTIONS)
    print(f"recorded {len(rows)} / {expected} evaluations in {per_run_csv}")
    if missing_on_hub:
        print(f"{len(missing_on_hub)} runs not on the Hub:")
        for t in missing_on_hub:
            print("   ", t)


if __name__ == "__main__":
    main()
