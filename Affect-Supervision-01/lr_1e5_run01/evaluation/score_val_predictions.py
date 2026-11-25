"""Score all low-LR validation predictions with the original pinned metric protocol."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from bert_score import BERTScorer
from torchmetrics.functional.text import bleu_score, rouge_score


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B0_Baseline", "B1_EPA")
SEEDS = (42, 43, 44)
CONDITIONS = ("matched", "mismatched")
CLASS_ORDER = ("negative", "neutral", "positive")


def classification(labels, probabilities):
    cm = np.zeros((3, 3), dtype=int)
    for true, probs in zip(labels, probabilities):
        cm[int(true), int(np.argmax(probs))] += 1
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3),
                   where=(support + predicted) > 0)
    return {"n": int(cm.sum()), "class_order": list(CLASS_ORDER),
            "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "confusion_matrix": cm.tolist(),
            "support": support.tolist()}


def sentiment_metrics(rows, key):
    selected = [row for row in rows if row[key] is not None]
    if not selected:
        return None
    trial = classification([row["label_id"] for row in selected],
                           [row[key] for row in selected])
    by_uid = defaultdict(list)
    for row in selected:
        by_uid[row["uid"]].append(row)
    sentence_labels = []
    sentence_probs = []
    for uid, group in sorted(by_uid.items()):
        labels = {row["label_id"] for row in group}
        if len(labels) != 1:
            raise RuntimeError(f"Conflicting sentiment labels for UID {uid}")
        sentence_labels.append(next(iter(labels)))
        sentence_probs.append(np.mean([row[key] for row in group], axis=0))
    return {"trial": trial, "sentence": classification(sentence_labels, sentence_probs)}


def scalar(value):
    return float(value.detach().cpu())


def lexical_metrics(rows):
    for row in rows:
        pred, ref = row["prediction"], row["reference"]
        row["bleu1"] = scalar(bleu_score([pred], [[ref]], n_gram=1))
        row["bleu2"] = scalar(bleu_score([pred], [[ref]], n_gram=2))
        row["bleu4"] = scalar(bleu_score([pred], [[ref]], n_gram=4))
        row["rouge_l"] = scalar(rouge_score([pred], [ref], rouge_keys="rougeL")["rougeL_fmeasure"])


def generation_metrics(rows):
    keys = ("bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1")
    uid_groups = defaultdict(list)
    for row in rows:
        uid_groups[row["uid"]].append(row)
    result = {"n_trials": len(rows), "n_uids": len(uid_groups)}
    result["trial_mean"] = {key: statistics.mean(row[key] for row in rows) for key in keys}
    result["uid_balanced_mean"] = {
        key: statistics.mean(statistics.mean(row[key] for row in group)
                             for group in uid_groups.values())
        for key in keys
    }
    preds = [row["prediction"] for row in rows]
    refs = [[row["reference"]] for row in rows]
    result["corpus_bleu"] = {f"bleu{n}": scalar(bleu_score(preds, refs, n_gram=n))
                             for n in (1, 2, 4)}
    return result


def one_condition(rows, inference):
    result = {"generation": generation_metrics(rows),
              "lm_loss": inference["lm_loss"],
              "emotion": {"v3_discrete_head": sentiment_metrics(rows, "disc_probs"),
                          "epa": sentiment_metrics(rows, "epa_probs")},
              "by_subject": {}}
    subjects = sorted({row["subject"] for row in rows})
    for subject in subjects:
        subset = [row for row in rows if row["subject"] == subject]
        result["by_subject"][subject] = {
            "generation": generation_metrics(subset),
            "lm_loss": inference["by_subject_lm"][subject]["loss"],
            "emotion": {"v3_discrete_head": sentiment_metrics(subset, "disc_probs"),
                        "epa": sentiment_metrics(subset, "epa_probs")},
        }
    return result


def main():
    model_record = json.loads((ROOT / "evaluation" / "bertscore_model.json").read_text(encoding="utf-8"))
    if model_record["model"] != "roberta-large":
        raise RuntimeError("Unexpected BERTScore backbone")
    data = []
    for seed in SEEDS:
        for arm in ARMS:
            output = ROOT / "evaluation" / "outputs" / f"{arm}_seed{seed}_val"
            inference = json.loads((output / "inference_summary.json").read_text(encoding="utf-8"))
            if inference["seed"] != seed or inference["arm"] != arm or inference["split"] != "val":
                raise RuntimeError(f"Inference metadata mismatch in {output}")
            for condition in CONDITIONS:
                path = output / f"{condition}_predictions.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                if len(rows) != 406 or len({row["uid"] for row in rows}) != 40:
                    raise RuntimeError(f"Incomplete validation predictions: {path}")
                data.append((output, arm, seed, condition, rows, inference["conditions"][condition]))
    scorer = BERTScorer(model_type=model_record["path"], num_layers=17,
                        batch_size=16, device="cuda", idf=False,
                        rescale_with_baseline=False, use_fast_tokenizer=False)
    all_rows = [row for _, _, _, _, rows, _ in data for row in rows]
    _, _, scores = scorer.score([row["prediction"] for row in all_rows],
                                [row["reference"] for row in all_rows],
                                batch_size=16, verbose=True)
    if len(scores) != len(all_rows):
        raise RuntimeError("BERTScore length mismatch")
    for row, score in zip(all_rows, scores.detach().float().cpu().tolist()):
        row["bert_score_f1"] = float(score)
    record = {"model": model_record, "num_layers": 17,
              "idf": False, "rescale_with_baseline": False,
              "bert_score_package_hash": scorer.hash,
              "reference": "raw ZuCo1 Task1 sentence",
              "lexical_metric_library": "torchmetrics 1.3.1",
              "metric_unit_primary": "mean of per-UID means across subject trials"}
    (ROOT / "evaluation" / "metric_protocol.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    for output, arm, seed, condition, rows, inference in data:
        lexical_metrics(rows)
        with (output / f"{condition}_scored.jsonl").open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        metric_path = output / "metrics.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            metrics = {"arm": arm, "seed": seed, "split": "val", "conditions": {}}
        metrics["conditions"][condition] = one_condition(rows, inference)
        metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SCORED {arm} seed={seed} {condition}", flush=True)


if __name__ == "__main__":
    main()
