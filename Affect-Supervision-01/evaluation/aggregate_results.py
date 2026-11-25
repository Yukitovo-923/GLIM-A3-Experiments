"""Summarize the three paired seeds and UID-level matched EEG controls."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B0_Baseline", "B1_EPA")
SEEDS = (42, 43, 44)
CONDITIONS = ("matched", "mismatched")
GENERATIVE = ("bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1")


def mean_sd(values):
    values = list(values)
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": list(values)}


def metric_path(arm, seed):
    return ROOT / "evaluation" / "outputs" / f"{arm}_seed{seed}_test" / "metrics.json"


def rows_path(arm, seed, condition):
    return ROOT / "evaluation" / "outputs" / f"{arm}_seed{seed}_test" / f"{condition}_scored.jsonl"


def uid_means(arm, seed, condition, key):
    rows = [json.loads(line) for line in rows_path(arm, seed, condition).read_text(
        encoding="utf-8").splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row[key])
    if len(grouped) != 40:
        raise RuntimeError("UID bootstrap requires all 40 test sentences")
    return {uid: statistics.mean(values) for uid, values in grouped.items()}


def bootstrap_uid_differences(left, right, key):
    per_uid = defaultdict(list)
    for seed in SEEDS:
        a = uid_means(*left(seed), key)
        b = uid_means(*right(seed), key)
        if set(a) != set(b):
            raise RuntimeError("Paired UID sets differ")
        for uid in a:
            per_uid[uid].append(a[uid] - b[uid])
    uid_deltas = np.array([statistics.mean(per_uid[uid]) for uid in sorted(per_uid)])
    rng = np.random.default_rng(20261111)
    draws = rng.choice(uid_deltas, size=(5000, len(uid_deltas)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean_delta": float(uid_deltas.mean()), "uid_bootstrap_95ci": [float(lo), float(hi)],
            "n_uids": int(len(uid_deltas)), "resamples": 5000,
            "interpretation": "UID sampling uncertainty; only three fixed training seeds"}


def collect():
    records = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = metric_path(arm, seed)
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["arm"] != arm or item["seed"] != seed:
                raise RuntimeError(f"Metric identity mismatch: {path}")
            records[(arm, seed)] = item
    return records


def write_subject_csv(records):
    path = ROOT / "evaluation" / "subject_results.csv"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=[
            "arm", "seed", "condition", "subject", "n_trials", "n_uids",
            "bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1", "lm_loss",
            "v3_disc_macro_f1", "v3_disc_balanced_accuracy",
            "epa_macro_f1", "epa_balanced_accuracy",
        ])
        writer.writeheader()
        for (arm, seed), item in records.items():
            for condition in CONDITIONS:
                by_subject = item["conditions"][condition]["by_subject"]
                for subject, result in sorted(by_subject.items()):
                    generation = result["generation"]
                    disc = result["emotion"]["v3_discrete_head"]["sentence"]
                    epa = result["emotion"]["epa"]
                    writer.writerow({"arm": arm, "seed": seed, "condition": condition,
                                     "subject": subject,
                                     "n_trials": generation["n_trials"],
                                     "n_uids": generation["n_uids"],
                                     **generation["uid_balanced_mean"],
                                     "lm_loss": result["lm_loss"],
                                     "v3_disc_macro_f1": disc["macro_f1"],
                                     "v3_disc_balanced_accuracy": disc["balanced_accuracy"],
                                     "epa_macro_f1": epa["sentence"]["macro_f1"] if epa else "",
                                     "epa_balanced_accuracy": epa["sentence"]["balanced_accuracy"]
                                     if epa else ""})


def main():
    records = collect()
    write_subject_csv(records)
    result = {"seeds": list(SEEDS), "arms": list(ARMS), "metric_unit_primary":
              "mean of per-UID means across subject trials", "per_seed": {},
              "summary": {}, "paired_arm_delta_b1_minus_b0": {},
              "eeg_matched_minus_mismatched": {}, "uid_bootstrap": {}}
    for arm in ARMS:
        result["per_seed"][arm] = {}
        result["summary"][arm] = {}
        for seed in SEEDS:
            item = records[(arm, seed)]
            per_condition = {}
            for condition in CONDITIONS:
                c = item["conditions"][condition]
                per_condition[condition] = {
                    **c["generation"]["uid_balanced_mean"],
                    "lm_loss": c["lm_loss"],
                    "v3_disc_sentence_macro_f1":
                        c["emotion"]["v3_discrete_head"]["sentence"]["macro_f1"],
                    "v3_disc_sentence_balanced_accuracy":
                        c["emotion"]["v3_discrete_head"]["sentence"]["balanced_accuracy"],
                }
                if c["emotion"]["epa"]:
                    per_condition[condition]["epa_sentence_macro_f1"] = (
                        c["emotion"]["epa"]["sentence"]["macro_f1"])
                    per_condition[condition]["epa_sentence_balanced_accuracy"] = (
                        c["emotion"]["epa"]["sentence"]["balanced_accuracy"])
            result["per_seed"][arm][str(seed)] = per_condition
        for condition in CONDITIONS:
            keys = result["per_seed"][arm][str(SEEDS[0])][condition]
            result["summary"][arm][condition] = {
                key: mean_sd(result["per_seed"][arm][str(seed)][condition][key] for seed in SEEDS)
                for key in keys
            }
        result["eeg_matched_minus_mismatched"][arm] = {
            key: mean_sd(result["per_seed"][arm][str(seed)]["matched"][key] -
                         result["per_seed"][arm][str(seed)]["mismatched"][key]
                         for seed in SEEDS)
            for key in (*GENERATIVE, "lm_loss")
        }
    result["paired_arm_delta_b1_minus_b0"] = {
        key: mean_sd(result["per_seed"]["B1_EPA"][str(seed)]["matched"][key] -
                     result["per_seed"]["B0_Baseline"][str(seed)]["matched"][key]
                     for seed in SEEDS)
        for key in (*GENERATIVE, "lm_loss", "v3_disc_sentence_macro_f1",
                    "v3_disc_sentence_balanced_accuracy")
    }
    for key in ("bert_score_f1", "rouge_l", "bleu4"):
        result["uid_bootstrap"][f"b1_minus_b0_{key}"] = bootstrap_uid_differences(
            lambda seed: ("B1_EPA", seed, "matched"),
            lambda seed: ("B0_Baseline", seed, "matched"), key)
        for arm in ARMS:
            result["uid_bootstrap"][f"{arm}_matched_minus_mismatched_{key}"] = (
                bootstrap_uid_differences(
                    lambda seed, a=arm: (a, seed, "matched"),
                    lambda seed, a=arm: (a, seed, "mismatched"), key))
    path = ROOT / "evaluation" / "comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
