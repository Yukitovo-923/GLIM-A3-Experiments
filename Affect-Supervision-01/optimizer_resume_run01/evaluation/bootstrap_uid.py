"""Paired UID bootstrap conditional on the three observed seeds."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ARMS = ("B0_Baseline", "B1_EPA")
OPTIMIZERS = ("reset", "restored")
METRICS = ("bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1")


def uid_scores(optimizer, arm, seed, eeg_condition):
    path = ROOT / "evaluation" / "outputs" / f"{optimizer}_{arm}_seed{seed}_val" / f"{eeg_condition}_scored.jsonl"
    by_uid = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_uid[int(row["uid"])].append(row)
    if len(by_uid) != 40:
        raise RuntimeError("Expected 40 validation UIDs")
    return {uid: {metric: float(np.mean([row[metric] for row in group]))
                  for metric in METRICS} for uid, group in by_uid.items()}


def summarize(delta, rng, n_boot=10000):
    picks = rng.integers(0, delta.shape[1], size=(n_boot, delta.shape[1]))
    boots = np.mean(delta[:, picks], axis=(0, 2))
    return {"mean": float(delta.mean()),
            "ci95_uid_bootstrap_conditional_on_three_seeds":
            [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "per_seed": delta.mean(axis=1).tolist()}


def main():
    scored = {(optimizer, arm, seed, eeg): uid_scores(optimizer, arm, seed, eeg)
              for optimizer in OPTIMIZERS for arm in ARMS for seed in SEEDS
              for eeg in ("matched", "mismatched")}
    uids = sorted(scored[("reset", "B0_Baseline", 42, "matched")])
    if any(sorted(value) != uids for value in scored.values()):
        raise RuntimeError("UID sets differ between paired conditions")
    rng = np.random.default_rng(20261117)
    result = {"split": "val", "uids": 40, "seeds": list(SEEDS),
              "bootstrap_replicates": 10000,
              "resampling_unit": "same sentence UID across optimizer, arm, EEG condition and seed",
              "confidence_scope": "conditional on these three seeds; does not include seed uncertainty",
              "restored_minus_reset": {arm: {} for arm in ARMS},
              "b1_minus_b0": {optimizer: {} for optimizer in OPTIMIZERS},
              "matched_minus_mismatched": {optimizer: {arm: {} for arm in ARMS}
                                            for optimizer in OPTIMIZERS}}
    for metric in METRICS:
        for arm in ARMS:
            delta = np.array([[scored[("restored", arm, seed, "matched")][uid][metric]
                               - scored[("reset", arm, seed, "matched")][uid][metric]
                               for uid in uids] for seed in SEEDS])
            result["restored_minus_reset"][arm][metric] = summarize(delta, rng)
        for optimizer in OPTIMIZERS:
            delta = np.array([[scored[(optimizer, "B1_EPA", seed, "matched")][uid][metric]
                               - scored[(optimizer, "B0_Baseline", seed, "matched")][uid][metric]
                               for uid in uids] for seed in SEEDS])
            result["b1_minus_b0"][optimizer][metric] = summarize(delta, rng)
            for arm in ARMS:
                delta = np.array([[scored[(optimizer, arm, seed, "matched")][uid][metric]
                                   - scored[(optimizer, arm, seed, "mismatched")][uid][metric]
                                   for uid in uids] for seed in SEEDS])
                result["matched_minus_mismatched"][optimizer][arm][metric] = summarize(delta, rng)
    (ROOT / "evaluation" / "uid_bootstrap.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"restored_minus_reset": {arm: {
        metric: result["restored_minus_reset"][arm][metric]
        for metric in ("rouge_l", "bert_score_f1")} for arm in ARMS},
                      "b1_minus_b0": {optimizer: {
        metric: result["b1_minus_b0"][optimizer][metric]
        for metric in ("rouge_l", "bert_score_f1")} for optimizer in OPTIMIZERS}}, indent=2))


if __name__ == "__main__":
    main()
