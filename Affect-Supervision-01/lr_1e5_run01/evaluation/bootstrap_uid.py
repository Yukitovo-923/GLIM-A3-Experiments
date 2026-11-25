"""Conditional paired-UID bootstrap for validation generation differences."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ARMS = ("B0_Baseline", "B1_EPA")
METRICS = ("bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1")


def uid_scores(arm, seed, condition):
    path = ROOT / "evaluation" / "outputs" / f"{arm}_seed{seed}_val" / f"{condition}_scored.jsonl"
    by_uid = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_uid[int(row["uid"])].append(row)
    if len(by_uid) != 40:
        raise RuntimeError("Expected 40 validation UIDs")
    return {uid: {metric: float(np.mean([row[metric] for row in group]))
                  for metric in METRICS} for uid, group in by_uid.items()}


def summarize(delta, rng, n_boot=10000):
    # delta: seeds x same 40 UIDs. Resampling shared UIDs keeps B0/B1 and seeds paired.
    picks = rng.integers(0, delta.shape[1], size=(n_boot, delta.shape[1]))
    boots = np.mean(delta[:, picks], axis=(0, 2))
    return {"mean": float(delta.mean()),
            "ci95_uid_bootstrap_conditional_on_three_seeds":
            [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "per_seed": delta.mean(axis=1).tolist()}


def main():
    scored = {(arm, seed, condition): uid_scores(arm, seed, condition)
              for arm in ARMS for seed in SEEDS for condition in ("matched", "mismatched")}
    uids = sorted(scored[(ARMS[0], SEEDS[0], "matched")])
    if any(sorted(value) != uids for value in scored.values()):
        raise RuntimeError("UID sets do not match across paired conditions")
    rng = np.random.default_rng(20261115)
    result = {"split": "val", "uids": len(uids), "seeds": list(SEEDS),
              "bootstrap_replicates": 10000,
              "resampling_unit": "sentence UID, shared across arms/conditions/seeds",
              "confidence_scope": "conditional on these three seeds; seed uncertainty is not included",
              "b1_minus_b0_matched": {}, "matched_minus_mismatched": {arm: {} for arm in ARMS}}
    for metric in METRICS:
        arr = np.array([[scored[("B1_EPA", seed, "matched")][uid][metric]
                         - scored[("B0_Baseline", seed, "matched")][uid][metric]
                         for uid in uids] for seed in SEEDS])
        result["b1_minus_b0_matched"][metric] = summarize(arr, rng)
        for arm in ARMS:
            arr = np.array([[scored[(arm, seed, "matched")][uid][metric]
                             - scored[(arm, seed, "mismatched")][uid][metric]
                             for uid in uids] for seed in SEEDS])
            result["matched_minus_mismatched"][arm][metric] = summarize(arr, rng)
    output = ROOT / "evaluation" / "uid_bootstrap.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({metric: result["b1_minus_b0_matched"][metric]
                      for metric in ("rouge_l", "bert_score_f1")}, indent=2))


if __name__ == "__main__":
    main()
