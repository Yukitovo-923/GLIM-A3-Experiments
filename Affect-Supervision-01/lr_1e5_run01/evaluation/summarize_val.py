"""Aggregate new validation metrics and compare with the prior LR=1e-4 run."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "failure_analysis_v2" / "validation_checkpoints" / "comparison.json"
ARMS = ("B0_Baseline", "B1_EPA")
SEEDS = (42, 43, 44)
GENERATION = ("bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1")
METRICS = ("lm_loss", *GENERATION, "v3_discrete_macro_f1", "epa_macro_f1")


def extract(condition):
    row = {"lm_loss": condition["lm_loss"],
           **condition["generation"]["uid_balanced_mean"],
           "v3_discrete_macro_f1": condition["emotion"]["v3_discrete_head"]["sentence"]["macro_f1"],
           "epa_macro_f1": (condition["emotion"]["epa"]["sentence"]["macro_f1"]
                            if condition["emotion"]["epa"] else None)}
    return row


def mean_sd(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values)}


def main():
    old = json.loads(PRIOR.read_text(encoding="utf-8"))
    if old["split"] != "val":
        raise RuntimeError("Prior reference is not validation")
    old_by_key = {(row["arm"], row["seed"]): row for row in old["rows"]}
    rows = []
    subject_rows = []
    epoch_rows = []
    for seed in SEEDS:
        for arm in ARMS:
            output = ROOT / "evaluation" / "outputs" / f"{arm}_seed{seed}_val"
            scored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            inferred = json.loads((output / "inference_summary.json").read_text(encoding="utf-8"))
            if scored["split"] != "val" or inferred["split"] != "val":
                raise RuntimeError("Wrong split")
            matched = extract(scored["conditions"]["matched"])
            mismatched = extract(scored["conditions"]["mismatched"])
            rows.append({"arm": arm, "seed": seed, "best_epoch_zero_based": inferred["checkpoint_epoch"],
                         "checkpoint_sha256": inferred["best_checkpoint_sha256"],
                         "matched": matched, "mismatched": mismatched})
            for subject, stats in scored["conditions"]["matched"]["by_subject"].items():
                subject_rows.append({"arm": arm, "seed": seed, "subject": subject,
                                     **extract(stats)})
            for path in sorted((ROOT / "runs" / arm / f"seed-{seed}" / "metrics").glob("epoch_*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                metrics = record["metrics"]
                epoch_rows.append({"arm": arm, "seed": seed, "epoch": record["epoch"],
                                   "learning_rate": record["learning_rate"],
                                   "epoch_elapsed_seconds": record["epoch_elapsed_seconds"],
                                   "train_lm": metrics.get("train/loss_lm"),
                                   "val_lm": metrics.get("val/loss_lm"),
                                   "train_original_emotion_ce": metrics.get("train/loss_disc_emotion"),
                                   "val_original_emotion_ce": metrics.get("val/loss_disc_emotion"),
                                   "train_epa_ce": metrics.get("train/loss_epa"),
                                   "val_epa_ce": metrics.get("val/loss_epa")})
    if len(rows) != 6:
        raise RuntimeError("Expected six runs")
    summary = {arm: {metric: mean_sd([row["matched"][metric] for row in rows
                                     if row["arm"] == arm]) for metric in METRICS
                     if all(row["matched"][metric] is not None for row in rows if row["arm"] == arm)}
               for arm in ARMS}
    paired_b1_b0 = {}
    for metric in METRICS:
        if metric == "epa_macro_f1":
            continue
        values = [next(r for r in rows if r["arm"] == "B1_EPA" and r["seed"] == seed)["matched"][metric]
                  - next(r for r in rows if r["arm"] == "B0_Baseline" and r["seed"] == seed)["matched"][metric]
                  for seed in SEEDS]
        paired_b1_b0[metric] = {"seed_42_43_44": values, **mean_sd(values)}
    low_vs_prior = {}
    for arm in ARMS:
        low_vs_prior[arm] = {}
        for metric in METRICS:
            if metric == "epa_macro_f1" and arm == "B0_Baseline":
                continue
            values = [next(r for r in rows if r["arm"] == arm and r["seed"] == seed)["matched"][metric]
                      - old_by_key[(arm, seed)][metric] for seed in SEEDS]
            low_vs_prior[arm][metric] = {"seed_42_43_44": values, **mean_sd(values)}
    eeg_effect = {}
    for arm in ARMS:
        eeg_effect[arm] = {}
        for metric in ("lm_loss", *GENERATION):
            values = [r["matched"][metric] - r["mismatched"][metric]
                      for r in rows if r["arm"] == arm]
            eeg_effect[arm][metric] = {"seed_42_43_44": values, **mean_sd(values)}
    result = {"experiment": "LR 1e-5 continuation", "split": "val",
              "primary_unit": "UID-balanced mean over 40 validation UIDs",
              "source_epoch46": old_by_key[("source", None)], "runs": rows,
              "arm_summary": summary, "paired_b1_minus_b0": paired_b1_b0,
              "new_lr_minus_old_lr": low_vs_prior,
              "matched_minus_mismatched_eeg": eeg_effect}
    (ROOT / "evaluation" / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "evaluation" / "epoch_losses.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(epoch_rows[0]))
        writer.writeheader(); writer.writerows(epoch_rows)
    with (ROOT / "evaluation" / "subject_matched.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(subject_rows[0]))
        writer.writeheader(); writer.writerows(subject_rows)
    print(json.dumps({"best_epochs": [(r["arm"], r["seed"], r["best_epoch_zero_based"]) for r in rows],
                      "arm_summary": summary, "paired_b1_minus_b0": paired_b1_b0}, indent=2))


if __name__ == "__main__":
    main()
