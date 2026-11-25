"""Aggregate 12 validation runs by arm, optimizer condition, seed and UID."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B0_Baseline", "B1_EPA")
OPTIMIZERS = ("reset", "restored")
SEEDS = (42, 43, 44)
GENERATION = ("bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1")
METRICS = ("lm_loss", *GENERATION, "v3_discrete_macro_f1", "epa_macro_f1")


def extract(condition):
    return {"lm_loss": condition["lm_loss"],
            **condition["generation"]["uid_balanced_mean"],
            "v3_discrete_macro_f1": condition["emotion"]["v3_discrete_head"]["sentence"]["macro_f1"],
            "epa_macro_f1": (condition["emotion"]["epa"]["sentence"]["macro_f1"]
                             if condition["emotion"]["epa"] else None)}


def mean_sd(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values)}


def main():
    rows, subjects, epochs = [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            for optimizer in OPTIMIZERS:
                output = ROOT / "evaluation" / "outputs" / f"{optimizer}_{arm}_seed{seed}_val"
                scored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                inferred = json.loads((output / "inference_summary.json").read_text(encoding="utf-8"))
                if (scored["split"] != "val" or inferred["split"] != "val" or
                        scored["optimizer_condition"] != optimizer or inferred["condition"] != optimizer):
                    raise RuntimeError("Validation or optimizer metadata mismatch")
                matched = extract(scored["conditions"]["matched"])
                mismatched = extract(scored["conditions"]["mismatched"])
                rows.append({"arm": arm, "optimizer": optimizer, "seed": seed,
                             "best_epoch_zero_based": inferred["checkpoint_epoch"],
                             "checkpoint_sha256": inferred["best_checkpoint_sha256"],
                             "matched": matched, "mismatched": mismatched})
                for subject, stats in scored["conditions"]["matched"]["by_subject"].items():
                    subjects.append({"arm": arm, "optimizer": optimizer, "seed": seed,
                                     "subject": subject, **extract(stats)})
                run = ROOT / "runs" / optimizer / arm / f"seed-{seed}"
                logs = sorted((run / "metrics").glob("epoch_*.json"))
                if not logs:
                    raise RuntimeError(f"No epoch metrics in {run}")
                for path in logs:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    metrics = record["metrics"]
                    epochs.append({"arm": arm, "optimizer": optimizer, "seed": seed,
                                   "epoch": record["epoch"], "learning_rate": record["learning_rate"],
                                   "epoch_elapsed_seconds": record["epoch_elapsed_seconds"],
                                   "train_lm": metrics.get("train/loss_lm"),
                                   "val_lm": metrics.get("val/loss_lm"),
                                   "train_original_emotion_ce": metrics.get("train/loss_disc_emotion"),
                                   "val_original_emotion_ce": metrics.get("val/loss_disc_emotion"),
                                   "train_epa_ce": metrics.get("train/loss_epa"),
                                   "val_epa_ce": metrics.get("val/loss_epa")})
    if len(rows) != 12:
        raise RuntimeError("Expected 12 completed validation runs")
    by_key = {(r["optimizer"], r["arm"], r["seed"]): r for r in rows}
    cell = {}
    optimizer_effect = {}
    epa_effect = {}
    eeg_effect = {}
    for optimizer in OPTIMIZERS:
        cell[optimizer] = {}
        epa_effect[optimizer] = {}
        eeg_effect[optimizer] = {}
        for arm in ARMS:
            selected = [by_key[(optimizer, arm, seed)] for seed in SEEDS]
            cell[optimizer][arm] = {metric: mean_sd([r["matched"][metric] for r in selected])
                                    for metric in METRICS
                                    if all(r["matched"][metric] is not None for r in selected)}
            eeg_effect[optimizer][arm] = {
                metric: {"per_seed": [r["matched"][metric] - r["mismatched"][metric]
                                      for r in selected],
                         **mean_sd([r["matched"][metric] - r["mismatched"][metric]
                                    for r in selected])}
                for metric in ("lm_loss", *GENERATION)}
        for metric in METRICS:
            if metric == "epa_macro_f1":
                continue
            differences = [by_key[(optimizer, "B1_EPA", seed)]["matched"][metric]
                           - by_key[(optimizer, "B0_Baseline", seed)]["matched"][metric]
                           for seed in SEEDS]
            epa_effect[optimizer][metric] = {"per_seed": differences, **mean_sd(differences)}
    for arm in ARMS:
        optimizer_effect[arm] = {}
        for metric in METRICS:
            if arm == "B0_Baseline" and metric == "epa_macro_f1":
                continue
            differences = [by_key[("restored", arm, seed)]["matched"][metric]
                           - by_key[("reset", arm, seed)]["matched"][metric]
                           for seed in SEEDS]
            optimizer_effect[arm][metric] = {"per_seed": differences, **mean_sd(differences)}
    result = {"experiment": "V3 Adam reset versus restoration, B0/B1",
              "split": "val", "primary_unit": "UID-balanced mean over 40 validation UIDs",
              "runs": rows, "cell_summary": cell,
              "paired_restored_minus_reset": optimizer_effect,
              "paired_b1_minus_b0": epa_effect,
              "matched_minus_mismatched_eeg": eeg_effect}
    (ROOT / "evaluation" / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    for filename, data in (("epoch_losses.csv", epochs), ("subject_matched.csv", subjects)):
        with (ROOT / "evaluation" / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    print(json.dumps({"cell_summary": cell,
                      "paired_restored_minus_reset": optimizer_effect,
                      "paired_b1_minus_b0": epa_effect}, indent=2), flush=True)


if __name__ == "__main__":
    main()
