"""Summarize and plot read-only EEG-side EPA train/validation diagnostics."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("negative", "neutral", "positive")


def main():
    records = []
    for seed in (42, 43, 44):
        source = ROOT / "tables" / f"eeg_epa_geometry_seed{seed}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        for split in ("train", "val"):
            for row in data[split]["records"]:
                records.append({"seed": seed, "split": split, **row})

    summary = []
    with (ROOT / "tables" / "eeg_epa_trial_scores.csv").open("w", encoding="utf-8", newline="") as target:
        cols = ["seed", "split", "uid", "subject", "true_class", "predicted_class",
                "logit_negative", "logit_neutral", "logit_positive",
                "prob_negative", "prob_neutral", "prob_positive", "true_margin",
                "cos_before_true", "cos_after_true", "projection_angle_degrees"]
        writer = csv.DictWriter(target, fieldnames=cols)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "seed": row["seed"], "split": row["split"], "uid": row["uid"],
                "subject": row["subject"], "true_class": CLASSES[row["label"]],
                "predicted_class": CLASSES[row["predicted"]],
                **{f"logit_{c}": row["logits"][i] for i, c in enumerate(CLASSES)},
                **{f"prob_{c}": row["probs"][i] for i, c in enumerate(CLASSES)},
                "true_margin": row["true_margin"],
                "cos_before_true": row["cos_before"][row["label"]],
                "cos_after_true": row["cos_after"][row["label"]],
                "projection_angle_degrees": row["projection_angle_degrees"],
            })

    for seed in (42, 43, 44):
        for split in ("train", "val"):
            for label, cls in enumerate(CLASSES):
                rows = [r for r in records if r["seed"] == seed and r["split"] == split and r["label"] == label]
                margins = np.array([r["true_margin"] for r in rows])
                angles = np.array([r["projection_angle_degrees"] for r in rows])
                predicted = [r["predicted"] for r in rows]
                summary.append({"seed": seed, "split": split, "true_class": cls,
                                "trial_count": len(rows), "mean_true_margin": float(margins.mean()),
                                "median_true_margin": float(np.median(margins)),
                                "true_margin_q05": float(np.quantile(margins, .05)),
                                "true_margin_q95": float(np.quantile(margins, .95)),
                                "mean_true_probability": float(np.mean([r["probs"][label] for r in rows])),
                                "mean_projection_angle_degrees": float(angles.mean()),
                                **{f"predicted_{c}_share": predicted.count(i) / len(rows)
                                   for i, c in enumerate(CLASSES)}})
    with (ROOT / "tables" / "eeg_epa_class_summary.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for split, color in (("train", "#23649b"), ("val", "#cb6547")):
        values = [r["true_margin"] for r in records if r["split"] == split]
        axes[0].hist(values, bins=35, alpha=.55, density=True, color=color, label=split)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set(xlabel="True-class logit minus strongest other class", ylabel="Density", title="EEG-side EPA margins")
    axes[0].legend()
    x = np.arange(3)
    width = .22
    for j, seed in enumerate((42, 43, 44)):
        rows = [r for r in records if r["seed"] == seed and r["split"] == "val"]
        frac = np.bincount([r["predicted"] for r in rows], minlength=3) / len(rows)
        axes[1].bar(x + (j - 1) * width, frac, width, label=f"seed {seed}")
    axes[1].set(xticks=x, xticklabels=CLASSES, ylim=(0, 1.05), ylabel="Predicted trial share", title="Validation prediction collapse")
    axes[1].legend()
    fig.savefig(ROOT / "figures" / "eeg_epa_margins_and_collapse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for split, color in (("train", "#23649b"), ("val", "#cb6547")):
        values = [r["projection_angle_degrees"] for r in records if r["split"] == split]
        axes[0].hist(values, bins=35, alpha=.55, density=True, color=color, label=split)
    axes[0].set(xlabel="Angle before/after EPA projection (degrees)", ylabel="Density", title="Projection changes EEG direction")
    axes[0].legend()
    for i, cls in enumerate(CLASSES):
        vals = [r["probs"][i] for r in records if r["split"] == "val" and r["label"] == i]
        axes[1].hist(vals, bins=30, alpha=.5, density=True, label=cls)
    axes[1].set(xlabel="Probability assigned to true class", ylabel="Density", title="Validation true-class confidence")
    axes[1].legend()
    fig.savefig(ROOT / "figures" / "eeg_epa_projection_and_confidence.png", dpi=180)
    plt.close(fig)
    full_path = ROOT / "tables" / "eeg_epa_geometry_seed42_full_train.json"
    if full_path.exists():
        full = json.loads(full_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
        for split, color in (("train", "#23649b"), ("val", "#cb6547")):
            vals = [r["true_margin"] for r in full[split]["records"]]
            ax.hist(vals, bins=48, density=True, alpha=.5, color=color,
                    label=f"{split} ({len(vals)} trials)")
        ax.axvline(0, color="black", lw=1)
        ax.set(xlabel="True-class EPA logit margin", ylabel="Density",
               title="Seed 42: all train trials versus validation trials")
        ax.legend()
        fig.savefig(ROOT / "figures" / "eeg_epa_full_train_vs_val.png", dpi=180)
        plt.close(fig)
        with (ROOT / "tables" / "eeg_epa_full_train_seed42_scores.csv").open(
                "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["split", "uid", "subject", "true_class",
                                                        "predicted_class", "true_margin", "true_probability"])
            writer.writeheader()
            for split in ("train", "val"):
                for row in full[split]["records"]:
                    writer.writerow({"split": split, "uid": row["uid"], "subject": row["subject"],
                                     "true_class": CLASSES[row["label"]],
                                     "predicted_class": CLASSES[row["predicted"]],
                                     "true_margin": row["true_margin"],
                                     "true_probability": row["probs"][row["label"]]})
    print(json.dumps({"rows": len(records), "class_rows": len(summary)}))


if __name__ == "__main__":
    main()
