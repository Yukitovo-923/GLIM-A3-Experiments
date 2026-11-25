"""Real-data diagnostic figures for the completed 12-run validation study."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ("B0_Baseline", "B1_EPA")
OPTIMIZERS = ("reset", "restored")
SEEDS = (42, 43, 44)
COLORS = {42: "#246494", 43: "#ce6c48", 44: "#6d934c"}


def main():
    rows = list(csv.DictReader((ROOT / "evaluation" / "epoch_losses.csv").open(encoding="utf-8")))
    comparison = json.loads((ROOT / "evaluation" / "comparison.json").read_text(encoding="utf-8"))
    by_key = {(r["optimizer"], r["arm"], r["seed"]): r for r in comparison["runs"]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True, layout="constrained")
    for i, arm in enumerate(ARMS):
        for j, optimizer in enumerate(OPTIMIZERS):
            ax = axes[i, j]
            for seed in SEEDS:
                selected = sorted((r for r in rows if r["arm"] == arm and
                                   r["optimizer"] == optimizer and int(r["seed"]) == seed),
                                  key=lambda r: int(r["epoch"]))
                ax.plot([int(r["epoch"]) for r in selected],
                        [float(r["val_lm"]) for r in selected],
                        marker="o", markersize=3, color=COLORS[seed], label=f"seed {seed}")
            ax.axhline(7.918644208639811, color="black", linestyle=":", linewidth=1,
                       label="source V3")
            ax.set(title=f"{arm}: Adam {optimizer}", xlabel="Epoch (zero-based)")
            if j == 0:
                ax.set_ylabel("Validation LM loss")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT / "validation_lm_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("rouge_l", "bert_score_f1"),
                                 ("ROUGE-L", "BERTScore F1")):
        for j, optimizer in enumerate(OPTIMIZERS):
            x = [seed - 43 + (-0.08 if optimizer == "reset" else 0.08) for seed in SEEDS]
            y = [by_key[(optimizer, "B1_EPA", seed)]["matched"][metric]
                 - by_key[(optimizer, "B0_Baseline", seed)]["matched"][metric]
                 for seed in SEEDS]
            ax.scatter(x, y, color="#246494" if optimizer == "reset" else "#ce6c48",
                       label=f"Adam {optimizer}")
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xticks=[-1, 0, 1], xticklabels=["42", "43", "44"],
               xlabel="Seed", ylabel="B1 minus B0", title=title)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "epa_generation_effect_by_seed.png", dpi=180)
    plt.close(fig)

    labels = []
    neg, neu, pos = [], [], []
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            output = ROOT / "evaluation" / "outputs" / f"{optimizer}_B1_EPA_seed{seed}_val"
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            cm = metrics["conditions"]["matched"]["emotion"]["epa"]["sentence"]["confusion_matrix"]
            counts = [sum(cm[true][pred] for true in range(3)) for pred in range(3)]
            labels.append(f"{optimizer}\n{seed}")
            neg.append(counts[0]); neu.append(counts[1]); pos.append(counts[2])
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    xs = list(range(len(labels)))
    ax.bar(xs, neg, color="#547fa7", label="Negative")
    ax.bar(xs, neu, bottom=neg, color="#a4a4a4", label="Neutral")
    ax.bar(xs, pos, bottom=[a + b for a, b in zip(neg, neu)],
           color="#d17f58", label="Positive")
    ax.set(xticks=xs, xticklabels=labels, ylim=(0, 40),
           ylabel="Predicted validation UIDs (of 40)", title="EPA sentence-level predictions")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "epa_uid_prediction_collapse.png", dpi=180)
    plt.close(fig)
    print("WROTE three real-data figures")


if __name__ == "__main__":
    main()
