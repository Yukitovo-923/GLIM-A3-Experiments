"""Plot actual low-LR validation curves and paired generation comparisons."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "failure_analysis_v2"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    old_epoch = list(csv.DictReader((OLD / "tables" / "epoch_losses.csv").open(encoding="utf-8")))
    new_epoch = list(csv.DictReader((ROOT / "evaluation" / "epoch_losses.csv").open(encoding="utf-8")))
    result = json.loads((ROOT / "evaluation" / "comparison.json").read_text(encoding="utf-8"))
    source_lm = result["source_epoch46"]["lm_loss"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, arm in zip(axes, ("B0_Baseline", "B1_EPA")):
        for seed, color in zip((42, 43, 44), ("#23649b", "#cb6547", "#72934d")):
            old = sorted((r for r in old_epoch if r["arm"] == arm and int(r["seed"]) == seed),
                         key=lambda r: int(r["epoch"]))
            new = sorted((r for r in new_epoch if r["arm"] == arm and int(r["seed"]) == seed),
                         key=lambda r: int(r["epoch"]))
            ax.plot([int(r["epoch"]) for r in old], [float(r["val_loss_lm"]) for r in old],
                    color=color, ls="--", alpha=.8, label=f"seed {seed}, LR 1e-4")
            ax.plot([int(r["epoch"]) for r in new], [float(r["val_lm"]) for r in new],
                    color=color, marker="o", ms=3, label=f"seed {seed}, LR 1e-5")
        ax.axhline(source_lm, color="black", lw=1, ls=":", label="source V3")
        ax.set(xlabel="Epoch (zero-based)", title=arm, xticks=range(5))
    axes[0].set_ylabel("Task1 validation LM loss")
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")
    fig.savefig(OUT / "validation_lm_old_vs_new_lr.png", dpi=180)
    plt.close(fig)

    old_rows = {(r["arm"], r["seed"]): r for r in
                json.loads((OLD / "validation_checkpoints" / "comparison.json")
                           .read_text(encoding="utf-8"))["rows"]}
    new_rows = {(r["arm"], r["seed"]): r["matched"] for r in result["runs"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for ax, metric in zip(axes, ("rouge_l", "bert_score_f1")):
        for arm, color, offset in (("B0_Baseline", "#23649b", -.07),
                                   ("B1_EPA", "#cb6547", .07)):
            for seed in (42, 43, 44):
                x = seed - 43 + offset
                ax.plot([x-.025, x+.025], [old_rows[(arm, seed)][metric],
                                          new_rows[(arm, seed)][metric]],
                        marker="o", ms=4, color=color,
                        label=arm if seed == 42 else None)
        ax.set(xticks=[-1, 0, 1], xticklabels=["42", "43", "44"],
               xlabel="Seed", ylabel=f"UID-balanced {metric}",
               title="LR 1e-4 (left) to 1e-5 (right)")
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "generation_old_vs_new_lr.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    for arm, color in (("B0_Baseline", "#23649b"), ("B1_EPA", "#cb6547")):
        matched = [next(r for r in result["runs"] if r["arm"] == arm and r["seed"] == seed)
                   for seed in (42, 43, 44)]
        ax.plot([42, 43, 44], [r["matched"]["rouge_l"] - r["mismatched"]["rouge_l"]
                               for r in matched], marker="o", color=color, label=arm)
    ax.axhline(0, color="black", lw=1)
    ax.set(xticks=[42, 43, 44], xlabel="Seed", ylabel="Matched minus mismatched UID ROUGE-L",
           title="Validation EEG contribution")
    ax.legend()
    fig.savefig(OUT / "eeg_matched_minus_mismatched_rouge.png", dpi=180)
    plt.close(fig)
    print("wrote three real-data validation figures")


if __name__ == "__main__":
    main()
