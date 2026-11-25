"""Summarize same-protocol source/B0/B1 Task1 validation and make figures."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("lm_loss", "bleu1", "bleu2", "bleu4", "rouge_l", "bert_score_f1",
           "v3_discrete_macro_f1", "epa_macro_f1")


def main():
    data = json.loads((ROOT / "validation_checkpoints" / "comparison.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    with (ROOT / "tables" / "validation_checkpoint_comparison.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["arm", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    source = next(r for r in rows if r["arm"] == "source")
    aggregates = {}
    for arm in ("B0_Baseline", "B1_EPA"):
        subset = [r for r in rows if r["arm"] == arm]
        aggregates[arm] = {metric: {
            "mean": statistics.mean(r[metric] for r in subset),
            "sd": statistics.stdev(r[metric] for r in subset),
        } for metric in METRICS if all(r[metric] is not None for r in subset)}
    paired = {}
    for metric in METRICS:
        if metric == "epa_macro_f1":
            continue
        deltas = []
        for seed in (42, 43, 44):
            b0 = next(r for r in rows if r["arm"] == "B0_Baseline" and r["seed"] == seed)
            b1 = next(r for r in rows if r["arm"] == "B1_EPA" and r["seed"] == seed)
            deltas.append(b1[metric] - b0[metric])
        paired[metric] = {"seed_42_43_44": deltas, "mean": statistics.mean(deltas),
                          "sd": statistics.stdev(deltas)}
    result = {"source": source, "arm_summary": aggregates, "paired_b1_minus_b0": paired}
    (ROOT / "tables" / "validation_derived_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    colors = {"B0_Baseline": "#23649b", "B1_EPA": "#cb6547"}
    for j, metric in enumerate(("lm_loss", "rouge_l", "bert_score_f1")):
        ax = axes[j]
        for seed in (42, 43, 44):
            b0 = next(r for r in rows if r["arm"] == "B0_Baseline" and r["seed"] == seed)
            b1 = next(r for r in rows if r["arm"] == "B1_EPA" and r["seed"] == seed)
            ax.plot([0, 1], [b0[metric], b1[metric]], marker="o", color="#66727e", alpha=.7)
            ax.annotate(str(seed), (0, b0[metric]), xytext=(-11, 3), textcoords="offset points", fontsize=8)
        ax.scatter([0], [source[metric]], color="black", marker="*", s=100,
                   label="V3 epoch 46" if j == 0 else None, zorder=5)
        ax.set(xticks=[0, 1], xticklabels=["B0", "B1"], ylabel=metric,
               title=f"Task1 validation {metric}")
        if j == 0:
            ax.legend(fontsize=8)
    fig.savefig(ROOT / "figures" / "same_protocol_validation_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"paired_b1_minus_b0": paired}, indent=2))


if __name__ == "__main__":
    main()
