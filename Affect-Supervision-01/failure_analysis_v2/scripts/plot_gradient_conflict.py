"""Plot real multi-batch read-only gradient diagnostics."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "tables" / "gradient_by_batch_layer.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    groups = ["EEG_encoder_all", "EEG_aligner", "all_shared_EEG",
              "EEG_input_projection", "encoder_in_block_0", "encoder_in_block_1",
              "encoder_in_block_2", "encoder_in_block_3", "encoder_in_block_4",
              "encoder_in_block_5", "encoder_out_block_0", "encoder_out_block_1",
              "encoder_out_block_2", "encoder_out_block_3", "encoder_out_block_4",
              "encoder_out_block_5", "aligner_input_projection", "aligner_q_x",
              "aligner_cross_attention_x"]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    values = [[float(r["cos_epa_glim"]) for r in rows if r["group"] == group and r["cos_epa_glim"]]
              for group in groups]
    ax.boxplot(values, tick_labels=groups, vert=False, showmeans=True)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Cosine(grad EPA, grad GLIM+VQ)", title="Eight training batches, seed 42")
    fig.savefig(ROOT / "figures" / "gradient_cosine_by_layer.png", dpi=180)
    plt.close(fig)

    selected = ["EEG_encoder_all", "EEG_aligner", "all_shared_EEG"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    x = np.arange(8)
    for group in selected:
        z = sorted((r for r in rows if r["group"] == group), key=lambda r: int(r["batch"]))
        axes[0].plot(x, [float(r["cos_epa_glim"]) for r in z], marker="o", label=group)
        axes[1].plot(x, [float(r["weighted_norm_epa"]) / float(r["norm_glim_plus_vq"])
                         for r in z], marker="o", label=group)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set(xlabel="Batch", ylabel="Gradient cosine", title="EPA versus GLIM+VQ")
    axes[1].set(xlabel="Batch", ylabel="Weighted EPA norm / GLIM+VQ norm", yscale="log",
                title="Relative gradient scale")
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT / "figures" / "gradient_batch_stability.png", dpi=180)
    plt.close(fig)

    shared = [r for r in rows if r["group"] == "all_shared_EEG"]
    result = {
        "batches": len(shared),
        "shared_cos_epa_glim_negative_count": sum(float(r["cos_epa_glim"]) < 0 for r in shared),
        "shared_cos_epa_original_ce_negative_count": sum(float(r["cos_epa_original_emotion_ce"]) < 0 for r in shared),
        "shared_weighted_epa_to_glim_norm_ratio_mean": float(np.mean([
            float(r["weighted_norm_epa"]) / float(r["norm_glim_plus_vq"]) for r in shared])),
        "shared_weighted_original_ce_to_glim_norm_ratio_mean": float(np.mean([
            float(r["weighted_norm_original_emotion_ce"]) / float(r["norm_glim_plus_vq"])
            for r in shared])),
    }
    (ROOT / "tables" / "gradient_derived_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
