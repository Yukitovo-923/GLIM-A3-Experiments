"""Compare existing EEG-side prototype readout before and after EPA projection."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def macro_f1(cm):
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * cm.diagonal(), denominator,
                           out=np.zeros(3), where=denominator > 0).mean())


def score(records, key):
    by_uid = defaultdict(list)
    for row in records:
        by_uid[int(row["uid"])].append(row)
    cm = np.zeros((3, 3), dtype=int)
    for group in by_uid.values():
        labels = {row["label"] for row in group}
        if len(labels) != 1:
            raise RuntimeError("Conflicting label within UID")
        if key == "cos_before":
            logits = np.array([row[key] for row in group]) / 0.07
            logits -= logits.max(axis=1, keepdims=True)
            values = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        else:
            values = np.array([row["probs"] for row in group])
        probs = values.mean(axis=0)
        cm[labels.pop(), int(probs.argmax())] += 1
    return {"uid_macro_f1": macro_f1(cm), "uid_confusion_matrix": cm.tolist(),
            "uid_positive_prediction_share": float(cm[:, 2].sum() / cm.sum())}


def main():
    rows = []
    for seed in (42, 43, 44):
        data = json.loads((ROOT / "tables" / f"eeg_epa_geometry_seed{seed}.json")
                          .read_text(encoding="utf-8"))
        for split in ("train", "val"):
            records = data[split]["records"]
            for stage, key in (("before", "cos_before"), ("after", "cos_after")):
                stats = score(records, key)
                rows.append({"seed": seed, "split": split, "stage": stage,
                             "n_trials": len(records), "n_uids": data[split]["uids"],
                             **stats})
    with (ROOT / "tables" / "projection_before_after_uid.csv").open(
            "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    x = np.arange(3)
    width = .20
    for i, (split, stage, color, hatch) in enumerate((
            ("train", "before", "#669ac3", ""),
            ("train", "after", "#23649b", "//"),
            ("val", "before", "#e6a68e", ""),
            ("val", "after", "#cb6547", "//"))):
        values = [next(r["uid_macro_f1"] for r in rows if r["seed"] == seed
                       and r["split"] == split and r["stage"] == stage)
                  for seed in (42, 43, 44)]
        ax.bar(x + (i - 1.5) * width, values, width, color=color, hatch=hatch,
               label=f"{split} {stage}")
    ax.set(xticks=x, xticklabels=["42", "43", "44"], xlabel="Seed",
           ylabel="UID Macro-F1", ylim=(0, 1.08),
           title="Frozen prototypes: EEG readout before and after EPA projection")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(ROOT / "figures" / "projection_before_after_uid.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
