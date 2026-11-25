"""Postprocess saved noise evaluation rows with sentence-cluster bootstrap."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("bleu1_mtv", "bleu2_mtv", "rouge1_mtv", "wer_raw")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    config = json.loads((root / "eval_config.json").read_text(encoding="utf-8"))
    eeg = read_rows(root / "samples_eeg.jsonl")
    noise = [read_rows(root / f"samples_noise_seed{s}.jsonl") for s in config["seeds"]]
    if any(len(x) != len(eeg) for x in noise):
        raise ValueError("Condition sample counts differ")
    groups = defaultdict(list)
    strata = defaultdict(list)
    deltas = {k: [] for k in METRICS}
    for i, row in enumerate(eeg):
        identity = (row["dataset"], row["task"], row["text_uid"], row["subject"], row["raw_text"])
        if any((n[i]["dataset"], n[i]["task"], n[i]["text_uid"], n[i]["subject"], n[i]["raw_text"]) != identity for n in noise):
            raise ValueError(f"Unpaired record at sample {i}")
        groups[(row["dataset"], row["task"], row["text_uid"])].append(i)
        strata[(row["dataset"], row["task"])].append(i)
        for metric in METRICS:
            deltas[metric].append(row[metric] - np.mean([n[i][metric] for n in noise]))
    rng = np.random.default_rng(20261105)
    group_index = list(groups.values())
    n_group = len(group_index)
    sampled = rng.integers(0, n_group, size=(2000, n_group))
    result = {"sample_count": len(eeg), "sentence_groups": n_group,
              "seeds": config["seeds"], "method": "Paired cluster bootstrap by dataset/task/text UID; 2000 replicates; resampled groups weighted by their observed pair counts",
              "metrics": {}, "by_dataset_task": {}}
    counts = np.array([len(g) for g in group_index], dtype=float)
    for metric in METRICS:
        values = np.array(deltas[metric], dtype=float)
        sums = np.array([values[g].sum() for g in group_index])
        resampled_means = sums[sampled].sum(1) / counts[sampled].sum(1)
        ci = np.quantile(resampled_means, [.025, .975]).tolist()
        result["metrics"][metric] = {"eeg_minus_noise": float(values.mean()),
                                     "sentence_cluster_bootstrap_95pct_ci": ci}
    for stratum, indices in strata.items():
        key = "/".join(stratum)
        result["by_dataset_task"][key] = {"count": len(indices),
                                           **{metric: float(np.array(deltas[metric])[indices].mean()) for metric in METRICS}}
    (root / "grouped_bootstrap.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
