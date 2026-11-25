"""Exploratory text-length and train-label-permutation checks on cached text features."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr

from text_space_audit import CLASSES, bootstrap_ci, fit_lr, metrics, normalized, prototypes, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    data = np.load(run / "text_embeddings.npz")
    df = pd.read_csv(run / "sentences_official_labels.csv")
    pca = pd.read_csv(run / "pca_coordinates.csv")
    if not np.array_equal(data["uid"], df["text uid"].to_numpy()):
        raise ValueError("UID ordering mismatch")
    if not np.array_equal(pca["text uid"].to_numpy(), df["text uid"].to_numpy()):
        raise ValueError("PCA UID ordering mismatch")
    y = df.target.to_numpy(int)
    ix = {s: np.flatnonzero(df.phase.to_numpy() == s) for s in ("train", "val", "test")}
    token = data["token_length"].astype(float)
    words = df["input text"].str.split().map(len).to_numpy(float)
    chars = df["input text"].str.len().to_numpy(float)
    scalar = np.stack((token, words, chars), axis=1)
    length = {s: {CLASSES[c]: {"n": int((y[ix[s]] == c).sum()),
                               "mean_token_length": float(token[ix[s]][y[ix[s]] == c].mean()),
                               "mean_word_length": float(words[ix[s]][y[ix[s]] == c].mean()),
                               "mean_char_length": float(chars[ix[s]][y[ix[s]] == c].mean())}
                  for c in range(3)} for s in ix}
    corr = {"PC1_vs_token_length": float(spearmanr(pca.PC1.to_numpy(), token).statistic),
            "PC2_vs_token_length": float(spearmanr(pca.PC2.to_numpy(), token).statistic)}
    train, val, test = (ix[s] for s in ("train", "val", "test"))
    mu = scalar[train].mean(0)
    std = scalar[train].std(0).clip(1e-6)
    z = (scalar - mu) / std
    c_grid = (0.01, 0.1, 1., 10.)
    options = []
    for C in c_grid:
        W, b, info = fit_lr(z[train], y[train], C)
        pred = (z[val] @ W + b).argmax(1)
        options.append((metrics(y[val], pred)["macro_f1"], C, W, b, info))
    best = max(options, key=lambda row: (row[0], -row[1]))
    _, C, W, b, fit = best
    length_probe = {"features": ["T5 token length", "whitespace word count", "character count"],
                    "selected_C_on_val": C, "fit": fit}
    for s in ("val", "test"):
        pred = (z[ix[s]] @ W + b).argmax(1)
        length_probe[s] = metrics(y[ix[s]], pred)
        length_probe[s]["macro_f1_bootstrap_95pct_ci"] = bootstrap_ci(y[ix[s]], pred)
    rng = np.random.default_rng(42)
    permutation = {}
    for key in ("clip_text", "encoder_mean"):
        x = normalized(data[key].astype(np.float64))
        proto, _ = prototypes(x[train], y[train])
        observed = {s: metrics(y[ix[s]], (x[ix[s]] @ proto.T).argmax(1))["macro_f1"]
                    for s in ("val", "test")}
        null = {"val": [], "test": []}
        for _ in range(1000):
            fake_y = rng.permutation(y[train])
            fake_proto, _ = prototypes(x[train], fake_y)
            for s in null:
                null[s].append(metrics(y[ix[s]], (x[ix[s]] @ fake_proto.T).argmax(1))["macro_f1"])
        permutation[key] = {s: {"observed_macro_f1": observed[s],
                                "null_mean_macro_f1": float(np.mean(null[s])),
                                "null_95pct_interval": np.quantile(null[s], [.025, .975]).tolist(),
                                "permutation_p_ge_observed": float((1 + np.sum(np.array(null[s]) >= observed[s])) / 1001)}
                            for s in null}
    result = {"status": "complete", "seed": 42, "label_permutations": 1000,
              "length_by_split_and_class": length, "pca_token_length_correlation": corr,
              "length_only_probe": length_probe, "prototype_train_label_permutation": permutation,
              "source_sha256": sha256(__file__)}
    save_json(run / "exploratory_controls.json", result)
    print(json.dumps({"length_probe": {s: length_probe[s]["macro_f1"] for s in ("val", "test")},
                      "permutation": permutation, "pca_length_correlation": corr}, indent=2))


if __name__ == "__main__":
    main()
