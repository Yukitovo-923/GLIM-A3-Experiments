"""Paired sentence bootstrap for predeclared audit representations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from text_space_audit import CLASS_ID, metrics, save_json


def predictions(path):
    frame = pd.read_csv(path)
    return frame.set_index("text uid").sort_index()


def compare(a, b, column_a, column_b, seed=42):
    if not a.index.equals(b.index):
        raise ValueError("Sentence UID mismatch")
    if not a["sentiment label"].equals(b["sentiment label"]):
        raise ValueError("Sentence label mismatch")
    test = a.phase.eq("test").to_numpy()
    y = a["sentiment label"].map(CLASS_ID).to_numpy(int)[test]
    x = a[column_a].map(CLASS_ID).to_numpy(int)[test]
    z = b[column_b].map(CLASS_ID).to_numpy(int)[test]
    observed = metrics(y, x)["macro_f1"] - metrics(y, z)["macro_f1"]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(5000)
    for k in range(5000):
        idx = rng.integers(0, len(y), len(y))
        bootstrap[k] = metrics(y[idx], x[idx])["macro_f1"] - metrics(y[idx], z[idx])["macro_f1"]
    return {"n_test_sentences": len(y), "macro_f1_difference": float(observed),
            "paired_bootstrap_95pct_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "different_predictions": int((x != z).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    v3_clip = predictions(root / "run-01/clip_text_predictions.csv")
    v3_mean = predictions(root / "run-01/encoder_mean_predictions.csv")
    released = predictions(root / "run-02-released-glim/clip_text_predictions.csv")
    result = {
        "v3_clip_vs_released_glim_clip_prototype": compare(v3_clip, released, "prototype_prediction", "prototype_prediction"),
        "v3_clip_vs_released_glim_clip_linear": compare(v3_clip, released, "linear_prediction", "linear_prediction"),
        "v3_encoder_mean_vs_v3_clip_prototype": compare(v3_mean, v3_clip, "prototype_prediction", "prototype_prediction"),
        "v3_encoder_mean_vs_v3_clip_linear": compare(v3_mean, v3_clip, "linear_prediction", "linear_prediction"),
        "interpretation": "Exploratory paired descriptive comparisons; no test labels were used for fitting or method selection."
    }
    save_json(root / "comparison.json", result)
    print(result)


if __name__ == "__main__":
    main()
