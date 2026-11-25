"""Paired audit of raw vs GLIM-prefixed text inputs; selection uses val only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from text_space_audit import CLASS_ID, metrics, save_json, sha256


ROOT = Path(__file__).resolve().parent
PAIRS = (("v3", "run-01", "run-03-v3-prefixed"),
         ("released_glim", "run-02-released-glim", "run-04-released-prefixed"))


def paired(raw, prefixed, phase, feature, method, seed=42, resamples=5000):
    a = pd.read_csv(ROOT / raw / f"{feature}_predictions.csv").set_index("text uid").sort_index()
    b = pd.read_csv(ROOT / prefixed / f"{feature}_predictions.csv").set_index("text uid").sort_index()
    if not a.index.equals(b.index) or not a["sentiment label"].equals(b["sentiment label"]):
        raise ValueError("Input-template comparison has mismatched sentence IDs or labels")
    if not a.phase.equals(b.phase):
        raise ValueError("Input-template comparison has mismatched split")
    idx = a.phase.eq(phase).to_numpy()
    y = a["sentiment label"].map(CLASS_ID).to_numpy(int)[idx]
    x = a[method].map(CLASS_ID).to_numpy(int)[idx]
    z = b[method].map(CLASS_ID).to_numpy(int)[idx]
    observed = metrics(y, x)["macro_f1"] - metrics(y, z)["macro_f1"]
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples)
    for k in range(resamples):
        draw = rng.integers(0, len(y), len(y))
        distribution[k] = metrics(y[draw], x[draw])["macro_f1"] - metrics(y[draw], z[draw])["macro_f1"]
    return {"n_sentences": len(y), "raw_minus_prefixed_macro_f1": float(observed),
            "paired_bootstrap_95pct_ci": np.quantile(distribution, [.025, .975]).tolist(),
            "different_predictions": int((x != z).sum())}


def main():
    result = {"selection_protocol": "Choose encoder_mean prototype input template by validation Macro-F1 only in this comparison; test descriptive.",
              "selection_blinded_to_all_prior_test_results": False,
              "selection_limit": "The unprefixed test result was already known from the earlier Text-space Audit; this is an exploratory comparison, not a fresh blind final test.",
              "seed": 42, "paired_bootstrap_resamples": 5000, "models": {}}
    for model, raw, prefixed in PAIRS:
        old = json.loads((ROOT / raw / "audit.json").read_text(encoding="utf8"))
        new = json.loads((ROOT / prefixed / "audit.json").read_text(encoding="utf8"))
        if old["checkpoint_sha256"] != new["checkpoint_sha256"]:
            raise ValueError("Different checkpoints")
        if old["manifest_sha256"] != new["manifest_sha256"]:
            raise ValueError("Different splits")
        if old["text_model_weight_sha256"] != new["text_model_weight_sha256"]:
            raise ValueError("Different text encoder weights")
        a, b = (np.load(ROOT / name / "text_embeddings.npz") for name in (raw, prefixed))
        if not np.array_equal(a["uid"], b["uid"]):
            raise ValueError("Different UID order")
        embedding_cosine = {feature: float(np.sum(a[feature] * b[feature], axis=1).mean())
                            for feature in ("encoder_mean", "clip_text")}
        prototype_cosine = {feature: np.sum(np.load(ROOT / raw / f"{feature}_prototypes.npy") *
                                             np.load(ROOT / prefixed / f"{feature}_prototypes.npy"), axis=1).tolist()
                            for feature in ("encoder_mean", "clip_text")}
        comparisons = {feature: {method: {phase: paired(raw, prefixed, phase, feature, method)
                                          for phase in ("val", "test")}
                                 for method in ("prototype_prediction", "linear_prediction")}
                       for feature in ("encoder_mean", "clip_text")}
        result["models"][model] = {"checkpoint_sha256": old["checkpoint_sha256"],
                                   "raw_run": raw, "prefixed_run": prefixed,
                                   "embedding_cosine_same_sentence_mean": embedding_cosine,
                                   "same_class_prototype_cosine_raw_vs_prefixed": prototype_cosine,
                                   "comparisons": comparisons}
    val = result["models"]["v3"]["comparisons"]["encoder_mean"]["prototype_prediction"]["val"]
    result["preferred_template_by_validation"] = "raw_unprefixed" if val["raw_minus_prefixed_macro_f1"] > 0 else "glim_prefixed"
    result["selection_margin_macro_f1"] = abs(val["raw_minus_prefixed_macro_f1"])
    result["selection_uncertainty"] = "paired validation bootstrap interval includes zero" if val["paired_bootstrap_95pct_ci"][0] <= 0 <= val["paired_bootstrap_95pct_ci"][1] else "paired validation bootstrap interval excludes zero"
    result["script_sha256"] = sha256(__file__)
    save_json(ROOT / "prefix_comparison.json", result)
    print(json.dumps({"preferred_template_by_validation": result["preferred_template_by_validation"],
                      "selection_margin_macro_f1": result["selection_margin_macro_f1"],
                      "selection_uncertainty": result["selection_uncertainty"],
                      "v3_comparisons": result["models"]["v3"]["comparisons"]["encoder_mean"],
                      "prototype_cosine": result["models"]["v3"]["same_class_prototype_cosine_raw_vs_prefixed"]["encoder_mean"]}, indent=2))


if __name__ == "__main__":
    main()
