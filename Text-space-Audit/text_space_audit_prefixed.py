"""Frozen GLIM text-space audit; one row per official ZuCo1 Task 1 sentence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from model.glim_multitask import GLIMMultitask

CLASSES = ("negative", "neutral", "positive")
CLASS_ID = {x: i for i, x in enumerate(CLASSES)}
FEATURES = ("clip_text", "encoder_mean")
SPLITS = ("train", "val", "test")
COLORS = ("#3164ad", "#8d8683", "#d8753a")
INPUT_TEMPLATE = "To English: <MASK>"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalized(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-10) or not np.isfinite(norm).all():
        raise ValueError("Zero or nonfinite text embedding")
    return x / norm


def metrics(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    support = cm.sum(1)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(3, float), where=cm.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(3, float), where=support > 0)
    f1 = np.divide(2 * tp, support + cm.sum(0), out=np.zeros(3, float), where=(support + cm.sum(0)) > 0)
    return {"macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "accuracy": float(tp.sum() / max(1, support.sum())),
            "precision": precision.tolist(), "recall": recall.tolist(), "class_f1": f1.tolist(),
            "support": support.tolist(), "predicted_counts": cm.sum(0).tolist(),
            "confusion": cm.tolist()}


def bootstrap_ci(y, pred, seed=42, n=2000):
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        values[i] = metrics(y[idx], pred[idx])["macro_f1"]
    return np.quantile(values, [0.025, 0.975]).tolist()


def read_sentences(manifest_path, reference_path):
    manifest = pd.read_csv(manifest_path)
    reference = pd.read_pickle(reference_path)
    required = {"dataset", "task", "text uid", "input text", "phase", "sentiment label"}
    if not required <= set(manifest) or not (required - {"phase"}) <= set(reference):
        raise ValueError("Missing sentence metadata columns")
    sentences = manifest[(manifest.dataset == "ZuCo1") & (manifest.task == "task1")].copy()
    sentences = sentences.sort_values("text uid").reset_index(drop=True)
    if len(sentences) != 400 or sentences["text uid"].nunique() != 400:
        raise ValueError("Expected exactly 400 distinct Task 1 sentence IDs")
    if sentences["input text"].nunique() != 400:
        raise ValueError("Duplicate canonical text in Task 1")
    if not set(sentences["sentiment label"]) == set(CLASSES):
        raise ValueError("Missing or unexpected sentiment class")
    split_count = sentences.phase.value_counts().to_dict()
    if split_count != {"train": 320, "val": 40, "test": 40}:
        raise ValueError(f"Unexpected E1 split: {split_count}")
    official = reference[(reference.dataset == "ZuCo1") & (reference.task == "task1")].copy()
    official = official[["text uid", "input text", "sentiment label", "raw label"]].drop_duplicates()
    if len(official) != 400 or official["text uid"].nunique() != 400:
        raise ValueError("Official reference does not have 400 unambiguous sentence IDs")
    check = sentences.merge(official, on="text uid", suffixes=("_manifest", "_reference"), validate="one_to_one")
    if len(check) != 400 or not (check["input text_manifest"] == check["input text_reference"]).all():
        raise ValueError("Manifest text differs from official reference")
    if not (check["sentiment label_manifest"] == check["sentiment label_reference"]).all():
        raise ValueError("Manifest emotion differs from official reference")
    check = check.rename(columns={"input text_manifest": "input text",
                                  "sentiment label_manifest": "sentiment label"})
    check = check.drop(columns=["input text_reference", "sentiment label_reference"])
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a, b = check[check.phase == left], check[check.phase == right]
        if set(a["text uid"]) & set(b["text uid"]) or set(a["input text"]) & set(b["input text"]):
            raise ValueError("Cross-split sentence leakage")
    check["target"] = check["sentiment label"].map(CLASS_ID).astype(int)
    return check


@torch.inference_mode()
def extract(model, sentences, batch_size):
    arrays = {key: [] for key in FEATURES}
    masks = []
    for start in range(0, len(sentences), batch_size):
        raw_texts = sentences["input text"].iloc[start:start + batch_size].tolist()
        texts = [INPUT_TEMPLATE.replace("<MASK>", text) for text in raw_texts]
        if len(texts) == 1:
            raise ValueError("GLIM Aligner.embed_text uses squeeze(); use a batch size avoiding singleton final batch")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids, mask = model.tokenize(texts, model.input_text_len - model.prompt_tuning_len)
            hidden, _ = model.encode_text(ids, mask)
            vector = model.aligner.embed_text(hidden, mask)
            mean = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        arrays["clip_text"].append(vector.float().cpu().numpy())
        arrays["encoder_mean"].append(mean.float().cpu().numpy())
        masks.append(mask.sum(1).cpu().numpy())
        if (start // batch_size + 1) % 8 == 0:
            print("encoded", min(start + batch_size, len(sentences)), "/", len(sentences), flush=True)
    arrays = {key: normalized(np.concatenate(value).astype(np.float64)) for key, value in arrays.items()}
    return arrays, np.concatenate(masks)


def geometry(x, y):
    cosine = np.clip(x @ x.T, -1, 1)
    distance = np.maximum(1 - cosine, 0)
    within, between = {}, {}
    within_sum = within_pairs = between_sum = between_pairs = 0
    for a, name in enumerate(CLASSES):
        idx = np.flatnonzero(y == a)
        sub = distance[np.ix_(idx, idx)]
        count = len(idx) * (len(idx) - 1)
        value = float((sub.sum() - np.trace(sub)) / count)
        within[name] = value
        within_sum += value * count
        within_pairs += count
        for b in range(a + 1, 3):
            jdx = np.flatnonzero(y == b)
            sub = distance[np.ix_(idx, jdx)]
            value = float(sub.mean())
            between[f"{name}|{CLASSES[b]}"] = value
            between_sum += value * sub.size
            between_pairs += sub.size
    s = np.zeros(len(y))
    for i in range(len(y)):
        own = np.flatnonzero(y == y[i]); other = [np.flatnonzero(y == k) for k in range(3) if k != y[i]]
        a = distance[i, own].sum() / (len(own) - 1)
        b = min(distance[i, j].mean() for j in other)
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return {"within_distance": within, "between_distance": between,
            "weighted_within_mean": float(within_sum / within_pairs),
            "weighted_between_mean": float(between_sum / between_pairs),
            "within_over_between_ratio": float((within_sum / within_pairs) / (between_sum / between_pairs)),
            "cosine_silhouette": float(s.mean()),
            "cosine_silhouette_by_class": {name: float(s[y == i].mean()) for i, name in enumerate(CLASSES)}}


def prototypes(x_train, y_train):
    proto = np.stack([normalized(x_train[y_train == i].mean(0)) for i in range(3)])
    return proto, {CLASSES[i]: int((y_train == i).sum()) for i in range(3)}


def fit_lr(x, y, C):
    n, d = x.shape
    hot = np.eye(3)[y]
    penalty = 1.0 / (C * n)
    def objective(flat):
        W = flat[:d * 3].reshape(d, 3)
        b = flat[d * 3:]
        logits = x @ W + b
        probability = softmax(logits, axis=1)
        loss = (logsumexp(logits, axis=1) - logits[np.arange(n), y]).mean()
        loss += 0.5 * penalty * np.square(W).sum()
        diff = (probability - hot) / n
        grad = np.r_[(x.T @ diff + penalty * W).ravel(), diff.sum(0)]
        return float(loss), grad
    result = minimize(objective, np.zeros(d * 3 + 3), jac=True, method="L-BFGS-B",
                      options={"maxiter": 400, "ftol": 1e-9})
    if not np.isfinite(result.fun):
        raise RuntimeError("Nonfinite logistic-regression fit")
    return result.x[:d * 3].reshape(d, 3), result.x[d * 3:], {"converged": bool(result.success),
                                                               "iterations": int(result.nit), "objective": float(result.fun)}


def evaluate(x, sentences, output, feature_name):
    split = {s: np.flatnonzero(sentences.phase.to_numpy() == s) for s in SPLITS}
    y = sentences.target.to_numpy(int)
    train, val, test = (split[s] for s in SPLITS)
    proto, counts = prototypes(x[train], y[train])
    sim = np.clip(proto @ proto.T, -1, 1)
    outcome = {"dimension": int(x.shape[1]), "train_class_counts": counts,
               "prototype_cosine_similarity": sim.tolist(),
               "geometry_train": geometry(x[train], y[train]),
               "geometry_all_descriptive": geometry(x, y),
               "prototype_classifier": {}, "linear_probe": {}}
    for phase, idx in (("val", val), ("test", test)):
        probability = softmax(x[idx] @ proto.T, axis=1)
        pred = probability.argmax(1)
        result = metrics(y[idx], pred)
        result["macro_f1_bootstrap_95pct_ci"] = bootstrap_ci(y[idx], pred)
        outcome["prototype_classifier"][phase] = result
    mu = x[train].mean(0)
    std = x[train].std(0).clip(1e-6)
    standardize = lambda arr: np.clip((arr - mu) / std, -20, 20)
    c_grid = (0.01, 0.1, 1.0, 10.0)
    candidates = []
    for C in c_grid:
        W, b, info = fit_lr(standardize(x[train]), y[train], C)
        pred = (standardize(x[val]) @ W + b).argmax(1)
        candidates.append((metrics(y[val], pred)["macro_f1"], C, W, b, info))
    best = max(candidates, key=lambda item: (item[0], -item[1]))
    _, C, W, b, fit_info = best
    outcome["linear_probe"] = {"C_grid": list(c_grid), "validation_grid_macro_f1":
        {str(item[1]): item[0] for item in candidates}, "selected_C": C,
        "train_only_standardizer": True, "fit": fit_info}
    for phase, idx in (("val", val), ("test", test)):
        probability = softmax(standardize(x[idx]) @ W + b, axis=1)
        pred = probability.argmax(1)
        result = metrics(y[idx], pred)
        result["macro_f1_bootstrap_95pct_ci"] = bootstrap_ci(y[idx], pred)
        outcome["linear_probe"][phase] = result
    majority = int(np.bincount(y[train], minlength=3).argmax())
    outcome["majority_baseline"] = {phase: metrics(y[idx], np.full(len(idx), majority, int))
                                    for phase, idx in (("val", val), ("test", test))}
    predictions = sentences[["text uid", "phase", "input text", "sentiment label", "target"]].copy()
    ps = x @ proto.T
    lp = softmax(standardize(x) @ W + b, axis=1)
    predictions["prototype_prediction"] = [CLASSES[i] for i in ps.argmax(1)]
    predictions["linear_prediction"] = [CLASSES[i] for i in lp.argmax(1)]
    for i, label in enumerate(CLASSES):
        predictions[f"prototype_cosine_{label}"] = ps[:, i]
        predictions[f"linear_probability_{label}"] = lp[:, i]
    predictions.to_csv(output / f"{feature_name}_predictions.csv", index=False)
    pd.DataFrame(proto.T, columns=CLASSES).to_csv(output / f"{feature_name}_prototypes.csv", index_label="dimension")
    np.save(output / f"{feature_name}_prototypes.npy", proto.astype(np.float32))
    return outcome


def pca_plot(x, sentences, output):
    train = sentences.phase.to_numpy() == "train"
    centered = x - x[train].mean(0)
    _, singular, vt = svd(centered[train], full_matrices=False, lapack_driver="gesdd")
    scores = centered @ vt[:2].T
    var = (singular ** 2) / (len(singular) - 1)
    explained = var[:2] / var.sum()
    result = sentences[["text uid", "phase", "sentiment label"]].copy()
    result["PC1"], result["PC2"] = scores[:, 0], scores[:, 1]
    result.to_csv(output / "pca_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, label in enumerate(CLASSES):
        idx = sentences["sentiment label"].to_numpy() == label
        ax.scatter(scores[idx, 0], scores[idx, 1], s=20, alpha=0.65,
                   color=COLORS[i], label=f"{label} (n={idx.sum()})")
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% train variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% train variance)")
    ax.set_title("GLIM CLIP text embeddings; PCA fitted on train sentences")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "pca_clip_text.png", dpi=180)
    plt.close(fig)
    return {"train_explained_variance_ratio_first_two": explained.tolist(),
            "fitted_on": "train only", "visualized": "all 400 sentences"}


def main():
    parser = argparse.ArgumentParser()
    for key in ("checkpoint", "text-model-path", "manifest", "labels", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sentences = read_sentences(Path(args.manifest), Path(args.labels))
    audit = {"status": "running", "unit": "independent sentence", "n_sentences": len(sentences),
             "split_counts": sentences.phase.value_counts().to_dict(),
             "class_counts": sentences["sentiment label"].value_counts().to_dict(),
             "split_class_counts": {s: sentences[sentences.phase == s]["sentiment label"].value_counts().to_dict() for s in SPLITS},
             "uid_overlap": 0, "canonical_text_overlap": 0,
             "text_input_template": INPUT_TEMPLATE,
             "first_model_input": INPUT_TEMPLATE.replace("<MASK>", sentences["input text"].iloc[0]),
             "audit_script_sha256": sha256(__file__),
             "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
             "manifest_sha256": sha256(args.manifest), "official_label_table_sha256": sha256(args.labels),
             "text_model_path": str(Path(args.text_model_path).resolve()),
             "text_model_weight_sha256": sha256(Path(args.text_model_path) / "model.safetensors"),
             "source_sha256": {name: sha256(name) for name in ("model/glim.py", "model/modules.py", "data/datamodule.py")}}
    save_json(output / "audit.json", audit)
    sentences.to_csv(output / "sentences_official_labels.csv", index=False)
    import os
    os.environ["GLIM_TEXT_MODEL_PATH"] = str(Path(args.text_model_path).resolve())
    model = GLIMMultitask.from_glim_checkpoint(args.checkpoint).cuda().eval()
    model.setup("predict")
    model.text_model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    smoke, smoke_length = extract(model, sentences.iloc[:2], 2)
    smoke_info = {"status": "PASS", "shapes": {key: list(value.shape) for key, value in smoke.items()},
                  "token_lengths": smoke_length.tolist(),
                  "finite": {key: bool(np.isfinite(value).all()) for key, value in smoke.items()}}
    if not all(smoke_info["finite"].values()):
        raise ValueError("Nonfinite smoke representation")
    save_json(output / "smoke.json", smoke_info)
    if args.smoke_only:
        audit["status"] = "smoke_pass_only"
        save_json(output / "audit.json", audit)
        return
    if len(sentences) % args.batch_size == 1:
        raise ValueError("Batch size would leave a singleton final batch")
    features, lengths = extract(model, sentences, args.batch_size)
    np.savez_compressed(output / "text_embeddings.npz", **{key: value.astype(np.float32) for key, value in features.items()},
                        uid=sentences["text uid"].to_numpy(), phase=sentences.phase.to_numpy(),
                        label=sentences["sentiment label"].to_numpy(), token_length=lengths)
    del model
    torch.cuda.empty_cache()
    report = {"method": "frozen Flan-T5-Large encoder + GLIM Aligner text query; L2 normalized at CLIP loss",
              "text_input_template": INPUT_TEMPLATE,
              "main_feature": "clip_text", "supplemental_feature": "encoder_mean",
              "seed": 42, "bootstrap_resamples": 2000, "representations": {}}
    for key in FEATURES:
        print("analyzing", key, flush=True)
        report["representations"][key] = evaluate(features[key], sentences, output, key)
        save_json(output / "results.json", report)
    report["pca"] = pca_plot(features["clip_text"], sentences, output)
    report["umap"] = "not_run: umap-learn is not installed in the existing environment"
    save_json(output / "results.json", report)
    audit["status"] = "complete"
    audit["feature_shapes"] = {key: list(value.shape) for key, value in features.items()}
    audit["token_length_summary"] = {"min": int(lengths.min()), "max": int(lengths.max()),
                                     "mean": float(lengths.mean())}
    save_json(output / "audit.json", audit)
    hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact_sha256.json"}
    save_json(output / "artifact_sha256.json", hashes)
    print("TEXT_SPACE_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
