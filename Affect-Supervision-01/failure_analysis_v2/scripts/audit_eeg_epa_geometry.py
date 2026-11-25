"""Read-only EPA geometry on train/val EEG; never reads test trials."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment_assets import build_arm, load_config, read_v3
from task1_data import Task1DataModule


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
CLASSES = ("negative", "neutral", "positive")


def summarize(records):
    probs = np.array([r["probs"] for r in records])
    labels = np.array([r["label"] for r in records])
    predicted = probs.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for true, guess in zip(labels, predicted):
        cm[true, guess] += 1
    uid = defaultdict(list)
    for record in records:
        uid[record["uid"]].append(record)
    uid_probs = np.array([np.mean([r["probs"] for r in group], axis=0)
                          for group in uid.values()])
    uid_labels = np.array([group[0]["label"] for group in uid.values()])
    uid_cm = np.zeros((3, 3), dtype=int)
    for true, guess in zip(uid_labels, uid_probs.argmax(axis=1)):
        uid_cm[true, guess] += 1
    support = uid_cm.sum(1)
    pred_count = uid_cm.sum(0)
    f1 = np.divide(2 * uid_cm.diagonal(), support + pred_count,
                   out=np.zeros(3), where=(support + pred_count) > 0)
    return {
        "trials": len(records), "uids": len(uid),
        "trial_confusion_matrix": cm.tolist(),
        "uid_confusion_matrix": uid_cm.tolist(),
        "uid_macro_f1": float(f1.mean()),
        "mean_probs_by_true_class": {
            CLASSES[c]: probs[labels == c].mean(axis=0).tolist() for c in range(3)},
        "mean_logit_by_true_class": {
            CLASSES[c]: np.mean([r["logits"] for r in records if r["label"] == c], axis=0).tolist()
            for c in range(3)},
        "mean_own_prototype_cosine_before": {
            CLASSES[c]: float(np.mean([r["cos_before"][c] for r in records if r["label"] == c]))
            for c in range(3)},
        "mean_own_prototype_cosine_after": {
            CLASSES[c]: float(np.mean([r["cos_after"][c] for r in records if r["label"] == c]))
            for c in range(3)},
        "records": records,
    }


def evaluate(model, frame, phase, device, batch_size, full_train=False):
    subset = frame.loc[frame.phase == phase].copy()
    if phase == "train" and not full_train:
        # One deterministic subject trial per UID keeps the CPU audit tractable.
        subset = subset.sort_values(["text uid", "subject"]).groupby("text uid").head(1)
    records = []
    for start in range(0, len(subset), batch_size):
        chunk = subset.iloc[start:start + batch_size]
        if len(chunk) == 1:
            raise RuntimeError("Singleton batch hits the unmodified V3 squeeze bug")
        eeg = torch.from_numpy(np.stack(chunk.eeg.to_numpy())).to(device)
        mask = torch.from_numpy(np.stack(chunk["mask"].to_numpy())).to(device)
        prompts = [["<NR>"] * len(chunk), ["ZuCo1"] * len(chunk), chunk.subject.tolist()]
        ids = model.p_embedder.encode(prompts, device=device)
        prompt = model.p_embedder(ids, model.eval_pembed)
        with torch.no_grad():
            hidden, _ = model.eeg_encoder(eeg, mask, prompt)
            x = model.aligner.in_proj(hidden)
            vector = model.aligner.embed_eeg(x, None)[1]
            before = F.normalize(vector.float(), dim=-1)
            after = F.normalize(model.epa_projector(vector).float(), dim=-1)
            cos_before = before @ model.epa_prototypes.float().T
            cos_after = after @ model.epa_prototypes.float().T
            logits = cos_after / model.epa_temperature
            probs = logits.softmax(-1)
        for i, (_, row) in enumerate(chunk.iterrows()):
            records.append({
                "uid": int(row["text uid"]), "subject": row.subject,
                "label": CLASSES.index(row["sentiment label"]),
                "logits": logits[i].cpu().tolist(), "probs": probs[i].cpu().tolist(),
                "cos_before": cos_before[i].cpu().tolist(),
                "cos_after": cos_after[i].cpu().tolist(),
                "predicted": int(probs[i].argmax().cpu()),
                "true_margin": float(logits[i, CLASSES.index(row["sentiment label"])]
                                     - logits[i, [j for j in range(3)
                                                  if j != CLASSES.index(row["sentiment label"])]]
                                     .max()),
                "projection_angle_degrees": float(torch.rad2deg(torch.acos(
                    (before[i] * after[i]).sum().clamp(-1, 1))).cpu()),
            })
    return summarize(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--full-train", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = load_config(ROOT / "configs" / "common.json")
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # This audit uses only the EEG encoder, q_x aligner and EPA projector.
    # Loading T5 would waste memory and cannot affect these EEG-only logits.
    model = build_arm(config, read_v3(config), "B1_EPA", args.seed)
    fit = json.loads((ROOT / "runs" / "B1_EPA" / f"seed-{args.seed}" /
                      "fit_complete.json").read_text())
    ckpt = torch.load(fit["best_model_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    data = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"], seed=args.seed,
        num_workers=0,
    )
    frame = data.load_frame()
    p = model.epa_prototypes.float().cpu().numpy()
    result = {"seed": args.seed, "device": device, "prototype_cosine": (p @ p.T).tolist(),
              "sampling": ("all training trials; all validation trials" if args.full_train else
                           "one deterministic subject trial per train UID; all validation trials"),
              "train": evaluate(model, frame, "train", device, 16, args.full_train),
              "val": evaluate(model, frame, "val", device, 16)}
    suffix = "_full_train" if args.full_train else ""
    path = OUT / "tables" / f"eeg_epa_geometry_seed{args.seed}{suffix}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "device": device,
                      "train_f1": result["train"]["uid_macro_f1"],
                      "val_f1": result["val"]["uid_macro_f1"]}))


if __name__ == "__main__":
    main()
