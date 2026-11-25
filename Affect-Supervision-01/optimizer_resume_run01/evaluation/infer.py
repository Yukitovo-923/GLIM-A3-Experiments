"""Evaluate the validation-selected checkpoint on every Task1 trial, with EEG derangement."""

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from transformers.modeling_outputs import BaseModelOutput

from data.datamodule import ZuCoDataset
from experiment_assets import build_arm, load_config, read_v3, sha256_file
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]


def derange_within_subject(dataset, seed):
    subjects = np.array([prompt[2] for prompt in dataset.data["prompt"]])
    uids = np.array(dataset.data["text uid"], dtype=int)
    mapping = np.arange(len(dataset), dtype=int)
    rng = np.random.default_rng(seed + 1000)
    for subject in sorted(set(subjects)):
        indices = np.flatnonzero(subjects == subject)
        if len(indices) < 2:
            raise RuntimeError(f"Subject {subject} has too few trials for EEG derangement")
        for _ in range(1000):
            shuffled = rng.permutation(indices)
            if np.all(uids[shuffled] != uids[indices]):
                mapping[indices] = shuffled
                break
        else:
            raise RuntimeError(f"Could not derange EEG within subject {subject}")
    if np.any(uids[mapping] == uids) or np.any(subjects[mapping] != subjects):
        raise RuntimeError("Invalid mismatched-EEG mapping")
    return mapping


def load_best_model(config, arm, seed, condition):
    run = ROOT / "runs" / condition / arm / f"seed-{seed}"
    completion = json.loads((run / "fit_complete.json").read_text(encoding="utf-8"))
    if completion["arm"] != arm or completion["seed"] != seed:
        raise RuntimeError("Training completion record does not match requested model")
    checkpoint = Path(completion["best_model_path"])
    if not checkpoint.is_file() or run.resolve() not in checkpoint.resolve().parents:
        raise RuntimeError("Best checkpoint is absent or outside its isolated run")
    package = read_v3(config)
    model = build_arm(config, package, arm, seed).cuda()
    model.setup("test")
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(trained["state_dict"], strict=True)
    if any(param.requires_grad for param in model.text_model.parameters()):
        raise RuntimeError("The text model was not frozen after restoring best checkpoint")
    model.eval()
    return model, checkpoint, trained


def run_condition(model, dataset, mapping, condition, output, batch_size, generation):
    if condition not in ("matched", "mismatched"):
        raise ValueError(condition)
    by_subject = defaultdict(list)
    for index, prompt in enumerate(dataset.data["prompt"]):
        by_subject[prompt[2]].append(index)
    rows = []
    total_lm_numerator = 0.0
    total_lm_tokens = 0
    subject_lm = {}
    for subject, indices in sorted(by_subject.items()):
        lm_numerator = 0.0
        lm_tokens = 0
        chunks = [indices[start:start + batch_size]
                  for start in range(0, len(indices), batch_size)]
        # The unmodified V3 aligner squeezes its batch dimension at size 1.
        # Merge a singleton tail into the preceding inference batch.
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        if any(len(chunk) < 2 for chunk in chunks):
            raise RuntimeError(f"Subject {subject} has no valid inference batch")
        for chunk in chunks:
            samples = [dataset[index] for index in chunk]
            if condition == "mismatched":
                for sample, index in zip(samples, chunk):
                    source = dataset[int(mapping[index])]
                    sample["eeg"] = source["eeg"]
                    sample["mask"] = source["mask"]
            batch = default_collate(samples)
            batch = {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor)
                     else value for key, value in batch.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
                if "memory_mask" not in out:
                    raise RuntimeError("V3 Attribute Token path was not preserved")
                ids = model.text_model.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=out["eeg_embeds"]),
                    attention_mask=out["memory_mask"],
                    num_beams=generation["num_beams"],
                    min_length=generation["min_length"],
                    max_length=generation["max_length"],
                )
            predictions = model.tokenizer.batch_decode(ids, skip_special_tokens=True)
            target_tokens = int((out["target_text_ids"] != model.text_model.config.pad_token_id).sum())
            lm_numerator += float(out["loss_lm"].detach().float().cpu()) * target_tokens
            lm_tokens += target_tokens
            disc_probs = out["disc_logits"].float().softmax(-1).detach().cpu().numpy()
            epa_probs = (out["epa_logits"].float().softmax(-1).detach().cpu().numpy()
                         if "epa_logits" in out else None)
            labels = out["emotion_targets"].detach().cpu().tolist()
            for position, index in enumerate(chunk):
                rows.append({
                    "dataset_index": int(index),
                    "uid": int(dataset.data["text uid"][index]),
                    "subject": subject,
                    "label_id": int(labels[position]),
                    "reference": dataset.data["raw input text"][index],
                    "prediction": predictions[position],
                    "eeg_source_uid": int(dataset.data["text uid"][int(mapping[index])])
                    if condition == "mismatched" else int(dataset.data["text uid"][index]),
                    "disc_probs": disc_probs[position].tolist(),
                    "epa_probs": epa_probs[position].tolist() if epa_probs is not None else None,
                })
            del out, ids, batch
        subject_lm[subject] = {"loss": lm_numerator / lm_tokens, "target_tokens": lm_tokens}
        total_lm_numerator += lm_numerator
        total_lm_tokens += lm_tokens
    rows.sort(key=lambda row: row["dataset_index"])
    if len(rows) != len(dataset) or len({row["dataset_index"] for row in rows}) != len(dataset):
        raise RuntimeError("Evaluation lost or duplicated a trial")
    with (output / f"{condition}_predictions.jsonl").open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"trial_count": len(rows), "uid_count": len({row["uid"] for row in rows}),
            "lm_loss": total_lm_numerator / total_lm_tokens,
            "lm_target_tokens": total_lm_tokens, "by_subject_lm": subject_lm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("B0_Baseline", "B1_EPA"), required=True)
    parser.add_argument("--condition", choices=("reset", "restored"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed not in config["training"]["seeds"]:
        parser.error("Seed not in locked configuration")
    if args.output.exists() or ROOT.resolve() not in args.output.resolve().parents:
        parser.error("Output must be a new directory under Affect-Supervision-01")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A800 BF16 is unavailable")
    args.output.mkdir(parents=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    torch.manual_seed(args.seed)
    model, checkpoint, trained = load_best_model(config, args.arm, args.seed, args.condition)
    data = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        audit_dir=args.output, seed=args.seed, num_workers=0,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"],
    )
    frame = data.load_frame()
    dataset = ZuCoDataset(frame, args.split)
    del frame
    gc.collect()
    expected = config["data"][f"{args.split}_trials"]
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} {args.split} trials, got {len(dataset)}")
    mapping = derange_within_subject(dataset, args.seed)
    (args.output / "eeg_derangement.json").write_text(
        json.dumps([{"dataset_index": i, "source_index": int(mapping[i]),
                     "target_uid": int(dataset.data["text uid"][i]),
                     "source_uid": int(dataset.data["text uid"][int(mapping[i])])}
                    for i in range(len(dataset))], indent=2), encoding="utf-8")
    results = {condition: run_condition(model, dataset, mapping, condition, args.output,
                                        config["training"]["batch_size_test"],
                                        config["generation"])
               for condition in ("matched", "mismatched")}
    summary = {"arm": args.arm, "condition": args.condition,
               "seed": args.seed, "split": args.split,
               "best_checkpoint": str(checkpoint), "best_checkpoint_sha256": sha256_file(checkpoint),
               "checkpoint_epoch": int(trained["epoch"]),
               "checkpoint_global_step": int(trained["global_step"]),
               "generation": config["generation"], "conditions": results}
    (args.output / "inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
