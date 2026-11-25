"""Verify canonical train targets with unchanged 8x sampling and validation data."""

import json
from collections import Counter
from pathlib import Path

from experiment_assets import load_config
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config(ROOT / "configs" / "common.json")
    data = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        seed=42, num_workers=0,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"],
    )
    data.setup("fit")
    train, val = data.train_set, data.val_set
    if len(train) != 3316 * 8 or len(val) != 406:
        raise RuntimeError("Training multiplicity or validation size changed")
    if any(target != raw for target, raw in zip(train.data["target text"],
                                                train.data["raw input text"])):
        raise RuntimeError("A train target is not the canonical original sentence")
    if any(target != raw for target, raw in zip(val.data["target text"],
                                                val.data["raw input text"])):
        raise RuntimeError("Validation canonical targets changed")
    train_counts = Counter(int(uid) for uid in train.data["text uid"])
    if len(train_counts) != 320 or any(train_counts[uid] != count * 8
                                       for uid, count in data.train_uid_counts.items()):
        raise RuntimeError("Train UID trial weighting changed")
    if len(data.train_dataloader()) != 367:
        raise RuntimeError("Optimizer steps per epoch changed")
    report = {"status": "PASS", "train_trials": 3316,
              "train_positions": len(train), "train_uids": len(train_counts),
              "train_unique_target_sentences": len(set(train.data["target text"])),
              "val_trials": len(val), "val_uids": len(set(val.data["text uid"])),
              "optimizer_steps_per_epoch": 367,
              "all_train_targets_equal_raw_original": True,
              "all_val_targets_equal_raw_original": True,
              "train_target_mode": config["data"]["train_target_mode"]}
    (ROOT / "reports" / "target_mode_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
