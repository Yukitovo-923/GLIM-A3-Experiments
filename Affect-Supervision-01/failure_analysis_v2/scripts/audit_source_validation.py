"""Evaluate the original V3 epoch-46 checkpoint on the Task1 validation split.

Uses the exact B0 model and inference protocol without training or changing weights.
Writes only inside failure_analysis_v2/validation_checkpoints/.
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.datamodule import ZuCoDataset
from experiment_assets import build_arm, load_config, read_v3, sha256_file
from infer import run_condition
from task1_data import Task1DataModule


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("An A800-class BF16 GPU is required")
    config = load_config(ROOT / "configs" / "common.json")
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    source = Path(config["checkpoint"]["path"])
    before = sha256_file(source)
    if before != config["checkpoint"]["sha256"]:
        raise RuntimeError("Source checkpoint SHA256 mismatch")
    data = Task1DataModule(
        data_path=config["assets"]["data"],
        labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        seed=42, num_workers=0,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"],
    )
    frame = data.load_frame()
    dataset = ZuCoDataset(frame, "val")
    del frame
    gc.collect()
    if len(dataset) != config["data"]["val_trials"]:
        raise RuntimeError("Validation trial count changed")
    model = build_arm(config, read_v3(config), "B0_Baseline", 42).cuda()
    model.setup("test")
    if any(p.requires_grad for p in model.text_model.parameters()):
        raise RuntimeError("Flan-T5 is not frozen")
    model.eval()
    output = OUT / "validation_checkpoints" / "source_v3_epoch46"
    output.mkdir(parents=True, exist_ok=False)
    stats = run_condition(model, dataset, np.arange(len(dataset)), "matched", output,
                          config["training"]["batch_size_test"], config["generation"])
    after = sha256_file(source)
    if before != after:
        raise RuntimeError("Source checkpoint changed")
    result = {"split": "val", "checkpoint": str(source), "checkpoint_sha256": before,
              "epoch": 46, "global_step": 93436, "model_arm": "B0_Baseline",
              "training_performed": False, "generation": config["generation"], "matched": stats}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
