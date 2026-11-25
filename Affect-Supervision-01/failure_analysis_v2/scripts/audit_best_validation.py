"""Same-protocol Task1 validation for all six selected continuation checkpoints."""

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.datamodule import ZuCoDataset
from experiment_assets import load_config, sha256_file
from infer import load_best_model, run_condition
from task1_data import Task1DataModule


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("An A800-class BF16 GPU is required")
    config = load_config(ROOT / "configs" / "common.json")
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    data = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"], seed=42, num_workers=0,
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
    output_root = OUT / "validation_checkpoints"
    for arm in config["arms"]:
        for seed in config["training"]["seeds"]:
            output = output_root / f"{arm}_seed{seed}"
            output.mkdir(parents=True, exist_ok=False)
            model, checkpoint, trained = load_best_model(config, arm, seed)
            before = sha256_file(checkpoint)
            stats = run_condition(model, dataset, np.arange(len(dataset)), "matched", output,
                                  config["training"]["batch_size_test"], config["generation"])
            after = sha256_file(checkpoint)
            if before != after:
                raise RuntimeError(f"Checkpoint changed: {checkpoint}")
            result = {"arm": arm, "seed": seed, "split": "val",
                      "checkpoint": str(checkpoint), "checkpoint_sha256": before,
                      "epoch_zero_based": int(trained["epoch"]),
                      "training_performed": False, "generation": config["generation"],
                      "matched": stats}
            (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"arm": arm, "seed": seed, "epoch": result["epoch_zero_based"],
                              "val_lm": stats["lm_loss"]}), flush=True)
            del model, trained
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
