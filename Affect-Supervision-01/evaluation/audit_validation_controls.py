"""Repeat within-subject EEG mismatches on validation data only, with locked best models."""

import gc
import json
import os
from pathlib import Path

import torch

from data.datamodule import ZuCoDataset
from experiment_assets import load_config
from infer import derange_within_subject, load_best_model, run_condition
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "validation_controls"


def main():
    if OUT.exists():
        raise RuntimeError("Validation control output exists; refuse to overwrite")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A800 BF16 required for validation controls")
    config = load_config(ROOT / "configs" / "common.json")
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    OUT.mkdir(parents=True)
    for seed in config["training"]["seeds"]:
        for arm in config["arms"]:
            model, checkpoint, _ = load_best_model(config, arm, seed)
            data = Task1DataModule(
                data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
                expected_manifest=config["assets"]["e1_manifest"],
                expected_sha256=config["assets"]["e1_manifest_sha256"],
                seed=seed, num_workers=0,
                bsz_train=config["training"]["batch_size_train"],
                bsz_val=config["training"]["batch_size_val"],
                bsz_test=config["training"]["batch_size_test"],
            )
            frame = data.load_frame()
            dataset = ZuCoDataset(frame, "val")
            del frame
            if len(dataset) != 406:
                raise RuntimeError("Expected 406 validation trials")
            run_dir = OUT / f"{arm}_seed{seed}"
            run_dir.mkdir()
            summary = {"arm": arm, "seed": seed, "split": "val",
                       "best_checkpoint": str(checkpoint), "conditions": {}}
            for repeat in range(4):
                name = "matched" if repeat == 0 else f"mismatched_{repeat}"
                condition = "matched" if repeat == 0 else "mismatched"
                # Same subject-specific permutation for both arms at a given seed/repeat.
                mapping = derange_within_subject(dataset, seed + 10000 * repeat)
                directory = run_dir / name
                directory.mkdir()
                result = run_condition(model, dataset, mapping, condition, directory,
                                       config["training"]["batch_size_val"],
                                       config["generation"])
                summary["conditions"][name] = result
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(arm, seed, name, "LM", result["lm_loss"], flush=True)
            del model, dataset, data
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
