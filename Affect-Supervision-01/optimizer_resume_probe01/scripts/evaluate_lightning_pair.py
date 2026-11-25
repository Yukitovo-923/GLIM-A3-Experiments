"""Generate on the same 40 validation UIDs for the two Lightning B0 probes."""

import argparse
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


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("reset", "restored"), required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / "common.json")
    output = ROOT / "evaluation" / "outputs" / args.condition
    if output.exists():
        parser.error("Evaluation output exists; refusing to overwrite")
    output.mkdir(parents=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    run_name = "seed-42" if args.condition == "reset" else "seed-42-restored"
    run = ROOT / "runs" / "B0_Baseline" / run_name
    complete = json.loads((run / "fit_complete.json").read_text(encoding="utf-8"))
    checkpoint = Path(complete["best_model_path"])
    if not checkpoint.is_file() or run.resolve() not in checkpoint.resolve().parents:
        raise RuntimeError("Checkpoint missing or outside isolated run")
    torch.manual_seed(42)
    model = build_arm(config, read_v3(config), "B0_Baseline", 42).cuda()
    model.setup("test")
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(trained["state_dict"], strict=True)
    model.eval()
    data = Task1DataModule(data_path=config["assets"]["data"],
                           labels_path=config["assets"]["labels"],
                           expected_manifest=config["assets"]["e1_manifest"],
                           expected_sha256=config["assets"]["e1_manifest_sha256"],
                           audit_dir=output, seed=42, num_workers=0,
                           bsz_train=config["training"]["batch_size_train"],
                           bsz_val=config["training"]["batch_size_val"],
                           bsz_test=config["training"]["batch_size_test"])
    frame = data.load_frame()
    dataset = ZuCoDataset(frame, "val")
    del frame
    gc.collect()
    if len(dataset) != 406 or len(set(dataset.data["text uid"])) != 40:
        raise RuntimeError("Validation set mismatch")
    result = run_condition(model, dataset, np.arange(406), "matched", output,
                           config["training"]["batch_size_test"], config["generation"])
    summary = {"condition": args.condition, "split": "val", "seed": 42,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "checkpoint_epoch": int(trained["epoch"]), "generation": config["generation"],
               "matched": result}
    (output / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
