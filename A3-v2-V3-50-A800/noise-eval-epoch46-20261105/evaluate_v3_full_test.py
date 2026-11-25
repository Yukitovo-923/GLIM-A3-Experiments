"""Evaluate every test row with free generation, including the final short batch."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datamodule import ZuCoDataset
from data.revised_labels import audit
from model.glim_multitask import GLIMMultitask
from noise_input_eval_v3 import digest, dump, evaluation_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--text-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["GLIM_TEXT_MODEL_PATH"] = str(args.text_model_path.resolve())
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    model = GLIMMultitask.from_glim_checkpoint(str(args.checkpoint))
    if model.variant != "A3-full" or model.hparams.vq_distance != "cosine":
        raise RuntimeError("Expected the saved V3-50 model")
    model.to(device)
    model.setup("test")
    model.eval()
    frame, manifest = audit(pd.read_pickle(args.data_path), pd.read_pickle(args.labels))
    manifest.to_csv(args.output / "split_manifest.csv", index=False)
    dataset = ZuCoDataset(frame, "test")
    loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    dump(args.output / "eval_config.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "split": "test", "sample_count": len(dataset), "batch_count": len(loader),
        "sampler": "sequential; includes final short batch",
        "decoding": {"num_beams": 2, "min_length": 0,
                     "max_length": model.tgt_text_len, "teacher_forcing": False}})
    with torch.inference_mode():
        rows, _, report = evaluation_pass(model, loader, args.output, "eeg", None, device, 104)
    if len(rows) != len(dataset):
        raise AssertionError("Not every test sample was evaluated")
    print("FULL TEST PASS", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
