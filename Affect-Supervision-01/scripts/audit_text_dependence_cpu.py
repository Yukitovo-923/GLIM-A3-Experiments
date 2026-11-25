"""Validation-only counterfactual: alter reference input text, hold EEG/prompt fixed."""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import default_collate
from transformers.modeling_outputs import BaseModelOutput

from data.datamodule import ZuCoDataset
from experiment_assets import build_arm, load_config, read_v3
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("audit_start", flush=True)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(ROOT / "configs" / "common.json")
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    print("config_loaded", flush=True)
    model = build_arm(config, read_v3(config), "B1_EPA", 42).to(device)
    print("v3_loaded", flush=True)
    model.setup("test")
    print("text_model_loaded", flush=True)
    run = ROOT / "runs" / "B1_EPA" / "seed-42"
    fit = json.loads((run / "fit_complete.json").read_text())
    checkpoint = torch.load(fit["best_model_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    print("best_loaded", flush=True)
    data = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"], seed=42, num_workers=0,
    )
    frame = data.load_frame()
    print("frame_loaded", flush=True)
    dataset = ZuCoDataset(frame, "val")
    batch = default_collate([dataset[0], dataset[1]])
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    altered = dict(batch)
    altered["input text"] = tuple(reversed(batch["input text"]))
    if list(altered["input text"]) == list(batch["input text"]):
        raise RuntimeError("Counterfactual texts did not change")
    with torch.no_grad():
        print("forward_start", flush=True)
        original = model(batch)
        changed = model(altered)
        print("forward_done", flush=True)
        args = dict(num_beams=2, min_length=0, max_length=16)
        original_ids = model.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=original["eeg_embeds"]),
            attention_mask=original["memory_mask"], **args)
        changed_ids = model.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=changed["eeg_embeds"]),
            attention_mask=changed["memory_mask"], **args)
    result = {
        "split": "val", "arm": "B1_EPA", "seed": 42, "device": str(device),
        "uid": [int(dataset.data["text uid"][i]) for i in (0, 1)],
        "counterfactual": "swap input text between two trials; EEG, mask, prompt, target fixed",
        "input_text_changed": True,
        "eeg_memory_max_absolute_difference": float((original["eeg_embeds"] - changed["eeg_embeds"]).abs().max()),
        "memory_mask_equal": bool(torch.equal(original["memory_mask"], changed["memory_mask"])),
        "generation_token_ids_equal": bool(torch.equal(original_ids, changed_ids)),
        "lm_loss_difference": float((original["loss_lm"] - changed["loss_lm"]).abs()),
        "clip_loss_difference": float((original["loss_clip"] - changed["loss_clip"]).abs()),
        "generation_original": model.tokenizer.batch_decode(original_ids, skip_special_tokens=True),
        "generation_counterfactual": model.tokenizer.batch_decode(changed_ids, skip_special_tokens=True),
    }
    path = ROOT / "reports" / "text_dependence_runtime.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if (result["eeg_memory_max_absolute_difference"] != 0 or
            not result["memory_mask_equal"] or
            not result["generation_token_ids_equal"] or
            result["lm_loss_difference"] != 0):
        raise RuntimeError("Generation path depended on reference input text")


if __name__ == "__main__":
    main()
