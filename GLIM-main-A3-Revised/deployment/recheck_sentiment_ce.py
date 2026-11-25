"""Recompute labeled-only sentiment CE from the real ZuCo validation split."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datamodule import GLIMDataModule
from model.glim_multitask import GLIMMultitask


CLASSES = {"negative": 0, "neutral": 1, "positive": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant = config.pop("variant")
    model = GLIMMultitask.from_glim_checkpoint(args.original_checkpoint, variant, **config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != 49 or saved["hyper_parameters"].get("variant") != variant:
        raise RuntimeError("Wrong pilot checkpoint")
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.cuda().eval()
    dm = GLIMDataModule(data_path=str(args.data_path), bsz_train=model.bsz_train,
                        bsz_val=model.bsz_val, bsz_test=model.bsz_retrieval,
                        num_workers=0)
    dm.label_reference = args.labels
    dm.manifest_path = args.output.with_name("val_recheck_manifest.csv")
    dm.setup("fit")
    confusion = torch.zeros(3, 3, dtype=torch.long)
    loss_sum = 0.0
    labeled = 0
    batches = 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in dm.val_dataloader():
            eeg = batch["eeg"].cuda(non_blocking=True)
            mask = batch["mask"].cuda(non_blocking=True)
            logits = model.predict_emotion(eeg, mask, batch["prompt"])["disc_logits"].float()
            targets = torch.tensor([CLASSES.get(label, -100)
                                    if task == "task1" and dataset == "ZuCo1" else -100
                                    for label, task, dataset in zip(batch["sentiment label"],
                                                                    batch["raw task key"], batch["prompt"][1])],
                                   device=logits.device)
            valid = targets.ge(0)
            if valid.any():
                loss_sum += F.cross_entropy(logits[valid], targets[valid], reduction="sum").item()
                labeled += int(valid.sum().item())
                prediction = logits.argmax(dim=-1)
                confusion += torch.bincount(3 * targets[valid].cpu() + prediction[valid].cpu(),
                                            minlength=9).reshape(3, 3)
            batches += 1
    if labeled != 406 or int(confusion.sum()) != labeled:
        raise RuntimeError(f"Labeled validation count mismatch: {labeled}")
    tp = confusion.diag().float()
    f1 = (2 * tp / (confusion.sum(0) + confusion.sum(1)).clamp_min(1)).mean().item()
    result = {"kind": "REAL_ZUCO_VALIDATION_SENTIMENT_RECHECK", "epoch_index": 49,
              "labeled_count": labeled, "validation_batches": batches,
              "labeled_only_cross_entropy": loss_sum / labeled,
              "macro_f1": f1, "confusion_matrix_negative_neutral_positive": confusion.tolist(),
              "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
