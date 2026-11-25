"""One-batch in-memory optimizer mapping and backward smoke for all four cells."""

import json
import os
from pathlib import Path

import lightning as L
import torch

from experiment_assets import build_arm, load_config, read_v3, verify_assets
from run_train import source_optimizer_for_arm
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config(ROOT / "configs" / "common.json")
    hashes = verify_assets(config, include_large_data=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    L.seed_everything(42, workers=True)
    package = read_v3(config)
    data = Task1DataModule(data_path=config["assets"]["data"],
                           labels_path=config["assets"]["labels"],
                           expected_manifest=config["assets"]["e1_manifest"],
                           expected_sha256=config["assets"]["e1_manifest_sha256"],
                           audit_dir=ROOT / "reports" / "smoke_data", seed=42,
                           bsz_train=72, bsz_val=24, bsz_test=24, num_workers=0)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    rows = []
    for arm in ("B0_Baseline", "B1_EPA"):
        for condition in ("reset", "restored"):
            L.seed_everything(42, workers=True)
            model = build_arm(config, package, arm, 42).cuda()
            model.train()
            model.setup("fit")
            if model.text_model.training or any(p.requires_grad for p in model.text_model.parameters()):
                raise RuntimeError("Frozen T5 mode/gradient changed")
            if arm == "B1_EPA":
                model.train_uid_counts = data.train_uid_counts
            mapped, mapping = source_optimizer_for_arm(package, model, arm)
            optimizer = model.configure_optimizers()
            if condition == "restored":
                optimizer.load_state_dict(mapped)
            before = len(optimizer.state)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
                loss = out["loss_total"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite one-batch loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if model.eeg_encoder.shared_queries.grad is None:
                raise RuntimeError("No shared EEG gradient")
            if arm == "B1_EPA" and model.epa_projector.weight.grad is None:
                raise RuntimeError("No EPA projector gradient")
            optimizer.step()
            after = len(optimizer.state)
            total_trainable = len([p for p in model.parameters() if p.requires_grad])
            if (before != (214 if condition == "restored" else 0) or
                    not (0 < after <= total_trainable and after >= before) or
                    (arm == "B1_EPA" and model.epa_projector.weight not in optimizer.state)):
                raise RuntimeError("Adam state count after one step is wrong")
            rows.append({"arm": arm, "condition": condition,
                         "mapping": mapping, "states_before_step": before,
                         "states_after_step": after, "one_batch_loss": float(loss.detach().cpu()),
                         "shared_eeg_gradient_norm": float(model.eeg_encoder.shared_queries.grad.norm())})
            del model, optimizer, out, loss
            torch.cuda.empty_cache()
    report = {"status": "PASS", "source_hashes": hashes, "split": "train only",
              "optimizer_step_on_source": False, "rows": rows}
    (ROOT / "reports" / "optimizer_smoke.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("OPTIMIZER_SMOKE_PASS", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
