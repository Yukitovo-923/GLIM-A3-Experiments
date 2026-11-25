"""One real Task1 EEG batch: strict V3 init, forward/backward, and original generation."""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from transformers.modeling_outputs import BaseModelOutput

from data.datamodule import ZuCoDataset
from experiment_assets import build_arm, load_config, read_v3, sha256_file, verify_assets
from task1_data import Task1DataModule


def grad_norm(module):
    gradients = [param.grad.detach().float() for param in module.parameters()
                 if param.grad is not None]
    if not gradients:
        return 0.0
    if any(not torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("Nonfinite gradient")
    return float(torch.linalg.vector_norm(torch.stack([v.norm() for v in gradients])).cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Smoke output already exists")
    config = load_config(args.config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A800 with BF16 is required for this smoke test")
    source_hashes = verify_assets(config)
    package = read_v3(config)
    seed = config["training"]["seeds"][0]
    b0 = build_arm(config, package, "B0_Baseline", seed)
    b1 = build_arm(config, package, "B1_EPA", seed)
    for key, value in b0.state_dict().items():
        torch.testing.assert_close(value, b1.state_dict()[key], rtol=0, atol=0)
    args.output.mkdir(parents=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    dm = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        audit_dir=args.output, seed=seed, num_workers=0,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"],
    )
    frame = dm.load_frame()
    train = frame.loc[frame.phase == "train"]
    chosen = train.drop_duplicates("text uid").groupby("sentiment label", sort=True).head(1)
    chosen = chosen.loc[chosen["sentiment label"].isin(("negative", "positive"))].head(2)
    if len(chosen) != 2 or chosen["text uid"].nunique() != 2:
        raise RuntimeError("Could not select two distinct real train UIDs")
    dataset = ZuCoDataset(chosen.copy(), "train")
    batch = default_collate([dataset[0], dataset[1]])
    if len(set(batch["text uid"].tolist())) != 2:
        raise RuntimeError("Smoke batch contains duplicate UIDs")
    del frame, train, chosen, dataset
    gc.collect()
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    report = {
        "kind": "AFFECT_SUPERVISION_01_REAL_BATCH",
        "status": "running",
        "source_hashes": source_hashes,
        "checkpoint_epoch": package["epoch"],
        "checkpoint_step": package["global_step"],
        "v3_shared_keys": len(package["state_dict"]),
        "shared_initialization_equal": True,
        "task1_uids": batch["text uid"].tolist(),
        "subjects": list(batch["prompt"][2]),
        "class_order": ["negative", "neutral", "positive"],
        "arms": {},
    }
    common_losses = None
    baseline_generation = None
    for arm, model in (("B0_Baseline", b0), ("B1_EPA", b1)):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = model.cuda()
        model.setup("fit")
        if arm == "B1_EPA":
            model.train_uid_counts = dm.train_uid_counts
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        losses = {key: float(value.detach().float().cpu()) for key, value in out.items()
                  if key.startswith("loss") and isinstance(value, torch.Tensor)}
        if not all(np.isfinite(value) for value in losses.values()):
            raise RuntimeError(f"Nonfinite {arm} loss")
        core = {key: out[key].detach().float().cpu() for key in
                ("loss_glim", "loss_clip", "loss_lm", "loss_vq", "loss_disc_emotion")}
        if common_losses is None:
            common_losses = core
            torch.testing.assert_close(
                out["loss_total"].float(),
                out["loss_glim"].float() + out["loss_vq"].float() + 0.2 * out["loss_disc_emotion"].float(),
                rtol=5e-3, atol=5e-3,
            )
        else:
            for key in core:
                torch.testing.assert_close(core[key], common_losses[key], rtol=0, atol=0)
            torch.testing.assert_close(
                out["loss_total"].float(),
                (out["loss_glim"] + out["loss_vq"] + 0.2 * out["loss_disc_emotion"]
                 + 0.2 * out["loss_epa"]).float(),
                rtol=5e-3, atol=5e-3,
            )
            if out["epa_projection"].shape != (2, 1024) or out["epa_logits"].shape != (2, 3):
                raise RuntimeError("EPA shape mismatch")
            eeg_grads = torch.autograd.grad(
                out["loss_epa"],
                tuple(param for param in model.eeg_encoder.parameters() if param.requires_grad),
                retain_graph=True, allow_unused=True,
            )
            if not any(value is not None and value.abs().sum() > 0 for value in eeg_grads):
                raise RuntimeError("EPA loss does not reach the EEG Encoder")
            if torch.autograd.grad(out["loss_epa"], model.aligner.q_y,
                                   retain_graph=True, allow_unused=True)[0] is not None:
                raise RuntimeError("EPA loss unexpectedly reaches the text-side q_y")
        out["loss_total"].backward()
        grads = {
            "eeg_encoder": grad_norm(model.eeg_encoder),
            "discrete_emotion_head": grad_norm(model.disc_head),
            "attribute_token": grad_norm(model.attribute),
            "vq_codebook": grad_norm(model.vq.codebook),
        }
        if arm == "B1_EPA":
            grads["epa_projector"] = grad_norm(model.epa_projector)
        if any(value <= 0 for value in grads.values()):
            raise RuntimeError(f"Missing expected gradient in {arm}: {grads}")
        if any(param.requires_grad or param.grad is not None
               for param in model.text_model.parameters()):
            raise RuntimeError("Frozen Flan-T5 gained a gradient")
        if arm == "B1_EPA" and model.epa_prototypes.grad is not None:
            raise RuntimeError("Frozen prototype gained a gradient")
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_out = model(batch)
            if "memory_mask" not in generated_out or generated_out["eeg_embeds"].shape[1] != 97:
                raise RuntimeError("Original V3 Attribute Token generation path changed")
            ids = model.text_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=generated_out["eeg_embeds"]),
                attention_mask=generated_out["memory_mask"], num_beams=2,
                min_length=0, max_length=16,
            )
            generated = model.tokenizer.batch_decode(ids, skip_special_tokens=True)
        if baseline_generation is None:
            baseline_generation = generated
        elif generated != baseline_generation:
            raise RuntimeError("B1 changed initial generation before any training")
        report["arms"][arm] = {"status": "PASS", "losses": losses,
                               "grad_norms": grads, "generated_text": generated,
                               "attribute_token_present": True}
        del generated_out, out, model
        gc.collect()
        torch.cuda.empty_cache()
    if verify_assets(config) != source_hashes:
        raise RuntimeError("An original source asset changed during smoke test")
    report["status"] = "PASS"
    report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
