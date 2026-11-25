"""One real ZuCo batch, all EPA arms; no optimizer step or training loop."""
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
from epa_assets import hash_file, load_common, load_model_for_arm, validate_asset_hashes
from epa_data import EPADataModule
from epa_model import masked_sentence_ce
from model.glim import GLIM


def grad_norm(module):
    gradients = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
    if not gradients:
        return 0.0
    if not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Nonfinite parameter gradient")
    return float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients])).cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--common-init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to reuse a smoke output directory")
    common = load_common(args.common)
    validate_asset_hashes(common)
    before = {name: hash_file(common["assets"][name]) for name in
              ("released_glim", "vq_initialization", "prototype")}
    args.output.mkdir(parents=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = common["assets"]["text_model"]
    seed = common["training"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Real EPA smoke requires the available A800")
    torch.cuda.reset_peak_memory_stats()
    dm = EPADataModule(data_path=common["assets"]["data"],
                       labels_path=common["assets"]["labels"],
                       expected_manifest=common["assets"]["e1_manifest"],
                       audit_dir=args.output, bsz_train=common["training"]["batch_size_train"],
                       bsz_val=common["training"]["batch_size_val"],
                       bsz_test=common["training"]["batch_size_test"], num_workers=0)
    frame = dm.load_frame()  # read the large paired pickle exactly once.
    sr = frame.loc[(frame.dataset == "ZuCo1") & (frame.task == "task1") &
                   (frame.phase == "train") & frame["sentiment label"].notna()]
    chosen = sr.drop_duplicates("text uid").groupby("sentiment label", sort=True).head(1)
    chosen = chosen.loc[chosen["sentiment label"].isin(("negative", "positive"))].head(2).copy()
    if len(chosen) != 2 or chosen["text uid"].nunique() != 2:
        raise ValueError("Could not select two distinct real labeled train sentences")
    if any(np.asarray(x).shape != (1280, 128) for x in chosen.eeg):
        raise ValueError("Unexpected real EEG tensor shape")
    dataset = ZuCoDataset(chosen, "train")
    batch = default_collate([dataset[0], dataset[1]])
    if len(set(batch["text uid"].tolist())) != 2:
        raise ValueError("Smoke batch contains duplicate sentence UID")
    del frame, sr, chosen, dataset
    gc.collect()
    batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    model = load_model_for_arm(args.common, args.common_init, "E0").cuda()
    model.set_train_uid_counts(dm.uid_trial_counts)
    model.setup("fit")
    model.text_model.requires_grad_(False)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = GLIM.shared_forward(model, batch)
        controlled = model(batch)
        for key in ("loss_clip", "loss_lm", "loss_commitment", "eeg_emb_vector", "eeg_embeds"):
            torch.testing.assert_close(base[key], controlled[key], rtol=0, atol=0)
        if "memory_mask" in controlled or hasattr(model, "attribute"):
            raise RuntimeError("Attribute conditioning changed the GLIM generation path")
        generated = model.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=controlled["eeg_embeds"]),
            max_length=16, num_beams=2)
        generated_text = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
    del base, controlled, generated
    # Check ignore_index on actual logits: changing an invalid row must not affect CE/gradient.
    labels = torch.tensor([-100, 2], device="cuda")
    logits = torch.randn(2, 3, device="cuda", requires_grad=True)
    uid = batch["text uid"]
    a = masked_sentence_ce(logits, labels, uid, dm.uid_trial_counts)
    replaced = logits.detach().clone()
    replaced[0] += 1000
    b = masked_sentence_ce(replaced, labels, uid, dm.uid_trial_counts)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward()
    if logits.grad[0].abs().sum() != 0 or logits.grad[1].abs().sum() == 0:
        raise RuntimeError("Invalid trial affected emotion CE gradient")
    report = {"kind": "REAL_ZUCO_EPA_SMOKE", "status": "running",
              "common_init_sha256": hash_file(args.common_init),
              "initialization_tensor_sha256": torch.load(args.common_init, map_location="cpu", weights_only=False)["tensor_sha256"],
              "manifest_sha256": hash_file(args.output / "split_manifest.csv"),
              "prototype_sha256": before["prototype"],
              "batch_size": len(batch["text uid"]),
              "train_sample_uids": batch["text uid"].tolist(),
              "subjects": list(batch["prompt"][2]),
              "generated_text": generated_text,
              "original_glim_path_identical": True,
              "attribute_token_absent": True,
              "invalid_emotion_trial_mask": "PASS",
              "arm_status": {}, "arms": {}}
    original_losses = None
    for arm in ("E0", "E1", "E2"):
        model.arm = arm
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        if out["epa_input_shape"] != (2, 1024) or out["epa_projection_shape"] != (2, 1024):
            raise ValueError("Wrong EPA EEG vector shape")
        if out["epa_logits"].shape != (2, 3) or out["disc_logits"].shape != (2, 3):
            raise ValueError("Wrong sentiment logits shape")
        losses = {k: float(v.detach().float().cpu()) for k, v in out.items()
                  if k.startswith("loss") and isinstance(v, torch.Tensor)}
        if not all(np.isfinite(v) for v in losses.values()):
            raise RuntimeError(f"Nonfinite {arm} loss")
        if original_losses is None:
            original_losses = {k: out[k].detach().clone() for k in ("loss_glim", "loss_clip", "loss_lm", "loss_vq")}
        else:
            for key, previous in original_losses.items():
                torch.testing.assert_close(out[key].detach(), previous, rtol=0, atol=0)
        if arm == "E2":
            encoder_grads = torch.autograd.grad(out["loss_epa"],
                                                tuple(p for p in model.eeg_encoder.parameters() if p.requires_grad),
                                                retain_graph=True, allow_unused=True)
            projector_grads = torch.autograd.grad(out["loss_epa"], tuple(model.epa_projector.parameters()),
                                                  retain_graph=True, allow_unused=True)
            if not any(g is not None and g.abs().sum() > 0 for g in encoder_grads):
                raise RuntimeError("EPA loss does not reach EEG Encoder")
            if not any(g is not None and g.abs().sum() > 0 for g in projector_grads):
                raise RuntimeError("EPA loss does not reach EPA projection")
            if torch.autograd.grad(out["loss_epa"], model.aligner.q_y,
                                   retain_graph=True, allow_unused=True)[0] is not None:
                raise RuntimeError("EPA loss unexpectedly reaches text q_y")
        out["loss_total"].backward()
        grads = {"eeg_encoder": grad_norm(model.eeg_encoder),
                 "epa_projection": grad_norm(model.epa_projector),
                 "discrete_emotion_head": grad_norm(model.disc_head),
                 "vq_codebook": grad_norm(model.vq.codebook)}
        if grads["eeg_encoder"] <= 0 or grads["vq_codebook"] <= 0:
            raise RuntimeError(f"Missing shared/VQ gradient in {arm}")
        if arm == "E1" and grads["discrete_emotion_head"] <= 0:
            raise RuntimeError("E1 emotion head has no gradient")
        if arm == "E2" and grads["epa_projection"] <= 0:
            raise RuntimeError("E2 EPA projection has no gradient")
        if arm != "E2" and grads["epa_projection"] != 0:
            raise RuntimeError("EPA projection updated in a non-EPA arm")
        if model.epa_prototypes.requires_grad or model.epa_prototypes.grad is not None:
            raise RuntimeError("Frozen text prototypes have gradients")
        if any(p.requires_grad or p.grad is not None for p in model.text_model.parameters()):
            raise RuntimeError("Frozen T5 has gradients")
        report["arm_status"][arm] = "PASS"
        report["arms"][arm] = {"losses": losses, "grad_norms": grads,
                                "epa_logits_shape": list(out["epa_logits"].shape),
                                "epa_projection_shape": list(out["epa_projection"].shape),
                                "labeled": int(out["emotion_targets"].ge(0).sum()),
                                "codebook_perplexity": float(out["codebook_perplexity"].detach().cpu())}
        del out
    after = {name: hash_file(common["assets"][name]) for name in before}
    if before != after:
        raise RuntimeError("An original checkpoint or prototype changed")
    report["source_sha256_unchanged"] = True
    report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["status"] = "PASS"
    (args.output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
