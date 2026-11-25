"""Real ZuCo A3 smoke test. Runs one batch and one optimizer update, never a training loop."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import default_collate
from transformers.modeling_outputs import BaseModelOutput

from data.revised_labels import ROOT, audit, mapped_step3


def inspect_data(path, label_path, step3):
    from data.datamodule import ZuCoDataset
    reference = pd.read_pickle(label_path)
    raw = mapped_step3(path) if step3 else pd.read_pickle(path)
    df, manifest = audit(raw, reference)
    counts = manifest.pivot_table(index="phase", columns="target", values="EEG rows", aggfunc="sum", fill_value=0)
    train = df.loc[df.phase.eq("train") & df["sentiment label"].notna()].drop_duplicates("text uid")
    if len(train) < 2:
        raise ValueError("Need two distinct labeled train text UIDs")
    chosen = train.iloc[:2]
    if step3:
        chosen = chosen.merge(reference.loc[reference.task.eq("task1")], validate="many_to_one",
                              on=["dataset","task","text uid","input text","sentiment label"])
    if not all(x.shape == (1280, 128) for x in chosen["eeg"]):
        raise ValueError("Expected (1280,128) real EEG samples")
    if not all(x.shape == (1280,) for x in chosen["mask"]):
        raise ValueError("Expected (1280,) EEG masks")
    if not all(x.any() and set(x.tolist()) <= {0, 1} for x in chosen["mask"]):
        raise ValueError("EEG masks must contain valid frames and only 0/1 values")
    chosen = chosen.copy()
    chosen["eeg"] = chosen["eeg"].map(lambda x: x.copy())
    dataset = ZuCoDataset(chosen, "train")
    batch = default_collate([dataset[0], dataset[1]])
    return batch, {str(phase): {str(label): int(n) for label, n in row.items()}
                   for phase, row in counts.to_dict(orient="index").items()}, len(df), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT/"data/tmp/zuco_label_8variants.df")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/a3_revised.json")
    parser.add_argument("--step3", action="store_true", help="Memory-map STEP3 and reproduce official grouped split")
    parser.add_argument("--tensor-only", action="store_true", help="Use test framework adapters; real model, real EEG")
    args = parser.parse_args()
    for path in (args.data_path, args.checkpoint):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if not args.text_model_path.is_dir():
        parser.error(f"Missing language model directory: {args.text_model_path}")
    if args.output.exists():
        parser.error(f"Refusing to reuse experiment directory: {args.output}")
    if args.tensor_only:
        import sys
        sys.path.insert(0, str(ROOT / "tests"))
        from synthetic_glim import tensor_only_imports
        adapter = tensor_only_imports()
        adapter.start()
    from model.glim import GLIM
    from model.glim_multitask import GLIMMultitask
    batch, distribution, n_rows, manifest = inspect_data(args.data_path, args.labels, args.step3)
    assert len(set(batch["text uid"].tolist())) == 2
    config = json.loads(args.config.read_text(encoding="utf-8"))
    os.environ["GLIM_TEXT_MODEL_PATH"] = str(args.text_model_path.resolve())
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = GLIMMultitask.from_glim_checkpoint(args.checkpoint, **config).to(device)
    if args.tensor_only:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        model.tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, local_files_only=True)
        model.text_model = T5ForConditionalGeneration.from_pretrained(args.text_model_path,
            torch_dtype=torch.bfloat16, local_files_only=True).to(device).requires_grad_(False)
    else:
        model.setup("fit")
    original_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    if model.encode_labels(batch["sentiment label"]).lt(0).any():
        raise ValueError("Selected train samples must have valid sentiment labels")

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda", cache_enabled=False)
    model.eval()
    with autocast, torch.no_grad():
        original = GLIM.shared_forward(model, batch)
        dual = model(batch)
        for key in ("loss_clip", "loss_commitment", "eeg_emb_vector"):
            torch.testing.assert_close(dual[key], original[key], rtol=0, atol=0)
        if hasattr(model, "attribute"):
            assert dual["memory_mask"][:, 0].all()
            torch.testing.assert_close(dual["eeg_embeds"][:,1:], original["eeg_embeds"])
        swapped = model({**batch, "sentiment label": ["nan","nan"]})
        torch.testing.assert_close(dual["logits_lm"], swapped["logits_lm"], rtol=0, atol=0)
        torch.testing.assert_close(dual["eeg_embeds"], swapped["eeg_embeds"], rtol=0, atol=0)
        generation = model.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=dual["eeg_embeds"]),
            attention_mask=dual.get("memory_mask"), max_length=16, num_beams=2)
        generated_text = model.tokenizer.batch_decode(generation, skip_special_tokens=True)
    del original, dual, generation
    model.train()
    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    old_codes = model.vq.codebook.weight.detach().clone() if hasattr(model,"vq") else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        out = model(batch)
        loss = out["loss_total"] if "loss_total" in out else model.λ*out["loss_clip"]+(1-model.λ)*out["loss_lm"]+model.ε*out["loss_commitment"]
    if not all(torch.isfinite(value).all() for key, value in out.items() if key.startswith("loss")):
        raise ValueError("Nonfinite loss")
    if any(out[k].shape != (2,3) for k in ("cont_logits","disc_logits") if k in out):
        raise ValueError("Emotion logits shape mismatch")
    if hasattr(model, "attribute") and model.hparams.detach_probs:
        assert torch.autograd.grad(out["loss_lm"], out["disc_logits"], retain_graph=True, allow_unused=True)[0] is None
    loss.backward()
    modules = {"encoder": model.eeg_encoder}
    if old_codes is not None:
        modules.update(codebook=model.vq.codebook, discrete_head=model.disc_head)
    if hasattr(model, "attribute"):
        modules["attribute"] = model.attribute
    for name, module in modules.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all(g.isfinite().all() for g in gradients) or not any(g.any() for g in gradients):
            raise ValueError(f"Missing or invalid {name} gradient")
    optimizer.step()
    if old_codes is not None and torch.equal(old_codes, model.vq.codebook.weight):
        raise ValueError("VQ codebook did not update")
    report = {
        "kind": "REAL_ZUCO_SMOKE", "status": "PASS", "variant": model.variant, "tensor_only": args.tensor_only, "rows": n_rows,
        "data_path": str(args.data_path.resolve()), "labels_path": str(args.labels.resolve()), "step3": args.step3,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "text_model_path": str(args.text_model_path.resolve()),
        "configuration": {key: getattr(model.hparams,key) for key in
                          ("lm_loss_weight","attribute_dim","temperature","gate_init","detach_probs","disc_emotion_weight","cont_emotion_weight","codebook_size", "code_dim", "beta", "emotion_hidden_dim",
                           "emotion_weight", "vq_weight", "clip_loss_weight",
                           "commitment_loss_weight", "lr")},
        "label_distribution_by_phase": distribution,
        "train_sample_text_uids": [int(x) for x in batch["text uid"].tolist()],
        "parameters": {"trainable": trainable_parameters, "frozen": original_parameters},
        "vq_shape": {"input": [2, 96, model.eeg_encoder.out_dim],
                     "code_ids": list(out["code_ids"].shape),
                     "code_dim": model.vq.codebook.embedding_dim} if old_codes is not None else None,
        "losses": {key: float(value.detach().cpu()) for key, value in out.items()
                   if key.startswith("loss")},
        "codebook_perplexity": float(out["codebook_perplexity"]) if old_codes is not None else None,
        "active_code_ratio": float(out["active_code_ratio"]) if old_codes is not None else None,
        "codebook_updated": old_codes is not None,
        "generation": generated_text,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
    }
    args.output.mkdir(parents=True)
    manifest.to_csv(args.output/"split_manifest.csv", index=False)
    (args.output / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
