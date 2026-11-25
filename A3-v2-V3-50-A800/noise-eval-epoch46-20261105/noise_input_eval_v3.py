"""Read-only, free-generation EEG versus Gaussian-noise validation control.

Run from the existing GLIM project root. This script never trains or writes to
the project, checkpoint, data, or source directories.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput

from data.datamodule import GLIMSampler, ZuCoDataset
from data.revised_labels import audit
from model.glim_multitask import GLIMMultitask


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scores(model, predictions, targets, raw_texts):
    metrics, _ = model.cal_gen_metrics(predictions, targets, raw_texts)
    return [{"bleu1_mtv": float(metrics["bleu1"][i]),
             "bleu2_mtv": float(metrics["bleu2"][i]),
             "rouge1_mtv": float(metrics["rouge1_recall"][i]),
             "wer_raw": float(metrics["wer"][i])}
            for i in range(len(predictions))]


def make_noise(real, mask, generator, signal_channels):
    if real.ndim != 3 or mask.shape != real.shape[:2]:
        raise ValueError(f"Unexpected EEG/mask shape: {real.shape}, {mask.shape}")
    if signal_channels > real.shape[-1]:
        raise ValueError("signal_channels exceeds EEG width")
    noise = real.clone()
    draw = torch.randn(real.shape, dtype=real.dtype, generator=generator)
    valid = mask.bool().unsqueeze(-1).expand_as(real).clone()
    valid[..., signal_channels:] = False  # STEP3 zero-padded channels
    noise[valid] = draw[valid]
    if not torch.equal(noise[~valid], real[~valid]):
        raise AssertionError("EEG padding changed")
    return noise, valid


@torch.inference_mode()
def encode_and_generate(model, batch, eeg, device):
    """The original EEG -> aligner -> predicted attribute -> T5 generate path.

    No target ids, source text encoder embeddings, or teacher-forced logits are
    supplied to the decoder. The same path runs for real EEG and noise.
    """
    mask = batch["mask"].to(device)
    prompts = batch["prompt"]
    prompt_ids = model.p_embedder.encode(prompts, device=device)
    prompt_emb = model.p_embedder(prompt_ids, model.eval_pembed)
    hidden, _ = model.eeg_encoder(eeg.to(device), mask, prompt_emb)
    memory, _ = model.aligner.embed_eeg(hidden)
    extra = {}
    if hasattr(model, "condition_memory") and hasattr(model, "attribute"):
        memory, extra = model.condition_memory(hidden, memory, mask,
                                               batch["raw task key"], prompts[1])
    ids = model.text_model.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=extra.get("memory_mask"), num_beams=2,
        min_length=0, max_length=model.tgt_text_len)
    return model.tokenizer.batch_decode(ids, skip_special_tokens=True), {
        "hidden": hidden.detach(), "memory": memory.detach(),
        "disc_logits": extra.get("disc_logits"),
        "memory_mask": extra.get("memory_mask"),
    }


def batch_references(batch):
    return [tuple(item) for item in zip(*batch["all target texts"])]


def running_stats(values):
    x = values.double()
    return float(x.sum()), float((x * x).sum()), int(x.numel())


def finish_stats(triple):
    total, squares, count = triple
    mean = total / count
    return {"count": count, "mean": mean,
            "std": math.sqrt(max(0.0, squares / count - mean * mean))}


def evaluation_pass(model, loader, out_dir, name, seed, device, signal_channels,
                    expected_keys=None):
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    rows, keys = [], []
    stat = (0.0, 0.0, 0)
    start = time.time()
    path = out_dir / f"samples_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as stream:
        for batch_idx, batch in enumerate(loader):
            real = batch["eeg"]
            mask = batch["mask"]
            valid = mask.bool().unsqueeze(-1).expand_as(real).clone()
            valid[..., signal_channels:] = False
            if seed is None:
                eeg = real
            else:
                eeg, _ = make_noise(real, mask, generator, signal_channels)
            part = running_stats(eeg[valid])
            stat = tuple(a + b for a, b in zip(stat, part))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated, _ = encode_and_generate(model, batch, eeg, device)
            refs = batch_references(batch)
            raw = list(batch["raw input text"])
            batch_scores = scores(model, generated, refs, raw)
            for i, (prediction, metrics) in enumerate(zip(generated, batch_scores)):
                uid = int(batch["text uid"][i])
                key = (uid, batch["prompt"][1][i], batch["prompt"][2][i],
                       batch["raw task key"][i], raw[i])
                keys.append(key)
                row = {"sample_index": len(rows), "text_uid": uid,
                       "dataset": key[1], "subject": key[2], "task": key[3],
                       "raw_text": raw[i], "references": refs[i],
                       "generated_text": prediction, **metrics}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            stream.flush()
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
                print(f"{name}: {batch_idx+1}/{len(loader)} batches, {len(rows)} samples, {time.time()-start:.1f}s", flush=True)
    if expected_keys is not None and keys != expected_keys:
        raise AssertionError(f"Sample/order mismatch for {name}")
    means = {k: float(np.mean([row[k] for row in rows])) for k in
             ("bleu1_mtv", "bleu2_mtv", "rouge1_mtv", "wer_raw")}
    result = {"condition": name, "seed": seed, "sample_count": len(rows),
              "batch_count": len(loader), "metrics": means,
              "input_distribution_valid_channels": finish_stats(stat),
              "elapsed_seconds": time.time()-start, "samples_path": str(path)}
    dump(out_dir / f"summary_{name}.json", result)
    return rows, keys, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--text-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--signal-channels", type=int, default=104)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.smoke_only and not (args.output / "smoke_report.json").exists():
            parser.error("Existing output directory lacks a passing smoke report")
    else:
        args.output.mkdir(parents=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = str(args.text_model_path.resolve())
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    checkpoint_sha = digest(args.checkpoint)
    model = GLIMMultitask.from_glim_checkpoint(str(args.checkpoint))
    if model.variant != "A3-full" or model.hparams.vq_distance != "cosine":
        raise RuntimeError(f"Expected V3 A3-full cosine checkpoint, got {model.variant}, {model.hparams.vq_distance}")
    model.to(device)
    model.setup("test")
    model.eval()
    frame, manifest = audit(pd.read_pickle(args.data_path), pd.read_pickle(args.labels))
    manifest.to_csv(args.output / "split_manifest.csv", index=False)
    dataset = ZuCoDataset(frame, "val")
    loader = DataLoader(dataset, batch_sampler=GLIMSampler(dataset, dataset.data["text uid"],
                                                          "val", args.batch_size), num_workers=0)
    batch = next(iter(loader))
    real = batch["eeg"]
    mask = batch["mask"]
    assert real.shape[1:] == (1280, 128) and mask.shape == real.shape[:2]
    assert torch.count_nonzero(real[..., args.signal_channels:]) == 0, "Padded channels not zero"
    noise, valid = make_noise(real, mask, torch.Generator().manual_seed(args.seeds[0]), args.signal_channels)
    assert torch.any(noise[valid] != real[valid])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        real_text, real_rep = encode_and_generate(model, batch, real, device)
        noise_text, noise_rep = encode_and_generate(model, batch, noise, device)
    hidden_delta = float((real_rep["hidden"]-noise_rep["hidden"]).float().abs().mean())
    memory_delta = float((real_rep["memory"]-noise_rep["memory"]).float().abs().mean())
    disc_delta = float((real_rep["disc_logits"]-noise_rep["disc_logits"]).float().abs().mean())
    assert hidden_delta > 0 and memory_delta > 0 and disc_delta > 0
    assert torch.equal(real_rep["memory_mask"], noise_rep["memory_mask"])
    smoke = {"status": "PASS", "batch_size": len(real), "real_eeg_shape": list(real.shape),
             "valid_time_counts": mask.sum(1).tolist(), "padding_preserved": True,
             "hidden_mean_absolute_delta": hidden_delta, "memory_mean_absolute_delta": memory_delta,
             "disc_logits_mean_absolute_delta": disc_delta,
             "attribute_memory_mask_identical": True,
             "real_generated_first": real_text[0], "noise_generated_first": noise_text[0],
             "checkpoint_sha256": checkpoint_sha}
    dump(args.output / "smoke_report.json", smoke)
    print("SMOKE PASS", json.dumps(smoke, ensure_ascii=False), flush=True)
    if args.smoke_only:
        return
    config = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
              "data_path": str(args.data_path.resolve()), "labels": str(args.labels.resolve()),
              "text_model_path": str(args.text_model_path.resolve()), "split": "val",
              "seeds": args.seeds, "batch_size": args.batch_size,
              "noise": "N(0,1) on valid time frames and 104 signal channels; original padding preserved",
              "decoding": {"method": "generate", "num_beams": 2, "min_length": 0,
                           "max_length": model.tgt_text_len, "teacher_forcing": False},
              "model_variant": model.variant, "vq_distance": model.hparams.vq_distance,
              "sample_count_expected": len(dataset), "batch_count": len(loader)}
    dump(args.output / "eval_config.json", config)
    eeg_rows, keys, eeg_result = evaluation_pass(model, loader, args.output, "eeg", None,
                                                  device, args.signal_channels)
    noise_results = []
    contrasts = []
    metric_keys = ("bleu1_mtv", "bleu2_mtv", "rouge1_mtv", "wer_raw")
    for seed in args.seeds:
        name = f"noise_seed{seed}"
        noise_rows, _, result = evaluation_pass(model, loader, args.output, name, seed,
                                                 device, args.signal_channels, keys)
        noise_results.append(result)
        for i, (eeg_row, noise_row) in enumerate(zip(eeg_rows, noise_rows)):
            contrasts.append({"sample_index": i, "seed": seed,
                              **{f"delta_{k}": eeg_row[k]-noise_row[k] for k in metric_keys}})
    with open(args.output / "paired_deltas.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    rng = np.random.default_rng(20261105)
    comparisons = {}
    for key in metric_keys:
        eeg_values = np.array([row[key] for row in eeg_rows])
        per_seed = np.array([[row[f"delta_{key}"] for row in contrasts if row["seed"] == seed]
                             for seed in args.seeds])
        paired_mean = per_seed.mean(axis=0)
        indices = rng.integers(0, len(eeg_rows), size=(1000, len(eeg_rows)))
        boot = paired_mean[indices].mean(axis=1)
        comparisons[key] = {"eeg": float(eeg_values.mean()),
                            "noise_mean": float(np.mean([r["metrics"][key] for r in noise_results])),
                            "noise_seed_std": float(np.std([r["metrics"][key] for r in noise_results], ddof=1)),
                            "delta_eeg_minus_noise": float(paired_mean.mean()),
                            "paired_bootstrap_95pct_ci": np.quantile(boot, [.025, .975]).tolist()}
    summary = {"status": "PASS", "checkpoint_sha256": checkpoint_sha,
               "eeg": eeg_result, "noise": noise_results, "comparisons": comparisons,
               "bootstrap": "1000 paired resamples of validation samples; noise averaged across seeds"}
    dump(args.output / "comparison.json", summary)
    lines = ["# V3-50 epoch 46: EEG vs Gaussian noise (validation)", "",
             f"Checkpoint SHA256: `{checkpoint_sha}`", "",
             "Free generation with beam=2; no target tokens are passed to the decoder.", "",
             "| Metric | EEG | Noise mean ± seed SD | EEG − Noise | Paired bootstrap 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for key, item in comparisons.items():
        lo, hi = item["paired_bootstrap_95pct_ci"]
        lines.append(f"| {key} | {item['eeg']:.5f} | {item['noise_mean']:.5f} ± {item['noise_seed_std']:.5f} | {item['delta_eeg_minus_noise']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ["", "WER is scored against the raw source sentence; other metrics use eight target variants.",
              "Confidence intervals resample observations and do not prove causal EEG dependence."]
    (args.output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("EVALUATION PASS", json.dumps(comparisons, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
