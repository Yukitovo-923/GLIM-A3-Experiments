"""B0 short continuation: reset Adam versus compatible V3 Adam state.

Only ZuCo1 Task1 train/validation are accessed. The source checkpoint is read only.
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import default_collate

from experiment_assets import build_arm, load_config, read_v3, verify_assets
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]
STEPS = (0, 1, 10, 50, 100, 367)


def scalar(value):
    return value.item() if isinstance(value, torch.Tensor) else value


def audit_optimizer(model, package, config):
    """Validate positional mapping before loading Adam state into B0."""
    states = package.get("optimizer_states", [])
    if len(states) != 1:
        raise RuntimeError(f"Expected one source optimizer, got {len(states)}")
    saved = states[0]
    groups = saved["param_groups"]
    names_and_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if len(groups) != 1 or len(groups[0]["params"]) != len(names_and_params):
        raise RuntimeError("Source Adam parameter group/count differs from B0")
    group = groups[0]
    if (group["lr"] != config["training"]["learning_rate"] or
            group["weight_decay"] != config["training"]["weight_decay"] or
            tuple(group["betas"]) != (0.9, 0.999) or group["eps"] != 1e-8):
        raise RuntimeError("Source Adam hyperparameters differ from B0")
    missing, mismatches, steps = [], [], []
    for (name, param), parameter_id in zip(names_and_params, group["params"]):
        state = saved["state"].get(parameter_id)
        if state is None:
            missing.append(name)
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state or tuple(state[key].shape) != tuple(param.shape):
                mismatches.append({"name": name, "slot": key,
                                   "parameter_shape": list(param.shape),
                                   "state_shape": list(state[key].shape) if key in state else None})
        steps.append(int(scalar(state["step"])))
    if missing or mismatches:
        raise RuntimeError(f"Adam state incompatible: missing={missing}, mismatches={mismatches}")
    return {"status": "PASS", "parameter_count": len(names_and_params),
            "names_and_shapes_sha256": hashlib.sha256(json.dumps(
                [(name, list(p.shape)) for name, p in names_and_params],
                separators=(",", ":")).encode()).hexdigest(),
            "first_names": [name for name, _ in names_and_params[:5]],
            "last_names": [name for name, _ in names_and_params[-5:]],
            "source_step_min": min(steps), "source_step_max": max(steps),
            "source_hyperparameters": {key: group[key] for key in
                                       ("lr", "betas", "eps", "weight_decay")}}


def make_data(config, condition, seed):
    return Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        audit_dir=ROOT / "runs" / condition, seed=seed,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"], num_workers=0)


@torch.no_grad()
def evaluate_lm(model, dataset, batch_size):
    """Match the existing Task1 inference protocol; omit generation."""
    was_training = model.training
    model.eval()
    by_subject = defaultdict(list)
    for index, prompt in enumerate(dataset.data["prompt"]):
        by_subject[prompt[2]].append(index)
    loss_sum = 0.0
    tokens = 0
    trials = 0
    for subject, indices in sorted(by_subject.items()):
        chunks = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        if any(len(chunk) < 2 for chunk in chunks):
            raise RuntimeError(f"Invalid validation batch for {subject}")
        for chunk in chunks:
            batch = default_collate([dataset[index] for index in chunk])
            batch = {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor)
                     else value for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            n_tokens = int((out["target_text_ids"] != model.text_model.config.pad_token_id).sum())
            loss_sum += float(out["loss_lm"].detach().float().cpu()) * n_tokens
            tokens += n_tokens
            trials += len(chunk)
            del out, batch
    model.train(was_training)
    # Lightning loads frozen T5 during setup after putting the trainable model
    # in train mode. from_pretrained leaves T5 in eval mode for epoch zero.
    model.text_model.eval()
    if trials != 406:
        raise RuntimeError(f"Validation trial count changed: {trials}")
    return {"lm_loss": loss_sum / tokens, "target_tokens": tokens,
            "trials": trials, "uids": 40}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("audit", "reset_t5eval", "restored_t5eval"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A800 BF16 GPU required")
    config = load_config(ROOT / "configs" / "common.json")
    if config["training"]["learning_rate"] != 1e-4 or config["data"]["train_text_variants_per_trial"] != 8:
        raise RuntimeError("Locked eight-rewrite, 1e-4 configuration changed")
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    hashes = verify_assets(config, include_large_data=False)
    L.seed_everything(args.seed, workers=True)
    package = read_v3(config)
    model = build_arm(config, package, "B0_Baseline", args.seed)
    audit = audit_optimizer(model, package, config)
    audit.update({"source_hashes": hashes, "source_epoch": package["epoch"],
                  "source_global_step": package["global_step"], "seed": args.seed,
                  "mapping_basis": "same V3 model parameter order; B0 adds no trainable parameters"})
    (ROOT / "reports" / "optimizer_compatibility.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print("OPTIMIZER_AUDIT_PASS", json.dumps(audit), flush=True)
    if args.condition == "audit":
        return
    output = ROOT / "runs" / args.condition
    if (output / "complete.json").exists():
        raise RuntimeError("Completed condition exists; refusing to overwrite")
    model.cuda()
    model.setup("fit")
    if model.text_model.training or any(p.requires_grad for p in model.text_model.parameters()):
        raise RuntimeError("Frozen T5 must begin in eval mode, as in Lightning epoch zero")
    optimizer = model.configure_optimizers()
    if args.condition == "restored_t5eval":
        optimizer.load_state_dict(package["optimizer_states"][0])
    if len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != audit["parameter_count"]:
        raise RuntimeError("Live optimizer mapping differs from audited B0 parameters")
    del package
    data = make_data(config, args.condition, args.seed)
    data.setup("fit")
    if len(data.train_set) != 26528 or len(data.val_set) != 406:
        raise RuntimeError("Train/validation counts changed")
    L.seed_everything(args.seed, workers=True)
    train_loader = data.train_dataloader()
    if len(train_loader) != 367:
        raise RuntimeError(f"Unexpected training batches: {len(train_loader)}")
    records = []
    order_hash = hashlib.sha256()
    started = time.monotonic()
    for step, batch in enumerate(train_loader, 1):
        if step == 1:
            result = evaluate_lm(model, data.val_set, config["training"]["batch_size_val"])
            records.append({"step": 0, **result, "elapsed_seconds": time.monotonic() - started})
            print("EVAL", args.condition, json.dumps(records[-1]), flush=True)
        order_hash.update(batch["text uid"].numpy().tobytes())
        order_hash.update("\n".join(batch["target text"]).encode("utf-8"))
        batch = {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor)
                 else value for key, value in batch.items()}
        model.train()
        model.text_model.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = out["loss_total"]
        loss.backward()
        optimizer.step()
        if step in STEPS:
            result = evaluate_lm(model, data.val_set, config["training"]["batch_size_val"])
            records.append({"step": step, "train_loss_last_batch": float(loss.detach().float().cpu()),
                            **result, "elapsed_seconds": time.monotonic() - started})
            (output / "trajectory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("EVAL", args.condition, json.dumps(records[-1]), flush=True)
        del out, loss, batch
    completion = {"condition": args.condition, "seed": args.seed,
                  "training_order_sha256": order_hash.hexdigest(),
                  "source_checkpoint_sha256": hashes["checkpoint"],
                  "optimizer_compatibility_sha256": hashlib.sha256(json.dumps(audit,
                      sort_keys=True).encode()).hexdigest(),
                  "records": records, "total_elapsed_seconds": time.monotonic() - started}
    (output / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print("COMPLETE", args.condition, completion["training_order_sha256"], flush=True)


if __name__ == "__main__":
    main()
