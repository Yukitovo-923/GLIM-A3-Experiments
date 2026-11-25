"""Reproducible EPA asset checks and one shared E0/E1/E2 initial state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from epa_model import ARMS, EPAExperiment


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_state(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_common(path):
    path = Path(path)
    common = json.loads(path.read_text(encoding="utf8"))
    for field in ("model", "epa", "training", "assets"):
        if field not in common:
            raise ValueError(f"Missing common configuration: {field}")
    return common


def validate_asset_hashes(common, include_large_text_weights=False):
    assets = common["assets"]
    for name in ("released_glim", "vq_initialization", "vq_full_initialization",
                 "prototype", "e1_manifest"):
        path = Path(assets[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hash_file(path)
        if actual != assets[f"{name}_sha256"]:
            raise RuntimeError(f"{name} SHA256 mismatch: {actual}")
    text_path = Path(assets["text_model"]) / "model.safetensors"
    if not text_path.is_file():
        raise FileNotFoundError(text_path)
    if include_large_text_weights and hash_file(text_path) != assets["text_model_weight_sha256"]:
        raise RuntimeError("Flan-T5 weight SHA256 mismatch")
    for name in ("data", "labels"):
        if not Path(assets[name]).is_file():
            raise FileNotFoundError(assets[name])


def model_configuration(common, released_hparams):
    t = common["training"]
    return {**dict(released_hparams), **common["model"],
            "bsz_train": t["batch_size_train"], "bsz_val": t["batch_size_val"],
            "bs_retrieval": t["batch_size_test"], "lr": t["learning_rate"],
            "weight_decay": t["weight_decay"], "full_val_interval": t["full_val_interval"]}


def construct_model(common, model_config, arm, metrics_dir=None):
    return EPAExperiment(arm=arm, prototypes_path=common["assets"]["prototype"],
                         prototype_sha256=common["assets"]["prototype_sha256"],
                         epa_tau=common["epa"]["tau"],
                         emotion_aux_weight=common["epa"]["emotion_aux_weight"],
                         metrics_dir=metrics_dir, **model_config)


def prepare(common_path, output):
    common = load_common(common_path)
    validate_asset_hashes(common, include_large_text_weights=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(common["training"]["seed"])
    assets = common["assets"]
    released = torch.load(assets["released_glim"], map_location="cpu", weights_only=False)
    full_vq = torch.load(assets["vq_full_initialization"], map_location="cpu", weights_only=False)
    vq_only = torch.load(assets["vq_initialization"], map_location="cpu", weights_only=False)
    init_metadata = full_vq.get("initialization")
    if not isinstance(init_metadata, dict) or init_metadata.get("initialization") != "spherical_kmeans":
        raise RuntimeError("VQ initialization is not spherical K-means")
    if full_vq.get("source_sha256") != assets["released_glim_sha256"]:
        raise RuntimeError("VQ full initialization came from a different GLIM checkpoint")
    original = released["state_dict"]
    if any(k not in full_vq["state_dict"] or not torch.equal(v, full_vq["state_dict"][k])
           for k, v in original.items()):
        raise RuntimeError("VQ initialization changed published EEG Encoder/Aligner weights")
    if set(vq_only) != {"projection.weight", "projection.bias", "codebook.weight"}:
        raise RuntimeError("Unexpected VQ initialization keys")
    if any(not torch.equal(v, full_vq["state_dict"]["vq." + k]) for k, v in vq_only.items()):
        raise RuntimeError("VQ-only and full original initialization disagree")
    config = model_configuration(common, released["hyper_parameters"])
    model = construct_model(common, config, "E0")
    state = model.state_dict()
    if any(k not in state or state[k].shape != v.shape for k, v in original.items()):
        raise RuntimeError("Published GLIM weights are not compatible with EPA model")
    for key, value in original.items():
        state[key] = value.detach().clone()
    for key, value in vq_only.items():
        state["vq." + key] = value.detach().clone()
    model.load_state_dict(state, strict=True)
    if hasattr(model, "attribute") or not torch.equal(model.epa_prototypes.cpu(),
                                                       torch.from_numpy(np.load(assets["prototype"]))):
        raise RuntimeError("Attribute token present or prototype mismatch")
    digest = hash_state(model.state_dict())
    output.mkdir(parents=True)
    path = output / "common_init.pt"
    torch.save({"kind": "EPA_COMMON_INITIALIZATION", "state_dict": model.state_dict(),
                "model_config": config, "common_config": common,
                "tensor_sha256": digest}, path)
    audit = {"status": "PASS", "seed": t if (t := common["training"]["seed"]) is not None else None,
             "common_init_path": str(path.resolve()), "common_init_sha256": hash_file(path),
             "tensor_sha256": digest, "state_keys": len(state),
             "released_glim_keys_preserved": len(original), "vq_keys_preserved": len(vq_only),
             "has_attribute_token": False, "arms": list(ARMS),
             "all_arms_load_the_same_state_file": True,
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "frozen_prototype_values": int(model.epa_prototypes.numel()),
             "prototypes_sha256": assets["prototype_sha256"],
             "prototype_space": assets["prototype_space"],
             "source": {name: assets[name + "_sha256"] for name in
                        ("released_glim", "vq_initialization", "vq_full_initialization", "e1_manifest")}}
    (output / "initialization_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    return audit


def load_model_for_arm(common_path, init_path, arm, metrics_dir=None):
    if arm not in ARMS:
        raise ValueError(arm)
    common = load_common(common_path)
    package = torch.load(init_path, map_location="cpu", weights_only=False)
    if package.get("kind") != "EPA_COMMON_INITIALIZATION":
        raise RuntimeError("Not an EPA common initialization")
    if package["common_config"] != common:
        raise RuntimeError("Common configuration differs from saved initialization")
    if hash_state(package["state_dict"]) != package["tensor_sha256"]:
        raise RuntimeError("Common initialization tensor digest mismatch")
    model = construct_model(common, package["model_config"], arm, metrics_dir)
    model.load_state_dict(package["state_dict"], strict=True)
    if hash_state(model.state_dict()) != package["tensor_sha256"]:
        raise RuntimeError("An EPA arm did not restore the exact common weights")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.common, args.output), indent=2))
