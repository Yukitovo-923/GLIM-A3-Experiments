"""Strict V3 checkpoint and asset loading for the two-arm experiment."""

import hashlib
import json
from pathlib import Path

import torch

from affect_model import ARMS, AffectGLIM


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["arms"] != list(ARMS):
        raise RuntimeError("Experiment must have only the B0/B1 arms")
    if config["training"]["learning_rate"] != 1e-5:
        raise RuntimeError("This isolated continuation experiment requires LR=1e-5")
    if (config["loss"]["discrete_emotion_weight"] != 0.2 or
            config["loss"]["epa_weight_b1_only"] != 0.2 or
            config["loss"]["epa_temperature"] != 0.07):
        raise RuntimeError("The confirmed V3 + EPA loss configuration changed")
    return config


def verify_assets(config, *, include_large_data=False):
    checks = {
        "checkpoint": (config["checkpoint"]["path"], config["checkpoint"]["sha256"]),
        "labels": (config["assets"]["labels"], config["assets"]["labels_sha256"]),
        "e1_manifest": (config["assets"]["e1_manifest"], config["assets"]["e1_manifest_sha256"]),
        "prototype": (config["assets"]["prototype"], config["assets"]["prototype_sha256"]),
        "text_model_weights": (str(Path(config["assets"]["text_model"]) / "model.safetensors"),
                               config["assets"]["text_model_weight_sha256"]),
    }
    if include_large_data:
        checks["eeg_data"] = (config["assets"]["data"], config["assets"]["data_sha256"])
    results = {}
    for name, (path, expected) in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA256 mismatch: {actual}")
        results[name] = actual
    if not Path(config["assets"]["data"]).is_file():
        raise FileNotFoundError(config["assets"]["data"])
    return results


def read_v3(config):
    package = torch.load(config["checkpoint"]["path"], map_location="cpu", weights_only=False)
    hparams = dict(package["hyper_parameters"])
    if package.get("epoch") != 46 or package.get("global_step") != 93436:
        raise RuntimeError("Wrong V3 checkpoint epoch or step")
    if hparams.get("variant") != "A3-full" or hparams.get("disc_emotion_weight") != 0.2:
        raise RuntimeError("V3 architecture or existing emotion CE differs from confirmed design")
    for key, expected in (("clip_loss_weight", 0.5), ("lm_loss_weight", 0.5),
                          ("vq_weight", 1.0), ("commitment_loss_weight", 0.0),
                          ("vq_distance", "cosine")):
        if hparams.get(key) != expected:
            raise RuntimeError(f"Unexpected V3 {key}: {hparams.get(key)}")
    training = config["training"]
    # The immutable source was trained at 1e-4. Its original hyperparameters
    # must remain verified even though this continuation deliberately uses 1e-5.
    for key, expected in (("lr", 1e-4),
                          ("weight_decay", training["weight_decay"]),
                          ("bsz_train", training["batch_size_train"]),
                          ("bsz_val", training["batch_size_val"]),
                          ("bs_retrieval", training["batch_size_test"]),
                          ("full_val_interval", training["full_val_interval"])):
        if hparams.get(key) != expected:
            raise RuntimeError(f"Training configuration differs from V3 {key}")
    return package


def build_arm(config, package, arm, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    # Only B1's new projector depends on this seed; all V3 tensors come from the same checkpoint.
    torch.manual_seed(seed)
    continuation_hparams = dict(package["hyper_parameters"])
    continuation_hparams["lr"] = config["training"]["learning_rate"]
    model = AffectGLIM(
        arm=arm,
        prototype_path=config["assets"]["prototype"] if arm == "B1_EPA" else None,
        epa_temperature=config["loss"]["epa_temperature"],
        epa_weight=config["loss"]["epa_weight_b1_only"],
        **continuation_hparams,
    )
    if model.lr != 1e-5:
        raise RuntimeError("Continuation model did not receive the locked LR")
    source = package["state_dict"]
    result = model.load_state_dict(source, strict=False)
    expected_missing = ({"epa_projector.weight", "epa_projector.bias", "epa_prototypes"}
                        if arm == "B1_EPA" else set())
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(f"Non-EPA checkpoint incompatibility: {result}")
    current = model.state_dict()
    if any(not torch.equal(current[key].cpu(), value.cpu()) for key, value in source.items()):
        raise RuntimeError("A V3 checkpoint tensor changed during initialization")
    if arm == "B1_EPA" and model.epa_prototypes.requires_grad:
        raise RuntimeError("Text prototypes are not frozen")
    return model
