"""Train one B0/B1 seed after a passing real-batch preflight; never evaluates test."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from deployment.local_logging import LocalWandbLogger
from experiment_assets import build_arm, load_config, read_v3, sha256_file, verify_assets
from task1_data import Task1DataModule


def source_optimizer_for_arm(package, model, arm):
    """Map the unchanged 214 V3 parameters positionally; leave B1 projector fresh."""
    source = package["optimizer_states"]
    if len(source) != 1 or len(source[0]["param_groups"]) != 1:
        raise RuntimeError("Expected one source V3 Adam parameter group")
    # torch.optim may update loaded step tensors in place; isolate every run.
    state = deepcopy(source[0])
    group = state["param_groups"][0]
    ids = group["params"]
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    expected_extra = [] if arm == "B0_Baseline" else ["epa_projector.weight", "epa_projector.bias"]
    if len(ids) != 214 or len(named) != 214 + len(expected_extra) or [n for n, _ in named[214:]] != expected_extra:
        raise RuntimeError("B0/B1 trainable parameter ordering differs from V3")
    if (group["lr"] != 1e-4 or group["weight_decay"] != 0 or
            tuple(group["betas"]) != (0.9, 0.999) or group["eps"] != 1e-8):
        raise RuntimeError("Source Adam hyperparameters differ")
    steps = []
    for (name, parameter), ident in zip(named[:214], ids):
        slots = state["state"].get(ident)
        if slots is None or any(tuple(slots[k].shape) != tuple(parameter.shape)
                                for k in ("exp_avg", "exp_avg_sq")):
            raise RuntimeError(f"Adam slot shape mismatch at {name}")
        steps.append(int(slots["step"]))
    if min(steps) != 93436 or max(steps) != 93436:
        raise RuntimeError("Source Adam step differs from V3 checkpoint")
    if not expected_extra:
        mapped = state
    else:
        new_ids = [max(ids) + 1, max(ids) + 2]
        if any(ident in state["state"] for ident in new_ids):
            raise RuntimeError("New EPA parameter ID collides with source state")
        mapped = {"state": state["state"], "param_groups": [dict(group, params=ids + new_ids)]}
    audit = {"status": "PASS", "arm": arm, "source_parameters": 214,
             "new_projector_parameters": len(expected_extra), "source_adam_step": 93436,
             "source_names_and_shapes_sha256": hashlib.sha256(json.dumps(
                 [(n, list(p.shape)) for n, p in named[:214]],
                 separators=(",", ":")).encode()).hexdigest(),
             "projector_names": expected_extra}
    return mapped, audit


class RecordOptimizerStart(Callback):
    def __init__(self, condition, mapped_state, output):
        self.condition = condition
        self.mapped_state = mapped_state
        self.output = Path(output)

    def on_train_start(self, trainer, pl_module):
        if len(trainer.optimizers) != 1 or trainer.global_step != 0:
            raise RuntimeError("Optimizer initialization occurred after training began")
        optimizer = trainer.optimizers[0]
        if self.condition == "restored":
            optimizer.load_state_dict(self.mapped_state)
        n_state = len(optimizer.state)
        expected = 214 if self.condition == "restored" else 0
        if n_state != expected:
            raise RuntimeError(f"Expected {expected} Adam parameter states, got {n_state}")
        steps = [int(item["step"]) for item in optimizer.state.values()]
        if steps and (min(steps) != 93436 or max(steps) != 93436):
            raise RuntimeError("Loaded Adam step changed")
        self.output.write_text(json.dumps({"condition": self.condition,
            "status": "PASS", "parameter_states_at_start": n_state,
            "source_step": 93436 if steps else None,
            "learning_rate": optimizer.param_groups[0]["lr"]}, indent=2), encoding="utf-8")


class SaveEpochMetrics(Callback):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        self.directory.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (float, int)):
                metrics[key] = float(value)
        path = self.directory / f"epoch_{trainer.current_epoch:02d}.json"
        path.write_text(json.dumps({"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                                    "learning_rate": pl_module.lr,
                                    "epoch_elapsed_seconds": time.monotonic() - self.epoch_start
                                    if self.epoch_start is not None else None,
                                    "metrics": metrics}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("B0_Baseline", "B1_EPA"), required=True)
    parser.add_argument("--condition", choices=("reset", "restored"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed not in config["training"]["seeds"]:
        parser.error("Seed not in locked common configuration")
    root = args.config.resolve().parent.parent
    expected_parent = (root / "runs" / args.condition / args.arm).resolve()
    if args.output.resolve().parent != expected_parent:
        parser.error("Output must be one seed directory inside the matching new arm directory")
    if args.output.exists():
        parser.error("Output already exists; refusing to overwrite a previous run")
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    if (smoke.get("status") != "PASS" or
            len(smoke.get("rows", [])) != 4 or
            {(item.get("arm"), item.get("condition")) for item in smoke["rows"]} !=
            {(arm, condition) for arm in ("B0_Baseline", "B1_EPA")
             for condition in ("reset", "restored")} or
            smoke.get("source_hashes", {}).get("checkpoint") != config["checkpoint"]["sha256"]):
        parser.error("The required four-cell optimizer smoke test has not passed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A800 BF16 is unavailable")
    source_hashes = verify_assets(config, include_large_data=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    L.seed_everything(args.seed, workers=True)
    package = read_v3(config)
    model = build_arm(config, package, args.arm, args.seed)
    mapped_optimizer, optimizer_audit = source_optimizer_for_arm(package, model, args.arm)
    # The B1 projector consumes random draws; reset stochastic training state for both arms.
    L.seed_everything(args.seed, workers=True)
    args.output.mkdir(parents=True)
    run_config = {"arm": args.arm, "condition": args.condition,
                  "seed": args.seed, "common": config,
                  "source_hashes": source_hashes,
                  "optimizer_mapping": optimizer_audit,
                  "smoke_report_sha256": sha256_file(args.smoke_report),
                  "v3_shared_keys_loaded": len(package["state_dict"])}
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2),
                                                    encoding="utf-8")
    data = Task1DataModule(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        audit_dir=args.output, seed=args.seed,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"],
        num_workers=config["training"]["num_workers"],
    )
    logger = LocalWandbLogger(project="GLIM-Affect-Supervision-01", group=args.arm,
                              name=f"{args.arm}-seed-{args.seed}",
                              save_dir=str(args.output), offline=True)
    early = config["training"]["early_stopping"]
    callbacks = [
        RecordOptimizerStart(args.condition, mapped_optimizer,
                             args.output / "optimizer_start.json"),
        ModelCheckpoint(dirpath=args.output / "checkpoints", save_last=True,
                        save_top_k=1, monitor=early["monitor"], mode=early["mode"],
                        every_n_epochs=1),
        EarlyStopping(monitor=early["monitor"], mode=early["mode"],
                      patience=early["patience"], min_delta=early["min_delta"]),
        SaveEpochMetrics(args.output / "metrics"),
    ]
    trainer = L.Trainer(
        accelerator="gpu", devices=1, precision=config["training"]["precision"],
        max_epochs=config["training"]["max_epochs"],
        use_distributed_sampler=False, num_sanity_val_steps=0,
        default_root_dir=str(args.output), logger=logger, callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data)
    (args.output / "fit_complete.json").write_text(
        json.dumps({"arm": args.arm, "seed": args.seed,
                    "epoch_reached": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "best_model_path": callbacks[1].best_model_path,
                    "best_model_score": float(callbacks[1].best_model_score)
                    if callbacks[1].best_model_score is not None else None,
                    "stopped_early": callbacks[2].stopped_epoch > 0},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
