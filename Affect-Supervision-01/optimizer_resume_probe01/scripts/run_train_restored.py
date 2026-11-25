"""Train one B0/B1 seed after a passing real-batch preflight; never evaluates test."""

import argparse
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
from probe_optimizer import audit_optimizer


class RestoreAdam(Callback):
    def __init__(self, source_state, evidence_path):
        self.source_state = source_state
        self.evidence_path = Path(evidence_path)

    def on_train_start(self, trainer, pl_module):
        if len(trainer.optimizers) != 1 or trainer.global_step != 0:
            raise RuntimeError("Cannot restore source Adam at the start of training")
        optimizer = trainer.optimizers[0]
        optimizer.load_state_dict(self.source_state)
        steps = [int(item["step"].item()) for item in optimizer.state.values()]
        if len(steps) != 214 or min(steps) != 93436 or max(steps) != 93436:
            raise RuntimeError("Restored Adam step/count differs from V3 checkpoint")
        self.evidence_path.write_text(json.dumps({"status": "PASS", "parameters": len(steps),
            "step_min": min(steps), "step_max": max(steps),
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed not in config["training"]["seeds"]:
        parser.error("Seed not in locked common configuration")
    root = args.config.resolve().parent.parent
    expected_parent = (root / "runs" / args.arm).resolve()
    if args.output.resolve().parent != expected_parent:
        parser.error("Output must be one seed directory inside the matching new arm directory")
    if args.output.exists():
        parser.error("Output already exists; refusing to overwrite a previous run")
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    if (smoke.get("kind") != "AFFECT_SUPERVISION_01_REAL_BATCH" or
            smoke.get("status") != "PASS" or
            set(smoke.get("arms", {})) != {"B0_Baseline", "B1_EPA"} or
            any(item.get("status") != "PASS" for item in smoke["arms"].values()) or
            smoke.get("source_hashes", {}).get("checkpoint") != config["checkpoint"]["sha256"]):
        parser.error("The required B0/B1 real-batch smoke test has not passed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A800 BF16 is unavailable")
    source_hashes = verify_assets(config, include_large_data=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    L.seed_everything(args.seed, workers=True)
    package = read_v3(config)
    model = build_arm(config, package, args.arm, args.seed)
    if args.arm != "B0_Baseline":
        parser.error("This probe restores Adam only for unchanged B0 model")
    optimizer_audit = audit_optimizer(model, package, config)
    if optimizer_audit["status"] != "PASS":
        raise RuntimeError("Optimizer state mapping audit failed")
    source_optimizer_state = package["optimizer_states"][0]
    # The B1 projector consumes random draws; reset stochastic training state for both arms.
    L.seed_everything(args.seed, workers=True)
    args.output.mkdir(parents=True)
    run_config = {"arm": args.arm, "seed": args.seed, "common": config,
                  "source_hashes": source_hashes,
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
        RestoreAdam(source_optimizer_state, args.output / "restored_optimizer_at_start.json"),
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
