"""Independent E0/E1/E2 Lightning entry point; requires a passing real-data preflight."""
import argparse
import json
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from deployment.local_logging import LocalWandbLogger
from epa_assets import hash_file, load_common, load_model_for_arm, validate_asset_hashes
from epa_data import EPADataModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "test"))
    parser.add_argument("--arm-config", type=Path, required=True)
    parser.add_argument("--common-init", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Required for test; optional Lightning resume for fit")
    parser.add_argument("--max-epochs", type=int,
                        help="Optional shorter total epoch budget; shared initialization remains unchanged")
    args = parser.parse_args()
    arm_config = json.loads(args.arm_config.read_text(encoding="utf8"))
    arm = arm_config["arm"]
    common_path = (args.arm_config.parent / arm_config["common"]).resolve()
    common = load_common(common_path)
    effective_max_epochs = args.max_epochs if args.max_epochs is not None else common["training"]["max_epochs"]
    if effective_max_epochs < 1 or effective_max_epochs > common["training"]["max_epochs"]:
        parser.error("--max-epochs must be between 1 and the common configuration's upper bound")
    validate_asset_hashes(common, include_large_text_weights=True)
    if args.output.exists():
        parser.error(f"Refusing to reuse output directory: {args.output}")
    if not args.common_init.is_file() or not args.smoke_report.is_file():
        parser.error("Missing common initialization or real-data smoke report")
    if args.mode == "test" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("Test mode needs a trained --checkpoint")
    smoke = json.loads(args.smoke_report.read_text(encoding="utf8"))
    if (smoke.get("kind") != "REAL_ZUCO_EPA_SMOKE" or smoke.get("status") != "PASS"
            or smoke.get("common_init_sha256") != hash_file(args.common_init)
            or set(smoke.get("arm_status", {})) != {"E0", "E1", "E2"}
            or any(value != "PASS" for value in smoke["arm_status"].values())
            or smoke.get("manifest_sha256") != common["assets"]["e1_manifest_sha256"]):
        parser.error("EPA smoke report does not certify this common init, three arms and E1 split")
    os.environ["GLIM_TEXT_MODEL_PATH"] = common["assets"]["text_model"]
    L.seed_everything(common["training"]["seed"], workers=True)
    model = load_model_for_arm(common_path, args.common_init, arm, args.output / "metrics")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run_config.json").write_text(json.dumps({
        "arm": arm, "common": common, "common_init_sha256": hash_file(args.common_init),
        "smoke_report_sha256": hash_file(args.smoke_report),
        "initialization_tensor_sha256": smoke["initialization_tensor_sha256"],
        "effective_max_epochs": effective_max_epochs,
        "resume_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "resume_checkpoint_sha256": hash_file(args.checkpoint) if args.checkpoint else None,
        "checkpoint_policy": {"monitor": "val/loss_lm", "mode": "min",
                              "save_top_k": 1, "save_last": True}},
        ensure_ascii=False, indent=2), encoding="utf8")
    dm = EPADataModule(data_path=common["assets"]["data"],
                       labels_path=common["assets"]["labels"],
                       expected_manifest=common["assets"]["e1_manifest"],
                       audit_dir=args.output,
                       bsz_train=common["training"]["batch_size_train"],
                       bsz_val=common["training"]["batch_size_val"],
                       bsz_test=common["training"]["batch_size_test"],
                       num_workers=common["training"]["num_workers"])
    logger = LocalWandbLogger(project="GLIM-EPA", group="EPA-01-" + arm,
                              save_dir=str(args.output), offline=True)
    callback = ModelCheckpoint(dirpath=args.output / "checkpoints", save_last=True,
                               save_top_k=1, monitor="val/loss_lm", mode="min",
                               every_n_epochs=1)
    trainer = L.Trainer(accelerator="gpu", devices=1,
                        precision=common["training"]["precision"],
                        max_epochs=effective_max_epochs,
                        use_distributed_sampler=False, num_sanity_val_steps=0,
                        logger=logger, callbacks=[callback],
                        default_root_dir=str(args.output))
    resume = str(args.checkpoint) if args.checkpoint else None
    if args.mode == "fit":
        trainer.fit(model, datamodule=dm, ckpt_path=resume)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=resume)


if __name__ == "__main__":
    main()
