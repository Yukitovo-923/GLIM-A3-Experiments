"""Single GPU Lightning entry point for the four GLIM experiments."""
import argparse
import json
import os
from pathlib import Path

from model.glim_multitask import GLIMMultitask, REVISED, VARIANTS
from data.revised_labels import ROOT


def local_asset(relative):
    for root in Path(__file__).resolve().parents:
        path = root / relative
        if path.exists():
            return str(path)
    return None


def main():
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from data.datamodule import GLIMDataModule
    from deployment.local_logging import LocalWandbLogger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "test"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--checkpoint", default=os.environ.get("GLIM_CHECKPOINT") or
                        local_asset("glim-zuco-epoch=199-step=49600.ckpt"))
    parser.add_argument("--text-model-path", default=os.environ.get("GLIM_TEXT_MODEL_PATH") or
                        local_asset("models/flan-t5-large"))
    parser.add_argument("--data-path", default=os.environ.get("GLIM_DATA_PATH", "data/tmp/zuco_eeg_label_8variants.df"))
    parser.add_argument("--output", type=Path, required=True, help="A new run directory")
    parser.add_argument("--smoke-report", type=Path, help="Passing real ZuCo smoke report required for dual fit")
    parser.add_argument("--config", type=Path, help="Model hyperparameters in JSON")
    parser.add_argument("--labels", type=Path, default=ROOT/"data/tmp/zuco_label_8variants.df")
    parser.add_argument("--epochs", type=int, default=200)
    for key, kind in (("emotion-weight", float), ("vq-weight", float), ("beta", float),
                      ("clip-loss-weight", float), ("commitment-loss-weight", float),
                      ("codebook-size", int), ("code-dim", int), ("emotion-hidden-dim", int),
                      ("lr", float)):
        parser.add_argument("--" + key, type=kind)
    args = parser.parse_args()
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        parser.error("Provide an existing GLIM --checkpoint")
    if not args.text_model_path or not Path(args.text_model_path).is_dir():
        parser.error("Provide the local Flan-T5 directory via --text-model-path")
    if not Path(args.data_path).is_file():
        parser.error("Provide the real paired EEG dataframe via --data-path")
    os.environ["GLIM_TEXT_MODEL_PATH"] = str(Path(args.text_model_path).resolve())
    L.seed_everything(42, workers=True)
    overrides = {k: v for k, v in vars(args).items() if k in {
        "emotion_weight", "vq_weight", "beta", "clip_loss_weight", "commitment_loss_weight",
        "codebook_size", "code_dim", "emotion_hidden_dim", "lr"} and v is not None}
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    variant = args.variant or config.get("variant")
    config.pop("variant", None)
    model = GLIMMultitask.from_glim_checkpoint(args.checkpoint, variant, **{**config, **overrides})
    if args.mode == "fit" and model.variant in ("dual", *REVISED):
        if not args.smoke_report or not args.smoke_report.is_file():
            parser.error("Run smoke_a3.py on real ZuCo data and supply --smoke-report before dual fit")
        report = json.loads(args.smoke_report.read_text(encoding="utf-8"))
        expected_paths = {
            "data_path": Path(args.data_path).resolve(),
            "checkpoint_path": Path(args.checkpoint).resolve(),
            "text_model_path": Path(args.text_model_path).resolve(),
        }
        configuration_keys = ("codebook_size", "code_dim", "beta", "emotion_hidden_dim",
                              "emotion_weight", "vq_weight", "clip_loss_weight",
                              "commitment_loss_weight", "lr")
        if model.variant in REVISED:
            configuration_keys += ("lm_loss_weight","attribute_dim","temperature","gate_init","detach_probs","disc_emotion_weight","cont_emotion_weight")
            expected_paths["labels_path"] = args.labels.resolve()
            if report.get("tensor_only") or report.get("step3"):
                parser.error("Server fit requires a framework smoke report on the paired training file")
        if (report.get("kind") != "REAL_ZUCO_SMOKE" or report.get("status") != "PASS"
                or report.get("variant") != model.variant
                or any(Path(report.get(key, "")).resolve() != value for key, value in expected_paths.items())
                or any(report.get("configuration", {}).get(key) != model.hparams[key]
                       for key in configuration_keys)):
            parser.error("Smoke report does not match this A3 experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    logger = LocalWandbLogger(project="glim", group=model.variant, save_dir=str(args.output), offline=True)
    dm = GLIMDataModule(data_path=args.data_path, bsz_train=model.bsz_train,
                        bsz_val=model.bsz_val, bsz_test=model.bsz_retrieval, num_workers=4)
    if model.variant in REVISED:
        dm.label_reference = args.labels
        dm.manifest_path = args.output / "split_manifest.csv"
    trainer = L.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", logger=logger,
                        max_epochs=args.epochs, use_distributed_sampler=False, num_sanity_val_steps=0,
                        callbacks=[ModelCheckpoint(dirpath=args.output / "checkpoints", save_last=True)])
    getattr(trainer, args.mode)(model, datamodule=dm)


if __name__ == "__main__":
    main()
