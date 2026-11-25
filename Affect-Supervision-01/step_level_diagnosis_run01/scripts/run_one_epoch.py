"""One-epoch B0 Lightning continuation with RNG-neutral step-level evaluation."""

import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, default_collate
from transformers.modeling_outputs import BaseModelOutput

from data.datamodule import ZuCoDataset
from deployment.local_logging import LocalWandbLogger
from experiment_assets import build_arm, read_v3, verify_assets
from task1_data import Task1DataModule


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "common.json"
REQUESTED = (0, 1, 10, 50, 100, 200, 367)
GENERATION = {0, 10, 50, 200, 367}
LOSS_KEYS = ("loss_lm", "loss_clip", "loss_vq", "loss_disc_emotion", "loss_total")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def snapshot_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


class DiagnosticData(Task1DataModule):
    def setup(self, stage):
        if stage != "fit":
            raise RuntimeError("Only the single fit stage is permitted")
        frame = self.load_frame()
        self.train_set = ZuCoDataset(frame, "train")
        self.val_set = ZuCoDataset(frame, "val")
        canonical = frame.loc[frame.phase == "train"].copy()
        canonical.loc[:, "phase"] = "val"
        self.train_canonical_set = ZuCoDataset(canonical, "val")
        self.n_target_text = self.val_set.n_target_text
        if (len(self.train_set), len(self.train_canonical_set), len(self.val_set)) != (26528, 3316, 406):
            raise RuntimeError("Unexpected trial or target expansion count")
        if len(set(self.train_canonical_set.data["text uid"])) != 320:
            raise RuntimeError("Canonical train UID count changed")
        if len(set(self.val_set.data["text uid"])) != 40:
            raise RuntimeError("Validation UID count changed")


def parameter_groups(model):
    result = {}
    for name, parameter in model.named_parameters():
        if name.startswith("eeg_encoder."):
            result[name] = ("encoder", parameter)
        elif name.startswith("aligner."):
            result[name] = ("aligner", parameter)
    if not any(group == "encoder" for group, _ in result.values()):
        raise RuntimeError("EEG Encoder group absent")
    if not any(group == "aligner" for group, _ in result.values()):
        raise RuntimeError("Aligner group absent")
    return result


class StepDiagnosis(Callback):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.rows = []
        self.drifts = []
        self.before_grad = {}
        self.previous_batch = None
        self.source = {}
        self.source_norm = {}
        self.last_step = None
        self.targets = None
        self.generation_steps = None
        self.initial_versions = None
        self.start_time = time.monotonic()

    def on_train_start(self, trainer, model):
        if trainer.global_step != 0 or len(trainer.optimizers) != 1:
            raise RuntimeError("Expected a fresh Adam continuation")
        if trainer.optimizers[0].state:
            raise RuntimeError("Adam state was not reset")
        if trainer.accumulate_grad_batches != 1:
            raise RuntimeError("Gradient accumulation changed")
        model.text_model.eval()
        if any(p.requires_grad for p in model.text_model.parameters()):
            raise RuntimeError("Flan-T5 is not frozen")
        self.last_step = int(trainer.num_training_batches)
        if self.last_step < 200:
            raise RuntimeError(f"Actual DataLoader only provides {self.last_step} steps")
        self.targets = tuple(sorted(set(s for s in REQUESTED if s <= self.last_step) | {self.last_step}))
        self.generation_steps = tuple(sorted(set(s for s in GENERATION if s <= self.last_step) | {self.last_step}))
        groups = parameter_groups(model)
        self.source = {name: p.detach().float().cpu().clone() for name, (_, p) in groups.items()}
        self.source_norm = {group: np.sqrt(sum(float(torch.sum(t.double() ** 2))
                                            for n, t in self.source.items() if groups[n][0] == group))
                            for group in ("encoder", "aligner")}
        write_json(ROOT / "reports" / "runtime_audit.json", {
            "actual_dataloader_batches": self.last_step,
            "planned_optimizer_steps": self.last_step,
            "requested_steps": REQUESTED,
            "executed_steps": self.targets,
            "generation_steps": self.generation_steps,
            "train_target_positions": len(self.data.train_set),
            "canonical_train_trials": len(self.data.train_canonical_set),
            "validation_trials": len(self.data.val_set),
            "adam_states_at_start": len(trainer.optimizers[0].state),
            "gradient_accumulation": trainer.accumulate_grad_batches,
            "text_model_mode": model.text_model.training,
        })
        self.evaluate(trainer, model, 0)

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if model.text_model.training or any(p.requires_grad for p in model.text_model.parameters()):
            raise RuntimeError("Frozen T5 entered train mode")

    def on_before_optimizer_step(self, trainer, model, optimizer):
        step = trainer.global_step + 1
        norms = [torch.sum(p.grad.detach().float().square()) for p in model.parameters()
                 if p.requires_grad and p.grad is not None]
        total = torch.sqrt(torch.stack(norms).sum()) if norms else torch.tensor(0.0, device=model.device)
        self.before_grad[step] = float(total.detach().cpu())
        if not np.isfinite(self.before_grad[step]):
            raise RuntimeError(f"Nonfinite gradient at step {step}")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step != batch_idx + 1:
            raise RuntimeError(f"Batch count and optimizer-step count diverged: {batch_idx}, {step}")
        if step in self.targets:
            self.evaluate(trainer, model, step)

    def on_train_end(self, trainer, model):
        if trainer.global_step != self.last_step or {r["step"] for r in self.rows} != set(self.targets):
            raise RuntimeError("One-epoch step coverage incomplete")
        write_json(ROOT / "reports" / "fit_complete.json", {
            "status": "complete", "optimizer_steps": int(trainer.global_step),
            "dataloader_batches": self.last_step, "checkpoints": list(self.targets),
            "elapsed_seconds": time.monotonic() - self.start_time,
        })

    def drift(self, model, step):
        groups = parameter_groups(model)
        d2 = defaultdict(float)
        for name, (group, p) in groups.items():
            delta = p.detach().float().cpu() - self.source[name]
            d2[group] += float(torch.sum(delta.double() ** 2))
        record = {"step": step}
        for group in ("encoder", "aligner"):
            record[f"{group}_delta_norm"] = np.sqrt(d2[group])
            record[f"{group}_source_norm"] = self.source_norm[group]
            record[f"{group}_relative_delta"] = record[f"{group}_delta_norm"] / self.source_norm[group]
        record["global_grad_norm_pre_update"] = self.before_grad.get(step)
        record["finite"] = all(np.isfinite(v) for v in record.values() if isinstance(v, (float, int)))
        self.drifts.append(record)
        write_csv(ROOT / "parameter_drift.csv", self.drifts, list(record))

    @staticmethod
    def batches(dataset, batch_size=24):
        indices = list(range(len(dataset)))
        chunks = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        for chunk in chunks:
            yield chunk, default_collate([dataset[i] for i in chunk])

    def evaluate_split(self, model, dataset, generate, step, split):
        numerators = defaultdict(float)
        tokens_total = 0
        trials_total = 0
        predictions = []
        for indices, batch in self.batches(dataset):
            batch = {key: value.to(model.device, non_blocking=True) if isinstance(value, torch.Tensor)
                     else value for key, value in batch.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.shared_forward(batch)
                if generate:
                    ids = model.text_model.generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=out["eeg_embeds"]),
                        attention_mask=out["memory_mask"],
                        **self.config["generation"],
                    )
            tokens = int((out["target_text_ids"] != model.text_model.config.pad_token_id).sum())
            tokens_total += tokens
            trials_total += len(indices)
            for key in LOSS_KEYS:
                numerators[key] += float(out[key].detach().float().cpu()) * (tokens if key == "loss_lm" else len(indices))
            if generate:
                texts = model.tokenizer.batch_decode(ids, skip_special_tokens=True)
                for i, text in zip(indices, texts):
                    predictions.append({"step": step, "uid": int(dataset.data["text uid"][i]),
                                        "subject": dataset.data["prompt"][i][2],
                                        "reference": dataset.data["raw input text"][i],
                                        "prediction": text})
            del out, batch
        expected = 3316 if split == "train_canonical" else 406
        if trials_total != expected:
            raise RuntimeError(f"Incomplete {split} evaluation: {trials_total}")
        result = {key: numerators[key] / (tokens_total if key == "loss_lm" else trials_total)
                  for key in LOSS_KEYS}
        result.update(trials=trials_total, target_tokens=tokens_total)
        if generate:
            path = ROOT / "evaluation" / f"predictions_step_{step:03d}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result

    def evaluate(self, trainer, model, step):
        rng = snapshot_rng()
        modes = [(m, m.training) for m in model.modules()]
        versions = {name: p._version for name, p in model.named_parameters()}
        optimizer_states = len(trainer.optimizers[0].state)
        started = time.monotonic()
        try:
            model.eval()
            if model.text_model.training:
                raise RuntimeError("Frozen T5 is not in eval during diagnostic")
            train = self.evaluate_split(model, self.data.train_canonical_set, False, step, "train_canonical")
            val = self.evaluate_split(model, self.data.val_set,
                                      step in self.generation_steps, step, "val")
        finally:
            for module, was_training in modes:
                module.training = was_training
            restore_rng(rng)
        if versions != {name: p._version for name, p in model.named_parameters()}:
            raise RuntimeError("Parameter changed during validation")
        if len(trainer.optimizers[0].state) != optimizer_states or model.text_model.training:
            raise RuntimeError("Validation changed Adam state or T5 mode")
        if trainer.global_step != step:
            raise RuntimeError("Validation changed the optimizer step counter")
        row = {"step": step, "train_batch_index": None if step == 0 else step - 1,
               "train_batch_lm_pre_update": None if step == 0 else model._diagnostic_last_train.get("loss_lm"),
               "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
               "global_grad_norm_pre_update": self.before_grad.get(step),
               "generation_ran": step in self.generation_steps,
               "eval_seconds": time.monotonic() - started}
        for prefix, values in (("train_canonical", train), ("val", val)):
            row.update({f"{prefix}_{key}": val for key, val in values.items()})
        if step > 0:
            row.update({f"train_batch_{k}_pre_update": v for k, v in model._diagnostic_last_train.items()})
        row["all_finite"] = all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int)) and not isinstance(v, bool))
        if not row["all_finite"]:
            raise RuntimeError(f"Nonfinite evaluation at step {step}")
        self.rows.append(row)
        fields = list(dict.fromkeys(key for record in self.rows for key in record))
        write_csv(ROOT / "step_metrics.csv", self.rows, fields)
        self.drift(model, step)
        trainer.save_checkpoint(ROOT / "runs" / f"step_{step:03d}.ckpt", weights_only=True)
        print("DIAG_STEP", json.dumps({"step": step, "train_lm": train["loss_lm"],
              "val_lm": val["loss_lm"], "elapsed_seconds": row["eval_seconds"]}), flush=True)


def main():
    if (ROOT / "reports" / "fit_complete.json").exists():
        raise RuntimeError("A completed run already exists; refusing to rerun")
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    if config["training"]["max_epochs"] != 1 or config["training"]["learning_rate"] != 1e-4:
        raise RuntimeError("One-epoch locked training config changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A800 BF16 required")
    checks = verify_assets(config, include_large_data=True)
    os.environ["GLIM_TEXT_MODEL_PATH"] = config["assets"]["text_model"]
    os.environ["WANDB_MODE"] = "offline"
    L.seed_everything(42, workers=True)
    package = read_v3(config)
    model = build_arm(config, package, "B0_Baseline", 42)
    L.seed_everything(42, workers=True)
    model._diagnostic_last_train = {}
    original = model.shared_forward

    def observed_forward(batch):
        out = original(batch)
        if model.training and torch.is_grad_enabled():
            model._diagnostic_last_train = {key: float(out[key].detach().float().cpu()) for key in LOSS_KEYS}
        return out

    model.shared_forward = observed_forward
    data = DiagnosticData(
        data_path=config["assets"]["data"], labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        audit_dir=ROOT / "reports", seed=42,
        bsz_train=72, bsz_val=24, bsz_test=24, num_workers=4,
    )
    write_json(ROOT / "reports" / "source_hashes.json", checks)
    logger = LocalWandbLogger(project="GLIM-Affect-Supervision-01", group="B0-step-diagnosis",
                              name="B0-step-diagnosis-seed-42", save_dir=str(ROOT / "runs"), offline=True)
    callback = StepDiagnosis(data, config)
    trainer = L.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=1,
                        use_distributed_sampler=False, num_sanity_val_steps=0,
                        limit_val_batches=0, logger=logger, callbacks=[callback],
                        default_root_dir=str(ROOT / "runs"))
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
