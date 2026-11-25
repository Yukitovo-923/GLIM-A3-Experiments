"""Minimal Lightning runtime check: frozen module inserted in setup in eval mode."""

import json
from pathlib import Path

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class Probe(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(1, 1)
        self.events = []

    def setup(self, stage):
        self.frozen = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.5)).requires_grad_(False).eval()
        self.events.append({"event": "setup", "epoch": self.current_epoch,
                            "parent_train": self.training, "frozen_train": self.frozen.training,
                            "dropout_train": self.frozen[1].training})

    def on_train_epoch_start(self):
        self.events.append({"event": "train_epoch_start", "epoch": self.current_epoch,
                            "parent_train": self.training, "frozen_train": self.frozen.training,
                            "dropout_train": self.frozen[1].training})

    def training_step(self, batch, batch_idx):
        x, y = batch
        return ((self.weight(x) - y) ** 2).mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", torch.tensor(float(self.current_epoch)))

    def configure_optimizers(self):
        return torch.optim.Adam(self.weight.parameters(), lr=1e-4)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=2)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=2)


def main():
    model = Probe()
    trainer = L.Trainer(max_epochs=2, limit_train_batches=1, limit_val_batches=1,
                        num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False,
                        accelerator="cpu", devices=1)
    trainer.fit(model)
    result = {"lightning_version": L.__version__, "events": model.events}
    (ROOT / "reports" / "frozen_mode_runtime_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
