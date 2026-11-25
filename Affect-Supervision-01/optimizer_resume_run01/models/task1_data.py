"""Strict ZuCo1 Task1-SR filtering with the approved sentence UID split."""

import hashlib
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from data.datamodule import GLIMDataModule, GLIMSampler
from data.revised_labels import audit


class Task1DataModule(GLIMDataModule):
    def __init__(self, *, data_path, labels_path, expected_manifest, expected_sha256,
                 audit_dir=None, seed=42, bsz_train=72, bsz_val=24, bsz_test=24,
                 num_workers=4):
        super().__init__(data_path=data_path, bsz_train=bsz_train, bsz_val=bsz_val,
                         bsz_test=bsz_test, num_workers=num_workers)
        self.labels_path = Path(labels_path)
        self.expected_manifest = Path(expected_manifest)
        self.expected_sha256 = expected_sha256
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.seed = int(seed)
        self.train_uid_counts = None

    def load_frame(self):
        frame = pd.read_pickle(self.data_path)
        if "input text" not in frame:
            raise RuntimeError("Source EEG frame lacks canonical input text")
        frame, manifest = audit(frame, pd.read_pickle(self.labels_path))
        actual_sha = hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest()
        expected_sha = hashlib.sha256(self.expected_manifest.read_bytes()).hexdigest()
        if actual_sha != self.expected_sha256 or expected_sha != actual_sha:
            raise RuntimeError("Source UID split or label manifest differs from approved E1 split")
        frame = frame.loc[(frame.dataset == "ZuCo1") & (frame.task == "task1") &
                          frame["sentiment label"].notna()].copy()
        counts = frame.groupby("phase")["text uid"].nunique().to_dict()
        trials = frame.phase.value_counts().to_dict()
        if counts != {"train": 320, "val": 40, "test": 40}:
            raise RuntimeError(f"Unexpected Task1 sentence split: {counts}")
        if trials != {"train": 3316, "val": 406, "test": 417}:
            raise RuntimeError(f"Unexpected Task1 trial split: {trials}")
        if frame.groupby("text uid").phase.nunique().gt(1).any():
            raise RuntimeError("A sentence UID appears in multiple splits")
        train = frame.loc[frame.phase == "train"]
        self.train_uid_counts = {int(k): int(v) for k, v in
                                 train.groupby("text uid").size().items()}
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            task1_manifest = manifest.loc[(manifest.dataset == "ZuCo1") &
                                          (manifest.task == "task1")]
            task1_manifest.to_csv(self.audit_dir / "task1_split_manifest.csv", index=False)
        return frame

    def train_dataloader(self):
        sampler = GLIMSampler(self.train_set, self.train_set.data["text uid"],
                              "train", self.bsz_train)
        sampler.seed = self.seed
        sampler.n_batches_per_device = sampler.estimate_len()
        sampler.n_batches = sampler.n_batches_per_device * sampler.num_replicas
        return DataLoader(self.train_set, batch_sampler=sampler, num_workers=self.num_workers,
                          pin_memory=True)
