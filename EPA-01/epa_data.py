"""Exact E1 split enforcement and sentence/subject audit without changing GLIM data code."""
import json
from pathlib import Path

import numpy as np

from data.datamodule import GLIMDataModule
from epa_assets import hash_file


class EPADataModule(GLIMDataModule):
    def __init__(self, *, data_path, labels_path, expected_manifest, audit_dir,
                 bsz_train=72, bsz_val=24, bsz_test=24, num_workers=4):
        super().__init__(data_path=data_path, bsz_train=bsz_train, bsz_val=bsz_val,
                         bsz_test=bsz_test, num_workers=num_workers)
        self.label_reference = Path(labels_path)
        self.expected_manifest = Path(expected_manifest)
        self.audit_dir = Path(audit_dir)
        self.manifest_path = self.audit_dir / "split_manifest.csv"
        self.uid_trial_counts = None

    def load_frame(self):
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        frame = super().load_frame()  # one real EEG pickle read; original audit preserves phase.
        actual = hash_file(self.manifest_path)
        expected = hash_file(self.expected_manifest)
        if actual != expected:
            raise RuntimeError(f"EPA split manifest differs from E1: {actual} != {expected}")
        sr = frame.loc[(frame.dataset == "ZuCo1") & (frame.task == "task1") &
                       frame["sentiment label"].notna()]
        if sr["text uid"].nunique() != 400 or sr.groupby("text uid").phase.nunique().gt(1).any():
            raise ValueError("Invalid Task 1 sentence UID split")
        expected_sentences = {"train": 320, "val": 40, "test": 40}
        actual_sentences = sr.groupby("phase")["text uid"].nunique().to_dict()
        if actual_sentences != expected_sentences:
            raise ValueError(f"Unexpected sentence split: {actual_sentences}")
        train = sr.loc[sr.phase == "train"]
        self.uid_trial_counts = {int(k): int(v) for k, v in train.groupby("text uid").size().items()}
        if len(self.uid_trial_counts) != 320:
            raise ValueError("Expected 320 train UID trial counts")
        report = {"manifest_sha256": actual, "manifest_matches_e1": True,
                  "task1_rows": int(len(sr)),
                  "sentences_by_split": actual_sentences,
                  "trials_by_split": {k: int(v) for k, v in sr.phase.value_counts().items()},
                  "sentiment_trial_counts_by_split": {
                      phase: {str(k): int(v) for k, v in group["sentiment label"].value_counts().items()}
                      for phase, group in sr.groupby("phase")},
                  "subjects_by_split": {
                      phase: {str(k): int(v) for k, v in group.subject.value_counts().items()}
                      for phase, group in sr.groupby("phase")},
                  "unique_subjects": int(sr.subject.nunique()),
                  "train_trial_count_per_uid": self.uid_trial_counts,
                  "train_trial_count_summary": {
                      "min": min(self.uid_trial_counts.values()),
                      "median": float(np.median(list(self.uid_trial_counts.values()))),
                      "max": max(self.uid_trial_counts.values()),
                      "mean": float(np.mean(list(self.uid_trial_counts.values())))},
                  "train_subject_trial_count": {str(k): int(v) for k, v in train.subject.value_counts().items()},
                  "train_unique_uid_by_subject": {
                      str(k): int(v) for k, v in train.groupby("subject")["text uid"].nunique().items()}}
        (self.audit_dir / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                         encoding="utf8")
        return frame
