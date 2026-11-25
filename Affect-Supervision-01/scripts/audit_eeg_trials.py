"""Read-only audit of ZuCo1 Task1-SR EEG trials and the approved UID split."""

import hashlib
import json

import numpy as np
import pandas as pd

from data.revised_labels import audit


DATA = "/root/autodl-tmp/GLIM-main-A3-v2/data/tmp/zuco_eeg_label_8variants.df"
LABELS = "/root/autodl-tmp/GLIM-main-A3-v2/data/tmp/zuco_label_8variants.df"
EXPECTED_MANIFEST_SHA256 = "13ba43e80a6e90365bbabf2ced66538198597c9c749d6fdd77a3eb6ec374bb17"


def main():
    frame, manifest = audit(pd.read_pickle(DATA), pd.read_pickle(LABELS))
    manifest_hash = hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"UID split or label manifest changed: {manifest_hash}")

    sr = frame.loc[(frame["dataset"] == "ZuCo1") & (frame["task"] == "task1")].copy()
    if sr["text uid"].nunique() != 400:
        raise RuntimeError("Unexpected independent sentence count")
    if sr.groupby("text uid")["phase"].nunique().gt(1).any():
        raise RuntimeError("One sentence UID appears in multiple splits")
    if sr.groupby("text uid")["input text"].nunique().gt(1).any():
        raise RuntimeError("One sentence UID has conflicting canonical texts")
    if sr.groupby("text uid")["sentiment label"].nunique().gt(1).any():
        raise RuntimeError("One sentence UID has conflicting sentiment labels")

    invalid = []
    shapes = {}
    for i, row in sr.iterrows():
        eeg = np.asarray(row["eeg"])
        mask = np.asarray(row["mask"]).astype(bool)
        shapes[str(tuple(eeg.shape))] = shapes.get(str(tuple(eeg.shape)), 0) + 1
        if eeg.ndim != 2 or mask.shape != (eeg.shape[0],) or not mask.any():
            invalid.append({"row": int(i), "reason": "shape or empty mask"})
        elif not np.isfinite(eeg[mask]).all():
            invalid.append({"row": int(i), "reason": "nonfinite valid EEG"})

    valid = sr.drop(index=[item["row"] for item in invalid])
    report = {
        "manifest_sha256": manifest_hash,
        "source_rows": int(len(sr)),
        "valid_eeg_trials": int(len(valid)),
        "invalid_trial_count": len(invalid),
        "invalid_examples": invalid[:10],
        "eeg_shapes": shapes,
        "independent_sentence_uids": int(sr["text uid"].nunique()),
        "sentence_split_counts": {str(k): int(v) for k, v in sr.groupby("phase")["text uid"].nunique().items()},
        "valid_trial_split_counts": {str(k): int(v) for k, v in valid["phase"].value_counts().items()},
        "sentence_label_counts": {
            str(k): int(v) for k, v in sr.drop_duplicates("text uid")["sentiment label"].value_counts().items()
        },
        "valid_trial_label_counts": {str(k): int(v) for k, v in valid["sentiment label"].value_counts().items()},
        "subject_valid_trial_counts": {str(k): int(v) for k, v in valid["subject"].value_counts().items()},
        "subject_counts_by_split": {
            str(phase): {str(k): int(v) for k, v in part["subject"].value_counts().items()}
            for phase, part in valid.groupby("phase")
        },
        "uid_split_overlap": False,
        "uid_label_conflict": False,
        "uid_canonical_text_conflict": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
