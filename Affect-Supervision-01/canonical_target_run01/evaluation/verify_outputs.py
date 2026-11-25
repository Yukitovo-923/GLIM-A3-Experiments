"""Verify six local best checkpoints, epoch logs and validation UID isolation."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "failure_analysis_v2" / "source_snapshots" / "v3_split_manifest.csv"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    with MANIFEST.open(encoding="utf-8", newline="") as source:
        val_uids = {int(row["text uid"]) for row in csv.DictReader(source)
                    if row["dataset"] == "ZuCo1" and row["task"] == "task1"
                    and row["phase"] == "val"}
    if len(val_uids) != 40:
        raise RuntimeError("Wrong validation UID count")
    rows = []
    for arm in ("B0_Baseline", "B1_EPA"):
        for seed in (42, 43, 44):
            run = ROOT / "runs" / arm / f"seed-{seed}"
            output = ROOT / "evaluation" / "outputs" / f"{arm}_seed{seed}_val"
            checkpoint = list((run / "checkpoints").glob("epoch=*.ckpt"))
            if len(checkpoint) != 1:
                raise RuntimeError("Expected exactly one best checkpoint")
            epochs = sorted((run / "metrics").glob("epoch_*.json"))
            if not (1 <= len(epochs) <= 10) or any(
                    json.loads(p.read_text())["learning_rate"] != 1e-4
                    for p in epochs):
                raise RuntimeError("Wrong epoch count or learning rate")
            inference = json.loads((output / "inference_summary.json").read_text())
            if (inference["split"] != "val" or
                    not 0 <= inference["checkpoint_epoch"] < len(epochs)):
                raise RuntimeError("Wrong split or best epoch")
            digest = sha(checkpoint[0])
            if digest != inference["best_checkpoint_sha256"]:
                raise RuntimeError("Best checkpoint SHA mismatch")
            matched = [json.loads(line) for line in
                       (output / "matched_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            mismatched = [json.loads(line) for line in
                          (output / "mismatched_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(matched) != 406 or len(mismatched) != 406 or {r["uid"] for r in matched} != val_uids:
                raise RuntimeError("Validation predictions incomplete or UID mismatch")
            if any(a["uid"] != b["uid"] or a["dataset_index"] != b["dataset_index"]
                   for a, b in zip(matched, mismatched)):
                raise RuntimeError("EEG mismatch condition is unpaired")
            rows.append({"arm": arm, "seed": seed, "best_checkpoint": str(checkpoint[0]),
                         "sha256": digest,
                         "best_epoch_zero_based": inference["checkpoint_epoch"],
                         "epochs_logged": len(epochs), "val_uids": len(val_uids),
                         "val_trials": len(matched),
                         "changed_predictions_with_mismatched_eeg": sum(
                             a["prediction"] != b["prediction"]
                             for a, b in zip(matched, mismatched))})
    with (ROOT / "reports" / "sha256_manifest.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (ROOT / "reports" / "verification.json").write_text(
        json.dumps({"status": "PASS", "split": "val", "runs": rows}, indent=2),
        encoding="utf-8")
    print("PASS: six hashes, complete epoch logs, and 12 paired validation prediction sets")


if __name__ == "__main__":
    main()
