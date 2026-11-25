"""Verify 12 best checkpoints, optimizer starts, validation UIDs and EEG pairs."""

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
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        val_uids = {int(r["text uid"]) for r in csv.DictReader(stream)
                    if r["dataset"] == "ZuCo1" and r["task"] == "task1" and r["phase"] == "val"}
    if len(val_uids) != 40:
        raise RuntimeError("Validation UID count differs from reviewed manifest")
    rows = []
    for seed in (42, 43, 44):
        for arm in ("B0_Baseline", "B1_EPA"):
            for optimizer in ("reset", "restored"):
                run = ROOT / "runs" / optimizer / arm / f"seed-{seed}"
                output = ROOT / "evaluation" / "outputs" / f"{optimizer}_{arm}_seed{seed}_val"
                start = json.loads((run / "optimizer_start.json").read_text())
                if (start["status"] != "PASS" or start["condition"] != optimizer or
                        start["parameter_states_at_start"] != (214 if optimizer == "restored" else 0)):
                    raise RuntimeError("Optimizer start audit mismatch")
                completion = json.loads((run / "fit_complete.json").read_text())
                checkpoint = Path(completion["best_model_path"])
                if not checkpoint.is_file() or run.resolve() not in checkpoint.resolve().parents:
                    raise RuntimeError("Best checkpoint missing or outside run")
                epochs = sorted((run / "metrics").glob("epoch_*.json"))
                if not 1 <= len(epochs) <= 10:
                    raise RuntimeError("Epoch logs incomplete")
                inference = json.loads((output / "inference_summary.json").read_text())
                digest = sha(checkpoint)
                if (inference["split"] != "val" or inference["condition"] != optimizer or
                        inference["best_checkpoint_sha256"] != digest or
                        not 0 <= inference["checkpoint_epoch"] < len(epochs)):
                    raise RuntimeError("Checkpoint/inference integrity mismatch")
                matched = [json.loads(line) for line in
                           (output / "matched_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                mismatched = [json.loads(line) for line in
                              (output / "mismatched_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                if (len(matched) != 406 or len(mismatched) != 406 or
                        {r["uid"] for r in matched} != val_uids or
                        any(a["uid"] != b["uid"] or a["dataset_index"] != b["dataset_index"]
                            for a, b in zip(matched, mismatched))):
                    raise RuntimeError("Validation UID or matched-EEG pairing mismatch")
                rows.append({"optimizer": optimizer, "arm": arm, "seed": seed,
                             "best_checkpoint": str(checkpoint), "sha256": digest,
                             "best_epoch_zero_based": inference["checkpoint_epoch"],
                             "epochs_logged": len(epochs), "val_uids": len(val_uids),
                             "val_trials": len(matched)})
    with (ROOT / "reports" / "sha256_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (ROOT / "reports" / "verification.json").write_text(
        json.dumps({"status": "PASS", "split": "val", "runs": rows}, indent=2), encoding="utf-8")
    print("PASS: 12 checkpoint hashes, optimizer starts and paired validation UID sets", flush=True)


if __name__ == "__main__":
    main()
