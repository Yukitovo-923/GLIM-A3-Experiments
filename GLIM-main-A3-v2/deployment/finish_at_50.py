"""Archive epoch 49 and gracefully end the existing A3 run after 50 epochs.

The training process is never reinitialized. No signal is sent until a full,
hash-verified Lightning checkpoint and the epoch 49 generation samples exist.
"""

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from extract_pilot_metrics import read_history
from pilot_checkpoint_watcher import archive_samples, checkpoint_metadata, copy_verified, digest


TARGET_EPOCH = 49  # Epoch indices are zero based: 0..49 is 50 completed epochs.


def status_write(path, status):
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def running_cmd(pid):
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return None
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def expected_train_process(pid, run_dir):
    cmd = running_cmd(pid)
    return bool(cmd and "run_multitask.py fit" in cmd and run_dir.name in cmd)


def archive_epoch(run_dir, source):
    archive_dir = run_dir / "pilot_artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    before = checkpoint_metadata(source)
    if before["epoch_index"] != TARGET_EPOCH:
        raise RuntimeError(f"Expected epoch {TARGET_EPOCH}, found {before['epoch_index']}")
    destination = archive_dir / f"epoch={TARGET_EPOCH}-step={before['global_step']}.ckpt"
    if not destination.exists():
        copy_verified(source, destination)
    saved = checkpoint_metadata(destination)
    if (saved["epoch_index"] != TARGET_EPOCH or saved["global_step"] != before["global_step"]
            or saved["variant"] != "A3-full" or not saved["has_optimizer_state"]
            or not saved["has_loop_state"] or digest(destination) != digest(source)):
        raise RuntimeError("Archived checkpoint failed resume, identity, or SHA256 validation")
    for suffix in ("json", "csv"):
        sample = run_dir / f"full_val_Samples.{suffix}"
        if not sample.is_file() or abs(source.stat().st_mtime - sample.stat().st_mtime) > 1800:
            raise RuntimeError(f"Epoch 49 generation samples are missing or stale: {sample}")
    samples = archive_samples(run_dir, archive_dir, TARGET_EPOCH)
    if len(samples) != 2:
        raise RuntimeError("Failed to archive both generation sample formats")
    saved.update({"checkpoint": destination.name, "checkpoint_sha256": digest(destination),
                  "generation_samples": samples})
    return saved


def save_epoch_metrics(run_dir, archive_dir):
    files = list((run_dir / "wandb").glob("offline-run-*/run-*.wandb"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one W&B offline run, found {len(files)}")
    rows = [row for row in read_history(files[0]) if row.get("epoch") == TARGET_EPOCH]
    if not any("val/loss_lm" in row for row in rows):
        raise RuntimeError("Epoch 49 validation LM loss is absent from offline history")
    if not any("val/disc_macro_f1" in row for row in rows):
        raise RuntimeError("Epoch 49 validation Macro-F1 is absent from offline history")
    path = archive_dir / f"epoch={TARGET_EPOCH}.metrics.json"
    path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": path.name, "sha256": digest(path), "records": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source = run_dir / "checkpoints" / "last.ckpt"
    status_path = run_dir / "pilot_artifacts" / "finish_at_50.status.json"
    if not expected_train_process(args.pid, run_dir):
        raise RuntimeError("A3 training PID and command line do not match")
    if args.dry_run:
        print(json.dumps({"train_pid": args.pid, "checkpoint": checkpoint_metadata(source),
                          "target_epoch_index": TARGET_EPOCH}), flush=True)
        return
    status = {"train_pid": args.pid, "target_completed_epochs": 50,
              "target_epoch_index": TARGET_EPOCH, "state": "watching"}
    status_write(status_path, status)
    seen = None
    while True:
        if not expected_train_process(args.pid, run_dir):
            status["state"] = "training_exited_before_target"
            status_write(status_path, status)
            return
        if source.exists():
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature != seen:
                time.sleep(3)
                if (source.stat().st_size, source.stat().st_mtime_ns) != signature:
                    continue
                try:
                    current = checkpoint_metadata(source)
                except Exception as exc:
                    print(f"Waiting for complete checkpoint: {exc}", flush=True)
                    continue
                epoch = current["epoch_index"]
                if epoch > TARGET_EPOCH:
                    status.update(state="missed_target_checkpoint", observed_epoch=epoch)
                    status_write(status_path, status)
                    return
                if epoch == TARGET_EPOCH:
                    try:
                        saved = archive_epoch(run_dir, source)
                    except Exception as exc:
                        print(f"Waiting for verified epoch 49 artifacts: {exc}", flush=True)
                        continue
                    status.update(state="checkpoint_and_samples_verified", checkpoint=saved)
                    status_write(status_path, status)
                    # This is the only signal path. The source checkpoint is no
                    # longer needed for resuming after this point.
                    if not expected_train_process(args.pid, run_dir):
                        status["state"] = "training_exited_after_archive"
                        status_write(status_path, status)
                        return
                    os.kill(args.pid, signal.SIGTERM)
                    status["state"] = "sigterm_sent_after_verified_epoch_49"
                    status_write(status_path, status)
                    print("Sent SIGTERM after verified epoch 49 checkpoint and samples", flush=True)
                    for _ in range(60):
                        if not expected_train_process(args.pid, run_dir):
                            break
                        time.sleep(5)
                    status["training_process_exited"] = not expected_train_process(args.pid, run_dir)
                    if status["training_process_exited"]:
                        try:
                            status["metrics"] = save_epoch_metrics(run_dir, run_dir / "pilot_artifacts")
                        except Exception as exc:
                            status["metrics_error"] = str(exc)
                    status["state"] = "finished" if status["training_process_exited"] else "sigterm_timeout"
                    status_write(status_path, status)
                    return
                seen = signature
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
