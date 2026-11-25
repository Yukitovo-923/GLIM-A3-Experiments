"""Archive A3 pilot checkpoints without changing the running Trainer.

This process only reads Lightning's rolling last.ckpt and writes separate
copies under pilot_artifacts. Epoch indices are zero based.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


TARGET_EPOCHS = (4, 5, 9, 10)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_metadata(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    gate = state.get("state_dict", {}).get("attribute.gate")
    raw = gate.item() if gate is not None else None
    return {
        "epoch_index": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "variant": state.get("hyper_parameters", {}).get("variant"),
        "has_optimizer_state": bool(state.get("optimizer_states")),
        "has_loop_state": "loops" in state,
        "gate_raw": raw,
        "gate_effective_sigmoid": 1 / (1 + math.exp(-raw)) if raw is not None else None,
    }


def copy_verified(source, dest):
    before = source.stat()
    tmp = dest.with_name(dest.name + ".partial")
    try:
        with source.open("rb") as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("source changed while copying")
        if tmp.stat().st_size != after.st_size or digest(tmp) != digest(source):
            raise RuntimeError("copy size or SHA256 mismatch")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def archive_samples(run_dir, archive_dir, epoch):
    # Full generation runs at epoch 0 and then every 10 epochs (index 9, 19...).
    if epoch != 0 and (epoch + 1) % 10 != 0:
        return []
    copied = []
    for suffix in ("json", "csv"):
        source = run_dir / f"full_val_Samples.{suffix}"
        if source.exists():
            dest = archive_dir / f"full_val_Samples.epoch={epoch}.{suffix}"
            if not dest.exists():
                copy_verified(source, dest)
            copied.append({"path": dest.name, "sha256": digest(dest)})
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source = run_dir / "checkpoints" / "last.ckpt"
    archive_dir = run_dir / "pilot_artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_samples(run_dir, archive_dir, 0)
    seen = None
    while True:
        if source.exists():
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature != seen:
                time.sleep(3)
                try:
                    if (source.stat().st_size, source.stat().st_mtime_ns) != signature:
                        continue
                    metadata = checkpoint_metadata(source)
                    epoch = metadata["epoch_index"]
                    if epoch in TARGET_EPOCHS:
                        dest = archive_dir / f"epoch={epoch}-step={metadata['global_step']}.ckpt"
                        if not dest.exists():
                            copy_verified(source, dest)
                        verified = checkpoint_metadata(dest)
                        if verified["epoch_index"] != epoch or not verified["has_optimizer_state"]:
                            raise RuntimeError("archived checkpoint is not resume-ready")
                        verified["checkpoint"] = dest.name
                        verified["checkpoint_sha256"] = digest(dest)
                        verified["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
                        verified["generation_samples"] = archive_samples(run_dir, archive_dir, epoch)
                        (archive_dir / f"epoch={epoch}.json").write_text(
                            json.dumps(verified, indent=2) + "\n", encoding="utf-8"
                        )
                        print(json.dumps(verified), flush=True)
                    seen = signature
                except Exception as exc:
                    print(f"retry after checkpoint read/copy error: {exc}", flush=True)
        if all(list(archive_dir.glob(f"epoch={epoch}-step=*.ckpt")) for epoch in TARGET_EPOCHS):
            print("all target epochs archived", flush=True)
            return
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            print("training process exited; watcher exiting", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
