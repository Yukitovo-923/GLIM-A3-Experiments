"""Read-only checkpoint integrity gate for the 10-epoch EPA queue."""
import hashlib
import json
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected_epoch = int(sys.argv[2])
expected_step = int(sys.argv[3])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if checkpoint.get("epoch") != expected_epoch or checkpoint.get("global_step") != expected_step:
    raise RuntimeError(f"Wrong epoch/global step: {checkpoint.get('epoch')}/{checkpoint.get('global_step')}")
if not checkpoint.get("optimizer_states") or not checkpoint.get("loops") or not checkpoint.get("state_dict"):
    raise RuntimeError("Missing model, optimizer or Lightning loop state")
hasher = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
print(json.dumps({"status": "PASS", "checkpoint": str(path.resolve()),
                  "bytes": path.stat().st_size, "sha256": hasher.hexdigest(),
                  "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
                  "optimizer_states": len(checkpoint["optimizer_states"]),
                  "state_keys": len(checkpoint["state_dict"])}))
