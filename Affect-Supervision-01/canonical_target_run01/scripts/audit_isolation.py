"""Verify only the canonical target selection differs from the original run."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((SOURCE / "configs" / "common.json").read_text(encoding="utf-8"))
    new = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8"))
    old_label, new_label = old.pop("experiment"), new.pop("experiment")
    mode = new["data"].pop("train_target_mode")
    if old != new or mode != "original_sentence_repeated_at_eight_positions":
        raise RuntimeError("Config differs beyond the target-mode label")
    copied = ["scripts/run_train.py", "scripts/smoke_b0_b1.py",
              "scripts/supervise_10ep.py", "evaluation/infer.py"]
    copied.extend(str(path.relative_to(ROOT)).replace("\\", "/")
                  for path in (ROOT / "models").rglob("*.py")
                  if path.name != "datamodule.py")
    hashes = {}
    for name in copied:
        source, target = SOURCE / name, ROOT / name
        if not source.is_file() or not target.is_file() or sha(source) != sha(target):
            raise RuntimeError(f"Copied source differs: {name}")
        hashes[name] = sha(target)
    changed = ROOT / "models" / "data" / "datamodule.py"
    if sha(changed) == sha(SOURCE / "models" / "data" / "datamodule.py"):
        raise RuntimeError("Target selection source did not change")
    result = {"status": "PASS", "source_experiment": old_label,
              "new_experiment": new_label, "target_mode": mode,
              "source_lr_and_new_lr": 1e-4,
              "identical_copied_files_sha256": hashes,
              "changed_training_file": "models/data/datamodule.py",
              "changed_training_file_sha256": sha(changed),
              "loader_exception": "scripts/experiment_assets.py checks target_mode metadata only"}
    (ROOT / "reports" / "isolation_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "identical_files": len(hashes),
                      "target_mode": mode}))


if __name__ == "__main__":
    main()
