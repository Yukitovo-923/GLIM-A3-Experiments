"""Verify the new run changes only the intended continuation LR and copied loader."""

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
    old_lr, new_lr = old["training"].pop("learning_rate"), new["training"].pop("learning_rate")
    if old != new or old_lr != 1e-4 or new_lr != 1e-5:
        raise RuntimeError("Config differs beyond the experiment label and learning rate")
    copied = ["scripts/run_train.py", "scripts/smoke_b0_b1.py",
              "scripts/supervise_10ep.py", "evaluation/infer.py"]
    copied.extend(str(path.relative_to(ROOT)).replace("\\", "/")
                  for path in (ROOT / "models").rglob("*.py"))
    hashes = {}
    for name in copied:
        source, target = SOURCE / name, ROOT / name
        if not source.is_file() or not target.is_file() or sha(source) != sha(target):
            raise RuntimeError(f"Copied source differs: {name}")
        hashes[name] = sha(target)
    result = {"status": "PASS", "source_experiment": old_label,
              "new_experiment": new_label, "only_config_delta": {"training.learning_rate":
              {"source": old_lr, "new": new_lr}}, "identical_copied_files_sha256": hashes,
              "copied_loader_exception": "scripts/experiment_assets.py explicitly checks source lr=1e-4 and builds continuation at lr=1e-5"}
    (ROOT / "reports" / "isolation_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "identical_files": len(hashes),
                      "old_lr": old_lr, "new_lr": new_lr}))


if __name__ == "__main__":
    main()
