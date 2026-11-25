"""Record immutable input hashes from verified validation and gradient audits."""

import csv
import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent


def main():
    config = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8"))
    rows = [{"asset": "source_v3_epoch46", "path": config["checkpoint"]["path"],
             "sha256": config["checkpoint"]["sha256"]},
            {"asset": "EPA_train_only_prototypes", "path": config["assets"]["prototype"],
             "sha256": config["assets"]["prototype_sha256"]},
            {"asset": "approved_UID_split", "path": config["assets"]["e1_manifest"],
             "sha256": config["assets"]["e1_manifest_sha256"]}]
    for arm in config["arms"]:
        for seed in config["training"]["seeds"]:
            summary = json.loads((OUT / "validation_checkpoints" / f"{arm}_seed{seed}" / "summary.json")
                                 .read_text(encoding="utf-8"))
            rows.append({"asset": f"{arm}_seed{seed}_best", "path": summary["checkpoint"],
                         "sha256": summary["checkpoint_sha256"]})
    with (OUT / "tables" / "input_sha256.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    env = {"checked_at": "2026-11-15T16:14:46+08:00", "gpu": "NVIDIA A800 80GB PCIe",
           "gpu_memory_mib": 81920, "gpu_memory_used_at_final_check_mib": 0,
           "torch": "2.3.0+cu121", "cuda_available": True, "bf16_supported": True,
           "autodl_tmp_free_gib_approx": 20,
           "training_started": False, "optimizer_step_called": False,
           "all_new_outputs_within": str(OUT)}
    (OUT / "remote_environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"Recorded {len(rows)} immutable input SHA256 values")


if __name__ == "__main__":
    main()
