"""Summarize all six completed runs, every epoch, and best/last checkpoint hashes."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B0_Baseline", "B1_EPA")
SEEDS = (42, 43, 44)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    status = json.loads((ROOT / "reports" / "training_status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete" or len(status["jobs"]) != 6:
        raise RuntimeError("Cannot summarize before all six runs complete")
    curve = []
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            directory = ROOT / "runs" / arm / f"seed-{seed}"
            finish = json.loads((directory / "fit_complete.json").read_text(encoding="utf-8"))
            config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
            if (finish["arm"], finish["seed"]) != (arm, seed):
                raise RuntimeError(f"Wrong completed run: {directory}")
            epochs = sorted((directory / "metrics").glob("epoch_*.json"))
            if not epochs or len(epochs) > 10:
                raise RuntimeError(f"Unexpected epoch logs in {directory}")
            for epoch_path in epochs:
                item = json.loads(epoch_path.read_text(encoding="utf-8"))
                m = item["metrics"]
                curve.append({"arm": arm, "seed": seed, "epoch": item["epoch"],
                              "global_step": item["global_step"],
                              "epoch_elapsed_seconds": item["epoch_elapsed_seconds"],
                              "learning_rate": item["learning_rate"],
                              "train_loss_total": m.get("train/loss_total"),
                              "train_loss_lm": m.get("train/loss_lm"),
                              "train_loss_clip": m.get("train/loss_clip"),
                              "train_loss_vq": m.get("train/loss_vq"),
                              "train_loss_disc_emotion": m.get("train/loss_disc_emotion"),
                              "train_loss_epa": m.get("train/loss_epa"),
                              "val_loss_lm": m["val/loss_lm"],
                              "val_loss_clip": m.get("val/loss_clip"),
                              "val_loss_vq": m.get("val/loss_vq"),
                              "val_loss_disc_emotion": m.get("val/loss_disc_emotion"),
                              "val_loss_epa": m.get("val/loss_epa"),
                              "val_disc_macro_f1": m.get("val/disc_macro_f1"),
                              "val_codebook_perplexity": m.get("val/codebook_perplexity")})
            best = Path(finish["best_model_path"])
            last = directory / "checkpoints" / "last.ckpt"
            if not best.is_file() or not last.is_file():
                raise RuntimeError(f"Missing best/last checkpoint in {directory}")
            runs.append({"arm": arm, "seed": seed, "epochs_completed": len(epochs),
                         "global_step": finish["global_step"],
                         "stopped_early": finish["stopped_early"],
                         "best_epoch": int(best.name.split("-")[0].split("=")[1]),
                         "best_val_lm_loss": finish["best_model_score"],
                         "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
                         "last_checkpoint": str(last), "last_checkpoint_sha256": sha256(last),
                         "source_checkpoint_sha256": config["source_hashes"]["checkpoint"],
                         "train_elapsed_seconds": sum(json.loads(p.read_text(encoding="utf-8"))
                                                      ["epoch_elapsed_seconds"] for p in epochs)})
    output = ROOT / "reports" / "training_summary.json"
    output.write_text(json.dumps({"status": "complete", "runs": runs}, indent=2), encoding="utf-8")
    with (ROOT / "reports" / "training_curve.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(output)


if __name__ == "__main__":
    main()
