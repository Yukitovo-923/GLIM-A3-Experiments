"""Analyze existing B0/B1 epoch logs and source V3 metadata without retraining."""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

OUT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = OUT.parent
ARMS = ("B0_Baseline", "B1_EPA")
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    curve = list(csv.DictReader((ROOT / "reports/training_curve.csv").open(encoding="utf-8")))
    config = read(ROOT / "configs/common.json")
    summary = read(ROOT / "reports/training_summary.json")
    source = read(OUT / "source_snapshots/v3_epoch46_metrics.json")
    optimizer = read(OUT / "source_snapshots/v3_checkpoint_optimizer.json")
    source_config = read(OUT / "source_snapshots/v3_model_config.json")
    source_manifest = OUT / "source_snapshots/v3_split_manifest.csv"
    assert len(curve) == 30 and len(summary["runs"]) == 6
    assert sha(source_manifest) == config["assets"]["e1_manifest_sha256"]
    assert optimizer["optimizer_param_state_count"] == 214
    assert source_config["lr"] == optimizer["optimizer_lr"] == config["training"]["learning_rate"]
    assert optimizer["scheduler_state_count"] == 0
    task1_source_metrics={}
    for name in ("bleu1_gen", "bleu2_gen", "bleu4_gen", "rouge1_fmeasure_gen",
                 "sentiment_cls_acc_top01"):
        vals=[float(value) for key,value in source.items()
              if key.startswith(f"full_val/{name}-") and "-ZuCo1-" in key and key.endswith("-task1")]
        if vals:
            task1_source_metrics[name]={"n_subjects":len(vals),"subject_mean":sum(vals)/len(vals)}
    counts = defaultdict(Counter)
    with source_manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] == "train":
                counts[(row["dataset"], row["task"])][row["sentiment label"]] += 1
    training = []
    for record in curve:
        z = {key: record[key] for key in ("arm", "seed", "epoch", "global_step")}
        for key in ("train_loss_lm", "val_loss_lm", "train_loss_disc_emotion",
                    "val_loss_disc_emotion", "train_loss_epa", "val_loss_epa"):
            z[key] = float(record[key]) if record[key] else None
        training.append(z)
    by_run = {}
    for arm in ARMS:
        for seed in SEEDS:
            z = sorted((r for r in training if r["arm"] == arm and int(r["seed"]) == seed),
                       key=lambda r: int(r["epoch"]))
            assert len(z) == 5
            best = min(z, key=lambda r: r["val_loss_lm"])
            by_run[f"{arm}_seed{seed}"] = {
                "best_epoch_zero_based": int(best["epoch"]),
                "best_val_lm": best["val_loss_lm"],
                "val_lm_change_epoch0_to4": z[-1]["val_loss_lm"] - z[0]["val_loss_lm"],
                "val_disc_ce_change_epoch0_to4": z[-1]["val_loss_disc_emotion"] - z[0]["val_loss_disc_emotion"],
                "val_epa_ce_change_epoch0_to4":
                    z[-1]["val_loss_epa"] - z[0]["val_loss_epa"] if arm == "B1_EPA" else None,
            }
    result = {
        "source_epoch46": {"epoch": optimizer["epoch"], "global_step": optimizer["global_step"],
             "reported_val_lm": source.get("val/loss_lm"),
             "metric_scope": "original V3 all-task validation; not Task1-only and not directly comparable",
             "same_scope_task1_direct_validation": "not available without model+EEG runtime",
             "task1_existing_full_val_subject_metrics":task1_source_metrics,
             "optimizer_state_tensors": optimizer["optimizer_param_state_count"],
             "optimizer_lr": optimizer["optimizer_lr"],
             "scheduler_count": optimizer["scheduler_state_count"]},
        "continuation": {"optimizer_reset": True,
             "reason": "run_train.py builds model from V3 state_dict, then Trainer.fit creates a new Adam; no optimizer_states load",
             "lr": config["training"]["learning_rate"],
             "scheduler": "none; GLIM.configure_optimizers returns Adam only",
             "source_train_tasks": [f"{d}:{t}" for d,t in sorted(counts)],
             "continuation_train_tasks": ["ZuCo1:task1"],
             "training_target_variants_per_trial": config["data"]["train_text_variants_per_trial"],
             "validation_target": "canonical original input text",
             "source_split_manifest_same_sha256": sha(source_manifest)},
        "per_run": by_run,
        "logging_caution": "train metrics from Lightning training_step default logging; val metrics are validation aggregates. Do not interpret their numerical gap as a matched train/val generalization gap.",
    }
    (OUT / "stability.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"tables"/"epoch_losses.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)

    fig,axes=plt.subplots(2,3,figsize=(13,6.5),layout="constrained")
    for col,seed in enumerate(SEEDS):
        for arm,color in ((ARMS[0],"#457b9d"),(ARMS[1],"#e76f51")):
            z=sorted((r for r in training if r["arm"]==arm and int(r["seed"])==seed),key=lambda r:int(r["epoch"]))
            ep=[int(r["epoch"])+1 for r in z]
            axes[0,col].plot(ep,[r["val_loss_lm"] for r in z],marker="o",color=color,label=arm)
            axes[1,col].plot(ep,[r["val_loss_disc_emotion"] for r in z],marker="o",color=color,label=arm)
        axes[0,col].set_title(f"Seed {seed}");axes[1,col].set_xlabel("Completed epoch")
        for ax in axes[:,col]:ax.set_xticks(range(1,6));ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Validation LM loss")
    axes[1,0].set_ylabel("Validation V3 emotion CE")
    axes[0,2].legend(fontsize=8)
    fig.savefig(OUT/"figures"/"validation_curves.png",dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    for seed,color in zip(SEEDS,("#3b6ca8","#c87834","#54a665")):
        z=sorted((r for r in training if r["arm"]=="B1_EPA" and int(r["seed"])==seed),key=lambda r:int(r["epoch"]))
        ax.plot(range(1,6),[r["val_loss_epa"] for r in z],marker="o",color=color,label=f"seed {seed}")
    ax.axhline(__import__("math").log(3),color="black",ls="--",lw=.8,label="uniform 3-class CE")
    ax.set_xlabel("Completed epoch");ax.set_ylabel("Validation EPA CE");ax.legend(fontsize=8)
    fig.savefig(OUT/"figures"/"validation_epa_ce.png",dpi=160);plt.close(fig)
    print("created stability.json, epoch loss CSV and validation figures")


if __name__ == "__main__":
    main()
