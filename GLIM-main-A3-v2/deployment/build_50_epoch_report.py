"""Build the A3 50 epoch Pilot report from completed offline run records."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from extract_pilot_metrics import read_history

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = (
    "train/loss_lm", "val/loss_lm", "val/loss_clip", "val/loss_vq",
    "val/loss_disc_emotion", "val/disc_accuracy", "val/disc_macro_f1",
    "val/disc_balanced_accuracy", "val/codebook_perplexity",
    "val/active_code_ratio", "full_val/mean_BLEU1@MTV",
    "full_val/mean_BLEU2@MTV", "full_val/mean_ROUGE1@MTV",
    "full_val/mean_ROUGE1@RAW",
)
COMMON = ("train/loss_lm", "val/loss_lm", "val/loss_clip",
          "full_val/mean_BLEU1@MTV", "full_val/mean_ROUGE1@MTV")


def scalar(rows, epoch, key):
    values = [r[key] for r in rows if r.get("epoch") == epoch
              and isinstance(r.get(key), (int, float))]
    if not values:
        return None
    if key.startswith("train/loss"):
        return sum(values) / len(values)  # Logged-step sample mean, not batch mean.
    return values[-1]


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def one_run(path):
    rows = read_history(path)
    return rows, {e: {key: scalar(rows, e, key) for key in FIELDS} for e in range(50)}


def samples_by_epoch(rows, run_file):
    tables = [r["full_val/Samples"] for r in rows if "full_val/Samples" in r]
    epochs = (0, 9, 19, 29, 39, 49)
    result = {}
    for epoch, ref in zip(epochs, tables):
        path = run_file.parent / "files" / ref["path"]
        if not path.is_file():
            result[epoch] = {"path": str(path), "exists": False}
            continue
        table = json.loads(path.read_text(encoding="utf-8"))
        columns = table["columns"]
        picked = []
        for row in table["data"]:
            data = dict(zip(columns, row))
            if data.get("Dataset") == "ZuCo1" and data.get("Raw Task Key") == "task1":
                picked.append({key: data.get(key) for key in
                               ("Input Text", "Generated Text", "Target Texts")})
                if len(picked) == 3:
                    break
        result[epoch] = {"path": str(path), "exists": True,
                         "rows": len(table["data"]), "examples": picked}
    return result


def plot_curves(a3, a0, gate, output):
    epochs = list(range(50))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, [a3[e]["val/loss_lm"] for e in epochs], label="A3")
    ax.plot(epochs, [a0[e]["val/loss_lm"] for e in epochs], label="A0")
    ax.set(title="Validation LM loss", xlabel="Epoch index", ylabel="Loss")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(epochs, [a3[e]["val/disc_macro_f1"] for e in epochs], label="Macro-F1")
    ax.plot(epochs, [a3[e]["val/disc_balanced_accuracy"] for e in epochs], label="Balanced accuracy")
    ax.set(title="A3 sentiment", xlabel="Epoch index", ylabel="Score", ylim=(0, 1))
    ax.legend()
    ax = axes[1, 0]
    ax.plot(epochs, [a3[e]["val/codebook_perplexity"] for e in epochs], label="Perplexity")
    ax.plot(epochs, [128 * a3[e]["val/active_code_ratio"]
                     if a3[e]["val/active_code_ratio"] is not None else None
                     for e in epochs], label="Active codes")
    ax.set(title="A3 codebook usage", xlabel="Epoch index", ylabel="Count / perplexity")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(sorted(gate), [gate[e] for e in sorted(gate)], marker="o")
    ax.axhline(0.1, linestyle="--", linewidth=1, color="gray", label="Initial gate")
    ax.set(title="A3 effective attribute gate", xlabel="Epoch index", ylabel="Sigmoid(gate)")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-run", type=Path, required=True)
    parser.add_argument("--a0-wandb", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    status = json.loads((out / "finish_at_50.status.json").read_text(encoding="utf-8"))
    if status.get("state") != "finished" or not status.get("training_process_exited"):
        raise RuntimeError("A3 has not safely completed and exited after 50 epochs")
    if status.get("checkpoint", {}).get("epoch_index") != 49:
        raise RuntimeError("No verified epoch 49 checkpoint")
    a3_files = list((args.a3_run / "wandb").glob("offline-run-*/run-*.wandb"))
    if len(a3_files) != 1:
        raise RuntimeError("Expected exactly one A3 offline log")
    a3_rows, a3 = one_run(a3_files[0])
    a0_rows, a0 = one_run(args.a0_wandb)
    if a3[49]["val/loss_lm"] is None or a3[49]["val/disc_macro_f1"] is None:
        raise RuntimeError("Epoch 49 metrics incomplete")
    gate = {}
    for path in out.glob("epoch=*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "gate_effective_sigmoid" in meta:
            gate[int(meta["epoch_index"])] = meta["gate_effective_sigmoid"]
    gate[49] = status["checkpoint"]["gate_effective_sigmoid"]
    for epoch in range(50):
        a3[epoch]["gate_effective_sigmoid"] = gate.get(epoch)
        a3[epoch]["a0_val_loss_lm"] = a0[epoch]["val/loss_lm"]
        a3[epoch]["a0_val_loss_clip"] = a0[epoch]["val/loss_clip"]
        a3[epoch]["a0_full_val_bleu1_mtv"] = a0[epoch]["full_val/mean_BLEU1@MTV"]
        a3[epoch]["a0_full_val_rouge1_mtv"] = a0[epoch]["full_val/mean_ROUGE1@MTV"]
    with (out / "metrics_50_epochs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["epoch", *a3[0].keys()])
        writer.writeheader()
        writer.writerows({"epoch": e, **a3[e]} for e in range(50))
    plot_curves(a3, a0, gate, out / "pilot_50_curves.png")
    hist = {int(key.removeprefix("val/code_usage/")): value
            for row in a3_rows if row.get("epoch") == 49
            for key, value in row.items() if key.startswith("val/code_usage/")
            and isinstance(value, (int, float)) and value > 0}
    final_classification = {key: value for row in a3_rows if row.get("epoch") == 49
                            for key, value in row.items()
                            if key.startswith("val/disc/") or key == "val/emotion_labeled"}
    finite_failures = [(e, key) for e in range(50) for key, value in a3[e].items()
                       if "loss" in key and isinstance(value, (int, float))
                       and not math.isfinite(value)]
    samples = samples_by_epoch(a3_rows, a3_files[0])
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    recheck_path = out / "epoch=49.sentiment_recheck.json"
    recheck = json.loads(recheck_path.read_text(encoding="utf-8")) if recheck_path.exists() else None
    if recheck and (recheck.get("epoch_index") != 49 or
                    recheck.get("labeled_count") != final_classification.get("val/emotion_labeled")):
        raise RuntimeError("Sentiment recheck does not match epoch 49 logged label count")
    valid_lm = [(e, a3[e]["val/loss_lm"]) for e in range(50)
                if a3[e]["val/loss_lm"] is not None]
    valid_f1 = [(e, a3[e]["val/disc_macro_f1"]) for e in range(50)
                if a3[e]["val/disc_macro_f1"] is not None]
    collapse_epochs = [e for e in range(50) if a3[e]["val/active_code_ratio"] is not None
                       and a3[e]["val/active_code_ratio"] <= 1 / 128 + 1e-6]
    best_lm = min(valid_lm, key=lambda item: item[1])
    best_f1 = max(valid_f1, key=lambda item: item[1])
    summary = {"status": status, "smoke": {"configuration": smoke["configuration"],
               "parameters": smoke["parameters"], "label_distribution_by_phase": smoke["label_distribution_by_phase"]},
               "gate_by_epoch": gate, "epoch_49_code_usage_nonzero": hist,
               "epoch_49_classification": final_classification,
               "epoch_49_labeled_only_sentiment_recheck": recheck,
               "best_validation_lm": {"epoch": best_lm[0], "value": best_lm[1]},
               "best_validation_macro_f1": {"epoch": best_f1[0], "value": best_f1[1]},
               "single_code_validation_epochs": collapse_epochs,
               "finite_loss_failures": finite_failures, "generated_samples": samples,
               "a3_epoch_metrics": a3, "a0_common_epoch_metrics":
               {e: {k: a0[e][k] for k in COMMON} for e in range(50)}}
    (out / "pilot_50_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)
                                                    + "\n", encoding="utf-8")
    lines = ["# A3 Revised: 50 epoch Pilot", "",
             "50 completed epochs: indices 0–49. A3 checkpoint and generation samples were verified before the training process exited.",
             "", f"Checkpoint: `{status['checkpoint']['checkpoint']}`",
             f"SHA256: `{status['checkpoint']['checkpoint_sha256']}`", "",
             "## Configuration", "",
             f"Trainable parameters: {smoke['parameters']['trainable']:,}; frozen: {smoke['parameters']['frozen']:,}.",
             f"Configuration: `{json.dumps(smoke['configuration'], sort_keys=True)}`", "",
             "## Matched epoch metrics", "",
             "Training LM is the mean of logged step samples. Validation metrics are epoch aggregates.", "",
             "| Epoch | A3 train LM | A3 val LM | A0 val LM | A3 val Macro-F1 | A3 val perplexity | A3 active codes | Gate |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for epoch in (0, 4, 5, 9, 10, 19, 29, 39, 49):
        m = a3[epoch]
        lines.append(f"| {epoch} | {fmt(m['train/loss_lm'])} | {fmt(m['val/loss_lm'])} | "
                     f"{fmt(a0[epoch]['val/loss_lm'])} | {fmt(m['val/disc_macro_f1'])} | "
                     f"{fmt(m['val/codebook_perplexity'])} | "
                     f"{fmt(128*m['val/active_code_ratio'] if m['val/active_code_ratio'] is not None else None)} | "
                     f"{fmt(m['gate_effective_sigmoid'])} |")
    lines += ["", "## Full validation text metrics", "",
              "| Epoch | A3 BLEU1@MTV | A0 BLEU1@MTV | A3 ROUGE1@MTV | A0 ROUGE1@MTV | Samples |",
              "|---:|---:|---:|---:|---:|---:|"]
    for epoch in (0, 9, 19, 29, 39, 49):
        lines.append(f"| {epoch} | {fmt(a3[epoch]['full_val/mean_BLEU1@MTV'])} | "
                     f"{fmt(a0[epoch]['full_val/mean_BLEU1@MTV'])} | "
                     f"{fmt(a3[epoch]['full_val/mean_ROUGE1@MTV'])} | "
                     f"{fmt(a0[epoch]['full_val/mean_ROUGE1@MTV'])} | "
                     f"{samples.get(epoch, {}).get('rows', 'missing')} |")
    lines += ["", "## Epoch 49 details", "",
              f"Validation LM loss: {fmt(a3[49]['val/loss_lm'])}; CLIP loss: {fmt(a3[49]['val/loss_clip'])}; "
              f"VQ loss: {fmt(a3[49]['val/loss_vq'])}.",
              f"Validation sentiment accuracy: {fmt(a3[49]['val/disc_accuracy'])}; "
              f"Macro-F1: {fmt(a3[49]['val/disc_macro_f1'])}; "
              f"balanced accuracy: {fmt(a3[49]['val/disc_balanced_accuracy'])}.",
              f"Validation code usage (nonzero entries): `{hist}`.",
              f"Per-class metrics and confusion counts: `{final_classification}`.",
              f"Nonfinite logged losses: `{finite_failures}`.", "",
              f"Best validation LM loss: {best_lm[1]:.4f} at epoch {best_lm[0]}; "
              f"best validation sentiment Macro-F1: {best_f1[1]:.4f} at epoch {best_f1[0]}.",
              f"Validation used only one code in {len(collapse_epochs)}/50 epochs.", "",
              "All 50 epoch metrics are in `metrics_50_epochs.csv`. Complete text samples remain in W&B offline media tables.",
              "", "![Pilot curves](pilot_50_curves.png)", ""]
    if recheck:
        lines += ["## Sentiment CE logging audit", "",
                  f"Logged epoch validation sentiment loss: {fmt(a3[49]['val/loss_disc_emotion'])}. "
                  "Lightning averaged batch losses using all samples, including zero-loss batches with no sentiment labels.",
                  f"Labeled-only CE recomputed on {recheck['labeled_count']} real ZuCo validation samples: "
                  f"{recheck['labeled_only_cross_entropy']:.4f}. "
                  f"Recomputed Macro-F1: {recheck['macro_f1']:.4f}.", ""]
    examples = samples.get(49, {}).get("examples", [])
    if examples:
        lines += ["## Epoch 49 generated examples (ZuCo1 Task 1)", ""]
        for index, example in enumerate(examples, 1):
            source = str(example.get("Input Text", "")).replace("\n", " ")[:220]
            generated = str(example.get("Generated Text", "")).replace("\n", " ")[:220]
            lines += [f"{index}. Input: {source}", "", f"   Generated: {generated}", ""]
    (out / "pilot_50_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(out / "pilot_50_report.md"),
                      "a3_records": len(a3_rows), "a0_records": len(a0_rows),
                      "generation_epochs_found": list(samples),
                      "finite_loss_failures": finite_failures}), flush=True)


if __name__ == "__main__":
    main()
