"""Render a factual validation report from the completed 12-run comparison."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fmt(summary, metric, digits=5):
    value = summary[metric]
    return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}"


def main():
    result = json.loads((ROOT / "evaluation" / "comparison.json").read_text(encoding="utf-8"))
    verification = json.loads((ROOT / "reports" / "verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "PASS" or len(result["runs"]) != 12:
        raise RuntimeError("Cannot report incomplete study")
    lines = [
        "# V3 optimizer-state continuation: validation results",
        "",
        "Three paired seeds (42/43/44), B0/B1, Adam reset/restored. All jobs start from the same V3 epoch-46 model weights. Restored B1 loads the 214 existing Adam states while its two new EPA projector states start empty. All other training and generation settings are shared. Best checkpoints are selected by validation LM loss. No test split was used.",
        "",
        "Metrics are UID-balanced over 40 ZuCo1 Task1 validation sentence UIDs (406 EEG trials), except LM loss, which is target-token weighted. ± is sample SD across three seeds. The source V3 validation LM loss under this protocol was 7.91864.",
        "",
        "| Adam | Arm | LM loss ↓ | BLEU-1 ↑ | BLEU-2 ↑ | BLEU-4 ↑ | ROUGE-L ↑ | BERTScore F1 ↑ | Original emotion F1 ↑ | EPA F1 ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for optimizer in ("reset", "restored"):
        for arm in ("B0_Baseline", "B1_EPA"):
            cell = result["cell_summary"][optimizer][arm]
            lines.append("| " + " | ".join((optimizer, arm,
                fmt(cell, "lm_loss"), fmt(cell, "bleu1"), fmt(cell, "bleu2"),
                fmt(cell, "bleu4"), fmt(cell, "rouge_l"), fmt(cell, "bert_score_f1"),
                fmt(cell, "v3_discrete_macro_f1"),
                fmt(cell, "epa_macro_f1") if arm == "B1_EPA" else "—")) + " |")
    lines.extend(["", "## Paired effects", "",
                  "Differences below are paired by seed; positive LM loss is worse, while positive generation metrics are better.",
                  "", "| Comparison | LM loss | ROUGE-L | BERTScore F1 | Original emotion F1 |",
                  "|---|---:|---:|---:|---:|"])
    for arm in ("B0_Baseline", "B1_EPA"):
        pair = result["paired_restored_minus_reset"][arm]
        lines.append("| Restored − reset, " + arm + " | " + " | ".join(
            fmt(pair, metric) for metric in ("lm_loss", "rouge_l", "bert_score_f1",
                                            "v3_discrete_macro_f1")) + " |")
    for optimizer in ("reset", "restored"):
        pair = result["paired_b1_minus_b0"][optimizer]
        lines.append("| B1 − B0, " + optimizer + " | " + " | ".join(
            fmt(pair, metric) for metric in ("lm_loss", "rouge_l", "bert_score_f1",
                                            "v3_discrete_macro_f1")) + " |")
    lines.extend(["", "## Checkpoints and evidence", "",
                  "| Adam | Arm | Seed | Best epoch (zero-based) | LM loss | ROUGE-L | BERTScore F1 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in result["runs"]:
        lines.append(f"| {row['optimizer']} | {row['arm']} | {row['seed']} | "
                     f"{row['best_epoch_zero_based']} | {row['matched']['lm_loss']:.5f} | "
                     f"{row['matched']['rouge_l']:.5f} | {row['matched']['bert_score_f1']:.5f} |")
    lines.extend(["", "The 12 best-checkpoint SHA256 hashes, optimizer-start audits, 40-UID membership, and paired matched/mismatched EEG prediction sets passed verification (`verification.json`). Per-epoch losses: `../evaluation/epoch_losses.csv`. Per-subject metrics: `../evaluation/subject_matched.csv`. Full predictions and scored rows: `../evaluation/outputs/`.",
                  "", "## Interpretation limits", "",
                  "The three seeds and 40 validation UIDs support a diagnostic comparison, not a final held-out test claim. B0 retains V3's original emotion CE weight 0.2, so B0 is not a no-emotion GLIM model. Validation selected checkpoints, and previously audited test sentences remain sealed in this follow-up. Teacher-forced LM loss and free-generation metrics may move in different directions.",
                  ""])
    (ROOT / "reports" / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE reports/RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
