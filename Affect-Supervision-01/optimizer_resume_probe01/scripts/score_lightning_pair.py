"""Score the two B0 validation generations using the pinned project metric protocol."""

import json
from pathlib import Path

from bert_score import BERTScorer

from score_val_predictions import generation_metrics, lexical_metrics, sentiment_metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    model_record = json.loads((ROOT / "evaluation" / "bertscore_model.json").read_text())
    entries = []
    for condition in ("reset", "restored"):
        output = ROOT / "evaluation" / "outputs" / condition
        rows = [json.loads(line) for line in
                (output / "matched_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != 406 or len({row["uid"] for row in rows}) != 40:
            raise RuntimeError("Validation predictions incomplete")
        entries.append((condition, output, rows))
    scorer = BERTScorer(model_type=model_record["path"], num_layers=17, batch_size=16,
                        device="cuda", idf=False, rescale_with_baseline=False,
                        use_fast_tokenizer=False)
    all_rows = [row for _, _, rows in entries for row in rows]
    _, _, scores = scorer.score([row["prediction"] for row in all_rows],
                                [row["reference"] for row in all_rows],
                                batch_size=16, verbose=True)
    for row, score in zip(all_rows, scores.detach().float().cpu().tolist()):
        row["bert_score_f1"] = float(score)
    summary = {"split": "val", "seed": 42, "model_record": model_record,
               "bert_score_hash": scorer.hash, "conditions": {}}
    for condition, output, rows in entries:
        lexical_metrics(rows)
        inference = json.loads((output / "inference_summary.json").read_text())
        result = {"lm_loss": inference["matched"]["lm_loss"],
                  "generation": generation_metrics(rows),
                  "emotion": sentiment_metrics(rows, "disc_probs"),
                  "checkpoint_sha256": inference["checkpoint_sha256"]}
        summary["conditions"][condition] = result
        (output / "matched_scored.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "evaluation" / "lightning_pair_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("SCORED", json.dumps({condition: {
        "lm_loss": summary["conditions"][condition]["lm_loss"],
        "uid_generation": summary["conditions"][condition]["generation"]["uid_balanced_mean"]}
        for condition in ("reset", "restored")}), flush=True)


if __name__ == "__main__":
    main()
