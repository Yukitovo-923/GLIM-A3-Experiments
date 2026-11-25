"""Score seven Task1 validation predictions with the original pinned protocol."""

import json
from pathlib import Path

from bert_score import BERTScorer
from score_predictions import lexical_metrics, one_condition


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent


def main():
    model_record = json.loads((ROOT / "evaluation" / "bertscore_model.json").read_text(encoding="utf-8"))
    if model_record["model"] != "roberta-large":
        raise RuntimeError("Unexpected BERTScore model")
    jobs = [("source_v3_epoch46", "source", None)]
    jobs.extend((f"{arm}_seed{seed}", arm, seed)
                for arm in ("B0_Baseline", "B1_EPA") for seed in (42, 43, 44))
    pending = []
    all_rows = []
    for folder, arm, seed in jobs:
        output = OUT / "validation_checkpoints" / folder
        meta = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if meta["split"] != "val" or meta["matched"]["trial_count"] != 406:
            raise RuntimeError("Wrong or incomplete validation set")
        rows = [json.loads(line) for line in (output / "matched_predictions.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        if len(rows) != 406 or len({row["uid"] for row in rows}) != 40:
            raise RuntimeError("Wrong validation prediction count")
        pending.append((output, arm, seed, rows, meta))
        all_rows.extend(rows)
    scorer = BERTScorer(model_type=model_record["path"], num_layers=17,
                        batch_size=16, device="cuda", idf=False,
                        rescale_with_baseline=False, use_fast_tokenizer=False)
    _, _, scores = scorer.score([row["prediction"] for row in all_rows],
                                [row["reference"] for row in all_rows],
                                batch_size=16, verbose=False)
    if len(scores) != len(all_rows):
        raise RuntimeError("Incomplete BERTScore output")
    for row, score in zip(all_rows, scores.detach().float().cpu().tolist()):
        row["bert_score_f1"] = float(score)
    summary = []
    for output, arm, seed, rows, meta in pending:
        lexical_metrics(rows)
        metrics = one_condition(rows, meta["matched"])
        record = {"arm": arm, "seed": seed, "split": "val", "checkpoint": meta["checkpoint"],
                  "checkpoint_sha256": meta["checkpoint_sha256"],
                  "bert_score_package_hash": scorer.hash, "protocol": "original score_predictions.py",
                  "matched": metrics}
        (output / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        primary = metrics["generation"]["uid_balanced_mean"]
        summary.append({"arm": arm, "seed": seed, "lm_loss": metrics["lm_loss"],
                        **primary,
                        "v3_discrete_macro_f1": metrics["emotion"]["v3_discrete_head"]["sentence"]["macro_f1"],
                        "epa_macro_f1": (metrics["emotion"]["epa"]["sentence"]["macro_f1"]
                                         if metrics["emotion"]["epa"] else None)})
    result = {"split": "val", "n_trials_per_checkpoint": 406,
              "n_uids_per_checkpoint": 40, "unit": "UID-balanced mean", "rows": summary}
    (OUT / "validation_checkpoints" / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
