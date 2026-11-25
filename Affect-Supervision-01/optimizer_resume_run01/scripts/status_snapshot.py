"""Read-only compact snapshot for hourly training/validation follow-up."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main():
    training = read(ROOT / "reports" / "training_status_attempt2.json")
    validation = read(ROOT / "evaluation" / "validation_status.json")
    finalization = read(ROOT / "reports" / "finalization_status.json")
    output = {"checked_at": datetime.now(TZ).isoformat(timespec="seconds"),
              "training": None, "validation": None, "finalization": None,
              "report_ready": (ROOT / "reports" / "RESULTS.md").is_file()}
    if training:
        jobs = training.get("jobs", [])
        active = next((job for job in jobs if job.get("status") == "running"), None)
        output["training"] = {"status": training["status"],
                              "jobs_complete": sum(j["status"] == "complete" for j in jobs),
                              "jobs_total": 12, "active": active and {
                                  k: active[k] for k in ("seed", "arm", "condition", "started_at")},
                              "completed": [],
                              "failed": [{k: j.get(k) for k in ("seed", "arm", "condition", "return_code")}
                                         for j in jobs if j["status"] in ("failed", "blocked")]}
        for job in jobs:
            if job["status"] == "complete":
                finished = read(Path(job["output"]) / "fit_complete.json")
                output["training"]["completed"].append({
                    "seed": job["seed"], "arm": job["arm"],
                    "condition": job["condition"],
                    "epochs": finished["epoch_reached"] if finished else None,
                    "best_val_lm": finished["best_model_score"] if finished else None})
        if active:
            metrics = sorted((Path(active["output"]) / "metrics").glob("epoch_*.json"))
            epochs = [read(p) for p in metrics]
            output["training"]["active_epochs_finished"] = len(epochs)
            output["training"]["active_best_val_lm"] = min(
                (e["metrics"]["val/loss_lm"] for e in epochs), default=None)
            output["training"]["active_latest_val_lm"] = (
                epochs[-1]["metrics"]["val/loss_lm"] if epochs else None)
    if validation:
        jobs = validation.get("jobs", [])
        output["validation"] = {"status": validation["status"],
                                "jobs_complete": sum(j["status"] == "complete" for j in jobs),
                                "jobs_total": 13,
                                "current": validation.get("current_job"),
                                "failed": [j.get("name") for j in jobs
                                           if j["status"] in ("failed", "blocked")]}
    if finalization:
        output["finalization"] = {"status": finalization["status"],
                                  "jobs_complete": sum(j.get("return_code") == 0
                                                       for j in finalization.get("jobs", [])),
                                  "jobs_total": 3}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
