"""Run locked test inference and scoring only after all six training jobs finish."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "evaluation" / "evaluation_status.json"
TZ = timezone(timedelta(hours=8))


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def save(value):
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def main():
    if STATUS.exists():
        raise RuntimeError("Refusing to overwrite an evaluation status")
    training = json.loads((ROOT / "reports" / "training_status.json").read_text(encoding="utf-8"))
    if training["status"] != "complete" or len(training["jobs"]) != 6:
        raise RuntimeError("All six training jobs must complete before test evaluation")
    if not (ROOT / "evaluation" / "bertscore_model.json").is_file():
        raise RuntimeError("Pinned BERTScore evaluator is missing")
    config = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8"))
    jobs = [(seed, arm) for seed in config["training"]["seeds"] for arm in config["arms"]]
    for seed, arm in jobs:
        if not (ROOT / "runs" / arm / f"seed-{seed}" / "fit_complete.json").is_file():
            raise RuntimeError(f"Missing training completion: {arm} seed {seed}")
    env = os.environ.copy()
    evaluation_site = ROOT / ".venv-eval" / "lib" / "python3.12" / "site-packages"
    env["PYTHONPATH"] = os.pathsep.join((str(evaluation_site), str(ROOT / "models"),
                                         str(ROOT / "scripts"), str(ROOT / "evaluation")))
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    output_root = ROOT / "evaluation" / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": now(), "current_job": None, "jobs": []}
    save(state)
    commands = [("summarize_training", [sys.executable,
                                       str(ROOT / "scripts" / "summarize_training.py")])]
    for seed, arm in jobs:
        output = output_root / f"{arm}_seed{seed}_test"
        commands.append((f"infer_{arm}_seed{seed}", [
            sys.executable, str(ROOT / "evaluation" / "infer.py"),
            "--config", str(ROOT / "configs" / "common.json"),
            "--arm", arm, "--seed", str(seed), "--split", "test",
            "--output", str(output),
        ]))
    commands.append(("score_predictions", [sys.executable,
                                           str(ROOT / "evaluation" / "score_predictions.py")]))
    commands.append(("aggregate_results", [sys.executable,
                                           str(ROOT / "evaluation" / "aggregate_results.py")]))
    for name, command in commands:
        entry = {"name": name, "started_at": now(), "status": "running",
                 "log": str(ROOT / "evaluation" / f"{name}.log")}
        state["current_job"] = name
        state["jobs"].append(entry)
        save(state)
        with Path(entry["log"]).open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        entry.update(return_code=completed.returncode, ended_at=now(),
                     status="complete" if completed.returncode == 0 else "failed")
        state["current_job"] = None
        if completed.returncode:
            state.update(status="failed", ended_at=now())
            save(state)
            raise SystemExit(completed.returncode)
        save(state)
    state.update(status="complete", ended_at=now())
    save(state)


if __name__ == "__main__":
    main()
