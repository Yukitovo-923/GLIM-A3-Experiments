"""After six training runs complete, evaluate validation only, including EEG mismatch."""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "reports" / "training_status.json"
STATUS = ROOT / "evaluation" / "validation_status.json"
TZ = timezone(timedelta(hours=8))


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def save(state):
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def wait_for_training():
    while True:
        state = json.loads(TRAINING.read_text(encoding="utf-8"))
        if state["status"] == "complete":
            if len(state["jobs"]) != 6 or any(job["status"] != "complete" for job in state["jobs"]):
                raise RuntimeError("Training status is inconsistent")
            return
        if state["status"] in ("failed", "blocked"):
            raise RuntimeError(f"Training ended with status {state['status']}")
        time.sleep(30)


def wait_for_idle_gpu():
    while True:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                 "--format=csv,noheader"], check=True,
                                capture_output=True, text=True)
        if not result.stdout.strip():
            return
        time.sleep(30)


def main():
    if STATUS.exists():
        raise RuntimeError("Refusing to overwrite prior validation status")
    config = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8"))
    if config["training"]["learning_rate"] != 1e-5:
        raise RuntimeError("Wrong experiment LR")
    jobs = [(seed, arm) for seed in config["training"]["seeds"] for arm in config["arms"]]
    if len(jobs) != 6:
        raise RuntimeError("Expected six runs")
    wait_for_training()
    for seed, arm in jobs:
        if not (ROOT / "runs" / arm / f"seed-{seed}" / "fit_complete.json").is_file():
            raise RuntimeError(f"Missing completed run: {arm} seed {seed}")
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("Insufficient disk for validation outputs")
    wait_for_idle_gpu()
    env = os.environ.copy()
    eval_site = ROOT.parent / ".venv-eval" / "lib" / "python3.12" / "site-packages"
    env["PYTHONPATH"] = os.pathsep.join((str(eval_site), str(ROOT / "models"),
                                         str(ROOT / "scripts"), str(ROOT / "evaluation")))
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    outputs = ROOT / "evaluation" / "outputs"
    outputs.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "split": "val", "started_at": now(),
              "current_job": None, "jobs": []}
    save(status)
    commands = []
    for seed, arm in jobs:
        output = outputs / f"{arm}_seed{seed}_val"
        commands.append((f"infer_{arm}_seed{seed}", [
            sys.executable, str(ROOT / "evaluation" / "infer.py"),
            "--config", str(ROOT / "configs" / "common.json"),
            "--arm", arm, "--seed", str(seed), "--split", "val",
            "--output", str(output),
        ]))
    commands.append(("score_val_predictions", [
        sys.executable, str(ROOT / "evaluation" / "score_val_predictions.py")]))
    for name, command in commands:
        entry = {"name": name, "status": "running", "started_at": now(),
                 "log": str(ROOT / "evaluation" / f"{name}.log")}
        status["current_job"] = name
        status["jobs"].append(entry)
        save(status)
        with Path(entry["log"]).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        entry.update(status="complete" if result.returncode == 0 else "failed",
                     return_code=result.returncode, ended_at=now())
        status["current_job"] = None
        if result.returncode:
            status.update(status="failed", ended_at=now())
            save(status)
            raise SystemExit(result.returncode)
        save(status)
    status.update(status="complete", ended_at=now())
    save(status)


if __name__ == "__main__":
    main()
