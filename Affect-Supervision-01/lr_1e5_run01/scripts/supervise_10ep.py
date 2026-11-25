"""Run the six paired, at-most-10-epoch jobs sequentially and record status."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "common.json"
SMOKE = ROOT / "reports" / "smoke_run_01" / "smoke_report.json"
STATUS = ROOT / "reports" / "training_status.json"
TZ = timezone(timedelta(hours=8))
MIN_FREE_BYTES = 5 * 1024**3


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def save_status(value):
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def check_resources():
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Only {free} bytes of data-disk space remain")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    if gpu.stdout.strip():
        raise RuntimeError(f"Another GPU compute process is active: {gpu.stdout.strip()}")
    return free


def main():
    if STATUS.exists():
        raise RuntimeError(f"Refusing to overwrite existing training status: {STATUS}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["training"]["max_epochs"] != 10:
        raise RuntimeError("This sequence requires the locked 10-epoch budget")
    if not SMOKE.is_file():
        raise FileNotFoundError(SMOKE)
    jobs = [(seed, arm) for seed in config["training"]["seeds"]
            for arm in config["arms"]]
    if len(jobs) != 6:
        raise RuntimeError("Expected three paired seeds and two arms")
    for seed, arm in jobs:
        output = ROOT / "runs" / arm / f"seed-{seed}"
        if output.exists():
            raise RuntimeError(f"Refusing to reuse output: {output}")
    state = {"experiment": "Affect-Supervision-01", "status": "running",
             "started_at": now(), "current_job": None, "jobs": [],
             "max_epochs_per_job": 10, "seeds": config["training"]["seeds"]}
    save_status(state)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "models"), str(ROOT / "scripts")))
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "offline"
    for seed, arm in jobs:
        entry = {"seed": seed, "arm": arm, "status": "running", "started_at": now(),
                 "output": str(ROOT / "runs" / arm / f"seed-{seed}"),
                 "log": str(ROOT / "runs" / arm / f"seed-{seed}.train.log")}
        try:
            entry["disk_free_bytes_at_start"] = check_resources()
        except Exception as error:
            entry.update(status="blocked", error=str(error), ended_at=now())
            state["jobs"].append(entry)
            state.update(status="blocked", current_job=None, ended_at=now())
            save_status(state)
            raise
        state["jobs"].append(entry)
        state["current_job"] = {"seed": seed, "arm": arm}
        save_status(state)
        command = [sys.executable, str(ROOT / "scripts" / "run_train.py"),
                   "--config", str(CONFIG), "--arm", arm, "--seed", str(seed),
                   "--smoke-report", str(SMOKE), "--output", entry["output"]]
        with Path(entry["log"]).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        entry.update(return_code=result.returncode, ended_at=now(),
                     status="complete" if result.returncode == 0 else "failed")
        state["current_job"] = None
        if result.returncode:
            state.update(status="failed", ended_at=now())
            save_status(state)
            raise SystemExit(result.returncode)
        save_status(state)
    state.update(status="complete", ended_at=now())
    save_status(state)


if __name__ == "__main__":
    main()
