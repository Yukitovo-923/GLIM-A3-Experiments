"""Build the Pilot report once finish_at_50 confirms a safe completed run."""

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/A3-revised-train-01"
OUT = RUN / "pilot_artifacts"
STATUS = OUT / "finish_at_50.status.json"
A0 = OUT / "a0_history_through_epoch_49.wandb"


def main():
    if not A0.is_file():
        raise FileNotFoundError(A0)
    while True:
        status = json.loads(STATUS.read_text(encoding="utf-8"))
        state = status.get("state")
        if state == "finished" and status.get("training_process_exited"):
            break
        if state in ("missed_target_checkpoint", "training_exited_before_target", "sigterm_timeout"):
            raise RuntimeError(f"A3 did not finish safely: {state}")
        time.sleep(30)
    command = [sys.executable, "-B", str(ROOT / "deployment/build_50_epoch_report.py"),
               "--a3-run", str(RUN), "--a0-wandb", str(A0),
               "--smoke-report", str(ROOT / "runs/A3-revised-server-smoke/smoke_report.json"),
               "--output-dir", str(OUT)]
    subprocess.run(command, cwd=ROOT, check=True)
    print("Pilot report completed", flush=True)


if __name__ == "__main__":
    main()
