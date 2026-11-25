"""Wait for validation completion, then summarize, verify and render results once."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "evaluation" / "validation_status.json"
STATUS = ROOT / "reports" / "finalization_status.json"
TZ = timezone(timedelta(hours=8))


def save(status):
    status["updated_at"] = datetime.now(TZ).isoformat(timespec="seconds")
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def main():
    if STATUS.exists():
        raise RuntimeError("Finalization status exists; refusing duplicate run")
    status = {"status": "waiting_for_validation", "jobs": []}
    save(status)
    while True:
        if VALIDATION.exists():
            validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
            if validation["status"] == "complete":
                break
            if validation["status"] in ("failed", "blocked"):
                status.update(status="blocked", reason="Validation failed or blocked")
                save(status)
                return
        time.sleep(120)
    status["status"] = "running"
    save(status)
    for script in ("summarize_val.py", "verify_outputs.py", "render_report.py"):
        with (ROOT / "evaluation" / f"{script}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, str(ROOT / "evaluation" / script)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status["jobs"].append({"script": script, "return_code": result.returncode})
        if result.returncode:
            status["status"] = "failed"
            save(status)
            raise SystemExit(result.returncode)
        save(status)
    status["status"] = "complete"
    save(status)


if __name__ == "__main__":
    main()
