"""Persist final report after the isolated pilot exits; never starts training."""
import os,subprocess,time,json
from pathlib import Path
root=Path('runs/A3-v2-pilot-5-r2')
pid=int(Path('logs/pilot-5-r2.pid').read_text())
while True:
    if (root/'COMPLETE.json').exists() or (root/'STOP.json').exists(): break
    try: os.kill(pid,0)
    except ProcessLookupError: break
    time.sleep(15)
result=subprocess.run(['.venv/bin/python','report_vq_pilot.py'],capture_output=True,text=True)
(root/'report-generation.log').write_text(result.stdout+result.stderr)
if not (root/'COMPLETE.json').exists() and not (root/'STOP.json').exists():
    (root/'UNEXPECTED_EXIT.json').write_text(json.dumps({'pid':pid,'message':'Process exited without completion or a caught exception; inspect server log.'},indent=2))
