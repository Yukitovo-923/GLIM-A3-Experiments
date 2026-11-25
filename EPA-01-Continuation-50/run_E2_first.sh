#!/usr/bin/env bash
# User-requested priority change: pause E0 at verified epoch 17, run E2 next.
# This script starts E2 only; E1 and E0 are not automatically restarted.
set -Eeuo pipefail

EPA=/root/autodl-tmp/EPA-01
BASE=/root/autodl-tmp/EPA-01-Continuation-50
PY=/root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python
SRC="$EPA/runs/E2-10-02/checkpoints/last.ckpt"
OUT="$BASE/runs/E2-50-from10-01"
LOG="$BASE/logs/E2-50-from10-01.log"
EXPECTED=04b4cf703a5bbbe6ac28fcc206c43e3e8a8773ca67fba0a3678ab0ab1d4a682e

export PYTHONPATH=/root/autodl-tmp/GLIM-main-A3-v2:/root/autodl-tmp/EPA-01
export PYTHONUNBUFFERED=1
exec > >(tee -a "$BASE/logs/E2-priority-queue.log") 2>&1
printf '%s E2_PRIORITY_START controller_pid=%s\n' "$(date -Is)" "$$"
printf '%s\n' "$$" > "$BASE/E2-priority.pid"

[[ ! -e "$OUT" && ! -e "$LOG" ]]
[[ -s "$SRC" ]]
[[ "$(sha256sum "$SRC" | cut -d ' ' -f1)" == "$EXPECTED" ]]
[[ "$(sha256sum "$EPA/run_epa_10.py" | cut -d ' ' -f1)" == 5ab05f96a5d0beb915ce49e37ce7a3dd4dfa7227e85f172d292b23d60532143b ]]
[[ "$(sha256sum "$EPA/epa_model.py" | cut -d ' ' -f1)" == e7a3c9ea320e0116079e796000e57d6aba99023ba84099f4e9599c04041e807b ]]
[[ "$(sha256sum "$EPA/initialization/common_init.pt" | cut -d ' ' -f1)" == b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e ]]
"$PY" "$EPA/check_epa_final_epoch.py" "$SRC" 9 19880 > "$BASE/reports/E2-source-check.json"
avail=$(df -B1 --output=avail "$BASE" | tail -n1 | tr -d ' ')
(( avail >= 8589934592 ))
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
printf '%s E2_START source_epoch=9 source_step=19880 free_bytes=%s output=%s\n' "$(date -Is)" "$avail" "$OUT"

"$PY" "$EPA/run_epa_10.py" fit \
  --arm-config "$EPA/config_E2.json" \
  --common-init "$EPA/initialization/common_init.pt" \
  --smoke-report "$EPA/smoke-run-02/smoke_report.json" \
  --checkpoint "$SRC" \
  --max-epochs 50 \
  --output "$OUT" > "$LOG" 2>&1 &
child=$!
printf '%s\n' "$child" > "$BASE/logs/E2-50-from10-01.pid"
printf '%s E2_TRAIN_PID %s\n' "$(date -Is)" "$child"

count=0
while kill -0 "$child" 2>/dev/null; do
  sleep 30
  count=$((count + 1))
  avail=$(df -B1 --output=avail "$BASE" | tail -n1 | tr -d ' ')
  if (( avail < 5368709120 )); then
    printf '%s E2_DISK_LOW free_bytes=%s\n' "$(date -Is)" "$avail" | tee "$BASE/reports/E2-incident.txt"
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
    exit 6
  fi
  if grep -Eiq 'CUDA out of memory|OutOfMemoryError|loss[^[:alnum:]]{0,24}[=:][[:space:]]*(nan|inf)([^[:alnum:]]|$)' "$LOG"; then
    printf '%s E2_NUMERICAL_OR_OOM_ALERT\n' "$(date -Is)" | tee "$BASE/reports/E2-incident.txt"
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
    exit 7
  fi
  if (( count % 10 == 0 )); then
    printf '%s E2_HEALTH pid=%s free_bytes=%s gpu=' "$(date -Is)" "$child" "$avail"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '
    printf '\n'
  fi
done
if wait "$child"; then
  final="$OUT/checkpoints/last.ckpt"
  [[ -s "$final" ]]
  "$PY" "$EPA/check_epa_final_epoch.py" "$final" 49 99400 > "$BASE/reports/E2-final-check.json"
  printf '%s E2_COMPLETE final_epoch=49 final_step=99400 final=%s\n' "$(date -Is)" "$final"
else
  status=$?
  printf '%s E2_FAIL exit=%s\n' "$(date -Is)" "$status" >&2
  exit "$status"
fi
