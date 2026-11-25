#!/usr/bin/env bash
# Resume each EPA arm from its verified epoch-9 Lightning checkpoint to a total
# of 50 epochs. All new files go to the independent continuation directory.
set -Eeuo pipefail

EPA=/root/autodl-tmp/EPA-01
BASE=/root/autodl-tmp/EPA-01-Continuation-50
PY=/root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python
export PYTHONPATH=/root/autodl-tmp/GLIM-main-A3-v2:/root/autodl-tmp/EPA-01
export PYTHONUNBUFFERED=1

declare -A EXPECTED=(
  [E0]=36ecf0b6807c8cc6155c55ab3833796ea58ba3df751b7ab3f9ab4410dddad2b9
  [E1]=356d980e562b1642771b0fed588fd231ebcc1c94ca3105708e2d17d6674ac080
  [E2]=04b4cf703a5bbbe6ac28fcc206c43e3e8a8773ca67fba0a3678ab0ab1d4a682e
)

mkdir -p "$BASE/logs" "$BASE/runs" "$BASE/reports"
exec > >(tee -a "$BASE/logs/queue50.log") 2>&1
printf '%s QUEUE_START pid=%s target_total_epochs=50\n' "$(date -Is)" "$$"
printf '%s\n' "$$" > "$BASE/queue.pid"
[[ "$(sha256sum "$EPA/run_epa_10.py" | cut -d ' ' -f1)" == 5ab05f96a5d0beb915ce49e37ce7a3dd4dfa7227e85f172d292b23d60532143b ]]
[[ "$(sha256sum "$EPA/epa_model.py" | cut -d ' ' -f1)" == e7a3c9ea320e0116079e796000e57d6aba99023ba84099f4e9599c04041e807b ]]
[[ "$(sha256sum "$EPA/initialization/common_init.pt" | cut -d ' ' -f1)" == b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e ]]

for arm in E0 E1 E2; do
  src="$EPA/runs/${arm}-10-02/checkpoints/last.ckpt"
  out="$BASE/runs/${arm}-50-from10-01"
  log="$BASE/logs/${arm}-50-from10-01.log"
  if [[ ! -s "$src" || -e "$out" || -e "$log" ]]; then
    printf '%s STOP %s missing source or existing output/log\n' "$(date -Is)" "$arm" >&2
    exit 2
  fi
  actual=$(sha256sum "$src" | cut -d ' ' -f1)
  if [[ "$actual" != "${EXPECTED[$arm]}" ]]; then
    printf '%s STOP %s source checkpoint SHA mismatch\n' "$(date -Is)" "$arm" >&2
    exit 3
  fi
  "$PY" "$EPA/check_epa_final_epoch.py" "$src" 9 19880 > "$BASE/reports/${arm}-source-check.json"
  avail=$(df -B1 --output=avail "$BASE" | tail -n1 | tr -d ' ')
  if (( avail < 8589934592 )); then
    printf '%s STOP %s less than 8 GiB disk available: %s bytes\n' "$(date -Is)" "$arm" "$avail" >&2
    exit 4
  fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    printf '%s STOP %s GPU already has another compute process\n' "$(date -Is)" "$arm" >&2
    exit 5
  fi
  printf '%s START %s source_epoch=9 source_step=19880 source_sha=%s free_bytes=%s output=%s\n' \
    "$(date -Is)" "$arm" "$actual" "$avail" "$out"
  "$PY" "$EPA/run_epa_10.py" fit \
    --arm-config "$EPA/config_${arm}.json" \
    --common-init "$EPA/initialization/common_init.pt" \
    --smoke-report "$EPA/smoke-run-02/smoke_report.json" \
    --checkpoint "$src" \
    --max-epochs 50 \
    --output "$out" > "$log" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$BASE/logs/${arm}-50-from10-01.pid"
  printf '%s TRAIN_PID %s %s\n' "$(date -Is)" "$arm" "$child"

  monitor_count=0
  while kill -0 "$child" 2>/dev/null; do
    sleep 30
    monitor_count=$((monitor_count + 1))
    avail=$(df -B1 --output=avail "$BASE" | tail -n1 | tr -d ' ')
    if (( avail < 5368709120 )); then
      printf '%s DISK_LOW %s free_bytes=%s\n' "$(date -Is)" "$arm" "$avail" | tee "$BASE/reports/${arm}-incident.txt"
      kill -TERM "$child" 2>/dev/null || true
      wait "$child" || true
      exit 6
    fi
    if grep -Eiq 'CUDA out of memory|OutOfMemoryError|loss[^[:alnum:]]{0,24}[=:][[:space:]]*(nan|inf)([^[:alnum:]]|$)' "$log"; then
      printf '%s NUMERICAL_OR_OOM_ALERT %s\n' "$(date -Is)" "$arm" | tee "$BASE/reports/${arm}-incident.txt"
      kill -TERM "$child" 2>/dev/null || true
      wait "$child" || true
      exit 7
    fi
    if (( monitor_count % 10 == 0 )); then
      printf '%s HEALTH %s pid=%s free_bytes=%s gpu=' "$(date -Is)" "$arm" "$child" "$avail"
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '
      printf '\n'
    fi
  done
  if wait "$child"; then
    final="$out/checkpoints/last.ckpt"
    if [[ ! -s "$final" ]]; then
      printf '%s STOP %s exited without final checkpoint\n' "$(date -Is)" "$arm" >&2
      exit 8
    fi
    "$PY" "$EPA/check_epa_final_epoch.py" "$final" 49 99400 \
      > "$BASE/reports/${arm}-final-check.json" || {
      printf '%s STOP %s final checkpoint invalid\n' "$(date -Is)" "$arm" >&2
      exit 9
    }
    printf '%s COMPLETE %s final_epoch=49 final_step=99400 final=%s\n' "$(date -Is)" "$arm" "$final"
  else
    status=$?
    printf '%s FAIL %s exit=%s; later arms not started\n' "$(date -Is)" "$arm" "$status" >&2
    exit "$status"
  fi
done
printf '%s QUEUE_COMPLETE E0 E1 E2\n' "$(date -Is)"
