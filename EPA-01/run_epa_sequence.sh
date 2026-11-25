#!/usr/bin/env bash
# Run the three preflighted EPA arms sequentially on one GPU. Stop on any failure.
set -Eeuo pipefail

EPA=/root/autodl-tmp/EPA-01
PY=/root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python
export PYTHONPATH=/root/autodl-tmp/GLIM-main-A3-v2:/root/autodl-tmp/EPA-01
export PYTHONUNBUFFERED=1

mkdir -p "$EPA/logs"
for arm in E0 E1 E2; do
    output="$EPA/runs/${arm}-01"
    log="$EPA/logs/${arm}-01.log"
    if [[ -e "$output" || -e "$log" ]]; then
        echo "$(date -Is) REFUSE existing output or log: $arm" >&2
        exit 2
    fi
done

for arm in E0 E1 E2; do
    free_bytes=$(df -B1 --output=avail "$EPA" | tail -n 1 | tr -d ' ')
    if (( free_bytes < 5368709120 )); then
        echo "$(date -Is) STOP before $arm: less than 5 GiB free" >&2
        exit 3
    fi
    echo "$(date -Is) START $arm; free_bytes=$free_bytes"
    "$PY" "$EPA/run_epa.py" fit \
        --arm-config "$EPA/config_${arm}.json" \
        --common-init "$EPA/initialization/common_init.pt" \
        --smoke-report "$EPA/smoke-run-02/smoke_report.json" \
        --output "$EPA/runs/${arm}-01" \
        > "$EPA/logs/${arm}-01.log" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$EPA/logs/${arm}-01.pid"
    if wait "$child"; then
        if [[ ! -s "$EPA/runs/${arm}-01/checkpoints/last.ckpt" ]]; then
            echo "$(date -Is) STOP: $arm exited without a final resumable checkpoint" >&2
            exit 4
        fi
        echo "$(date -Is) COMPLETE $arm; last checkpoint: $EPA/runs/${arm}-01/checkpoints/last.ckpt"
    else
        status=$?
        echo "$(date -Is) FAIL $arm; exit_status=$status; later arms not started" >&2
        exit "$status"
    fi
done
echo "$(date -Is) COMPLETE E0 E1 E2"
