#!/usr/bin/env bash
# Exact total budget: E0 resumes from completed epoch 1; E1/E2 start from common init.
set -Eeuo pipefail

EPA=/root/autodl-tmp/EPA-01
PY=/root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python
export PYTHONPATH=/root/autodl-tmp/GLIM-main-A3-v2:/root/autodl-tmp/EPA-01
export PYTHONUNBUFFERED=1

resume="$EPA/runs/E0-01/checkpoints/epoch=1-step=3976.ckpt"
expected_resume_sha=b103118f9af77c314eccba4c283913469aabff5b1d1ca1d5e2ab9b357bc3972d
actual_resume_sha=$(sha256sum "$resume" | cut -d ' ' -f 1)
if [[ "$actual_resume_sha" != "$expected_resume_sha" ]]; then
    echo "E0 resume checkpoint SHA256 changed; refusing to start" >&2
    exit 2
fi

for arm in E0 E1 E2; do
    name="${arm}-10-02"
    if [[ -e "$EPA/runs/$name" || -e "$EPA/logs/$name.log" ]]; then
        echo "Existing output or log for $name; refusing to overwrite" >&2
        exit 3
    fi
done

for arm in E0 E1 E2; do
    name="${arm}-10-02"
    free_bytes=$(df -B1 --output=avail "$EPA" | tail -n 1 | tr -d ' ')
    if (( free_bytes < 5368709120 )); then
        echo "$(date -Is) STOP before $arm: less than 5 GiB free" >&2
        exit 4
    fi
    extra=()
    if [[ "$arm" == E0 ]]; then
        extra=(--checkpoint "$resume")
    fi
    echo "$(date -Is) START $arm; total_max_epochs=10; free_bytes=$free_bytes; resume=${extra[*]:-none}"
    "$PY" "$EPA/run_epa_10.py" fit \
        --arm-config "$EPA/config_${arm}.json" \
        --common-init "$EPA/initialization/common_init.pt" \
        --smoke-report "$EPA/smoke-run-02/smoke_report.json" \
        --max-epochs 10 \
        --output "$EPA/runs/$name" \
        "${extra[@]}" \
        > "$EPA/logs/$name.log" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$EPA/logs/$name.pid"
    if wait "$child"; then
        last="$EPA/runs/$name/checkpoints/last.ckpt"
        if [[ ! -s "$last" ]]; then
            echo "$(date -Is) STOP: $arm lacks a final resumable checkpoint" >&2
            exit 5
        fi
        "$PY" "$EPA/check_epa_final_epoch.py" "$last" 9 19880 \
            > "$EPA/runs/$name/checkpoint_integrity.json" || {
            echo "$(date -Is) STOP: $arm last checkpoint did not complete epoch 9" >&2
            exit 6
        }
        echo "$(date -Is) COMPLETE $arm; epoch=9; global_step=19880; last=$last"
    else
        status=$?
        echo "$(date -Is) FAIL $arm; exit_status=$status; later arms not started" >&2
        exit "$status"
    fi
done
echo "$(date -Is) COMPLETE E0 E1 E2; total_epochs_per_arm=10"
