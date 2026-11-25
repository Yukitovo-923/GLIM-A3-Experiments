#!/usr/bin/env bash
set -Eeuo pipefail

EPA=/root/autodl-tmp/EPA-01
PY=/root/autodl-tmp/GLIM-main-A3-v2/.venv/bin/python
declare -A EXPECTED=(
  [E0]=36ecf0b6807c8cc6155c55ab3833796ea58ba3df751b7ab3f9ab4410dddad2b9
  [E1]=356d980e562b1642771b0fed588fd231ebcc1c94ca3105708e2d17d6674ac080
  [E2]=04b4cf703a5bbbe6ac28fcc206c43e3e8a8773ca67fba0a3678ab0ab1d4a682e
)

printf 'DATE '; date -Is
printf 'HOST '; hostname
printf 'DISK '; df -h /root/autodl-tmp | tail -n 1
printf 'GPU\n'; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
printf 'GPU_PROCESSES\n'; nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>&1 || true
printf 'TRAINING_PROCESSES\n'; pgrep -af 'run_epa(_10)?\.py|sequence.*epa|train.py' || true
printf 'SOURCE_SHA\n'; sha256sum "$EPA/run_epa_10.py" "$EPA/epa_model.py" "$EPA/common.json"

[[ -x "$PY" ]]
[[ -f "$EPA/smoke-run-02/smoke_report.json" ]]
[[ -f "$EPA/initialization/common_init.pt" ]]
[[ "$(sha256sum "$EPA/initialization/common_init.pt" | cut -d ' ' -f1)" == b35322f8dc910a415469c60cd1d56e948c868c3fbd710f7ba16fa209c925b02e ]]

for arm in E0 E1 E2; do
  ckpt="$EPA/runs/${arm}-10-02/checkpoints/last.ckpt"
  [[ -s "$ckpt" ]]
  [[ "$(sha256sum "$ckpt" | cut -d ' ' -f1)" == "${EXPECTED[$arm]}" ]]
  "$PY" "$EPA/check_epa_final_epoch.py" "$ckpt" 9 19880
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/root/autodl-tmp/EPA-01')
config = json.loads((root/'common.json').read_text())
assert config['training']['max_epochs'] == 50
assert config['training']['batch_size_train'] == 72
assert config['training']['learning_rate'] == 1e-4
assert config['training']['precision'] == 'bf16-mixed'
assert config['training']['optimizer'] == 'Adam'
for arm in ('E0','E1','E2'):
    assert json.loads((root/f'config_{arm}.json').read_text())['arm'] == arm
    assert not (Path('/root/autodl-tmp/EPA-01-Continuation-50')/'runs'/f'{arm}-50-from10-01').exists()
for name in ('data', 'labels', 'text_model', 'e1_manifest', 'prototype'):
    assert Path(config['assets'][name]).exists(), name
print('CONFIG_ASSETS_OK max_epochs=50 batch=72 lr=1e-4 precision=bf16-mixed')
PY

printf 'PREFLIGHT_PASS\n'
