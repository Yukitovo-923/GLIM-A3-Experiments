"""Read-only compact summary of the completed E2 priority continuation."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/root/autodl-tmp/EPA-01-Continuation-50/runs/E2-50-from10-01')
metrics = root / 'metrics'
selected = (0, 6, 9, 10, 12, 15, 20, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 49)
all_sentences = []
for path in sorted(metrics.glob('val_epoch=*.json')):
    data = json.loads(path.read_text())
    score = data['heads']['epa']['sentence_level']
    all_sentences.append((data['epoch'], score))
for epoch, score in all_sentences:
    if epoch in selected:
        print('SENTENCE', json.dumps({
            'epoch': epoch, 'n': score['n'], 'macro_f1': score['macro_f1'],
            'balanced_accuracy': score['balanced_accuracy'],
            'accuracy': score['accuracy'], 'class_f1': score['class_f1'],
            'predicted': score['predicted'], 'confusion': score['confusion'],
        }, ensure_ascii=False))
best_f1_epoch, best_f1 = max(all_sentences, key=lambda p: p[1]['macro_f1'])
print('BEST_VAL_SENTENCE_F1', best_f1_epoch, best_f1['macro_f1'])

run = next(root.glob('wandb/offline-run-*/run-*.wandb'))
datastore = DataStore()
datastore.open_for_scan(str(run))
keys = Counter()
val_history = []
last_train = {}
train_at_val = []
train_lm_by_epoch = defaultdict(list)
train_epa_by_epoch = defaultdict(list)
try:
    while raw := datastore.scan_data():
        record = wandb_internal_pb2.Record()
        record.ParseFromString(raw)
        if not record.HasField('history'):
            continue
        row = {}
        for item in record.history.item:
            key = item.key or '/'.join(item.nested_key)
            if not key:
                continue
            try:
                row[key] = json.loads(item.value_json)
            except (ValueError, TypeError):
                row[key] = item.value_json
            keys[key] += 1
        if 'val/loss_lm' in row:
            val_history.append(row)
            train_at_val.append(last_train.copy())
        if 'epoch' in row and 'train/loss_lm' in row:
            train_lm_by_epoch[int(row['epoch'])].append(float(row['train/loss_lm']))
        if 'epoch' in row and 'train/loss_epa' in row:
            train_epa_by_epoch[int(row['epoch'])].append(float(row['train/loss_epa']))
        for key, value in row.items():
            if key.startswith('train/') and any(x in key for x in (
                    'perplexity', 'active_code', 'top1', 'projection', 'grad_norm')):
                last_train[key] = value
        hist = [row.get(f'train/code_usage/{i}') for i in range(128)]
        if all(value is not None for value in hist) and sum(hist):
            last_train['train/top1_assignment_ratio'] = max(hist) / sum(hist)
finally:
    datastore.close()

print('LOGGED_PROJECTION_KEYS', json.dumps(sorted(k for k in keys if 'projection' in k)))
print('VAL_HISTORY_COUNT', len(val_history))
for epoch in selected:
    losses = train_lm_by_epoch.get(epoch, [])
    epa_losses = train_epa_by_epoch.get(epoch, [])
    if losses:
        print('TRAIN_HISTORY', epoch, json.dumps({
            'logged_batches': len(losses),
            'mean_lm_loss': sum(losses) / len(losses),
            'last_lm_loss': losses[-1],
            'mean_epa_loss': sum(epa_losses) / len(epa_losses) if epa_losses else None,
        }))
vq_by_epoch = []
for i, row in enumerate(val_history):
    epoch = row.get('epoch')
    vq = train_at_val[i]
    ratio = vq.get('train/active_code_ratio')
    perplexity = vq.get('train/codebook_perplexity')
    vq_by_epoch.append((epoch, ratio, perplexity))
    if epoch in selected:
        print('VAL_HISTORY', epoch, json.dumps({
            'epoch_recorded': row.get('epoch'), 'step': row.get('_step'),
            'lm': row.get('val/loss_lm'), 'clip': row.get('val/loss_clip'),
            'epa_loss': row.get('val/loss_epa'), 'total': row.get('val/loss_total'),
            'vq_loss': row.get('val/loss_vq'),
            'vq_codebook_loss': row.get('val/loss_vq_codebook'),
            'vq_commitment_loss': row.get('val/loss_vq_commitment'),
            'active_codes_train_epoch': round(ratio * 128) if ratio is not None else None,
            'perplexity_train_epoch': perplexity,
            'top1_assignment_ratio_train_epoch': vq.get('train/top1_assignment_ratio'),
        }, ensure_ascii=False))
best_lm_idx = min(range(len(val_history)), key=lambda i: val_history[i]['val/loss_lm'])
print('BEST_VAL_LM', val_history[best_lm_idx].get('epoch'), val_history[best_lm_idx]['val/loss_lm'])
for limit in (64, 16, 2, 1):
    first = next(((e, round(r * 128), p) for e, r, p in vq_by_epoch
                  if r is not None and r * 128 <= limit), None)
    print('FIRST_ACTIVE_LE', limit, first)
