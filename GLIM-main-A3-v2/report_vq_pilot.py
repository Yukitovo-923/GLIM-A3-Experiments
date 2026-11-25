"""Read-only experiment comparison; never launches training."""
import csv,json,math
from pathlib import Path

root=Path('runs/A3-v2-pilot-5-r2')
init=json.loads(Path('runs/A3-v2-initialization/report.json').read_text())
rows=[json.loads(s) for s in (root/'diagnostics.jsonl').read_text().splitlines()]
old=list(csv.DictReader(Path('original_A3_metrics.csv').open()))
complete=(root/'COMPLETE.json').exists()
stopped=json.loads((root/'STOP.json').read_text()) if (root/'STOP.json').exists() else None
lines=['# A3-v2 initialization and five-epoch pilot','',
       f'Status: {"COMPLETE" if complete else "STOPPED" if stopped else "RUNNING"}; selected {init["selection"]["variant"]}.',
       '','## Initialization (train split only)','',
       '| Variant | Active codes | Perplexity | Top-1 ratio | Real smoke |',
       '|---|---:|---:|---:|---|']
for k,v in init['variants'].items():
    lines.append(f'| {k} | {v["active_codes"]} | {v["perplexity"]:.4f} | {v["top1_ratio"]:.4%} | {v["real_batch_smoke"]["status"]} |')
lines+=['','V3 was selected by active-code coverage, then entropy, then top-one dominance. V2 has slightly higher entropy; V3 is not proven superior in downstream performance.',
        f'Frozen-encoder linear probe: validation Macro-F1={init["probe"]["macro_f1"]:.4f}, accuracy={init["probe"]["accuracy"]:.4f}. Fit used {init["probe"]["train_labeled"]} labeled training rows, not the complete training split.',
        '', '## Matched epochs (validation metrics)', '',
        '| Epoch | A3 LM | v2 LM | A3 F1 | v2 F1 | A3 perplexity | v2 perplexity | v2 active codes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
for p in sorted(root.glob('epoch-*-metrics.json')):
    ep=int(p.name.split('-')[1]); m=json.loads(p.read_text()); a=old[ep]
    lines.append('| '+ ' | '.join(map(str,[ep,a['val/loss_lm'],m.get('val/loss_lm'),a['val/disc_macro_f1'],m.get('val/disc_macro_f1'),a['val/codebook_perplexity'],m.get('val/codebook_perplexity'),128*m.get('val/active_code_ratio',0)]))+' |')
lines+=['','## Monitored training checkpoints','',
        '| Step | Active | Perplexity | Top-1 | Projection std | Updated codes/window | Encoder/projection/codebook gradient norm |',
        '|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    if r['step'] in (1,100,500) or r['step']%1000==0 or r is rows[-1]:
        lines.append(f'| {r["step"]} | {r["active_codes"]} | {r["perplexity"]:.4f} | {r["top1_ratio"]:.4%} | {r["projection_std"]:.6f} | {r["codebook_update_count"]} | {r["gradient_norm"]} |')
lines+=['','Diagnostics aggregate assignment histograms across each 100-step window (first point is step 1). Projection std and gradient norms are the last batch in the window. Updated-code count measures changed parameter rows, including Adam momentum; it is not the number of newly selected codes.',
        '','## Interpretation','',
        'High code utilization alone does not establish semantic or sentiment coding: learned query position can also explain assignments. Compare Macro-F1 and generation quality before claiming task improvements.',
        'Cosine uses normalized vectors for assignment, quantized outputs and VQ losses. Its loss scale differs from Euclidean MSE, so total loss and VQ loss magnitude are not directly comparable.',
        'Original A3 epoch 49 checkpoint was preserved. Dataset split manifests were byte-identical. Only train features initialized codebooks; validation labels were used only to evaluate the frozen linear probe.',
        'Generation retained the original full-validation schedule: only epoch 0 generated new text in this five-epoch run. Later epoch sample directories contain preserved copies of epoch 0 samples, not new generation evaluations.',
        'No 50-epoch training has been started.']
if stopped: lines+=['',f'Stopping diagnostic: `{stopped}`']
(root/'comparison_report.md').write_text('\n'.join(lines)+'\n')
stable=complete and not stopped and (root/'proposed-50-epoch-model-config.json').exists() and all(r['active_codes']>1 and math.isfinite(r['loss']) for r in rows)
for p in root.glob('epoch-*-metrics.json'):
    m=json.loads(p.read_text())
    stable=stable and all(math.isfinite(v) for v in m.values()) and m.get('val/active_code_ratio',0)>1/128
if stable:
    plan=dict(status='PREPARED_NOT_STARTED',max_epochs=50,seed=42,
              checkpoint=str((root/'checkpoints/last.ckpt').resolve()),resume=True,
              output='runs/A3-v2-train-50',
              model_config=str(root/'proposed-50-epoch-model-config.json'),
              prerequisite='Human approval and matching real smoke report for final pilot checkpoint; do not auto-launch')
    (root/'proposed-50-epoch-plan.json').write_text(json.dumps(plan,indent=2))
print('\n'.join(lines))
