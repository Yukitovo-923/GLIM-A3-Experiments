"""Add directly usable model configs and verify initialization state round trips."""
import hashlib,json
from pathlib import Path
import torch
from model.glim_multitask import GLIMMultitask

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

for folder in sorted(Path('runs/A3-v2-initialization').glob('V[0-3]')):
    checkpoint=folder/'initialization.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GLIMMultitask.from_glim_checkpoint(checkpoint)
    assert not any(model.load_report.values())
    assert model.vq.distance==saved['hyper_parameters']['vq_distance']
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,saved['state_dict'][k],rtol=0,atol=0)
    (folder/'model_config.json').write_text(json.dumps(saved['hyper_parameters'],indent=2))
    (folder/'SHA256SUMS').write_text('\n'.join(digest(p)+'  '+p.name for p in sorted(folder.iterdir()) if p.is_file() and p.name!='SHA256SUMS')+'\n')
    print(folder.name,'PASS exact initialization restoration',flush=True)
