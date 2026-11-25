"""Verify saved model and Adam states without performing a training step."""
import json,sys,hashlib
from pathlib import Path
import torch
from model.glim_multitask import GLIMMultitask

path=Path(sys.argv[1])
saved=torch.load(path,map_location='cpu',weights_only=False)
model=GLIMMultitask.from_glim_checkpoint(path)
assert not any(model.load_report.values())
opt=model.configure_optimizers(); opt.load_state_dict(saved['optimizer_states'][0])
assert saved['loops'] and saved['global_step']>0
for state in opt.state.values():
    for v in state.values():
        if torch.is_tensor(v): assert v.isfinite().all()
for key,value in model.state_dict().items():
    torch.testing.assert_close(value,saved['state_dict'][key],rtol=0,atol=0)
report=dict(status='PASS',kind='MODEL_AND_ADAM_RESTORE_WITHOUT_STEP',checkpoint=str(path),
            epoch=saved['epoch'],global_step=saved['global_step'],distance=model.vq.distance,
            optimizer_parameter_states=len(opt.state),loop_state_present=True,
            batch_size=model.bsz_train,lr=opt.param_groups[0]['lr'])
report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
report['gate_effective']=float(model.attribute.gate.sigmoid().detach())
report['codebook_raw_norm_mean']=float(model.vq.codebook.weight.detach().norm(dim=1).mean())
print(json.dumps(report,indent=2))
