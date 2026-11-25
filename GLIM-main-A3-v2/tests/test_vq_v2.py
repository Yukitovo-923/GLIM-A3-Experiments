"""Synthetic tensor tests; not a real EEG experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from model.vq import TokenVQ

torch.manual_seed(42)
for distance in ('euclidean','cosine'):
    vq=TokenVQ(256,distance=distance)
    h=torch.randn(2,96,256,requires_grad=True)
    mask=torch.ones(2,96,dtype=torch.bool); mask[0,40:]=False
    out=vq(h,mask)
    assert out['quantized'].shape==h.shape and out['code_usage_histogram'].sum()==136
    assert out['code_ids'][~mask].eq(-1).all()
    assert out['quantized'][~mask].eq(0).all()
    z=vq.projection(h[mask]); c=vq.codebook.weight
    if distance=='cosine': z=F.normalize(z,dim=-1); c=F.normalize(c,dim=-1)
    ids=torch.cdist(z,c).argmin(-1)
    assert torch.equal(ids,out['code_ids'][mask])
    torch.testing.assert_close(out['loss_vq_codebook'],F.mse_loss(c[ids],z.detach()))
    torch.testing.assert_close(out['loss_vq_commitment'],.25*F.mse_loss(z,c[ids].detach()))
    out['quantized'].square().sum().backward(retain_graph=True)
    assert h.grad[mask].norm()>0 and h.grad[~mask].eq(0).all()
    assert vq.codebook.weight.grad is None
    out['loss_vq'].backward()
    assert vq.codebook.weight.grad.norm()>0
    empty=vq(h,torch.zeros_like(mask))
    assert empty['loss_vq']==0 and empty['active_code_ratio']==0
    if distance=='cosine':
        with torch.no_grad(): vq.codebook.weight.mul_(3)
        assert torch.equal(out['code_ids'],vq(h,mask)['code_ids'])
    print(distance,'PASS mask/nearest-neighbor/loss/STE/codebook-gradient/empty-mask')
