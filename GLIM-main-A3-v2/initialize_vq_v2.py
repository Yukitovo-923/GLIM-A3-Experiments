"""Train-split-only VQ initialization comparison; no training of GLIM."""
import json, hashlib, copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data.datamodule import GLIMDataModule
from model.glim_multitask import GLIMMultitask
from model.vq import usage_stats

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    torch.manual_seed(42); np.random.seed(42)
    out=Path('runs/A3-v2-initialization'); out.mkdir(parents=True,exist_ok=False)
    cfg=json.load(open('configs/a3_revised.json')); variant=cfg.pop('variant')
    source=Path('checkpoints/glim-zuco-epoch=199-step=49600.ckpt')
    model=GLIMMultitask.from_glim_checkpoint(source,variant,**cfg).cuda().eval()
    initial=copy.deepcopy(model.state_dict())
    dm=GLIMDataModule('data/tmp/zuco_eeg_label_8variants.df',bsz_train=model.bsz_train,bsz_val=model.bsz_val,num_workers=0)
    dm.label_reference=Path('data/tmp/zuco_label_8variants.df'); dm.manifest_path=out/'split_manifest.csv'
    dm.setup('fit')
    features=[]; hidden=[]; train_x=[]; train_y=[]; val_x=[]; val_y=[]; uids=[]; smoke=None
    for phase,loader in [('train',dm.train_dataloader()),('val',dm.val_dataloader())]:
        for i,b in enumerate(loader):
            if phase=='train' and i>=32: break
            if phase=='train' and smoke is None: smoke=b
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                eeg=b['eeg'].cuda(); mask=b['mask'].cuda()
                ids=model.p_embedder.encode(b['prompt'],device=eeg.device)
                h,_=model.eeg_encoder(eeg,mask,model.p_embedder(ids,model.eval_pembed))
                valid=mask.bool().any(1)
                z=model.vq.projection(h).float()
                if phase=='train':
                    hidden.append(h[valid].float().cpu()); features.append(z[valid].cpu()); uids.extend(list(b['text uid']))
                pooled=h.float().mean(1).cpu().numpy()
            for j,(lab,task,dataset) in enumerate(zip(b['sentiment label'],b['raw task key'],b['prompt'][1])):
                if valid[j] and task=='task1' and dataset=='ZuCo1' and lab in model.sentiment_labels:
                    (train_x if phase=='train' else val_x).append(pooled[j]); (train_y if phase=='train' else val_y).append(model.sentiment_labels.index(lab))
    h=torch.cat(hidden).reshape(-1,256); z=torch.cat(features).reshape(-1,256)
    g=torch.Generator().manual_seed(42); pick=torch.randperm(len(z),generator=g)[:32768]
    x=z[pick].cuda()
    tx=torch.tensor(np.array(train_x)); ty=torch.tensor(train_y)
    vx=torch.tensor(np.array(val_x)); vy=torch.tensor(val_y)
    mean=tx.mean(0); std=tx.std(0).clamp_min(1e-6)
    probe=torch.nn.Linear(256,3); opt=torch.optim.LBFGS(probe.parameters(),max_iter=200,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss=F.cross_entropy(probe((tx-mean)/std),ty)+1e-4*probe.weight.square().mean(); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad(): pred=probe((vx-mean)/std).argmax(1)
    cm=torch.bincount(3*vy+pred,minlength=9).reshape(3,3).float()
    precision=cm.diag()/cm.sum(0).clamp_min(1); recall=cm.diag()/cm.sum(1).clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    report={'seed':42,'sampling':{'split':'train','batches':32,'EEG_rows':len(uids),'tokens_available':len(z),'tokens_fit':len(x),'text_uids':sorted(set(map(int,uids)))},'H':{'mean':h.mean().item(),'std':h.std().item()},'Z':{'mean':z.mean().item(),'std':z.std().item()},'probe':{'train_labeled':len(train_y),'val_labeled':len(val_y),'accuracy':float(cm.diag().sum()/cm.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'confusion':cm.tolist(),'method':'train-only standardized frozen H, linear softmax, LBFGS200; sampled train subset'},'variants':{}}
    torch.save({'state_dict':probe.state_dict(),'mean':mean,'std':std},out/'linear_probe.pt')
    def cluster(spherical):
        a=F.normalize(x,dim=-1) if spherical else x
        centers=a[torch.randperm(len(a),generator=torch.Generator().manual_seed(42))[:128].cuda()].clone()
        for _ in range(30):
            idx=(centers.square().sum(-1)-2*a@centers.T).argmin(-1)
            counts=torch.bincount(idx,minlength=128); sums=torch.zeros_like(centers).index_add_(0,idx,a)
            new=sums/counts.clamp_min(1)[:,None]
            new[counts==0]=centers[counts==0]
            centers=F.normalize(new,dim=-1) if spherical else new
        return centers
    for name,dist,km in [('V0','euclidean',False),('V1','cosine',False),('V2','euclidean',True),('V3','cosine',True)]:
        model.load_state_dict(initial); model.vq.distance=dist; model.hparams.vq_distance=dist
        if km:
            with torch.no_grad(): model.vq.codebook.weight.copy_(cluster(dist=='cosine'))
        folder=out/name; folder.mkdir()
        with torch.no_grad():
            a=F.normalize(x,dim=-1) if dist=='cosine' else x
            c=model.vq.codebook.weight.float(); c=F.normalize(c,dim=-1) if dist=='cosine' else c
            idx=(c.square().sum(-1)-2*a@c.T).argmin(-1); hist=torch.bincount(idx,minlength=128); perp,active=usage_stats(hist)
            stats={'active_codes':int(hist.gt(0).sum()),'perplexity':perp.item(),'top1_ratio':hist.max().item()/len(idx),'mean_nearest_distance':(a-c[idx]).norm(dim=-1).mean().item(),'histogram':hist.tolist(),'code_norms':model.vq.codebook.weight.norm(dim=-1).tolist()}
        configuration={**dict(model.hparams),'variant':variant,'vq_distance':dist,'initialization':('spherical_kmeans' if dist=='cosine' else 'kmeans') if km else 'random','seed':42,'K':128,'D':256,'kmeans_iterations':30}
        artifact={'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'hyper_parameters':dict(model.hparams),'initialization':configuration,'sampling':report['sampling'],'source_sha256':digest(source)}
        torch.save(artifact,folder/'initialization.ckpt')
        torch.save(model.vq.state_dict(),folder/'vq_projection_codebook.pt')
        (folder/'config.json').write_text(json.dumps(configuration,indent=2))
        (folder/'model_config.json').write_text(json.dumps(dict(model.hparams),indent=2))
        (folder/'statistics.json').write_text(json.dumps({**stats,'H':report['H'],'Z':report['Z'],'sampling':report['sampling']},indent=2))
        (folder/'SHA256SUMS').write_text('\n'.join(digest(p)+'  '+p.name for p in folder.iterdir() if p.is_file()))
        report['variants'][name]=stats
    # Real batch full-model forwards and gradient checks, using the same initial states.
    import os
    os.environ['GLIM_TEXT_MODEL_PATH']=str(Path('models/flan-t5-large').resolve())
    model.setup('fit'); model.eval()
    b={k:v.cuda() if torch.is_tensor(v) else v for k,v in smoke.items()}
    for name in report['variants']:
        saved=torch.load(out/name/'initialization.ckpt',map_location='cpu',weights_only=False)
        missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
        assert not unexpected and all(k.startswith('text_model.') for k in missing)
        model.vq.distance=saved['hyper_parameters']['vq_distance']; model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): result=model.shared_forward(b)
        assert result['disc_logits'].shape==(len(b['eeg']),3)
        assert all(torch.isfinite(v).all() for k,v in result.items() if k.startswith('loss'))
        result['loss_total'].backward()
        grad=model.vq.projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
        report['variants'][name]['real_batch_smoke']={'status':'PASS','batch_size':len(b['eeg']),'loss':result['loss_total'].item(),'projection_grad_norm':grad.norm().item(),'codebook_rows_with_gradient':int(model.vq.codebook.weight.grad.norm(dim=1).gt(0).sum())}
        optimizer=model.configure_optimizers(); old=model.vq.codebook.weight.detach().clone()
        optimizer.step(); assert not torch.equal(old,model.vq.codebook.weight)
        from transformers.modeling_outputs import BaseModelOutput
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            gen=model.text_model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=result['eeg_embeds'][:2]),attention_mask=result['memory_mask'][:2],max_new_tokens=16,num_beams=2)
        report['variants'][name]['real_batch_smoke'].update(adam_update=True,generation=model.tokenizer.batch_decode(gen,skip_special_tokens=True))
        del optimizer,result
    (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:{x:v for x,v in s.items() if x not in ('histogram','code_norms')} for k,s in report['variants'].items()}),flush=True)
    # Rank only passing candidates by broad coverage, then assignment entropy,
    # then low top-one dominance. Euclidean/cosine distances are not comparable.
    best=max(report['variants'],key=lambda k:(report['variants'][k]['active_codes'],report['variants'][k]['perplexity'],-report['variants'][k]['top1_ratio']))
    report['selection']={'variant':best,'rule':'active_codes, then perplexity, then negative top1_ratio; all real smokes must pass'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    from vq_pilot import train_pilot
    train_pilot(model,dm,out/best/'initialization.ckpt',report)
if __name__=='__main__': main()
