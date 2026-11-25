"""E1.1 frozen V3 emotion head and pooling linear readability diagnostics."""
import argparse, json, os, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from data.revised_labels import audit
from model.glim_multitask import GLIMMultitask
from e1_linear_readability_v2 import LABELS, fit_lr, grouped, metrics, bootstrap, hashfile, make_donors

REPRS=('encoder','aligner','quantized')
POOLS=('mean','max','mean_std')
CONTROLS=('real','gaussian','temporal','paired')

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def full_metrics(y,p):
    d=metrics(y,p);cm=np.asarray(d['confusion']);tp=np.diag(cm);pred=cm.sum(0);support=cm.sum(1)
    d.update(accuracy=float(tp.sum()/support.sum()),
             precision=np.divide(tp,pred,out=np.zeros(3,float),where=pred>0).tolist(),
             recall=np.divide(tp,support,out=np.zeros(3,float),where=support>0).tolist(),
             predicted_counts=pred.tolist(),predicted_fraction=(pred/pred.sum()).tolist())
    return d

def pool(x,valid):
    valid=valid.bool(); f=x.float();masked=f.masked_fill(~valid[:,:,None],0)
    count=valid.sum(1,keepdim=True).clamp_min(1)
    mean=masked.sum(1)/count
    maximum=f.masked_fill(~valid[:,:,None],float('-inf')).max(1).values
    maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    variance=(f-mean[:,None,:]).square().masked_fill(~valid[:,:,None],0).sum(1)/count
    return {'mean':mean,'max':maximum,'mean_std':torch.cat([mean,variance.clamp_min(0).sqrt()],1)}

@torch.inference_mode()
def extract(model,sub,condition,seed,stats,output,batch_size=32):
    rng=np.random.default_rng(seed);eegs=sub.eeg.tolist();masks=sub['mask'].tolist()
    donors=make_donors(sub,seed) if condition=='paired' else None
    prompts=[('<NR>',d,s) for d,s in zip(sub.dataset,sub.subject)]
    result={f'{rep}_{p}':[] for rep in REPRS for p in POOLS};result['head_logits']=[]
    code_hist=np.zeros(128,np.int64);shape_check=None
    for start in range(0,len(sub),batch_size):
        end=min(start+batch_size,len(sub));x=np.stack(eegs[start:end]).astype(np.float32,copy=True)
        mask=np.stack(masks[start:end]).astype(bool)
        if condition=='gaussian':
            draw=rng.normal(stats['mean'],stats['std'],size=x[:,:,:104].shape).astype(np.float32)
            x[:,:,:104]=np.where(mask[:,:,None],draw,x[:,:,:104])
        elif condition=='temporal':
            for i in range(len(x)):
                valid=np.flatnonzero(mask[i]);x[i,valid,:104]=x[i,rng.permutation(valid),:104]
        elif condition=='paired':
            x=np.stack([eegs[donors[i]] for i in range(start,end)]).astype(np.float32)
            mask=np.stack([masks[donors[i]] for i in range(start,end)]).astype(bool)
        if np.any(x[:,:,104:]!=0):raise AssertionError('Padded channels modified')
        dev=torch.device('cuda:0');xt=torch.from_numpy(x).to(dev);mt=torch.from_numpy(mask).to(dev)
        prompt=list(zip(*prompts[start:end]));ids=model.p_embedder.encode(prompt,device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            H,_=model.eeg_encoder(xt,mt,model.p_embedder(ids,model.eval_pembed))
            qmask=mt.any(1,keepdim=True).expand(H.shape[:2]);A=model.aligner.in_proj(H)
            V=model.vq(H,qmask);Q=V['quantized'];L=model.disc_head(Q,qmask)
        if shape_check is None:shape_check={'eeg':list(xt.shape),'mask':list(mt.shape),'H':list(H.shape),'aligner':list(A.shape),'quantized':list(Q.shape),'head_logits':list(L.shape),'valid_queries':qmask.sum(1).tolist()}
        for name,tensor in [('encoder',H),('aligner',A),('quantized',Q)]:
            for pooling,vec in pool(tensor,qmask).items():result[f'{name}_{pooling}'].append(vec.cpu().numpy())
        result['head_logits'].append(L.float().cpu().numpy())
        codes=V['code_ids'].cpu().numpy();code_hist+=np.bincount(codes[codes>=0],minlength=128)
    arr={k:np.concatenate(v,axis=0) for k,v in result.items()}
    if any(len(v)!=len(sub) for v in arr.values()):raise AssertionError('Row alignment lost')
    np.savez_compressed(output,**arr,code_hist=code_hist)
    return arr,{'rows':len(sub),'shapes':{k:list(v.shape) for k,v in arr.items()},
                'feature_std':{k:float(v.std()) for k,v in arr.items() if k!='head_logits'},
                'effective_dims':{k:int((v.std(axis=0)>1e-6).sum()) for k,v in arr.items() if k!='head_logits'},
                'active_codes':int((code_hist>0).sum()),'smoke_shapes':shape_check}

def comparison(y,uid,real_logits,control_logits,seed):
    ids,yy,rr=grouped(y,softmax(real_logits,axis=1),uid)
    ids2,yy2,cc=grouped(y,softmax(control_logits,axis=1),uid)
    if not np.array_equal(ids,ids2) or not np.array_equal(yy,yy2):raise AssertionError('UID pairing changed')
    p=rr.argmax(1);q=cc.argmax(1)
    out=full_metrics(yy,q);out['delta_macro_f1']=full_metrics(yy,p)['macro_f1']-out['macro_f1']
    out['paired_bootstrap_95pct_ci']=bootstrap(ids,yy,p,q,seed)
    return out

def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','data-path','labels','e1-run','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--smoke-only',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(a.checkpoint);sha=hashfile(checkpoint)
    if sha!='fe30e8cfc308aa464caa6d460b82bd1720c857d029c9944758b9d0661884c337':raise ValueError('V3 checkpoint SHA mismatch')
    model=GLIMMultitask.from_glim_checkpoint(str(checkpoint)).cuda().eval()
    if model.variant!='A3-full' or model.vq.distance!='cosine' or not hasattr(model,'disc_head'):raise ValueError('Wrong V3 model')
    for param in model.parameters():param.requires_grad_(False)
    frame,manifest=audit(pd.read_pickle(a.data_path),pd.read_pickle(a.labels));manifest.to_csv(out/'split_manifest.csv',index=False)
    sr=frame[(frame.dataset=='ZuCo1')&(frame.task=='task1')&frame['sentiment label'].isin(LABELS)]
    split={phase:sr[sr.phase==phase].reset_index(drop=True) for phase in ('train','val','test')}
    e1=Path(a.e1_run);old=json.loads((e1/'audit.json').read_text())
    for phase,sub in split.items():
        if len(sub)!=old['split'][phase]['trials'] or sub['text uid'].nunique()!=old['split'][phase]['sentences']:raise ValueError('E1 split differs')
    sets=[set(v['text uid']) for v in split.values()]
    if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Sentence leakage')
    st=json.loads((e1/'train_noise_stats.json').read_text());stats={k:np.asarray(st[k]) for k in ('mean','std')}
    test2=split['test'].iloc[:2].reset_index(drop=True)
    x,smoke=extract(model,test2,'real',42,stats,out/'smoke_real.npz',2)
    n,noise_smoke=extract(model,test2,'gaussian',42,stats,out/'smoke_noise.npz',2)
    assert x['head_logits'].shape==(2,3) and np.any(x['head_logits']!=n['head_logits'])
    dump(out/'smoke.json',{'status':'PASS','real':smoke,'noise':noise_smoke,'head_delta':float(np.abs(x['head_logits']-n['head_logits']).mean())})
    if a.smoke_only:return
    y={phase:sub['sentiment label'].map(LABELS).to_numpy(int) for phase,sub in split.items()}
    u={phase:sub['text uid'].to_numpy(int) for phase,sub in split.items()}
    counts=Counter(u['train']);weights=np.asarray([1/counts[uid] for uid in u['train']]);weights/=weights.sum()
    data={};diagnostics={}
    for phase,sub in split.items():
        conditions=('real',) if phase=='train' else CONTROLS
        for cond in conditions:
            arr,diag=extract(model,sub,cond,42,stats,out/f'{phase}_{cond}.npz');data[phase,cond]=arr;diagnostics[f'{phase}_{cond}']=diag
    # E1 mean feature equality verifies trial ordering and label correspondence.
    align={}
    for phase in split:
        original=np.load(e1/f'{phase}_real.npz')
        align[phase]={k:float(np.max(np.abs(data[phase,'real'][f'{k}_mean']-original[k]))) for k in REPRS}
    dump(out/'feature_diagnostics.json',{'conditions':diagnostics,'e1_mean_max_abs_difference':align})
    report={'checkpoint_sha256':sha,'seed':42,'train_sentences':len(counts),'test_sentences':len(sets[2]),
            'original_head':{},'probes':{},'e1_mean_alignment':align,'C_grid':[.01,.1,1,10]}
    head_real=data['test','real']['head_logits'];ids,yt,pr=grouped(y['test'],softmax(head_real,axis=1),u['test'])
    report['original_head']['test_sentence']=full_metrics(yt,pr.argmax(1))
    report['original_head']['test_trial']=full_metrics(y['test'],head_real.argmax(1))
    val_head=data['val','real']['head_logits'];_,yv,pv=grouped(y['val'],softmax(val_head,axis=1),u['val'])
    report['original_head']['val_sentence']=full_metrics(yv,pv.argmax(1))
    report['original_head']['val_trial']=full_metrics(y['val'],val_head.argmax(1))
    report['original_head']['controls']={cond:comparison(y['test'],u['test'],head_real,data['test',cond]['head_logits'],42) for cond in CONTROLS[1:]}
    for key in (f'{rep}_{pool}' for rep in REPRS for pool in POOLS):
        X=data['train','real'][key].astype(float);mu=np.average(X,axis=0,weights=weights)
        sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=weights)).clip(1e-6)
        normalize=lambda v:np.clip((v.astype(float)-mu)/sd,-20,20)
        trials=[]
        for C in report['C_grid']:
            W,b,fit=fit_lr(normalize(X),y['train'],weights,C,len(counts))
            _,yv,pv=grouped(y['val'],softmax(normalize(data['val','real'][key])@W+b,axis=1),u['val'])
            trials.append((full_metrics(yv,pv.argmax(1))['macro_f1'],C,W,b,fit))
        val,C,W,b,fit=max(trials,key=lambda z:(z[0],-z[1]))
        predicted={cond:normalize(data['test',cond][key])@W+b for cond in CONTROLS}
        _,yt,pt=grouped(y['test'],softmax(predicted['real'],axis=1),u['test'])
        report['probes'][key]={'selected_C':C,'val_macro_f1':val,'fit':fit,
           'test_sentence':full_metrics(yt,pt.argmax(1)),
           'test_trial':full_metrics(y['test'],predicted['real'].argmax(1)),
           'controls':{cond:comparison(y['test'],u['test'],predicted['real'],predicted[cond],42) for cond in CONTROLS[1:]}}
        dump(out/'report_partial.json',report)
    dump(out/'report.json',report)
    print('E1.1 PASS',json.dumps({'original_head':report['original_head']['test_sentence']['macro_f1'],
           **{k:v['test_sentence']['macro_f1'] for k,v in report['probes'].items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
