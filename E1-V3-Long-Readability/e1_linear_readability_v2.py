"""E1: frozen V3 representations and grouped sentence-level linear probes."""
import argparse, hashlib, json, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from model.glim_multitask import GLIMMultitask
from data.revised_labels import audit

LABELS = {'negative': 0, 'neutral': 1, 'positive': 2}
FEATURES = ('encoder', 'aligner', 'quantized')

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def hashfile(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def metrics(y,p):
    cm=np.zeros((3,3),int)
    for a,b in zip(y,p):cm[int(a),int(b)]+=1
    support=cm.sum(1); tp=cm.diagonal()
    rec=np.divide(tp,support,out=np.zeros(3,float),where=support>0)
    f1=np.divide(2*tp,support+cm.sum(0),out=np.zeros(3,float),where=(support+cm.sum(0))>0)
    return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(rec[support>0].mean()),
            'class_f1':f1.tolist(),'support':support.tolist(),'confusion':cm.tolist()}

def grouped(y, prob, uid):
    groups=defaultdict(list)
    for i,u in enumerate(uid):groups[int(u)].append(i)
    ys=[]; ps=[]; ids=[]
    for u,ii in groups.items():
        if len(set(y[ii]))!=1:raise ValueError('Conflicting sentence labels')
        ids.append(u);ys.append(int(y[ii[0]]));ps.append(prob[ii].mean(0))
    return np.array(ids),np.array(ys),np.array(ps)

def fit_lr(x,y,weights,C,independent_groups):
    # Exact multinomial softmax with L2 penalty on coefficients, no penalty on bias.
    n,d=x.shape; w=weights/weights.sum(); one=np.eye(3)[y]
    def fg(flat):
        W=flat[:d*3].reshape(d,3); b=flat[d*3:]
        logits=x@W+b; probs=softmax(logits,axis=1)
        penalty=1/(C*independent_groups)
        loss=float(np.sum(w*(logsumexp(logits,axis=1)-logits[np.arange(n),y]))+
                   .5*penalty*np.sum(W*W))
        diff=(probs-one)*w[:,None]
        return loss,np.r_[((x.T@diff)+penalty*W).ravel(),diff.sum(0)]
    result=minimize(fg,np.zeros(d*3+3),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-9})
    if not np.isfinite(result.fun):raise RuntimeError('Nonfinite probe objective')
    return result.x[:d*3].reshape(d,3),result.x[d*3:],{'converged':bool(result.success),'iterations':int(result.nit),'objective':float(result.fun)}

def make_donors(sub, seed):
    rng=np.random.default_rng(seed); u=sub['text uid'].to_numpy();s=sub['subject'].to_numpy()
    by_subject={name:np.flatnonzero(s==name) for name in np.unique(s)}
    donors=[]
    for i in range(len(sub)):
        pool=by_subject[s[i]];pool=pool[u[pool]!=u[i]]
        if not len(pool):raise ValueError('No different-sentence donor in same subject')
        donors.append(int(rng.choice(pool)))
    return np.array(donors)

@torch.inference_mode()
def extract(model, sub, condition, seed, train_stats, batch_size, out):
    rng=np.random.default_rng(seed); donors=make_donors(sub,seed) if condition=='paired' else None
    eegs=sub['eeg'].tolist();masks=sub['mask'].tolist()
    prompts=[('<NR>',d,s) for d,s in zip(sub['dataset'],sub['subject'])]
    arrays={k:[] for k in FEATURES};hist=np.zeros(128,np.int64)
    pos_hist=np.zeros((96,128),np.int64); subjects=sub['subject'].tolist()
    for start in range(0,len(sub),batch_size):
        end=min(start+batch_size,len(sub))
        x=np.stack(eegs[start:end]).astype(np.float32,copy=True)
        mask=np.stack(masks[start:end]).astype(bool)
        if condition=='gaussian':
            draw=rng.normal(train_stats['mean'],train_stats['std'],size=x[:,:,:104].shape).astype(np.float32)
            x[:,:,:104]=np.where(mask[:,:,None],draw,x[:,:,:104])
        elif condition=='temporal':
            for j in range(len(x)):
                valid=np.flatnonzero(mask[j]);x[j,valid,:104]=x[j,rng.permutation(valid),:104]
        elif condition=='paired':
            x=np.stack([eegs[donors[i]] for i in range(start,end)]).astype(np.float32)
            mask=np.stack([masks[donors[i]] for i in range(start,end)]).astype(bool)
        x=torch.from_numpy(x).cuda();m=torch.from_numpy(mask).cuda()
        prompt=list(zip(*prompts[start:end])); ids=model.p_embedder.encode(prompt,device=x.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            H,_=model.eeg_encoder(x,m,model.p_embedder(ids,model.eval_pembed))
            qmask=m.any(1,keepdim=True).expand(H.shape[:2])
            align=model.aligner.in_proj(H)
            vq=model.vq(H,qmask)
        for key,t in [('encoder',H),('aligner',align),('quantized',vq['quantized'])]:
            pooled=(t.float()*qmask[:,:,None]).sum(1)/qmask.sum(1,keepdim=True).clamp_min(1)
            arrays[key].append(pooled.cpu().numpy())
        codes=vq['code_ids'].cpu().numpy()
        hist+=np.bincount(codes[codes>=0],minlength=128)
        for j in range(codes.shape[1]):pos_hist[j]+=np.bincount(codes[:,j][codes[:,j]>=0],minlength=128)
        if (start//batch_size+1)%20==0:print(condition,len(sub),end,flush=True)
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    np.savez_compressed(out,**arrays,code_hist=hist,position_code_hist=pos_hist)
    return arrays,{'rows':len(sub),'shapes':{k:list(v.shape) for k,v in arrays.items()},
                   'active_codes':int((hist>0).sum()),'code_hist':hist.tolist(),
                   'subject_counts':dict(Counter(subjects))}

def bootstrap(uid,y,p_real,p_control,seed=42):
    rng=np.random.default_rng(seed); n=len(uid); reps=[]
    for _ in range(1000):
        ix=rng.integers(0,n,n);reps.append(metrics(y[ix],p_real[ix])['macro_f1']-metrics(y[ix],p_control[ix])['macro_f1'])
    return [float(v) for v in np.quantile(reps,[.025,.975])]

def main():
    a=argparse.ArgumentParser();
    for key in ('checkpoint','data-path','labels','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--batch-size',type=int,default=32);a.add_argument('--smoke-only',action='store_true')
    args=a.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ck=Path(args.checkpoint);sha=hashfile(ck)
    model=GLIMMultitask.from_glim_checkpoint(str(ck)).cuda().eval()
    if model.variant!='A3-full' or model.vq.distance!='cosine':raise ValueError('Not V3 cosine checkpoint')
    for p in model.parameters():p.requires_grad_(False)
    df,manifest=audit(pd.read_pickle(args.data_path),pd.read_pickle(args.labels))
    manifest.to_csv(out/'split_manifest.csv',index=False)
    sr=df[(df.dataset=='ZuCo1')&(df.task=='task1')].copy()
    sr=sr[sr['sentiment label'].isin(LABELS)].reset_index(drop=True)
    split={k:sr[sr.phase==k].reset_index(drop=True) for k in ('train','val','test')}
    uid_sets={k:set(v['text uid']) for k,v in split.items()}
    if any(uid_sets[a]&uid_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Sentence split overlap')
    audit_report={'checkpoint':str(ck),'sha256':sha,'epoch':46,'source_versions':{},'split':{k:{'trials':len(v),'sentences':len(uid_sets[k]),'classes':v['sentiment label'].value_counts().to_dict()} for k,v in split.items()},'uid_overlap':0}
    for name in ('model/glim.py','model/vq.py','model/glim_multitask.py','data/datamodule.py'):
        audit_report['source_versions'][name]=hashfile(Path(name))
    save(out/'audit.json',audit_report)
    train=split['train']; sums=np.zeros(104);squares=np.zeros(104);count=0
    for eeg,mask in zip(train.eeg,train['mask']):
        valid=np.asarray(eeg)[np.asarray(mask).astype(bool),:104].astype(np.float64)
        sums+=valid.sum(0);squares+=(valid*valid).sum(0);count+=len(valid)
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,1e-8))
    stats={'mean':mean,'std':std};save(out/'train_noise_stats.json',{'count':count,'mean':mean.tolist(),'std':std.tolist()})
    smoke=split['val'].iloc[:2].reset_index(drop=True)
    _,info=extract(model,smoke,'real',42,stats,2,out/'smoke_features.npz')
    _,noise_info=extract(model,smoke,'gaussian',42,stats,2,out/'smoke_noise_features.npz')
    save(out/'smoke.json',{'status':'PASS','real':info,'noise':noise_info})
    if args.smoke_only:return
    features={};diagnostics={}
    for phase in split:
        conditions=['real'] if phase=='train' else ['real','gaussian','temporal','paired']
        for condition in conditions:
            arr,info=extract(model,split[phase],condition,42,stats,args.batch_size,out/f'{phase}_{condition}.npz')
            features[phase,condition]=arr;diagnostics[f'{phase}_{condition}']=info
    save(out/'feature_diagnostics.json',diagnostics)
    report={'audit':audit_report,'features':{},'seed':42,'probe':'L2 multinomial logistic regression (SciPy L-BFGS-B)','C_grid':[.01,.1,1,10]}
    y={k:v['sentiment label'].map(LABELS).to_numpy(int) for k,v in split.items()}
    uid={k:v['text uid'].to_numpy(int) for k,v in split.items()}
    counts=Counter(uid['train']);weights=np.array([1/counts[u] for u in uid['train']]);weights/=weights.sum()
    for key in FEATURES:
        xtr=features['train','real'][key].astype(np.float64)
        mu=np.average(xtr,axis=0,weights=weights);sigma=np.sqrt(np.average((xtr-mu)**2,axis=0,weights=weights)).clip(1e-6)
        normalize=lambda x:np.clip((x.astype(np.float64)-mu)/sigma,-20,20)
        xtr=normalize(xtr)
        candidates=[]
        for C in report['C_grid']:
            W,b,fit=fit_lr(xtr,y['train'],weights,C,len(counts))
            _,yv,pv=grouped(y['val'],softmax(normalize(features['val','real'][key])@W+b,axis=1),uid['val'])
            candidates.append((metrics(yv,pv.argmax(1))['macro_f1'],C,W,b,fit))
        best=max(candidates,key=lambda z:(z[0],-z[1]));_,C,W,b,fit=best
        key_report={'selected_C':C,'validation_macro_f1':best[0],'fit':fit,'controls':{}}
        _,yt,pr=grouped(y['test'],softmax(normalize(features['test','real'][key])@W+b,axis=1),uid['test'])
        real_pred=pr.argmax(1);key_report['test_real']=metrics(yt,real_pred)
        for condition in ('gaussian','temporal','paired'):
            _,yc,pc=grouped(y['test'],softmax(normalize(features['test',condition][key])@W+b,axis=1),uid['test'])
            control=metrics(yc,pc.argmax(1));control['delta_macro_f1']=key_report['test_real']['macro_f1']-control['macro_f1']
            control['delta_bootstrap_95pct_ci']=bootstrap(np.unique(uid['test']),yt,real_pred,pc.argmax(1))
            key_report['controls'][condition]=control
        majority=Counter(y['train']).most_common(1)[0][0]
        key_report['majority_baseline']=metrics(yt,np.full_like(yt,majority))
        rng=np.random.default_rng(42);train_group={u:y['train'][np.flatnonzero(uid['train']==u)[0]] for u in np.unique(uid['train'])}
        perm=rng.permutation(list(train_group.values())); shuffled=dict(zip(train_group,perm)); ysh=np.array([shuffled[u] for u in uid['train']])
        Ws,bs,_=fit_lr(xtr,ysh,weights,C,len(counts))
        key_report['shuffled_label_baseline']=metrics(yt,grouped(y['test'],softmax(normalize(features['test','real'][key])@Ws+bs,axis=1),uid['test'])[2].argmax(1))
        report['features'][key]=key_report
        save(out/'report.json',report)
    print('E1 PASS',json.dumps({k:v['test_real']['macro_f1'] for k,v in report['features'].items()}),flush=True)

if __name__=='__main__':main()
