"""Repeat E1 control inference with fixed probes and independent seeds."""
import argparse, json, os
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from data.revised_labels import audit
from model.glim_multitask import GLIMMultitask
from e1_linear_readability_v2 import LABELS, FEATURES, fit_lr, grouped, metrics, bootstrap, extract

def main():
    p=argparse.ArgumentParser()
    for k in ('checkpoint','data-path','labels','run'):p.add_argument('--'+k,required=True)
    a=p.parse_args();root=Path(a.run)
    report=json.loads((root/'report.json').read_text());stats=json.loads((root/'train_noise_stats.json').read_text())
    stats={k:np.array(v) for k,v in stats.items() if k in ('mean','std')}
    frame,_=audit(pd.read_pickle(a.data_path),pd.read_pickle(a.labels))
    sr=frame[(frame.dataset=='ZuCo1')&(frame.task=='task1')&frame['sentiment label'].isin(LABELS)]
    train=sr[sr.phase=='train'].reset_index(drop=True);test=sr[sr.phase=='test'].reset_index(drop=True)
    ytr=train['sentiment label'].map(LABELS).to_numpy(int); yt=test['sentiment label'].map(LABELS).to_numpy(int)
    utr=train['text uid'].to_numpy(int);ut=test['text uid'].to_numpy(int)
    counts=Counter(utr);w=np.array([1/counts[u] for u in utr]);w/=w.sum()
    saved_train=np.load(root/'train_real.npz');saved_test=np.load(root/'test_real.npz')
    probes={};real={}
    for key in FEATURES:
        x=saved_train[key].astype(float);mu=np.average(x,axis=0,weights=w)
        sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w)).clip(1e-6)
        z=lambda a:np.clip((a.astype(float)-mu)/sd,-20,20)
        C=report['features'][key]['selected_C'];W,b,_=fit_lr(z(x),ytr,w,C,len(counts))
        probes[key]=(mu,sd,W,b)
        _,gy,pr=grouped(yt,softmax(z(saved_test[key])@W+b,axis=1),ut)
        real[key]=(gy,pr.argmax(1))
    model=GLIMMultitask.from_glim_checkpoint(a.checkpoint).cuda().eval()
    for param in model.parameters():param.requires_grad_(False)
    results={key:{} for key in FEATURES}
    for seed in (43,44):
        for condition in ('gaussian','temporal','paired'):
            arrays,diagnostics=extract(model,test,condition,seed,stats,32,root/f'test_{condition}_seed{seed}.npz')
            for key in FEATURES:
                mu,sd,W,b=probes[key]
                z=np.clip((arrays[key].astype(float)-mu)/sd,-20,20)
                _,gy,pc=grouped(yt,softmax(z@W+b,axis=1),ut)
                yr,pr=real[key]
                if not np.array_equal(gy,yr):raise ValueError('Sentence pairing changed')
                m=metrics(gy,pc.argmax(1));m['delta_macro_f1']=metrics(yr,pr)['macro_f1']-m['macro_f1']
                m['delta_bootstrap_95pct_ci']=bootstrap(np.unique(ut),yr,pr,pc.argmax(1),seed)
                results[key][f'{condition}_seed{seed}']=m
    (root/'controls_multiseed.json').write_text(json.dumps(results,indent=2)+'\n')
    print('MULTISEED PASS',json.dumps({k:{n:round(v['delta_macro_f1'],4) for n,v in d.items()} for k,d in results.items()}),flush=True)

if __name__=='__main__':main()
