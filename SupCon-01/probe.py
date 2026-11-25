"""Paired CPU probes on the existing frozen-V3 cache."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from geometry import embedding_analysis

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "Affect-Codebook-01"
MANIFEST = ROOT.parent / "E1-V3-Long-Readability/run-02/split_manifest.csv"
SEEDS = (42,43,44)
STEPS, PER_CLASS = 52, 21
LAMBDA, TAU = .1, .1


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write(name, obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


class Data:
    def __init__(self, split, manifest):
        path = SOURCE/(split+".pt")
        cache = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.x = cache["x"].float()
        self.y = cache["label"].long()
        self.uid = cache["uid"].long()
        self.subject = np.asarray(cache["subject"])
        self.words = np.asarray([len(s.split()) for s in cache["reference"]])
        self.uids, self.inverse = np.unique(self.uid.numpy(),return_inverse=True)
        self.trials = [np.flatnonzero(self.inverse==i) for i in range(len(self.uids))]
        self.labels = np.array([int(self.y[t[0]]) for t in self.trials])
        self.lengths = np.array([self.words[t[0]] for t in self.trials])
        assert self.x.shape == ({"train":3316,"val":406}[split],1024)
        assert len(self.uids)=={"train":320,"val":40}[split]
        assert torch.isfinite(self.x).all() and set(self.labels)=={0,1,2}
        expected = manifest.loc[manifest.phase==split].set_index("text uid",verify_integrity=True)
        assert set(self.uids)==set(expected.index)
        for u,indices,label in zip(self.uids,self.trials,self.labels):
            row = expected.loc[u]
            assert int(row["target"])==label and len(indices)==row["EEG rows"]
            assert self.y[indices].eq(label).all()
            assert all(cache["reference"][i]==row["input text"] for i in indices)
        assignment = pd.read_csv(SOURCE/("stage1_"+split+"_assignments.csv"))
        assert np.array_equal(assignment.uid,self.uid) and np.array_equal(assignment.label,self.y)
        assert np.array_equal(assignment.subject,self.subject)
        self.audit = dict(file_hash=digest(path), features=list(self.x.shape),
                          subject_count=int(len(set(self.subject))),
                          class_uids=np.bincount(self.labels).tolist(),class_trials=torch.bincount(self.y).tolist(),
                          words_by_class=[self.lengths[self.labels==k].mean().item() for k in range(3)])


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(1024,256)
        self.classifier = nn.Sequential(nn.Linear(256,12),nn.GELU(),nn.Linear(12,3))

    def forward(self,x):
        z = F.normalize(self.project(x),dim=-1)
        return self.classifier(16*z),z


def supcon(z,y,uid):
    legal = uid[:,None]!=uid[None,:]
    positive = legal & y[:,None].eq(y[None,:])
    valid = positive.any(1)
    if not valid.any():
        return z.sum()*0
    scores = (z[valid] @ z.T)/TAU
    logden = scores.masked_fill(~legal[valid],-torch.inf).logsumexp(dim=1)
    pos = positive[valid]
    return (logden-(scores*pos).sum(1)/pos.sum(1)).mean()


def schedule(data,seed):
    rng = np.random.default_rng(seed)
    subjects,subj_idx = np.unique(data.subject,return_inverse=True)
    uid_count = np.zeros(len(data.uids),int)
    trial_count = np.zeros(len(data.x),int)
    exposures = np.zeros((3,len(subjects)),int)
    batches = []
    for _ in range(STEPS):
        chosen = []
        for k in range(3):
            pool = rng.permutation(np.flatnonzero(data.labels==k))
            ordered = pool[np.argsort(uid_count[pool],kind="stable")][:PER_CLASS]
            current = np.zeros(len(subjects),int)
            for i in ordered:
                ts = rng.permutation(data.trials[i])
                # Among minimally used trials, balance subject counts in this class/batch,
                # then overall subject exposures. Random permutation resolves final ties.
                ss = subj_idx[ts]
                order = np.lexsort((exposures[k,ss],current[ss],trial_count[ts]))
                t = ts[order[0]]
                trial_count[t] += 1
                current[subj_idx[t]] += 1
                exposures[k,subj_idx[t]] += 1
                uid_count[i] += 1
                chosen.append(t)
        batch = rng.permutation(chosen)
        assert len(set(data.uid[batch].tolist()))==len(batch)
        assert torch.bincount(data.y[batch],minlength=3).tolist()==[PER_CLASS]*3
        batches.append(batch)
    arr = np.stack(batches)
    np.save(ROOT/f"batches_{seed}.npy",arr)
    return arr, dict(sha256=digest(ROOT/f"batches_{seed}.npy"),batches=STEPS,
        distinct_trials=int((trial_count>0).sum()),distinct_uids=int((uid_count>0).sum()),
        uid_uses_by_class=[[int(a.min()),int(a.max())] for k in range(3) for a in [uid_count[data.labels==k]]],
        subject_order=subjects.tolist(),subject_by_class=exposures.tolist(),
        mean_words_by_class=[float(data.words[arr][data.y.numpy()[arr]==k].mean()) for k in range(3)])


def state_hash(head):
    h=hashlib.sha256()
    for p in head.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def smoke(head,train,ix):
    h=copy.deepcopy(head)
    logits,z = h(train.x[ix])
    y,uid=train.y[ix],train.uid[ix]
    ce,sc=F.cross_entropy(logits,y),supcon(z,y,uid)
    grad=torch.autograd.grad(sc,h.project.weight,retain_graph=True)[0].norm().item()
    total=ce+LAMBDA*sc
    total.backward()
    grads={k:p.grad.norm().item() for k,p in h.named_parameters()}
    assert grad>0 and np.isfinite(grad) and min(grads.values())>0 and np.isfinite(list(grads.values())).all()
    legal=uid[:,None]!=uid[None,:]
    pos=legal & y[:,None].eq(y[None,:])
    assert pos.sum(1).eq(20).all() and (legal & ~pos).sum(1).eq(42).all()
    assert supcon(z,torch.arange(len(z)),uid).item()==0
    return dict(ce=ce.item(),supcon=sc.item(),total=total.item(),supcon_project_grad=grad,
                all_grads=grads,positive=int(pos.sum()),negative=int((legal & ~pos).sum()),
                missing_anchor=int((~pos.any(1)).sum()))


def classification(y,p):
    cm=np.bincount(3*y+p,minlength=9).reshape(3,3)
    recall=cm.diagonal()/cm.sum(1)
    return dict(f1=float(np.mean(2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))),ba=float(recall.mean()),
                recall=recall.tolist(),cm=cm.tolist(),pred_counts=cm.sum(0).tolist(),
                pred_share=(cm.sum(0)/len(y)).tolist())


def fit(head,method,seed,batches,tr,va):
    opt=torch.optim.Adam(head.parameters(),lr=3e-4)
    log=[]
    for ix in batches:
        logits,z=head(tr.x[ix])
        ce=F.cross_entropy(logits,tr.y[ix])
        loss=ce
        sc=None
        if method=="S1":
            sc=supcon(z,tr.y[ix],tr.uid[ix])
            loss=ce+LAMBDA*sc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        assert torch.isfinite(loss)
        opt.step()
        log.append([ce.item(),None if sc is None else sc.item(),norm.item()])
    write(f"loss_{seed}_{method}.json",log)
    projected, result={},{}
    for split,ds in [("train",tr),("val",va)]:
        with torch.no_grad():
            logits,z=head(ds.x)
            probs=logits.softmax(-1).numpy()
        uid_probs=np.stack([probs[ts].mean(0) for ts in ds.trials])
        pred=uid_probs.argmax(1)
        result[split]=classification(ds.labels,pred)
        projected[split]=z.numpy()
        pd.DataFrame(dict(uid=ds.uids,label=ds.labels,pred=pred,
                          p_neg=uid_probs[:,0],p_neu=uid_probs[:,1],p_pos=uid_probs[:,2])).to_csv(
                              ROOT/f"uid_{seed}_{method}_{split}.csv",index=False)
    result["gap_f1"]=result["train"]["f1"]-result["val"]["f1"]
    result["gap_ba"]=result["train"]["ba"]-result["val"]["ba"]
    result["embedding"]=embedding_analysis(tr,va,projected,ROOT,f"{seed}_{method}")
    return result


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    provenance=json.loads((SOURCE/"preflight.json").read_text())
    ck=provenance["checkpoint"]
    assert ck["sha256"]==digest(ck["path"])=="fe30e8cfc308aa464caa6d460b82bd1720c857d029c9944758b9d0661884c337"
    assert digest(MANIFEST)==provenance["split_sha256"]
    for s in ("train","val"):
        assert provenance["splits"][s]["first_batch_native_memory_equal"]
    m=pd.read_csv(MANIFEST)
    m=m.loc[m.dataset.eq("ZuCo1") & m.task.eq("task1") & m.phase.isin(["train","val"])]
    tr,va=Data("train",m),Data("val",m)
    assert not set(tr.uids)&set(va.uids)
    center,scale=tr.x.mean(0),tr.x.std(0).clamp_min(1e-5)
    for ds in [tr,va]:
        ds.x=(ds.x-center)/scale
        assert torch.isfinite(ds.x).all()
    write("preflight.json",dict(train=tr.audit,val=va.audit,checkpoint=ck,split_hash=digest(MANIFEST),
          provenance_hash=digest(SOURCE/"preflight.json"),producer_hash=digest(SOURCE/"prepare.py"),
          config_hash=digest(ROOT/"CONFIG.md"),torch_version=torch.__version__,device=str(tr.x.device)))
    records,audits={},{}
    for seed in SEEDS:
        torch.manual_seed(seed)
        base=Head()
        assert sum(p.numel() for p in base.parameters())==265523
        batches,audits[seed]=schedule(tr,seed)
        audits[seed]["initial_sha256"]=state_hash(base)
        if seed==42:
            write("smoke.json",smoke(base,tr,batches[0]))
        records[seed]={}
        for method in ("S0","S1"):
            head=copy.deepcopy(base)
            assert state_hash(head)==audits[seed]["initial_sha256"]
            records[seed][method]=fit(head,method,seed,batches,tr,va)
            print(seed,method,records[seed][method]["val"],flush=True)
    write("sampling.json",audits)
    write("results.json",records)


if __name__=="__main__":
    main()
