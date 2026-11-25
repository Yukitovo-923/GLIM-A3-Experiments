"""Seed 42 paired pilot. No label or reference argument in either conditioning path."""
import copy
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare import ROOT, base_model, generate, write


class Expert(nn.Module):
    def __init__(self, kind, center, scale):
        super().__init__()
        self.kind = kind
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.project = nn.Linear(len(center), 256)
        if kind == "A2":
            self.codes = nn.Parameter(torch.randn(12, 256) / 16)
        else:
            self.classifier = nn.Sequential(nn.Linear(256, 12), nn.GELU(), nn.Linear(12, 3))
        self.fusion = nn.Linear(256, 1024, bias=False)
        nn.init.zeros_(self.fusion.weight)

    def forward(self, x, *, intervention=False):
        with torch.autocast(x.device.type, enabled=False):
            z = F.normalize(self.project((x.float()-self.center)/self.scale), dim=-1)
            if self.kind == "A1":
                return self.classifier(z * 16), z, None
            codes = F.normalize(self.codes, dim=-1)
            scores = z @ codes.T / 0.1
            logits = scores.reshape(-1, 3, 4).logsumexp(-1) - math.log(4)
            idx = scores.argmax(-1)
            if intervention:
                idx = (idx + 4) % 12
            hard = F.one_hot(idx, 12).float()
            prob = scores.softmax(-1)
            assignment = hard + (prob - prob.detach())
            return logits, assignment @ codes, idx

    def condition(self, x, memory, *, intervention=False):
        logits, rep, idx = self(x, intervention=intervention)
        token = memory[:, 0] + self.fusion(rep * 16).to(memory.dtype)
        return torch.cat([token[:,None], memory[:,1:]], dim=1), logits, idx


def load(split):
    store = torch.load(ROOT / (split + ".pt"), weights_only=True)
    return {k: v.cuda() if torch.is_tensor(v) else v for k,v in store.items()}


def metrics(logits, ds, idx=None):
    uid, y = ds["uid"].cpu().numpy(), ds["label"].cpu().numpy()
    probs = logits.detach().float().softmax(-1).cpu().numpy()
    cm = np.zeros((3,3), dtype=int)
    for u in np.unique(uid):
        mask = uid == u
        assert np.unique(y[mask]).size == 1
        cm[y[mask][0], probs[mask].mean(0).argmax()] += 1
    tp = cm.diagonal()
    recall = tp / cm.sum(1)
    record = dict(macro_f1=float((2*tp/(cm.sum(1)+cm.sum(0))).mean()),
                  balanced_accuracy=float(recall.mean()), recall=recall.tolist(),
                  matrix=cm.tolist(), predicted_uid=cm.sum(0).tolist())
    if idx is not None:
        ids = idx.cpu().numpy()
        table = np.bincount(12*y + ids, minlength=36).reshape(3,12)
        hist = table.sum(0)
        p = hist[hist>0] / len(ids)
        record.update(code_counts=hist.tolist(), codes_by_true_class=table.tolist(),
                      selected_groups=hist.reshape(3,4).sum(1).tolist(),
                      active=int((hist>0).sum()), dead=np.flatnonzero(hist==0).tolist(),
                      perplexity=float(np.exp(-sum(p*np.log(p)))), top_share=float(p.max()))
    return record


@torch.no_grad()
def evaluate(head, train, val):
    record = {}
    for key, ds in [("train", train), ("val", val)]:
        logits, _, idx = head(ds["x"])
        record[key] = metrics(logits, ds, idx)
    record["gap"] = record["train"]["macro_f1"] - record["val"]["macro_f1"]
    return record


def batches(n, seed, size):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).split(size)


def finite_step(loss, optimizer, head):
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def smoke(head, train):
    head = copy.deepcopy(head)
    selection = batches(len(train["x"]), 42, 64)[0]
    x, y = train["x"][selection], train["label"][selection]
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    start = head.codes.detach().clone()
    losses = []
    for step in range(12):
        logits, _, idx = head(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        finite_step(loss, opt, head)
        if step == 0:
            grads = [p.grad.norm().item() for p in (head.codes, head.project.weight)]
    logits, _, idx = head(x)
    delta = (head.codes.detach()-start).norm().item()
    final = F.cross_entropy(logits,y).item()
    assert min(grads) > 0 and delta > 0 and final < losses[0] and idx.unique().numel() > 1
    write("smoke.json", dict(loss_before=losses[0], loss_after=final, code_delta=delta,
                             code_grad=grads[0], projection_grad=grads[1], active=int(idx.unique().numel()),
                             steps=12, finite=True))


def stage1(heads, train, val):
    opts = {k: torch.optim.Adam(h.parameters(), lr=3e-4) for k,h in heads.items()}
    history, best, selected, states = [], -1., 0, None
    for epoch in range(1,7):
        losses = {k: [] for k in heads}
        for ix in batches(len(train["x"]), 42+epoch, 64):
            for k,h in heads.items():
                logits,_,_ = h(train["x"][ix])
                loss = F.cross_entropy(logits, train["label"][ix])
                finite_step(loss, opts[k], h)
                losses[k].append((float(loss.detach()),len(ix)))
        record = {k:evaluate(h,train,val) for k,h in heads.items()}
        for k in heads:
            record[k]["train_ce"] = sum(l*n for l,n in losses[k])/len(train["x"])
        history.append(record)
        criterion = np.mean([r["val"]["macro_f1"] for r in record.values()])
        if criterion > best:
            best, selected = criterion, epoch
            states = {k:copy.deepcopy(h.state_dict()) for k,h in heads.items()}
        if epoch-selected >= 2:
            break
    for k,h in heads.items():
        h.load_state_dict(states[k])
    write("stage1.json", dict(history=history, selected_epoch=selected,
                              updates_per_epoch=math.ceil(len(train["x"])/64)))
    torch.save(states,ROOT/"stage1_heads.pt")


def stage2(model, heads, train, val):
    history = {}
    for kind, head in heads.items():
        opt = torch.optim.Adam(head.parameters(),lr=3e-4)
        log = []
        for step,ix in enumerate(batches(len(train["x"]), 42, 32)[:32],1):
            mem, logits, _ = head.condition(train["x"][ix], train["memory"][ix])
            if step == 1:
                assert torch.equal(mem,train["memory"][ix])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lm,_ = model.text_decoder_forward(mem,None,train["targets"][ix],memory_mask=train["mask"][ix])
            ce = F.cross_entropy(logits,train["label"][ix])
            record = dict(step=step,lm=lm.item(),ce=ce.item())
            if step in (1,2,32):
                params = {"fusion":head.fusion.weight,"projection":head.project.weight}
                if kind=="A2":
                    params["codes"] = head.codes
                gradients = torch.autograd.grad(lm,list(params.values()),retain_graph=True)
                record["lm_grad_norm"] = {k:g.norm().item() for k,g in zip(params,gradients)}
                assert all(np.isfinite(list(record["lm_grad_norm"].values())))
                assert record["lm_grad_norm"]["fusion"] > 0
            record["grad"] = finite_step(.5*lm+.2*ce,opt,head)
            log.append(record)
        history[kind] = dict(steps=log,classification=evaluate(head,train,val))
        assert not any(p.grad is not None or p.requires_grad for p in model.parameters())
    write("stage2.json",history)
    torch.save({k:h.state_dict() for k,h in heads.items()}, ROOT/"stage2_heads.pt")


def mismatch(val):
    uid, subject = val["uid"].cpu().numpy(), np.array(val["subject"])
    mapping = np.arange(len(uid))
    rng = np.random.default_rng(1042)
    for s in sorted(set(subject)):
        indices = np.flatnonzero(subject==s)
        for _ in range(1000):
            shuffled = rng.permutation(indices)
            if np.all(uid[shuffled]!=uid[indices]):
                mapping[indices] = shuffled
                break
    assert np.all(uid[mapping]!=uid) and np.array_equal(subject[mapping],subject)
    return mapping


@torch.no_grad()
def inference(model,heads,val):
    donors = mismatch(val)
    for name, head in {"A0":None, **heads}.items():
        conditions = ["matched","mismatched"]
        if name=="A2":
            conditions.append("intervention")
        for cond in conditions:
            rows = []
            for targets in np.array_split(np.arange(len(donors)), math.ceil(len(donors)/24)):
                source = donors[targets] if cond=="mismatched" else targets
                mem, mask = val["memory"][source], val["mask"][source]
                idx = None
                if head is not None:
                    mem,_,idx = head.condition(val["x"][source], mem, intervention=cond=="intervention")
                texts = generate(model, mem, mask)
                for j,(i,d) in enumerate(zip(targets,source)):
                    rows.append(dict(trial=int(i), donor=int(d), uid=int(val["uid"][i]),
                                     donor_uid=int(val["uid"][d]), subject=val["subject"][i],
                                     reference=val["reference"][i], prediction=texts[j],
                                     code=int(idx[j]) if idx is not None else None))
            write(name+"_"+cond+".json",rows)
            print("generated",name,cond,len(rows),flush=True)


def main():
    assert (ROOT/"preflight.json").is_file()
    train,val = load("train"),load("val")
    x = train["x"].float()
    center,scale = x.mean(0), x.std(0).clamp_min(1e-5)
    heads = {}
    for k in ("A1","A2"):
        torch.manual_seed(42)
        heads[k] = Expert(k,center,scale).cuda()
    write("parameters.json",{k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()})
    smoke(heads["A2"],train)
    model = base_model()
    write("original_head.json", {k:metrics(ds["original_logits"],ds) for k,ds in [("train",train),("val",val)]})
    stage1(heads,train,val)
    print("stage1 complete",flush=True)
    stage2(model,heads,train,val)
    print("stage2 complete",flush=True)
    inference(model,heads,val)


if __name__ == "__main__":
    main()
