"""Replay the trained heads on raw validation EEG and intervene with train-active codes."""
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pilot import Expert, load
from prepare import ROOT, base_model, datasets, generate, write


@torch.no_grad()
def decode_eeg(model, head, eeg, mask, prompts, tasks):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        prompt = model.p_embedder(model.p_embedder.encode(prompts,device=eeg.device),model.eval_pembed)
        hidden,_ = model.eeg_encoder(eeg,mask,prompt)
        memory,x = model.aligner.embed_eeg(hidden)
        memory,extra = model.condition_memory(hidden,memory,mask,tasks,prompts[1])
    if head is not None:
        memory,_,_ = head.condition(x.reshape(len(eeg),-1),memory)
    return generate(model,memory,extra["memory_mask"])


@torch.no_grad()
def main():
    import json
    model = base_model()
    val = load("val")
    states = torch.load(ROOT/"stage2_heads.pt",weights_only=True)
    heads = {}
    for key,s in states.items():
        heads[key] = Expert(key,s["center"],s["scale"]).cuda()
        heads[key].load_state_dict(s)
    ds = next(ds for split,ds in datasets() if split=="val")
    batch = next(iter(DataLoader(ds,batch_size=24)))
    parity = {}
    for k,h in {"A0":None,**heads}.items():
        direct = decode_eeg(model,h,batch["eeg"].cuda(),batch["mask"].cuda(),batch["prompt"],batch["raw task key"])
        saved = json.loads((ROOT/(k+"_matched.json")).read_text())
        assert direct == [r["prediction"] for r in saved[:len(direct)]]
        parity[k] = len(direct)
    h = heads["A2"]
    counts = json.loads((ROOT/"stage2.json").read_text())["A2"]["classification"]["train"]["code_counts"]
    major,alternate = sorted(range(12),key=lambda c:-counts[c])[:2]
    assert counts[alternate]>0
    rows = json.loads((ROOT/"A2_matched.json").read_text())
    for ids in np.array_split(np.arange(len(rows)),math.ceil(len(rows)/24)):
        memory = val["memory"][ids]
        _,_,natural = h(val["x"][ids])
        forced = torch.where(natural==major,alternate,major)
        rep = F.normalize(h.codes,dim=-1)[forced]
        token = memory[:,0]+h.fusion(rep*16).to(memory.dtype)
        fused = torch.cat([token[:,None],memory[:,1:]],dim=1)
        texts = generate(model,fused,val["mask"][ids])
        for i,c,p in zip(ids,forced.tolist(),texts):
            assert c!=rows[i]["code"]
            rows[i].update(code=c,prediction=p)
    write("A2_active_swap.json",rows)
    write("mechanism.json",dict(raw_eeg_generation_parity=parity, active_swap_codes=[major,alternate],
                                train_counts=[counts[major],counts[alternate]]))


if __name__=="__main__":
    main()
