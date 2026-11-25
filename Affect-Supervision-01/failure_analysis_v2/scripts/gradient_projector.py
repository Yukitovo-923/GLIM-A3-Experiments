"""Read-only check of gradient concentration in the EPA-only projection layer.

No optimizer is created. No backward() or optimizer.step() is called.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiment_assets import load_config
from infer import load_best_model
from task1_data import Task1DataModule


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
LOSSES = ("glim_plus_vq", "original_emotion_ce", "epa")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def group_names():
    groups = {
        "EPA_projector": lambda n: n.startswith("epa_projector."),
        "EEG_encoder_all": lambda n: n.startswith("eeg_encoder."),
        "EEG_aligner": lambda n: n.startswith(("aligner.in_proj", "aligner.q_x", "aligner.cross_attn_x")),
        "all_shared_EEG": lambda n: n.startswith("eeg_encoder.") or n.startswith(("aligner.in_proj", "aligner.q_x", "aligner.cross_attn_x")),
        "EEG_input_projection": lambda n: n.startswith("eeg_encoder.x_proj"),
        "aligner_input_projection": lambda n: n.startswith("aligner.in_proj"),
        "aligner_q_x": lambda n: n == "aligner.q_x",
        "aligner_cross_attention_x": lambda n: n.startswith("aligner.cross_attn_x"),
    }
    for i in range(6):
        groups[f"encoder_in_block_{i}"] = lambda n, i=i: n.startswith(f"eeg_encoder.in_blocks.{i}.")
        groups[f"encoder_out_block_{i}"] = lambda n, i=i: n.startswith(f"eeg_encoder.out_blocks.{i}.")
    return groups


def metrics(gradients, indices):
    def norm_and_dot(a, b=None):
        total=0.0
        for i in indices:
            x=gradients[a][i]
            if x is None:continue
            if b is None:
                total += float(torch.sum(x.float()**2))
            else:
                y=gradients[b][i]
                if y is not None:total += float(torch.sum(x.float()*y.float()))
        return total
    norms={loss:math.sqrt(norm_and_dot(loss)) for loss in LOSSES}
    def cosine(a,b):
        return norm_and_dot(a,b)/(norms[a]*norms[b]) if norms[a]>0 and norms[b]>0 else None
    return {**{f"norm_{k}":v for k,v in norms.items()},
            "weighted_norm_original_emotion_ce":.2*norms["original_emotion_ce"],
            "weighted_norm_epa":.2*norms["epa"],
            "cos_epa_glim":cosine("epa","glim_plus_vq"),
            "cos_epa_original_emotion_ce":cosine("epa","original_emotion_ce")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    parser.add_argument("--batches",type=int,default=8)
    parser.add_argument("--samples-per-batch",type=int,default=8)
    args=parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A800 or equivalent large BF16 GPU required")
    if args.batches<3 or args.samples_per_batch<2:
        raise ValueError("Need multiple batches with at least two trials each")
    config=load_config(ROOT/"configs/common.json")
    os.environ["GLIM_TEXT_MODEL_PATH"]=config["assets"]["text_model"]
    model,checkpoint,_=load_best_model(config,"B1_EPA",args.seed)
    before=sha(checkpoint)
    data=Task1DataModule(
        data_path=config["assets"]["data"],labels_path=config["assets"]["labels"],
        expected_manifest=config["assets"]["e1_manifest"],
        expected_sha256=config["assets"]["e1_manifest_sha256"],
        seed=args.seed,num_workers=0,
        bsz_train=config["training"]["batch_size_train"],
        bsz_val=config["training"]["batch_size_val"],
        bsz_test=config["training"]["batch_size_test"])
    data.setup("fit")
    model.train_uid_counts=data.train_uid_counts
    model.train()
    named=[(name,p) for name,p in model.named_parameters()
           if p.requires_grad and (name.startswith("eeg_encoder.") or name.startswith("aligner.")
                                   or name.startswith("epa_projector."))]
    names=[name for name,_ in named]
    params=[p for _,p in named]
    groups={name:[i for i,key in enumerate(names) if predicate(key)]
            for name,predicate in group_names().items()}
    if any(not indices for indices in groups.values()):
        raise RuntimeError("An expected EEG parameter group is empty")
    loader=data.train_dataloader()
    batch_indices=iter(loader.batch_sampler)
    rows=[]
    for batch_number in range(args.batches):
        indices=next(batch_indices)[:args.samples_per_batch]
        batch=default_collate([data.train_set[int(i)] for i in indices])
        batch={key:value.cuda(non_blocking=True) if isinstance(value,torch.Tensor) else value
               for key,value in batch.items()}
        torch.manual_seed(args.seed+batch_number)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model(batch)
            losses={"glim_plus_vq":out["loss_glim"]+out["loss_vq"],
                    "original_emotion_ce":out["loss_disc_emotion"],
                    "epa":out["loss_epa"]}
        gradients={}
        for j,(name,loss) in enumerate(losses.items()):
            raw=torch.autograd.grad(loss,params,retain_graph=j<len(losses)-1,
                                    allow_unused=True)
            gradients[name]=[g.detach().float().cpu() if g is not None else None for g in raw]
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Gradients accumulated on model parameters unexpectedly")
        for group,positions in groups.items():
            rows.append({"seed":args.seed,"batch":batch_number,"n_trials":len(indices),
                         "n_uids":len(set(int(batch["text uid"][k]) for k in range(len(indices)))),
                         "group":group,**metrics(gradients,positions)})
        print("completed batch",batch_number+1,"of",args.batches,flush=True)
        del out,gradients,batch
        torch.cuda.empty_cache()
    if before != sha(checkpoint):
        raise RuntimeError("Checkpoint bytes changed during read-only audit")
    with (OUT/"tables"/"gradient_projector_by_batch_layer.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for group in groups:
        z=[r for r in rows if r["group"]==group]
        summary[group]={}
        for key in ("norm_glim_plus_vq","norm_original_emotion_ce","norm_epa",
                    "weighted_norm_original_emotion_ce","weighted_norm_epa",
                    "cos_epa_glim","cos_epa_original_emotion_ce"):
            values=[r[key] for r in z if r[key] is not None]
            summary[group][key]={"mean":statistics.mean(values),
                                 "sd":statistics.stdev(values) if len(values)>1 else 0.0,
                                 "values":values} if values else None
    result={"status":"complete","seed":args.seed,"batches":args.batches,
            "samples_per_batch":args.samples_per_batch,"split":"train only",
            "checkpoint_sha256_unchanged":before,
            "loss_definitions":{"glim_plus_vq":"0.5 CLIP + 0.5 LM + VQ",
                                "original_emotion_ce":"raw V3 discrete CE; training weight 0.2",
                                "epa":"raw EPA CE; training weight 0.2"},
            "mode":"train(); frozen T5 parameters, actual dropout mode; fixed seed per batch",
            "optimizer_created":False,"optimizer_step_called":False,"summary":summary}
    (OUT/"gradient_projector.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("wrote gradient_projector.json and per-batch layer table")


if __name__=="__main__":
    main()
