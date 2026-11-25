"""Score the fixed validation generations and attribute assignments."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pilot import Expert, load
from prepare import ASSETS, ROOT, write

sys.path.insert(0, str(ASSETS/"evaluation"))
sys.path.insert(0, str(ASSETS/".venv-eval/lib/python3.12/site-packages"))
from score_predictions import lexical_metrics
from bert_score import BERTScorer


def read(name):
    return json.loads((ROOT/name).read_text())


KEYS = ["bleu1","bleu2","bleu4","rouge_l","bert_score_f1","length_words","repeated_bigram_rate"]


@torch.no_grad()
def attributes():
    result = {}
    for split in ("train","val"):
        ds = load(split)
        for stage in ("stage1","stage2"):
            states = torch.load(ROOT/(stage+"_heads.pt"),weights_only=True)
            state = states["A2"]
            head = Expert("A2",state["center"],state["scale"]).cuda()
            head.load_state_dict(state)
            logits,_,idx = head(ds["x"])
            frame = pd.DataFrame(dict(uid=ds["uid"].tolist(),label=ds["label"].tolist(),code=idx.tolist(),
                                      subject=ds["subject"]))
            counts = pd.crosstab(frame.uid,frame.code).reindex(columns=range(12),fill_value=0)
            shares = counts.div(counts.sum(axis=1),axis=0)
            labels = frame.groupby("uid").label.first()
            weights = shares.mean().to_numpy()
            positive = weights[weights>0]
            result[stage+"_"+split] = dict(code_share=weights.tolist(),
                code_share_by_label=shares.groupby(labels).mean().to_numpy().tolist(),
                dominant_counts=np.bincount(shares.to_numpy().argmax(1),minlength=12).tolist(),
                uid_perplexity=float(np.exp(-np.sum(positive*np.log(positive)))),
                multi_code_uids=int((counts.gt(0).sum(axis=1)>1).sum()))
            for j,p in enumerate(logits.softmax(-1).T):
                frame["p"+str(j)] = p.tolist()
            frame.to_csv(ROOT/(stage+"_"+split+"_assignments.csv"),index=False)
    write("uid_code_usage.json",result)


def score():
    config = json.loads((ASSETS/"step_level_diagnosis_run01/configs/bertscore_model.json").read_text())
    scorer = BERTScorer(model_type=config["path"],num_layers=17,batch_size=16,device="cuda",
                        idf=False,rescale_with_baseline=False,use_fast_tokenizer=False)
    groups = {}
    for arm in ("A0","A1","A2"):
        for condition in (["matched","mismatched","intervention","active_swap"] if arm=="A2" else ["matched","mismatched"]):
            key = arm+"_"+condition
            rows = read(key+".json")
            assert len(rows)==406 and len({r["uid"] for r in rows})==40
            lexical_metrics(rows)
            *_,bs = scorer.score([r["prediction"] for r in rows],[r["reference"] for r in rows])
            for row,b in zip(rows,bs.tolist()):
                words = row["prediction"].lower().split()
                bigrams = list(zip(words,words[1:]))
                row.update(bert_score_f1=b,length_words=len(words),
                           repeated_bigram_rate=(len(bigrams)-len(set(bigrams)))/max(1,len(bigrams)))
            frame = pd.DataFrame(rows).set_index("trial").sort_index()
            frame.to_csv(ROOT/(key+"_scored.csv"))
            groups[key] = frame
    def uid_scores(df):
        return df.groupby("uid")[KEYS].mean().sort_index()
    means = {k:uid_scores(v).mean().to_dict() for k,v in groups.items()}
    rng = np.random.default_rng(42)
    draws = rng.integers(40,size=(5000,40))
    def summarize(diff):
        assert len(diff)==40
        return {k:dict(mean=float(a.mean()),ci=np.quantile(a[draws].mean(1),[.025,.975]).tolist())
                for k in KEYS for a in [diff[k].to_numpy()]}
    contrasts = {}
    for k1,k2 in [("A2_matched","A1_matched"),("A2_matched","A0_matched"),("A1_matched","A0_matched"),
                  ("A0_matched","A0_mismatched"),("A1_matched","A1_mismatched"),
                  ("A2_matched","A2_mismatched"),("A2_matched","A2_intervention"),("A2_matched","A2_active_swap")]:
        left,right = groups[k1],groups[k2]
        assert left[["uid","subject","reference"]].equals(right[["uid","subject","reference"]])
        changed = left.prediction.ne(right.prediction)
        contrasts[k1+"-"+k2] = dict(scores=summarize(uid_scores(left)-uid_scores(right)),
                changed_trials=int(changed.sum()),uid_change_rate=float(changed.groupby(left.uid).mean().mean()))
    effects = {k:uid_scores(groups[k+"_matched"])-uid_scores(groups[k+"_mismatched"]) for k in ("A0","A1","A2")}
    intervention = groups["A2_intervention"]
    matched = groups["A2_matched"]
    assert ((matched.code+4)%12==intervention.code).all()
    write("generation.json",dict(means=means,contrasts=contrasts,bert_model=config,
                                 paired_eeg_effect={k:summarize(effects["A2"]-effects[k]) for k in ("A0","A1")}))


if __name__=="__main__":
    attributes()
    score()
