"""Recompute EPA text geometry from frozen, already extracted sentence embeddings.

Only train/validation UIDs are analyzed. No model training or test-set tuning.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

OUT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = OUT.parents[1]
AUDIT = ROOT / "A3-v2-results" / "Text-space-Audit"
EXPERIMENT = OUT.parent
CLASSES = ("negative", "neutral", "positive")
TAU = 0.07


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x)>1 else 0.0,
            "q05": float(np.quantile(x, .05)), "median": float(np.median(x)),
            "q95": float(np.quantile(x, .95))}


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=1, keepdims=True)


def load(name):
    directory = AUDIT / name
    # These NPZ files are trusted artifacts from the user's prior local audit.
    with np.load(directory / "text_embeddings.npz", allow_pickle=True) as data:
        x = np.asarray(data["encoder_mean"], dtype=np.float64)
        uid = np.asarray(data["uid"], dtype=np.int64)
        phase = np.asarray(data["phase"], dtype=str)
        label = np.asarray(data["label"], dtype=str)
        token_length = np.asarray(data["token_length"], dtype=int)
    p = np.asarray(np.load(directory / "encoder_mean_prototypes.npy", allow_pickle=False),
                   dtype=np.float64)
    assert x.shape == (400, 1024) and p.shape == (3, 1024)
    assert set(label) == set(CLASSES) and set(phase) == {"train", "val", "test"}
    assert len(set(uid)) == 400
    assert np.max(np.abs(np.linalg.norm(x, axis=1)-1)) < 1e-5
    assert np.max(np.abs(np.linalg.norm(p, axis=1)-1)) < 1e-5
    train = phase == "train"
    y = np.array([CLASSES.index(v) for v in label], dtype=int)
    made = normalize(np.stack([x[train & (y==k)].mean(0) for k in range(3)]))
    assert np.max(np.abs(made - p)) < 2e-7, "Stored prototype cannot be rebuilt from train only"
    return {"directory": directory, "x": x, "p": p, "uid": uid, "phase": phase,
            "label": label, "y": y, "token_length": token_length,
            "prototype_reconstruction_max_abs_diff": float(np.max(np.abs(made-p)))}


def class_pair_stats(x, y):
    out = {"within": {}, "between": {}}
    for c, name in enumerate(CLASSES):
        z = x[y == c]
        sims = (z @ z.T)[np.triu_indices(len(z), 1)]
        out["within"][name] = summarize(sims)
    for c in range(3):
        for d in range(c+1, 3):
            sims = (x[y==c] @ x[y==d].T).ravel()
            out["between"][f"{CLASSES[c]}|{CLASSES[d]}"] = summarize(sims)
    return out


def anisotropy(x):
    mu = x.mean(0)
    sims = x @ x.T
    pair = sims[np.triu_indices(len(x), 1)]
    centered = x - mu
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / np.sum(singular**2)
    entropy = -np.sum(variance[variance>0] * np.log(variance[variance>0]))
    return {"mean_vector_norm": float(np.linalg.norm(mu)),
            "mean_pairwise_cosine": float(pair.mean()),
            "pairwise_cosine_q05_q95": [float(np.quantile(pair,.05)),float(np.quantile(pair,.95))],
            "centered_pc1_variance_fraction": float(variance[0]),
            "centered_top10_variance_fraction": float(variance[:10].sum()),
            "centered_effective_rank": float(np.exp(entropy))}


def centered_control(bundle):
    """Diagnostic only: remove train mean, then fit prototypes on train sentences."""
    x,y,phase = bundle["x"],bundle["y"],bundle["phase"]
    train=phase=="train";val=phase=="val"
    z=normalize(x-x[train].mean(0))
    proto=normalize(np.stack([z[train & (y==c)].mean(0) for c in range(3)]))
    pred=(z[val]@proto.T).argmax(1)
    cm=np.zeros((3,3),dtype=int)
    for true,guess in zip(y[val],pred):cm[true,guess]+=1
    support=cm.sum(1);predicted=cm.sum(0)
    f1=np.divide(2*cm.diagonal(),support+predicted,out=np.zeros(3),
                 where=(support+predicted)>0)
    return {"val_macro_f1":float(f1.mean()),"val_confusion_matrix":cm.tolist(),
            "prototype_cosine":(proto@proto.T).tolist(),
            "interpretation":"train-only mean-centering diagnostic; no test rows used"}


def score_rows(bundle, name):
    x,p,y,uid,phase = (bundle[k] for k in ("x","p","y","uid","phase"))
    rows=[]
    for i in np.flatnonzero(phase != "test"):
        proto = p
        # Train accuracy uses leave-one-out prototypes to avoid self-inclusion.
        if phase[i] == "train":
            proto = p.copy()
            c=y[i]
            members=x[(phase=="train") & (y==c)]
            proto[c]=normalize((members.sum(0)-x[i])/(len(members)-1))
        sim=x[i]@proto.T
        logits=sim/TAU
        prob=softmax(logits[None,:])[0]
        other=max(logits[k] for k in range(3) if k != y[i])
        top=np.sort(logits)
        rows.append({"representation":name,"uid":int(uid[i]),"split":str(phase[i]),
                     "true_class":CLASSES[y[i]],"predicted_class":CLASSES[int(logits.argmax())],
                     "token_length":int(bundle["token_length"][i]),
                     **{f"cosine_{CLASSES[k]}":float(sim[k]) for k in range(3)},
                     **{f"logit_{CLASSES[k]}":float(logits[k]) for k in range(3)},
                     **{f"prob_{CLASSES[k]}":float(prob[k]) for k in range(3)},
                     "top1_minus_top2_logit":float(top[-1]-top[-2]),
                     "true_minus_best_other_logit":float(logits[y[i]]-other),
                     "true_probability":float(prob[y[i]])})
    return rows


def score_summary(rows):
    out={}
    for split in ("train","val"):
        z=[r for r in rows if r["split"]==split]
        cm=np.zeros((3,3),dtype=int)
        for r in z:cm[CLASSES.index(r["true_class"]),CLASSES.index(r["predicted_class"])]+=1
        support=cm.sum(1);pred=cm.sum(0)
        f1=np.divide(2*cm.diagonal(),support+pred,out=np.zeros(3),where=(support+pred)>0)
        out[split]={"n":len(z),"confusion_matrix":cm.tolist(),
                    "macro_f1":float(f1.mean()),"accuracy":float(cm.trace()/cm.sum()),
                    "predicted_counts":dict(zip(CLASSES,map(int,cm.sum(0)))),
                    "overall_true_logit_margin":summarize([r["true_minus_best_other_logit"] for r in z]),
                    "by_class":{c:{"n":sum(r["true_class"]==c for r in z),
                         "own_cosine":summarize([r[f"cosine_{c}"] for r in z if r["true_class"]==c]),
                         "true_logit_margin":summarize([r["true_minus_best_other_logit"] for r in z if r["true_class"]==c]),
                         "true_probability":summarize([r["true_probability"] for r in z if r["true_class"]==c]),
                         "logits":{k:summarize([r[f"logit_{k}"] for r in z if r["true_class"]==c]) for k in CLASSES},
                         "probabilities":{k:summarize([r[f"prob_{k}"] for r in z if r["true_class"]==c]) for k in CLASSES}}
                         for c in CLASSES}}
    return out


def plots(bundles, all_rows):
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,(name,b) in zip(axes,bundles.items()):
        m=b["p"]@b["p"].T
        ax.imshow(m,vmin=.85,vmax=1,cmap="viridis")
        ax.set_xticks(range(3),CLASSES,rotation=25);ax.set_yticks(range(3),CLASSES)
        ax.set_title(f"{name}: prototype cosine")
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{m[i,j]:.3f}",ha="center",va="center",
                        color="black" if m[i,j]>.97 else "white")
    fig.savefig(OUT/"figures"/"prototype_cosines.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for ax,(name,b) in zip(axes,bundles.items()):
        idx=b["phase"]=="train";x=b["x"][idx];y=b["y"][idx]
        data=[];labels=[]
        for c in range(3):
            z=x[y==c];data.append((z@z.T)[np.triu_indices(len(z),1)]);labels.append("within "+CLASSES[c][:3])
        for c,d in ((0,1),(0,2),(1,2)):
            data.append((x[y==c]@x[y==d].T).ravel());labels.append(f"between {c}-{d}")
        ax.boxplot(data,tick_labels=labels,showfliers=False);ax.tick_params(axis="x",rotation=35)
        ax.set_title(f"{name}: train sentence pair cosine");ax.set_ylabel("Cosine")
    fig.savefig(OUT/"figures"/"within_between.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for col,name in enumerate(bundles):
        for row,split in enumerate(("train","val")):
            ax=axes[row,col]
            z=[r for r in all_rows if r["representation"]==name and r["split"]==split]
            groups=[[r["true_minus_best_other_logit"] for r in z if r["true_class"]==c] for c in CLASSES]
            ax.boxplot(groups,tick_labels=CLASSES,showfliers=False)
            ax.axhline(0,color="black",lw=.8);ax.set_title(f"{name} {split}: true-class logit margin")
            ax.set_ylabel("True minus best other (tau=0.07)")
    fig.savefig(OUT/"figures"/"logit_margins.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,name in zip(axes,bundles):
        z=[r for r in all_rows if r["representation"]==name and r["split"]=="val"]
        means=np.array([[np.mean([r[f"prob_{pred}"] for r in z if r["true_class"]==true])
                         for pred in CLASSES] for true in CLASSES])
        im=ax.imshow(means,vmin=0,vmax=1,cmap="Blues")
        ax.set_xticks(range(3),CLASSES,rotation=25);ax.set_yticks(range(3),CLASSES)
        ax.set_xlabel("Prototype probability");ax.set_ylabel("True class")
        ax.set_title(f"{name}: validation mean probability")
        for i in range(3):
            for j in range(3):ax.text(j,i,f"{means[i,j]:.2f}",ha="center",va="center")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8)
    fig.savefig(OUT/"figures"/"validation_probabilities.png",dpi=160);plt.close(fig)

    left,right=bundles.values()
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    for split,color in (("train","#457b9d"),("val","#e76f51")):
        idx=left["phase"]==split
        cosine=np.sum(left["x"][idx]*right["x"][idx],axis=1)
        ax.hist(cosine,bins=np.linspace(.6,1,21),alpha=.55,label=f"{split} ({len(cosine)} UIDs)",color=color)
    ax.set_xlabel("Cosine: unprefixed vs 'To English:' embedding for same UID")
    ax.set_ylabel("Sentence count");ax.legend()
    fig.savefig(OUT/"figures"/"prefix_embedding_shift.png",dpi=160);plt.close(fig)


def main():
    original=AUDIT/"text_space_audit.py"
    prefixed=AUDIT/"text_space_audit_prefixed.py"
    assert 'texts = sentences["input text"]' in original.read_text(encoding="utf-8")
    assert 'INPUT_TEMPLATE = "To English: <MASK>"' in prefixed.read_text(encoding="utf-8")
    bundles={"EPA_used_unprefixed":load("run-01"),
             "V3_style_prefixed":load("run-03-v3-prefixed")}
    expected=EXPERIMENT/"configs"/"common.json"
    config=json.loads(expected.read_text(encoding="utf-8"))
    assert sha256(bundles["EPA_used_unprefixed"]["directory"]/"encoder_mean_prototypes.npy")==config["assets"]["prototype_sha256"]
    left,right=bundles.values()
    assert np.array_equal(left["uid"],right["uid"])
    assert np.array_equal(left["phase"],right["phase"])
    assert np.array_equal(left["label"],right["label"])
    all_rows=[];result={"scope":"train/val only; test array rows excluded from all statistics",
                       "class_order":CLASSES,"tau":TAU,"representations":{}}
    for name,b in bundles.items():
        train=b["phase"]=="train";val=b["phase"]=="val"
        row=score_rows(b,name);all_rows.extend(row)
        cosine=b["p"]@b["p"].T
        result["representations"][name]={
            "embedding_sha256":sha256(b["directory"]/"text_embeddings.npz"),
            "prototype_sha256":sha256(b["directory"]/"encoder_mean_prototypes.npy"),
            "prototype_reconstruction_max_abs_diff":b["prototype_reconstruction_max_abs_diff"],
            "prototype_cosine":cosine.tolist(),
            "prototype_euclidean_distance":np.sqrt(np.maximum(0,2-2*cosine)).tolist(),
            "prototype_angular_degrees":np.degrees(np.arccos(np.clip(cosine,-1,1))).tolist(),
            "raw_class_centroid_norm":{CLASSES[c]:float(np.linalg.norm(b["x"][train & (b["y"]==c)].mean(0))) for c in range(3)},
            "train_pair_cosine":class_pair_stats(b["x"][train],b["y"][train]),
            "val_pair_cosine":class_pair_stats(b["x"][val],b["y"][val]),
            "train_anisotropy":anisotropy(b["x"][train]),
            "train_mean_centered_control":centered_control(b),
            "validation_classifier":score_summary(row),
        }
    idx=left["phase"]!="test"
    shift=np.sum(left["x"][idx]*right["x"][idx],axis=1)
    result["preprocessing_comparison"]={
        "finding":"EPA used unprefixed raw sentences; V3 training input uses 'To English: ' prefix",
        "uid_aligned_train_val_embedding_cosine":summarize(shift),
        "prototype_cross_style_same_class_cosine":[float(np.dot(left["p"][c],right["p"][c])) for c in range(3)],
        "old_source_code_sha256":{"unprefixed":sha256(original),"prefixed":sha256(prefixed)},
    }
    (OUT/"text_geometry.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"tables"/"train_val_sentence_scores.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    class_rows=[];pair_rows=[]
    for name,z in result["representations"].items():
        for i in range(3):
            for j in range(i+1,3):
                pair_rows.append({"representation":name,"class_a":CLASSES[i],"class_b":CLASSES[j],
                    "prototype_cosine":z["prototype_cosine"][i][j],
                    "prototype_euclidean_distance":z["prototype_euclidean_distance"][i][j],
                    "prototype_angular_degrees":z["prototype_angular_degrees"][i][j]})
        for split in ("train","val"):
            for c in CLASSES:
                m=z["validation_classifier"][split]["by_class"][c]
                class_rows.append({"representation":name,"split":split,"class":c,
                    "n_sentences":m["n"],
                    "within_pair_cosine_mean":z[f"{split}_pair_cosine"]["within"][c]["mean"],
                    "own_prototype_cosine_mean":m["own_cosine"]["mean"],
                    "true_logit_margin_mean":m["true_logit_margin"]["mean"],
                    "true_probability_mean":m["true_probability"]["mean"],
                    "predicted_negative_probability_mean":m["probabilities"]["negative"]["mean"],
                    "predicted_neutral_probability_mean":m["probabilities"]["neutral"]["mean"],
                    "predicted_positive_probability_mean":m["probabilities"]["positive"]["mean"],
                    "macro_f1_all_classes":z["validation_classifier"][split]["macro_f1"]})
    for path,table in ((OUT/"tables"/"class_geometry_summary.csv",class_rows),
                       (OUT/"tables"/"prototype_pairs.csv",pair_rows)):
        with path.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plots(bundles,all_rows)
    print("created text_geometry.json, sentence scores CSV and five figures")
    for name,z in result["representations"].items():
        print(name,"val macro-F1",z["validation_classifier"]["val"]["macro_f1"],
              "train LOO macro-F1",z["validation_classifier"]["train"]["macro_f1"])
    print("prefix shift",result["preprocessing_comparison"]["uid_aligned_train_val_embedding_cosine"])


if __name__=="__main__":
    main()
