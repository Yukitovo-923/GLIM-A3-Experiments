"""UID-balanced geometry, retrieval and confound sensitivity diagnostics."""
import numpy as np
import pandas as pd


def centroids(ds,z,subjects=None):
    result=[]
    for ts in ds.trials:
        if subjects is not None:
            ts=ts[np.isin(ds.subject[ts],subjects)]
        if not len(ts):
            result.append(np.full(z.shape[1],np.nan))
            continue
        vector=z[ts].mean(0,dtype=np.float64)
        result.append(vector/max(np.linalg.norm(vector),1e-12))
    return np.stack(result)


def distribution(similarity,mask):
    valid=mask.any(1)
    matrix=similarity[valid]
    mask=mask[valid]
    weights=np.broadcast_to(1/mask.sum(1,keepdims=True),mask.shape)[mask]
    values=matrix[mask]
    order=np.argsort(values)
    cumulative=np.cumsum(weights[order])/weights.sum()
    return dict(anchors=int(valid.sum()),directed_pairs=int(mask.sum()),
                mean=float(np.average(values,weights=weights)),
                q10_q50_q90=np.interp([.1,.5,.9],cumulative,values[order]).tolist())


def geometry(a,b,uid_a,uid_b,y_a,y_b):
    matrix=a@b.T
    legal=uid_a[:,None]!=uid_b
    same=y_a[:,None]==y_b
    out={key:distribution(matrix,legal & m) for key,m in [("same",same),("different",~same)]}
    out["separation"]=out["same"]["mean"]-out["different"]["mean"]
    return out


def neighbors(a,b,legal):
    ok=legal & np.isfinite(a).all(1)[:,None] & np.isfinite(b).all(1)[None,:]
    if not ok.any(1).all():
        raise ValueError("No eligible training neighbor for a validation UID")
    return np.where(ok,a@b.T,-np.inf).argmax(1)


def embedding_analysis(tr,va,z,root,tag):
    a,b=centroids(va,z["val"]),centroids(tr,z["train"])
    np.savez_compressed(root/f"centroids_{tag}.npz",train=b,val=a,train_uid=tr.uids,
                        val_uid=va.uids,train_label=tr.labels,val_label=va.labels)
    legal=va.uids[:,None]!=tr.uids
    nn=neighbors(a,b,legal)
    accuracy=float((va.labels==tr.labels[nn]).mean())
    edges=np.quantile(tr.lengths,[1/3,2/3])
    length_match=np.digitize(va.lengths,edges)[:,None]==np.digitize(tr.lengths,edges)
    length_nn=neighbors(a,b,legal & length_match)
    length_acc=float((va.labels==tr.labels[length_nn]).mean())
    subjects=np.array(sorted(set(tr.subject)&set(va.subject)))
    halves=(subjects[::2],subjects[1::2])
    cross=[]
    evidence=dict(uid=va.uids,label=va.labels,neighbor_uid=tr.uids[nn],neighbor_label=tr.labels[nn],
                  words=va.lengths,donor_words=tr.lengths[nn],length_matched_uid=tr.uids[length_nn])
    for i in range(2):
        query=centroids(va,z["val"],halves[i])
        gallery=centroids(tr,z["train"],halves[1-i])
        idx=neighbors(query,gallery,legal & length_match)
        evidence[f"cross_subject_length_uid_{i}"]=tr.uids[idx]
        cross.append(float((va.labels==tr.labels[idx]).mean()))
    pd.DataFrame(evidence).to_csv(root/f"retrieval_{tag}.csv",index=False)
    # Shuffle UID labels as a whole, maintaining reused-donor dependencies.
    rng=np.random.default_rng(1942)
    null=np.array([(rng.permutation(tr.labels)[nn]==va.labels).mean() for _ in range(5000)])
    return dict(nearest=accuracy,correct=int((va.labels==tr.labels[nn]).sum()),
                unique_donors=int(np.unique(nn).size),length_matched=length_acc,
                length_edges=edges.tolist(),length_and_disjoint_subjects=cross,
                subject_halves=[h.tolist() for h in halves],
                frequency_baseline=float(np.dot(np.bincount(tr.labels)/len(b),np.bincount(va.labels)/len(a))),
                majority_baseline=float((va.labels==np.bincount(tr.labels).argmax()).mean()),
                perm_mean=float(null.mean()),perm_q025_q975=np.quantile(null,[.025,.975]).tolist(),
                perm_p_ge=float((1+(null>=accuracy).sum())/(1+len(null))),
                train_train=geometry(b,b,tr.uids,tr.uids,tr.labels,tr.labels),
                val_train=geometry(a,b,va.uids,tr.uids,va.labels,tr.labels),
                val_val=geometry(a,a,va.uids,va.uids,va.labels,va.labels))
