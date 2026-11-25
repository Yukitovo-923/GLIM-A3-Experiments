"""Descriptive query-position and subject diagnostics for saved E1 features."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from data.revised_labels import audit

def fraction_between(x, labels):
    grand=x.mean(0);total=((x-grand)**2).sum()
    if total==0:return 0.0
    between=0.0
    for label in np.unique(labels):
        sub=x[labels==label]
        between+=len(sub)*((sub.mean(0)-grand)**2).sum()
    return float(between/total)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-path',required=True);p.add_argument('--labels',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    root=Path(a.run)
    frame,_=audit(pd.read_pickle(a.data_path),pd.read_pickle(a.labels))
    sr=frame[(frame.dataset=='ZuCo1')&(frame.task=='task1')&frame['sentiment label'].isin(['negative','neutral','positive'])]
    report={'note':'Descriptive variance decomposition; not a causal explanation or leakage test','splits':{}}
    for phase in ('train','val','test'):
        sub=sr[sr.phase==phase].reset_index(drop=True)
        subjects=sub.subject.to_numpy();sentences=sub['text uid'].to_numpy()
        data=np.load(root/f'{phase}_real.npz')
        info={}
        for key in ('encoder','aligner','quantized'):
            x=data[key].astype(float)
            info[key]={'subject_between_variance_fraction':fraction_between(x,subjects),
                       'sentence_between_variance_fraction':fraction_between(x,sentences),
                       'feature_std_all_elements':float(x.std())}
        h=data['position_code_hist'].astype(float)
        joint=h/h.sum();p_pos=joint.sum(1,keepdims=True);p_code=joint.sum(0,keepdims=True)
        nonzero=joint>0
        mi=float(np.sum(joint[nonzero]*np.log2(joint[nonzero]/(p_pos*p_code)[nonzero])))
        entropy=float(-np.sum(p_code[p_code>0]*np.log2(p_code[p_code>0])))
        info['code_query_position']={'mutual_information_bits':mi,'code_entropy_bits':entropy,
             'mi_over_code_entropy':mi/entropy if entropy else None,
             'mean_top1_code_ratio_by_position':float(np.mean(h.max(1)/h.sum(1))),
             'distinct_position_dominant_codes':int(len(set(h.argmax(1))))}
        report['splits'][phase]=info
    (root/'confounds.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
