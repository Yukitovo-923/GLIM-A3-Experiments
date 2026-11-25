"""Sentence bootstrap intervals for unpaired metrics from exact confusion counts."""
import json,sys
import numpy as np
from pathlib import Path

root=Path(sys.argv[1]); report=json.loads((root/'report.json').read_text());rng=np.random.default_rng(42)
for feature,entry in report['features'].items():
    for name,record in [('real',entry['test_real']),('majority',entry['majority_baseline']),
                        *list(entry['controls'].items())]:
        cm=np.array(record['confusion']); pairs=np.array([(i,j) for i in range(3) for j in range(3) for _ in range(cm[i,j])])
        if len(pairs)!=40:raise ValueError('Expected one row per test sentence')
        draws=[]
        for _ in range(2000):
            sample=pairs[rng.integers(0,len(pairs),len(pairs))]
            tab=np.zeros((3,3),int)
            np.add.at(tab,(sample[:,0],sample[:,1]),1)
            support=tab.sum(1);tp=tab.diagonal();den=support+tab.sum(0)
            f1=np.divide(2*tp,den,out=np.zeros(3,float),where=den>0)
            rec=np.divide(tp,support,out=np.zeros(3,float),where=support>0)
            draws.append((f1.mean(),rec[support>0].mean()))
        d=np.array(draws)
        record['macro_f1_bootstrap_95pct_ci']=np.quantile(d[:,0],[.025,.975]).tolist()
        record['balanced_accuracy_bootstrap_95pct_ci']=np.quantile(d[:,1],[.025,.975]).tolist()
(root/'report_with_metric_ci.json').write_text(json.dumps(report,indent=2)+'\n')
print('METRIC_CI_PASS')
