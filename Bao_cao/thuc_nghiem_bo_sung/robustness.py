"""Predeclared report-only noise experiment on frozen CNNs, without fitting.

1,000 stratified test images per dataset; Gaussian noise in [0,1] space;
sigma in [0,.05,.1,.2], seeds 2026/2027/2028. Same clean indices per condition.
"""
from pathlib import Path
import json,os,sys,time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
ROOT=Path('D:/Tieuluanfinal');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from experiments.images import infer_batches,checkpoint
from experiments.common import sha


def main():
    records=[];samples={};start=time.perf_counter()
    configs=[('mnist','torch_improved'),('fashion','basic_cnn'),('cifar10','resnet_like')]
    for dataset,name in configs:
        data=np.load(ROOT/f'data/images/{dataset}.npz',allow_pickle=False)
        indexes,_=train_test_split(np.arange(len(data['y_test'])),train_size=1000,stratify=data['y_test'],random_state=2026)
        x=data['x_test'][indexes];y=data['y_test'][indexes];samples[dataset]=indexes.tolist()
        for sigma in [0.,.05,.1,.2]:
            for seed in [2026,2027,2028]:
                noise=np.random.default_rng(seed).normal(0,sigma,x.shape)
                noisy=np.rint(np.clip(x.astype('float32')/255+noise,0,1)*255).astype('uint8')
                pred=infer_batches(dataset,name,noisy).argmax(1)
                row=dict(dataset=dataset,model=name,n=len(y),sigma=sigma,seed=seed,accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')))
                records.append(row);print(json.dumps(row),flush=True)
    summary=[]
    for dataset,name in configs:
        for sigma in [0.,.05,.1,.2]:
            rows=[r for r in records if r['dataset']==dataset and r['sigma']==sigma]
            summary.append(dict(dataset=dataset,model=name,sigma=sigma,n=1000,repeats=len(rows),accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),
                accuracy_sd=float(np.std([r['accuracy'] for r in rows],ddof=1)),macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows]))))
    result=dict(protocol=__doc__,code_sha256=sha(Path(__file__)),checkpoint_sha256={d:sha(checkpoint(d,n)) for d,n in configs},
                samples=samples,records=records,summary=summary,elapsed_seconds=time.perf_counter()-start,
                limits='SD across noise realizations on a fixed subset and checkpoint; not training-seed or population uncertainty. No threshold/model tuning.')
    (HERE/'data/robustness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Report-only robustness experiment complete',flush=True)


if __name__=='__main__':main()
