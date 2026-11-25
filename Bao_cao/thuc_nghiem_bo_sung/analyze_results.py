"""Derive report evidence from the frozen submission; no model selection or fitting.

Run in the original experiment Python 3.13 environment, not the document renderer.
"""
from pathlib import Path
import hashlib,json,os,platform,sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
ROOT=Path('D:/Tieuluanfinal')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,brier_score_loss,mean_absolute_error,mean_squared_error,f1_score,accuracy_score
from experiments.tabular import load_data


def dump(name,value):
    target=HERE/'data'/name;target.parent.mkdir(parents=True,exist_ok=True)
    def cast(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=cast),encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    submission=read(ROOT/'runs/verification/submission.json')
    evidence=dict(source_verification=submission['verified_at'],tests=submission['tests'],tabular={},cnn={},sequences={},rag={},agents={},rl={})
    for spec in submission['tabular']:
        task=spec['task'];folder=ROOT/spec['run'];x,y,groups,source_rows,profile=load_data(task)
        raw=pd.read_csv(ROOT/'data/raw'/dict(diabetes='diabete.csv',house='house.csv',ecommerce='customerData.csv')[task])
        split=pd.read_csv(folder/'split.csv',dtype={'group_hash':'uint64'})
        records={name:pd.read_csv(folder/f'{name}_predictions.csv') for name in ['baseline','ml','mlp']}
        for key,records_frame in records.items():
            records_frame.to_csv(HERE/'data'/f'{task}_{key}_test.csv',index=False)
        data=dict(run=spec['run'],profile=profile,split=spec['split'],metrics=spec['metrics'],selection=spec['selection'],
            config=read(folder/'config.json'),architecture=read(folder/'architecture.json'),
            validation=pd.read_csv(folder/'validation.csv').to_dict('records'),history=pd.read_csv(folder/'history.csv').to_dict('records'),
            original_shape=list(raw.shape),full_row_duplicates=int(raw.duplicated().sum()),unique_profiles=int(len(np.unique(groups))),
            missing_original={k:int(v) for k,v in raw.isna().sum().items()},elapsed=read(folder/'complete.json')['elapsed_seconds'])
        indexes=np.flatnonzero(split.split.to_numpy()=='test')
        if task!='house':
            data['test_label_counts']={str(k):int(v) for k,v in pd.Series(y[indexes]).value_counts().sort_index().items()}
            data['classification']={}
            for name,r in records.items():
                data['classification'][name]=dict(confusion_matrix=confusion_matrix(r.actual,r.prediction,labels=[0,1]).tolist(),
                    brier=float(brier_score_loss(r.actual,r.score)))
            if task=='diabetes':
                test=x.iloc[indexes].reset_index(drop=True);test['actual']=records['mlp'].actual;test['pred']=records['mlp'].prediction
                data['age_groups']=[dict(age=int(age),n=len(g),positive_rate=float(g.actual.mean()),f1=float(f1_score(g.actual,g.pred,zero_division=0)),
                    accuracy=float(accuracy_score(g.actual,g.pred))) for age,g in test.groupby('Age')]
                data['bmi_quantiles']=x.BMI.quantile([0,.25,.5,.75,1]).to_dict()
            else:
                test=x.iloc[indexes].reset_index(drop=True);test['actual']=records['mlp'].actual;test['pred']=records['mlp'].prediction
                data['device_groups']=[dict(device=device,n=len(g),f1=float(f1_score(g.actual,g.pred,zero_division=0))) for device,g in test.groupby('PreferredDevice')]
                data['invalid_negative']={c:int((pd.to_numeric(raw[c],errors='coerce')<0).sum()) for c in ['NumberOfPurchases','TimeSpentOnWebsite','CustomerTenureYears','LastPurchaseDaysAgo']}
        else:
            r=records['mlp'];data['error_quantiles']={str(q):float(v) for q,v in (r.prediction-r.actual).abs().quantile([.5,.75,.9,.95,.99]).items()}
            bins=pd.qcut(r.actual,4,duplicates='drop')
            data['price_groups']=[dict(group=str(interval),n=int(mask.sum()),mae=float(mean_absolute_error(r.actual[mask],r.prediction[mask])),
                rmse=float(np.sqrt(mean_squared_error(r.actual[mask],r.prediction[mask])))) for interval in bins.cat.categories for mask in [bins==interval]]
        evidence['tabular'][task]=data
        # No label information is used to choose the sample used for EDA scatter plots.
        sample=x.assign(target=y).sample(min(4000,len(x)),random_state=2026)
        sample.to_csv(HERE/'data'/f'{task}_eda_sample.csv',index=False)
    for dataset in ['mnist','fashion','cifar10']:
        rows=[]
        for row in [r for r in submission['cnn'] if r['dataset']==dataset]:
            folder=ROOT/row['run'];pred=pd.read_csv(folder/'predictions.csv')
            rows.append({**row,'confusion_matrix':confusion_matrix(pred.actual,pred.prediction,labels=list(range(10))).tolist(),
                         'classification':read(folder/'classification_report.json')})
        evidence['cnn'][dataset]=rows
    for spec in submission['sequences']:
        folder=ROOT/spec['run'];records=pd.read_csv(folder/'predictions.csv');cfg=read(folder/'config.json');task=spec['task']
        name=f"{task}_{spec['framework']}_{spec['architecture']}";data={**spec,'metadata':cfg['metadata'],'n':len(records)}
        records.to_csv(HERE/'data'/f'{name}.csv',index=False)
        if task=='stock':
            data['chronological_quarters']=[]
            for i,index in enumerate(np.array_split(np.arange(len(records)),4)):
                sub=records.iloc[index]
                data['chronological_quarters'].append(dict(part=i+1,n=len(sub),first=sub.date.iloc[0],last=sub.date.iloc[-1],
                    rmse=float(np.sqrt(mean_squared_error(sub.actual,sub.prediction))),baseline_rmse=float(np.sqrt(mean_squared_error(sub.actual,sub.baseline)))))
        else:
            data['confusion_matrix']=confusion_matrix(records.actual,records.prediction,labels=[0,1,2]).tolist()
            data['class_counts']={str(k):int(v) for k,v in records.actual.value_counts().sort_index().items()}
        evidence['sequences'][name]=data
    rag_folder=ROOT/submission['rag']['run']
    evidence['rag']={**submission['rag'],'cases':read(ROOT/'experiments/fixtures/rag_cases.json'),
                     'retrieval_predictions':read(rag_folder/'retrieval_predictions.json'),
                     'generation_predictions':read(rag_folder/'generation_predictions.json'),'config':read(rag_folder/'config.json')}
    agent_folder=ROOT/submission['agents']['run']
    evidence['agents']={**submission['agents'],'routing_predictions':read(agent_folder/'routing_predictions.json'),
                        'fault_predictions':read(agent_folder/'fault_predictions.json')}
    benchmark=read(ROOT/'runs/verification/rl_benchmark.json')
    evidence['rl']={**benchmark,'results':{r['run_id']:read(ROOT/'runs/rl'/r['run_id']/'result.json') for r in benchmark['rows']}}
    from demo.knowledge import DOCUMENTS,NODES,EDGES
    from demo.domain_knowledge import domain_config
    evidence['knowledge']=dict(documents=DOCUMENTS,nodes=NODES,edges=EDGES,domains={d:domain_config(d) for d in ['diabetes','stock','gold']})
    evidence['architectures']=read(ROOT/'assets/data/architecture_registry.json')
    evidence['sources']=read(ROOT/'data/sources.json')
    evidence['environment']=dict(python=platform.python_version(),platform=platform.platform(),versions=read(ROOT/'runs/tabular/diabetes'/Path(submission['tabular'][0]['run']).name/'complete.json')['versions'])
    dump('evidence.json',evidence)
    print(json.dumps(dict(tabular=len(evidence['tabular']),cnn=sum(map(len,evidence['cnn'].values())),sequences=len(evidence['sequences']),rag_test=len(evidence['rag']['cases']['test']),rl_runs=len(benchmark['rows'])),ensure_ascii=False))


if __name__=='__main__':
    (HERE/'data').mkdir(parents=True,exist_ok=True)
    main()
