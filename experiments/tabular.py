"""Comparable ML/MLP protocol with feature-group splits and frozen test evaluation.

python -m experiments.tabular train --task all
python -m experiments.tabular evaluate --run runs/tabular/<task>/<run-id>
"""
import argparse, copy, json, random, time
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .common import ROOT, sha, write_json, finish_run, check_run

RAW = ROOT/'data/raw'
SPECS = {
    'diabetes': dict(file='diabete.csv', target='Diabetes_binary', kind='classification'),
    'house': dict(file='house.csv', target='price', kind='regression'),
    'ecommerce': dict(file='customerData.csv', target='PurchaseStatus', kind='classification'),
}
HOUSE_NUM = ['floor','area','rooms','population','longitude','latitude']
HOUSE_CAT = ['offer_type','offer_type_of_building','market','city_name','voivodeship','month']
NONNEGATIVE = ['NumberOfPurchases','TimeSpentOnWebsite','CustomerTenureYears','LastPurchaseDaysAgo']
PROTOCOL = dict(version=1, seed=42, split='feature-group-shuffle-70-15-15',
                mlp_hidden=[64,32,16,8], epochs=35, patience=6, batch_size=1024,
                learning_rate=.001, refit_train_validation=False, threshold=.5)


class TabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,output_dim))
    def forward(self, x): return self.net(x)


@lru_cache(maxsize=1)
def city_mapping():
    profiles = json.loads((ROOT/'assets/data/city_profiles.json').read_text(encoding='utf-8'))
    return {p['raw_city']: city for city,p in profiles.items()}


def canonical_city(value):
    return city_mapping().get(value,value)


def load_data(task):
    spec = SPECS[task]
    df = pd.read_csv(RAW/spec['file'])
    original_rows = len(df)
    df.insert(0,'source_row',np.arange(len(df)))
    if task == 'house':
        for c in HOUSE_NUM+['price']: df[c] = pd.to_numeric(df[c],errors='coerce')
        df = df[df.price.between(10000,10000000) & (df.area.isna() | df.area.between(10,500))].copy()
        df['city_name'] = df.city_name.map(canonical_city)
        features = HOUSE_NUM+HOUSE_CAT
    else:
        features = [c for c in df if c not in ['source_row',spec['target']]]
        if task == 'ecommerce':
            for col in NONNEGATIVE: df.loc[df[col]<0,col] = np.nan
    df = df.dropna(subset=[spec['target']]).reset_index(drop=True)
    x = df[features].copy()
    y = df[spec['target']].astype(float if task=='house' else int).to_numpy()
    groups = pd.util.hash_pandas_object(x,index=False).to_numpy()
    profile = dict(original_rows=original_rows,clean_rows=len(df),features=features,
                   missing={c:int(n) for c,n in x.isna().sum().items()},
                   duplicate_feature_rows=int(x.duplicated().sum()),
                   target_counts={str(k):int(v) for k,v in pd.Series(y).value_counts().items()} if task!='house' else None,
                   target_min=float(y.min()),target_max=float(y.max()))
    return x,y,groups,df.source_row.to_numpy(),profile


def split_groups(groups, seed):
    train,temp = next(GroupShuffleSplit(n_splits=1,train_size=.70,random_state=seed).split(groups,groups=groups))
    a,b = next(GroupShuffleSplit(n_splits=1,train_size=.50,random_state=seed).split(temp,groups=groups[temp]))
    parts = dict(train=train,validation=temp[a],test=temp[b])
    sets = {k:set(groups[v]) for k,v in parts.items()}
    assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
    return parts


def preprocessor(x):
    nums = x.select_dtypes(include='number').columns.tolist()
    cats = [c for c in x if c not in nums]
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats),
    ],sparse_threshold=0)


def metric(y,p,score=None,regression=False):
    if regression:
        return dict(MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),R2=float(r2_score(y,p)))
    d = dict(Accuracy=float(accuracy_score(y,p)),Precision=float(precision_score(y,p,zero_division=0)),
             Recall=float(recall_score(y,p,zero_division=0)),F1=float(f1_score(y,p,zero_division=0)),
             Macro_F1=float(f1_score(y,p,average='macro',zero_division=0)))
    if score is not None:
        d.update(ROC_AUC=float(roc_auc_score(y,score)),Average_Precision=float(average_precision_score(y,score)))
    return d


def train_mlp(xtrain,ytrain,xval,yval,regression,cfg):
    torch.set_num_threads(4)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    target = StandardScaler().fit(ytrain.reshape(-1,1)) if regression else None
    yt = target.transform(ytrain.reshape(-1,1)).astype('float32') if regression else ytrain.astype('int64')
    xv = torch.from_numpy(xval)
    tx = torch.from_numpy(xtrain);ty = torch.from_numpy(yt)
    model = TabularMLP(xtrain.shape[1],1 if regression else 2)
    weights = len(ytrain)/(2*np.bincount(ytrain.astype(int))) if not regression else None
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    optim = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    best = float('inf') if regression else -1.;state=None;history=[];stale=0
    rng = np.random.default_rng(cfg['seed'])
    for epoch in range(cfg['epochs']):
        model.train();total=0.;seen=0
        for indexes in np.array_split(rng.permutation(len(tx)),max(1,int(np.ceil(len(tx)/cfg['batch_size'])))):
            optim.zero_grad();loss=loss_fn(model(tx[indexes]),ty[indexes]);loss.backward();optim.step()
            total += float(loss.detach())*len(indexes);seen += len(indexes)
        model.eval()
        with torch.inference_mode(): out=model(xv).numpy()
        pred = target.inverse_transform(out).ravel() if regression else out.argmax(1)
        score = float(np.sqrt(mean_squared_error(yval,pred))) if regression else float(f1_score(yval,pred,zero_division=0))
        history.append(dict(epoch=epoch+1,train_loss=total/seen,validation_score=score))
        improved = score<best if regression else score>best
        if improved:best=score;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=epoch+1
        else:stale+=1
        print(f'  MLP epoch={epoch+1} val={score:.6f}',flush=True)
        if stale>=cfg['patience']:break
    model.load_state_dict(state);model.eval()
    return model,target,history,best_epoch,weights


def mlp_predictions(model,target,x,regression):
    with torch.inference_mode(): out=model(torch.from_numpy(x))
    if regression:return target.inverse_transform(out.numpy()).ravel(),None
    return out.argmax(1).numpy(),torch.softmax(out,dim=1)[:,1].numpy()


def draw_figures(folder,y,rows,predictions,history,regression):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=folder/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    if regression:axes[0].hist(y,bins=45);axes[0].set_xlabel('Target (PLN)')
    else:
        labels,counts=np.unique(y,return_counts=True);axes[0].bar(labels.astype(str),counts);axes[0].set_xlabel('Class')
    axes[0].set_title('Full cleaned dataset');axes[0].set_ylabel('Rows')
    axes[1].plot([r['epoch'] for r in history],[r['validation_score'] for r in history]);axes[1].set_xlabel('Epoch')
    axes[1].set_title('MLP validation RMSE' if regression else 'MLP validation F1')
    fig.tight_layout();fig.savefig(figures/'data_and_learning.png',dpi=150);plt.close(fig)
    for name,(actual,pred,score) in predictions.items():
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        if regression:
            axes[0].scatter(actual,pred,s=3,alpha=.2);lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max());axes[0].plot([lo,hi],[lo,hi],'k--')
            axes[0].set(xlabel='Actual PLN',ylabel='Predicted PLN',title='Test predictions')
            axes[1].hist(pred-actual,bins=50);axes[1].set(xlabel='Residual PLN',title='Test error')
        else:
            cm=confusion_matrix(actual,pred,labels=[0,1]);axes[0].imshow(cm,cmap='Blues')
            for (i,j),v in np.ndenumerate(cm):axes[0].text(j,i,str(v),ha='center',va='center')
            axes[0].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title='Test confusion matrix')
            if score is not None:
                precision,recall,_=precision_recall_curve(actual,score);axes[1].plot(recall,precision)
                axes[1].set(xlabel='Recall',ylabel='Precision',title='Test PR curve')
        fig.suptitle(name);fig.tight_layout();fig.savefig(figures/f'{name}_test.png',dpi=150);plt.close(fig)


def train(task):
    cfg={**PROTOCOL,'task':task};spec=SPECS[task];regression=spec['kind']=='regression'
    data_hash=sha(RAW/spec['file']);code_hash=sha(Path(__file__))
    signature=__import__('hashlib').sha256(json.dumps([cfg,data_hash,code_hash],sort_keys=True).encode()).hexdigest()[:12]
    folder=ROOT/'runs/tabular'/task/signature
    registry=ROOT/f'assets/models/submission/{task}/registry.json'
    if (folder/'complete.json').is_file():
        check_run(folder);print(f'{task}: verified cached run {signature}',flush=True)
        write_json(registry,dict(run=folder.relative_to(ROOT).as_posix(),protocol=1));return
    if folder.exists():raise RuntimeError(f'Incomplete run exists at {folder}; inspect it before retrying.')
    folder.mkdir(parents=True)
    started=time.perf_counter();print(f'Training {task} run={signature}',flush=True)
    x,y,groups,source_rows,profile=load_data(task);split=split_groups(groups,cfg['seed'])
    write_json(folder/'config.json',cfg);write_json(folder/'dataset.json',dict(file=f'data/raw/{spec["file"]}',sha256=data_hash,**profile))
    pd.DataFrame(dict(source_row=source_rows,group_hash=groups,split=np.select([np.isin(np.arange(len(x)),split[k]) for k in ['train','validation']],['train','validation'],default='test'))).to_csv(folder/'split.csv',index=False)
    write_json(folder/'split.json',dict(method=cfg['split'],seed=cfg['seed'],group_overlap=0,
        sizes={k:len(v) for k,v in split.items()},target_means={k:float(y[v].mean()) for k,v in split.items()},
        note='Approximate fractions by groups; identical feature rows stay together, even if labels differ.'))
    pre=preprocessor(x.iloc[split['train']]);pre.fit(x.iloc[split['train']])
    xx={k:pre.transform(x.iloc[indexes]).astype('float32') for k,indexes in split.items()}
    yy={k:y[v] for k,v in split.items()}
    joblib.dump(pre,folder/'preprocessor.joblib')
    candidates = ({'ridge_1':Ridge(alpha=1), 'ridge_10':Ridge(alpha=10),
        'random_forest':RandomForestRegressor(n_estimators=100,max_depth=20,min_samples_leaf=2,random_state=42,n_jobs=4)}
        if regression else {f'logistic_C{c}':LogisticRegression(C=c,class_weight='balanced',max_iter=1000,random_state=42) for c in [.1,1.,10.]})
    fitted={};validation=[]
    for name,model in candidates.items():
        print('  fitting',name,flush=True);model.fit(xx['train'],yy['train']);pred=model.predict(xx['validation'])
        score=model.predict_proba(xx['validation'])[:,1] if not regression else None
        validation.append(dict(model=name,**metric(yy['validation'],pred,score,regression)));fitted[name]=model
    selection='RMSE' if regression else 'F1'
    chosen=sorted(validation,key=lambda r:r[selection],reverse=not regression)[0]['model']
    model,target,history,best_epoch,weights=train_mlp(xx['train'],yy['train'],xx['validation'],yy['validation'],regression,cfg)
    p,s=mlp_predictions(model,target,xx['validation'],regression)
    validation.append(dict(model='mlp',**metric(yy['validation'],p,s,regression)))
    pd.DataFrame(validation).to_csv(folder/'validation.csv',index=False)
    # Freeze all choices before evaluating TEST. ML/MLP use the same training rows.
    write_json(folder/'selection.json',dict(metric=selection,ml=chosen,mlp_epoch=best_epoch,
        mlp_class_weights=None if weights is None else weights.tolist(),test_used_for_selection=False))
    joblib.dump(fitted[chosen],folder/'ml.joblib')
    torch.save(model.state_dict(),folder/'mlp.pth')
    if target is not None:joblib.dump(target,folder/'target_scaler.joblib')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    architecture=dict(input_dim=xx['train'].shape[1],output_dim=1 if regression else 2,hidden=cfg['mlp_hidden'],
        learned_layers=5,parameters=sum(p.numel() for p in model.parameters()),features=x.columns.tolist())
    write_json(folder/'architecture.json',architecture)
    baseline=DummyRegressor(strategy='median') if regression else DummyClassifier(strategy='prior')
    baseline.fit(xx['train'],yy['train'])
    predictions={};rows=[]
    for name,engine in [('baseline',baseline),('ml',fitted[chosen]),('mlp',None)]:
        if name=='mlp':p,s=mlp_predictions(model,target,xx['test'],regression)
        else:p=engine.predict(xx['test']);s=engine.predict_proba(xx['test'])[:,1] if not regression else None
        m=metric(yy['test'],p,s,regression);rows.append(dict(model=name,**m))
        records=dict(source_row=source_rows[split['test']],actual=yy['test'],prediction=p)
        if s is not None:records['score']=s
        pd.DataFrame(records).to_csv(folder/f'{name}_predictions.csv',index=False)
        predictions[name]=(yy['test'],p,s)
    pd.DataFrame(rows).to_csv(folder/'test_metrics.csv',index=False)
    write_json(folder/'metrics.json',{r['model']:{k:v for k,v in r.items() if k!='model'} for r in rows})
    draw_figures(folder,y,rows,predictions,history,regression)
    # Save explicit hard examples; they remain held-out examples, not training data.
    errors=pd.read_csv(folder/'mlp_predictions.csv')
    errors['error']=abs(errors.prediction-errors.actual)
    errors.sort_values('error',ascending=False).head(30).to_csv(folder/'error_examples.csv',index=False)
    finish_run(folder,dict(task=task,kind=spec['kind'],code_sha256=code_hash,dataset_sha256=data_hash,
                          elapsed_seconds=round(time.perf_counter()-started,2),selection=selection,ml_model=chosen))
    write_json(registry,dict(run=folder.relative_to(ROOT).as_posix(),protocol=1))
    print(f'{task}: complete {folder}',flush=True)


def evaluate(run):
    folder=(ROOT/run).resolve() if not Path(run).is_absolute() else Path(run)
    meta=check_run(folder);task=meta['task'];regression=meta['kind']=='regression'
    x,y,groups,source_rows,_=load_data(task)
    saved=pd.read_csv(folder/'split.csv',dtype={'group_hash':'uint64'})
    assert saved.source_row.tolist()==source_rows.tolist()
    assert np.array_equal(saved.group_hash.to_numpy(),groups)
    group_sets={k:set(saved.loc[saved.split==k,'group_hash']) for k in ['train','validation','test']}
    assert all(not(group_sets[a]&group_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
    indexes=np.flatnonzero(saved.split.to_numpy()=='test');pre=joblib.load(folder/'preprocessor.joblib')
    xt=pre.transform(x.iloc[indexes]).astype('float32');architecture=json.loads((folder/'architecture.json').read_text())
    torch.set_num_threads(4);model=TabularMLP(architecture['input_dim'],architecture['output_dim'])
    model.load_state_dict(torch.load(folder/'mlp.pth',map_location='cpu',weights_only=True));model.eval()
    target=joblib.load(folder/'target_scaler.joblib') if regression else None
    check=[]
    for name in ['ml','mlp']:
        if name=='ml':
            engine=joblib.load(folder/'ml.joblib');p=engine.predict(xt);s=None if regression else engine.predict_proba(xt)[:,1]
        else:p,s=mlp_predictions(model,target,xt,regression)
        expected=pd.read_csv(folder/f'{name}_predictions.csv')
        np.testing.assert_allclose(p,expected.prediction,atol=.05 if regression else 1e-6,rtol=1e-6)
        if s is not None:np.testing.assert_allclose(s,expected.score,atol=1e-6)
        check.append(dict(model=name,metrics=metric(y[indexes],p,s,regression),rows=len(p)))
    write_json(folder/'verification.json',dict(passed=True,group_overlap=0,reloaded_checkpoint=True,checks=check))
    print(json.dumps(dict(run=str(folder),passed=True,checks=check),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train');p.add_argument('--task',choices=[*SPECS,'all'],default='all')
    p=sub.add_parser('evaluate');p.add_argument('--run',required=True)
    args=parser.parse_args()
    if args.command=='train':
        for task in SPECS if args.task=='all' else [args.task]:train(task)
    else:evaluate(args.run)


if __name__=='__main__':main()
