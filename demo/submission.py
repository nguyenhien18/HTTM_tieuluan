"""Frozen submission models, selected offline using validation only."""
from functools import lru_cache
import json
import math
import joblib
import numpy as np
import torch
import pandas as pd
from .config import ROOT, profiles
from .artifacts import artifact


@lru_cache(maxsize=3)
def load(task):
    registry=ROOT/f'assets/models/submission/{task}/registry.json'
    if not registry.exists():return None
    record=json.loads(registry.read_text(encoding='utf-8'));folder=(ROOT/record['run']).resolve()
    folder.relative_to(ROOT/'runs/tabular')
    if not (folder/'complete.json').is_file():raise ValueError('Submission model run is incomplete')
    architecture=json.loads((folder/'architecture.json').read_text(encoding='utf-8'))
    from experiments.tabular import TabularMLP
    model=TabularMLP(architecture['input_dim'],architecture['output_dim'])
    model.load_state_dict(torch.load(folder/'mlp.pth',map_location='cpu',weights_only=True));model.eval()
    return dict(folder=folder,architecture=architecture,ml=joblib.load(folder/'ml.joblib'),mlp=model,
                pre=joblib.load(folder/'preprocessor.joblib'),
                target=joblib.load(folder/'target_scaler.joblib') if task=='house' else None,
                metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8')),
                selection=json.loads((folder/'selection.json').read_text(encoding='utf-8')),
                split=json.loads((folder/'split.json').read_text(encoding='utf-8')))


def predict(task,engine,values):
    bundle=load(task)
    if bundle is None:return None
    row=values.copy()
    if task=='house':row.update({k:profiles[row['city_name']][k] for k in ['population','longitude','latitude','voivodeship']})
    frame=pd.DataFrame([{k:row[k] for k in bundle['architecture']['features']}])
    x=bundle['pre'].transform(frame).astype('float32')
    files=[bundle['folder']/'preprocessor.joblib'];regression=task=='house'
    if engine=='ml':
        pred=float(bundle['ml'].predict(x)[0]);score=None if regression else float(bundle['ml'].predict_proba(x)[0,1])
        files.append(bundle['folder']/'ml.joblib');name=bundle['selection']['ml']
    else:
        with torch.inference_mode():out=bundle['mlp'](torch.from_numpy(x))
        if regression:
            pred=float(bundle['target'].inverse_transform(out.numpy()).ravel()[0]);score=None
            files.append(bundle['folder']/'target_scaler.joblib')
        else:pred=int(out.argmax(1)[0]);score=float(torch.softmax(out,dim=1)[0,1])
        files.append(bundle['folder']/'mlp.pth');name='MLP · 4 tầng ẩn + đầu ra'
    from .models import InputError
    if not math.isfinite(pred) or (regression and pred<=0) or (score is not None and (not math.isfinite(score) or not 0<=score<=1)):
        raise InputError('Mô hình trả kết quả ngoài miền hợp lệ cho hồ sơ này. Kiểm tra đầu vào.')
    label=(f'{pred:,.2f} PLN' if regression else ('Mua hàng' if pred else 'Không mua hàng') if task=='ecommerce' else ('Lớp tiểu đường (1)' if pred else 'Lớp không tiểu đường (0)'))
    return dict(engine=engine,model=name,prediction=pred,label=label,score=score,metrics=bundle['metrics'][engine],
                artifacts=[artifact(p) for p in files],experiment_run=bundle['folder'].relative_to(ROOT).as_posix(),
                protocol='Cùng split nhóm đặc trưng cho ML và MLP; fit trên TRAIN; chọn bằng VALIDATION; TEST không dùng chọn model.',
                split=bundle['split'])


def current_metrics(original):
    result={k:dict(v) if isinstance(v,dict) else v for k,v in original.items()}
    for task in ['diabetes','house','ecommerce']:
        bundle=load(task)
        if bundle:
            for engine in ['ml','mlp']:
                result[engine][task]=dict(Model=bundle['selection']['ml'] if engine=='ml' else 'MLP · 4 tầng ẩn + đầu ra',**bundle['metrics'][engine])
    return result
