from functools import lru_cache
from io import BytesIO
import base64
import math
import threading
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import DATA, MODELS, ASSETS, TASKS, CNN, profiles, read_json
from .artifacts import artifact

torch.set_num_threads(4)
LOCK=threading.RLock()

class InputError(ValueError):
    pass

def checked_probabilities(values):
    p=np.asarray(values,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or (p<0).any() or (p>1.00001).any() or abs(p.sum()-1)>.001:raise InputError('Đầu ra model không phải phân bố xác suất hợp lệ. Kiểm tra checkpoint.')
    return p

def validate(task, payload):
    if not isinstance(task,str) or task not in TASKS:raise InputError('Bài toán không hợp lệ.')
    if not isinstance(payload,dict):raise InputError('Đầu vào cần là một hồ sơ.')
    cleaned={}
    for f in TASKS[task]['fields']:
        name=f['name'];v=payload.get(name)
        if v is None or v=='':raise InputError(f"Bạn chưa nhập: {f['label']}.")
        if f['kind']=='select':
            values=[o['value'] for o in f['options']]
            if isinstance(values[0],int):
                if isinstance(v,bool):raise InputError(f"{f['label']}: lựa chọn không hợp lệ.")
                try:v=float(v)
                except (ValueError,TypeError):raise InputError(f"{f['label']}: lựa chọn không hợp lệ.")
                if not math.isfinite(v) or not v.is_integer():raise InputError(f"{f['label']}: lựa chọn không hợp lệ.")
                v=int(v)
            if v not in values:raise InputError(f"{f['label']}: lựa chọn không hợp lệ.")
        else:
            if isinstance(v,bool):raise InputError(f"{f['label']}: cần nhập số.")
            try:v=float(v)
            except (ValueError,TypeError):raise InputError(f"{f['label']}: cần nhập số.")
            if not math.isfinite(v) or not f['min']<=v<=f['max']:raise InputError(f"{f['label']}: nhập từ {f['min']} đến {f['max']}.")
            if f['step']==1 and not v.is_integer():raise InputError(f"{f['label']}: cần nhập số nguyên.")
        cleaned[name]=v
    return cleaned

class DeepMLP(nn.Module):
    def __init__(self,d,out=2):
        super().__init__()
        self.fc1=nn.Linear(d,64);self.fc2=nn.Linear(64,32);self.fc3=nn.Linear(32,16);self.fc4=nn.Linear(16,8);self.fc5=nn.Linear(8,out);self.relu=nn.ReLU()
    def forward(self,x):
        for layer in [self.fc1,self.fc2,self.fc3,self.fc4]:x=self.relu(layer(x))
        return self.fc5(x)

class MnistBasic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.ReLU(),nn.Conv2d(4,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(8*14*14,10))
    def forward(self,x):return self.net(x)

class MnistImproved(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Dropout(.25),nn.Linear(32*7*7,10))
    def forward(self,x):return self.net(x)

class CustomerRNN(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(3,8);self.rnn=nn.RNN(8,32,batch_first=True);self.fc=nn.Linear(32,3)
    def forward(self,x):
        _,h=self.rnn(self.embedding(x));return self.fc(h[-1])

class StockRNN(nn.Module):
    def __init__(self,hidden_size):
        super().__init__();self.rnn=nn.RNN(1,hidden_size,batch_first=True);self.fc=nn.Linear(hidden_size,1)
    def forward(self,x):
        _,h=self.rnn(x);return self.fc(h[-1]).squeeze(-1)

def torch_weights(path):return torch.load(path,map_location='cpu',weights_only=True)

@lru_cache(maxsize=12)
def pipeline(task):
    if task.endswith('_a01'):return joblib.load(MODELS/'a01'/('diabetes_model.pkl' if task=='diabetes_a01' else 'house_price_model.pkl'))
    filename={'diabetes':'diabetes_pipeline.joblib','house':'house_price_pipeline.joblib','ecommerce':'ecommerce_purchase_pipeline.joblib'}[task]
    return joblib.load(MODELS/'a02'/filename)['pipeline']

@lru_cache(maxsize=3)
def mlp(task):
    folder=MODELS/'a03';info=joblib.load(folder/f'{task}_model_info.joblib')
    model=DeepMLP(info['input_dim'],1 if task=='house' else 2);model.load_state_dict(torch_weights(folder/f'{task}_deep_mlp_5layers.pth'));model.eval()
    pre=joblib.load(folder/('diabetes_scaler.joblib' if task=='diabetes' else f'{task}_preprocessor.joblib'))
    target=joblib.load(folder/'house_target_scaler.joblib') if task=='house' else None
    return model,pre,target,info

def model_frame(task,values,engine):
    row=values.copy()
    if task=='house':
        profile=profiles[row['city_name']]
        row.update({k:profile[k] for k in ['population','longitude','latitude','voivodeship']})
        if engine=='mlp':row['city_name']=profile['raw_city']
    if engine=='mlp':
        info=mlp(task)[3];names=info.get('features',info.get('raw_features',info.get('numerical_features',[])+info.get('categorical_features',[])))
    else:
        names=list(pipeline(task).feature_names_in_)
    return pd.DataFrame([{k:row[k] for k in names}])

def predict_tabular(task,engine,payload):
    values=validate(task,payload)
    if engine not in TASKS[task]['engines']:raise InputError('Mô hình không hỗ trợ bài toán này.')
    engines=['ml','mlp'] if engine=='compare' else [engine];rows=[]
    metrics=read_json(DATA/'experiment_metrics.json')
    for selected in engines:
        if task in ['diabetes','house','ecommerce']:
            from .submission import predict
            updated=predict(task,selected,values)
            if updated is not None:
                rows.append(updated);continue
        frame=model_frame(task,values,selected)
        if selected=='ml':
            model=pipeline(task);pred=float(model.predict(frame)[0]);score=float(model.predict_proba(frame)[0,list(model.classes_).index(1)]) if hasattr(model,'predict_proba') else None
            name=metrics['ml'][task]['Model']
        else:
            model,pre,target,info=mlp(task);x=pre.transform(frame)
            if hasattr(x,'toarray'):x=x.toarray()
            with torch.inference_mode():logits=model(torch.tensor(np.asarray(x,dtype=np.float32)))
            if task=='house':pred=float(target.inverse_transform(logits.numpy()).ravel()[0]);score=None
            else:score=float(torch.softmax(logits,dim=1)[0,1]);pred=int(logits.argmax(1)[0])
            name='MLP · 4 tầng ẩn + đầu ra'
        if not math.isfinite(pred) or (score is not None and (not math.isfinite(score) or not 0<=score<=1)):raise InputError('Model trả giá trị không hợp lệ. Kiểm tra checkpoint và dữ liệu đầu vào.')
        regression=TASKS[task]['kind']=='regression'
        label=(f'{pred:,.2f} {TASKS[task]["unit"]}' if regression else ('Mua hàng' if pred else 'Không mua hàng') if task=='ecommerce' else ('Lớp tiểu đường (1)' if pred else 'Lớp không tiểu đường (0)'))
        if selected=='ml':
            model_path=MODELS/'a01'/('diabetes_model.pkl' if task=='diabetes_a01' else 'house_price_model.pkl') if task.endswith('_a01') else MODELS/'a02'/dict(diabetes='diabetes_pipeline.joblib',house='house_price_pipeline.joblib',ecommerce='ecommerce_purchase_pipeline.joblib')[task]
            files=[model_path]
        else:
            files=[MODELS/f'a03/{task}_deep_mlp_5layers.pth',MODELS/'a03'/('diabetes_scaler.joblib' if task=='diabetes' else f'{task}_preprocessor.joblib')]
            if task=='house':files.append(MODELS/'a03/house_target_scaler.joblib')
        rows.append(dict(engine=selected,model=name,prediction=pred,label=label,score=score,metrics=metrics[selected][task],artifacts=[artifact(p) for p in files]))
    return dict(task=task,input=values,results=rows,note=TASKS[task]['note'])

@lru_cache(maxsize=8)
def keras_model(path):
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
    from tensorflow import keras
    return keras.models.load_model(path,compile=False,safe_mode=True)

@lru_cache(maxsize=2)
def mnist_torch(name):
    model=MnistBasic() if name=='torch_basic' else MnistImproved()
    model.load_state_dict(torch_weights(MODELS/f'a04/mnist_pytorch_{"basic" if name=="torch_basic" else "improved"}_fixed.pth'));model.eval();return model

def scratch_mnist(x):
    from numpy.lib.stride_tricks import sliding_window_view
    with np.load(MODELS/'a04/mnist_scratch_fixed.npz',allow_pickle=False) as m:
        for n in [1,2]:
            w=sliding_window_view(np.pad(x,((0,0),(0,0),(1,1),(1,1))),(3,3),axis=(2,3))
            x=np.maximum(np.einsum('nchwkl,fckl->nfhw',w,m[f'conv{n}_W'])+m[f'conv{n}_b'][None,:,None,None],0)
        n,c,h,w=x.shape;x=x.reshape(n,c,h//2,2,w//2,2).max(axis=(3,5)).reshape(n,-1)
        return x@m['fc_W']+m['fc_b']

def image_data_url(image):
    out=BytesIO();image.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode('ascii')

def prepare_image(image,dataset,drawing=False):
    image=ImageOps.exif_transpose(image)
    if image.width*image.height>16000000:raise InputError('Ảnh quá lớn. Chọn ảnh dưới 16 megapixel.')
    if dataset=='cifar10':return image.convert('RGB').resize((32,32),Image.Resampling.LANCZOS)
    image=image.convert('L')
    if dataset=='fashion':return image.resize((28,28),Image.Resampling.LANCZOS)
    if drawing:
        a=np.asarray(image);corners=[int(a[0,0]),int(a[0,-1]),int(a[-1,0]),int(a[-1,-1])]
        if np.mean(corners)>128:image=ImageOps.invert(image)
        a=np.asarray(image);coords=np.argwhere(a>35)
        if len(coords)<5:raise InputError('Bạn hãy vẽ một chữ số trước khi nhận diện.')
        top,left=coords.min(0);bottom,right=coords.max(0)+1
        crop=image.crop((int(left),int(top),int(right),int(bottom)));crop.thumbnail((20,20),Image.Resampling.LANCZOS)
        image=Image.new('L',(28,28),0);image.paste(crop,((28-crop.width)//2,(28-crop.height)//2))
        a=np.asarray(image,dtype=float);ys,xs=np.indices(a.shape);mass=a.sum()
        dx=int(round(13.5-(xs*a).sum()/mass));dy=int(round(13.5-(ys*a).sum()/mass))
        shifted=Image.new('L',(28,28),0);shifted.paste(image,(dx,dy));image=shifted
    else:image=image.resize((28,28),Image.Resampling.LANCZOS)
    return image

def predict_image(dataset,name,sample_id=None,encoded=None,drawing=False):
    if not isinstance(dataset,str) or dataset not in CNN or not isinstance(name,str) or name not in dict(CNN[dataset]['models']):raise InputError('Dataset hoặc mô hình CNN không hợp lệ.')
    actual=None
    if sample_id:
        entry=next((s for s in read_json(DATA/'image_gallery.json') if s['id']==sample_id and s['dataset']==dataset),None)
        if not entry:raise InputError('Ảnh mẫu không tồn tại.')
        image=Image.open(ASSETS/'samples'/entry['path']);actual=entry['label']
    else:
        try:
            if not isinstance(encoded,str) or not encoded.startswith('data:image/'):raise ValueError()
            raw=base64.b64decode(encoded.split(',',1)[1],validate=True)
            if len(raw)>5_000_000:raise InputError('Ảnh cần nhỏ hơn 5 MB.')
            image=Image.open(BytesIO(raw));image.load()
        except (ValueError,UnidentifiedImageError,OSError,Image.DecompressionBombError):raise InputError('Không đọc được ảnh. Hãy chọn PNG hoặc JPEG.')
    processed=prepare_image(image,dataset,drawing)
    x=np.asarray(processed,dtype=np.float32)/255.
    with LOCK:
        if dataset=='mnist' and name.startswith('torch'):
            with torch.inference_mode():p=torch.softmax(mnist_torch(name)(torch.tensor(x[None,None])),dim=1).numpy()[0]
        elif dataset=='mnist' and name=='numpy_scratch':
            z=scratch_mnist(x[None,None])[0];p=np.exp(z-z.max());p/=p.sum()
        else:
            if dataset=='mnist':path=MODELS/f'a04/mnist_tensorflow_{"basic" if name=="keras_basic" else "improved"}_fixed.keras'
            else:path=MODELS/f'a05/{"fashion_mnist" if dataset=="fashion" else "cifar10"}_{name}.keras'
            if x.ndim==2:x=x[...,None]
            p=keras_model(str(path))(x[None],training=False).numpy()[0]
    p=checked_probabilities(p);predicted=int(np.argmax(p));classes=CNN[dataset]['classes']
    if dataset=='mnist':
        if name.startswith('torch'):model_path=MODELS/f'a04/mnist_pytorch_{"basic" if name=="torch_basic" else "improved"}_fixed.pth'
        elif name=='numpy_scratch':model_path=MODELS/'a04/mnist_scratch_fixed.npz'
        else:model_path=MODELS/f'a04/mnist_tensorflow_{"basic" if name=="keras_basic" else "improved"}_fixed.keras'
    else:model_path=MODELS/f'a05/{"fashion_mnist" if dataset=="fashion" else "cifar10"}_{name}.keras'
    return dict(dataset=dataset,model=dict(CNN[dataset]['models'])[name],artifact=artifact(model_path),prediction=predicted,label=classes[predicted],score=float(p[predicted]),probabilities=[dict(label=label,value=float(p[i])) for i,label in enumerate(classes)],actual=classes[actual] if actual is not None else None,correct=(predicted==actual) if actual is not None else None,processed=image_data_url(processed))

@lru_cache(maxsize=2)
def rnn_torch(task):
    prefix='AMZN' if task=='stock' else 'Customer';meta=read_json(MODELS/f'a06/{prefix}_PyTorch_SimpleRNN.json')
    model=StockRNN(meta['hidden_size']) if task=='stock' else CustomerRNN();model.load_state_dict(torch_weights(MODELS/f'a06/{prefix}_PyTorch_SimpleRNN.pth')['model_state_dict']);model.eval();return model,meta

def predict_customer(events,framework='pytorch',example=None,architecture='rnn'):
    labels=['view','addtocart','transaction']
    if architecture not in ['rnn','lstm','gru']:raise InputError('Kiến trúc không hợp lệ.')
    if framework not in ['pytorch','keras'] or (architecture!='rnn' and framework!='keras'):raise InputError('LSTM/GRU dùng checkpoint Keras; SimpleRNN hỗ trợ cả hai framework.')
    if not isinstance(events,list) or len(events)!=5 or any(a not in labels for a in events):raise InputError('Cần đúng 5 hành vi: view, addtocart hoặc transaction.')
    x=np.asarray([[labels.index(e) for e in events]],dtype=np.int64)
    with LOCK:
        if framework=='pytorch':
            with torch.inference_mode():p=torch.softmax(rnn_torch('customer')[0](torch.tensor(x)),dim=1).numpy()[0]
        else:p=keras_model(str(MODELS/f'a06/Customer_Keras_{"SimpleRNN" if architecture=="rnn" else architecture.upper()}.keras'))(x,training=False).numpy()[0]
    p=checked_probabilities(p)
    examples=read_json(DATA/'customer_examples.json');actual=None
    if isinstance(example,int) and not isinstance(example,bool) and 0<=example<len(examples) and events==examples[example]['events']:actual=examples[example]['actual']
    idx=int(p.argmax())
    arch='SimpleRNN' if architecture=='rnn' else architecture.upper();model_path=MODELS/f'a06/Customer_{"PyTorch" if framework=="pytorch" else "Keras"}_{arch}.{ "pth" if framework=="pytorch" else "keras"}'
    return dict(framework=framework,architecture=architecture,artifact=artifact(model_path),label=labels[idx],score=float(p[idx]),events=events,actual=actual,probabilities=[dict(label=l,value=float(p[i])) for i,l in enumerate(labels)],note='Mô hình dự đoán một hành vi tiếp theo từ lịch sử 5 hành vi. Điểm lớp chưa được hiệu chỉnh xác suất.')

@lru_cache(maxsize=1)
def stock_data():
    d=pd.read_csv(DATA/'AMZN.csv');d['Date']=pd.to_datetime(d.Date);return d[d.Date>='2015-01-01'].sort_values('Date').drop_duplicates('Date').reset_index(drop=True)

def predict_stock(date=None,framework='pytorch',architecture='rnn'):
    if architecture not in ['rnn','lstm','gru']:raise InputError('Kiến trúc không hợp lệ.')
    if framework not in ['pytorch','keras'] or (architecture!='rnn' and framework!='keras'):raise InputError('LSTM/GRU dùng checkpoint Keras; SimpleRNN hỗ trợ cả hai framework.')
    d=stock_data();start=int(len(d)*.85)
    if date is None:i=len(d)-1
    else:
        matches=np.flatnonzero(d.Date.dt.strftime('%Y-%m-%d').to_numpy()==date)
        if not len(matches) or matches[0]<start:raise InputError('Chọn một phiên trong tập test hiển thị trên giao diện.')
        i=int(matches[0])
    prefix='PyTorch' if framework=='pytorch' else 'Keras';arch='SimpleRNN' if architecture=='rnn' else architecture.upper();meta=read_json(MODELS/f'a06/AMZN_{prefix}_{arch}.json')
    raw=d.Close.iloc[i-30:i].to_numpy();sc=meta['scaler'];x=(raw*sc['scale'][0]+sc['min'][0]).astype(np.float32)[None,:,None]
    with LOCK:
        if framework=='pytorch':
            with torch.inference_mode():z=float(rnn_torch('stock')[0](torch.tensor(x))[0])
        else:z=float(keras_model(str(MODELS/f'a06/AMZN_Keras_{arch}.keras'))(x,training=False).numpy().ravel()[0])
    pred=(z-sc['min'][0])/sc['scale'][0]
    if not math.isfinite(pred):raise InputError('Model trả giá trị không hữu hạn. Kiểm tra checkpoint.')
    hist=d.iloc[max(0,i-59):i+1]
    model_path=MODELS/f'a06/AMZN_{prefix}_{arch}.{ "pth" if framework=="pytorch" else "keras"}'
    return dict(date=str(d.Date.iloc[i].date()),predicted=pred,actual=float(d.Close.iloc[i]),baseline=float(raw[-1]),absolute_error=abs(pred-float(d.Close.iloc[i])),history=[dict(date=str(r.Date.date()),value=float(r.Close)) for _,r in hist.iterrows()],framework=framework,architecture=architecture,artifact=artifact(model_path),note='Dự đoán một phiên lịch sử từ 30 phiên có sẵn trước đó. Đây không phải giá hiện tại hay dự báo nhiều bước.')
