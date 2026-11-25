"""Evaluate the validation-selected checkpoint of a new image training run.

python -m experiments.trained_images --run runs/cnn_training/fashion/basic_cnn/<run>
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score
from .common import ROOT,check_run,write_json,finish_run,sha
from .images import DATA,figures


def evaluate(run):
    folder=(ROOT/run).resolve() if not Path(run).is_absolute() else Path(run)
    meta=check_run(folder);cfg=json.loads((folder/'config.json').read_text());task=cfg['dataset'];name=cfg['model']
    data=np.load(DATA/f'{task}.npz',allow_pickle=False);x=data['x_test'];y=data['y_test'];out=[]
    if name.startswith('torch'):
        from demo.models import MnistBasic,MnistImproved
        model=MnistBasic() if name=='torch_basic' else MnistImproved()
        model.load_state_dict(torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=True));model.eval();torch.set_num_threads(4)
    elif name=='numpy_scratch':
        from .manual_cnn import ManualMNISTCNN
        model=ManualMNISTCNN();weights=np.load(folder/'checkpoint.npz',allow_pickle=False)
        for key,layer in [('conv1',model.conv1),('conv2',model.conv2),('fc',model.fc)]:layer.W=weights[key+'_W'];layer.b=weights[key+'_b']
    else:
        from tensorflow import keras
        model=keras.models.load_model(folder/'checkpoint.keras',compile=False,safe_mode=True)
    for start in range(0,len(x),128):
        batch=x[start:start+128].astype('float32')/255
        if name.startswith('torch'):
            with torch.inference_mode():p=torch.softmax(model(torch.from_numpy(batch.transpose(0,3,1,2))),dim=1).numpy()
        elif name=='numpy_scratch':
            z=model.forward(batch.transpose(0,3,1,2));p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        else:p=model(batch,training=False).numpy()
        out.append(p)
    p=np.concatenate(out);pred=p.argmax(1);destination=folder/'evaluation';destination.mkdir(exist_ok=True)
    records=pd.DataFrame(dict(test_index=np.arange(len(y)),actual=y,prediction=pred))
    for i in range(10):records[f'prob_{i}']=p[:,i]
    records.to_csv(destination/'predictions.csv',index=False)
    metrics=dict(Accuracy=float(accuracy_score(y,pred)),Macro_F1=float(f1_score(y,pred,average='macro')),test_samples=len(y),
                 majority_baseline_accuracy=float(np.mean(y==np.bincount(data['y_train']).argmax())))
    write_json(destination/'metrics.json',metrics);figures(destination,x,y,pred,p)
    finish_run(destination,dict(operation='held-out-test evaluation',training_run=folder.relative_to(ROOT).as_posix(),
        checkpoint_selected_using='validation loss',dataset_sha256=cfg['dataset_sha256']))
    # Preserve the original training completion manifest. Evaluation has its own manifest.
    print(json.dumps(dict(run=str(destination),**metrics),indent=2));return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);args=parser.parse_args();evaluate(args.run)
