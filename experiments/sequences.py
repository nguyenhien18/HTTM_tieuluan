"""Reload all eight sequence checkpoints on their complete original test sets."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,mean_squared_error,mean_absolute_error
from .common import ROOT,sha,write_json,finish_run,check_run


def evaluate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT/'scripts/source_a06'))
    import compare_architectures as source
    from demo.models import keras_model,rnn_torch
    torch.set_num_threads(4)
    summary=[]
    for task,prepare in [('stock',source.stock_inputs),('customer',source.customer_inputs)]:
        data=prepare();xt=data['X_test'];yt=data['y_test'];prefix='AMZN' if task=='stock' else 'Customer'
        for framework,arch in [('Keras','SimpleRNN'),('PyTorch','SimpleRNN'),('Keras','LSTM'),('Keras','GRU')]:
            path=ROOT/f'assets/models/a06/{prefix}_{framework}_{arch}.{ "pth" if framework=="PyTorch" else "keras"}'
            signature=__import__('hashlib').sha256((sha(path)+sha(Path(__file__))+sha(ROOT/'scripts/source_a06/compare_architectures.py')).encode()).hexdigest()[:12]
            folder=ROOT/'runs/sequences'/task/f'{framework}_{arch}'/signature
            if (folder/'complete.json').exists():
                check_run(folder);m=json.loads((folder/'metrics.json').read_text());print('Verified',task,framework,arch,flush=True)
            else:
                folder.mkdir(parents=True,exist_ok=True);predictions=[]
                for start in range(0,len(xt),1024):
                    batch=xt[start:start+1024]
                    if framework=='Keras':out=keras_model(str(path))(batch,training=False).numpy()
                    else:
                        with torch.inference_mode():out=rnn_torch(task)[0](torch.from_numpy(batch)).numpy()
                        if task=='customer':out=torch.softmax(torch.from_numpy(out),dim=1).numpy()
                    predictions.append(out)
                out=np.concatenate(predictions)
                if not np.isfinite(out).all():raise ValueError('Nonfinite sequence output')
                if task=='stock':
                    pred=data['scaler'].inverse_transform(out.reshape(-1,1)).ravel();actual=data['actual'];base=data['previous']
                    m=dict(MAE=float(mean_absolute_error(actual,pred)),RMSE=float(np.sqrt(mean_squared_error(actual,pred))),baseline_RMSE=float(np.sqrt(mean_squared_error(actual,base))))
                    records=pd.DataFrame(dict(date=data['dates'].dt.strftime('%Y-%m-%d'),actual=actual,prediction=pred,baseline=base))
                    fig,axes=plt.subplots(2,1,figsize=(10,6));axes[0].plot(actual,label='Actual');axes[0].plot(pred,label=arch);axes[0].plot(base,label='Persistence',alpha=.6);axes[0].legend()
                    axes[1].plot(pred-actual);axes[1].set(xlabel='Chronological test index',ylabel='Prediction error')
                else:
                    assert np.allclose(out.sum(1),1,atol=1e-4)
                    pred=out.argmax(1);m=dict(Accuracy=float(accuracy_score(yt,pred)),Macro_F1=float(f1_score(yt,pred,average='macro')),
                        majority_accuracy=float(accuracy_score(yt,np.full(len(yt),np.bincount(data['y_train']).argmax()))),
                        last_event_macro_f1=float(f1_score(yt,xt[:,-1],average='macro')))
                    records=pd.DataFrame(dict(target_time=data['test_times'],actual=yt,prediction=pred))
                    for i in range(3):records[f'prob_{i}']=out[:,i]
                    fig,ax=plt.subplots(figsize=(5,4));cm=confusion_matrix(yt,pred,labels=[0,1,2]);ax.imshow(cm,cmap='Blues')
                    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
                    ax.set(xlabel='Predicted',ylabel='Actual',xticks=[0,1,2],yticks=[0,1,2])
                fig.tight_layout();fig.savefig(folder/'test_analysis.png',dpi=150);plt.close(fig)
                records.to_csv(folder/'predictions.csv',index=False);write_json(folder/'metrics.json',m)
                meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                # Compare loaded checkpoints with their stored full-test prediction CSV.
                reference=(ROOT/f'assets/data/a06_results/{prefix}_{framework}_predictions.csv' if arch=='SimpleRNN' else path.with_name(path.stem+'_predictions.csv'))
                ref=pd.read_csv(reference)
                if task=='stock':np.testing.assert_allclose(pred,ref.RNN_Predicted_Close,atol=1e-3)
                else:
                    expected=ref.predicted_event.map({'view':0,'addtocart':1,'transaction':2}).to_numpy()
                    np.testing.assert_array_equal(pred,expected)
                write_json(folder/'config.json',dict(task=task,framework=framework,architecture=arch,checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),metadata=meta,full_test_matches_saved=True))
                finish_run(folder,dict(task=task,framework=framework,architecture=arch,test_samples=len(xt),full_test_matches_saved=True))
            summary.append(dict(task=task,framework=framework,architecture=arch,run=folder.relative_to(ROOT).as_posix(),**m))
    write_json(ROOT/'assets/data/sequences_verified.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':evaluate()
