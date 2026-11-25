"""Prepare exact local image data, re-evaluate frozen checkpoints, or train new runs.

python -m experiments.images prepare --mnist D:/ASignment4/data --fashion D:/Assignment5/Assignment5_CNN_Project/data/fashion_mnist --cifar D:/Assignment5/Assignment5_CNN_Project/data/cifar10
python -m experiments.images evaluate --dataset all
python -m experiments.images train --dataset fashion --model basic_cnn
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from .common import ROOT, sha, write_json, finish_run, check_run

DATA=ROOT/'data/images'
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def prepare(args):
    DATA.mkdir(parents=True,exist_ok=True)
    for task,folder,train_file,test_file in [('mnist',args.mnist,'mnist_train.csv','mnist_test.csv'),
            ('fashion',args.fashion,'fashion-mnist_train.csv','fashion-mnist_test.csv')]:
        destination=DATA/f'{task}.npz'
        if destination.exists():continue
        arrays={};sources=[]
        for split,filename in [('train',train_file),('test',test_file)]:
            path=Path(folder)/filename;frame=pd.read_csv(path)
            labels=frame.iloc[:,0].to_numpy(dtype='int64');pixels=frame.iloc[:,1:].to_numpy(dtype='uint8')
            assert pixels.shape[1]==784 and set(labels)==set(range(10))
            arrays[f'x_{split}']=pixels.reshape(-1,28,28,1);arrays[f'y_{split}']=labels
            sources.append(dict(file=filename,sha256=sha(path),rows=len(frame)))
        np.savez_compressed(destination,**arrays)
        write_json(DATA/f'{task}.json',dict(dataset=task,sha256=sha(destination),sources=sources,
            normalization='uint8 / 255.0',class_order=list(range(10)),test_is_official_assignment_test=True))
        print('Prepared',task,flush=True)
    if not (DATA/'cifar10.npz').exists():
        arrays={};indexes=[]
        for split in ['train','test']:
            pictures=[];labels=[]
            for label,name in enumerate(CLASSES):
                paths=sorted(p for p in (Path(args.cifar)/split/name).iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
                for p in paths:
                    with Image.open(p) as image:
                        assert image.size==(32,32)
                        pictures.append(np.asarray(image.convert('RGB'),dtype='uint8'))
                    labels.append(label);indexes.append(dict(split=split,path=p.relative_to(Path(args.cifar)).as_posix(),label=label))
            arrays[f'x_{split}']=np.stack(pictures);arrays[f'y_{split}']=np.asarray(labels,dtype='int64')
            assert len(labels)==(50000 if split=='train' else 10000)
        np.savez_compressed(DATA/'cifar10.npz',**arrays)
        pd.DataFrame(indexes).to_csv(DATA/'cifar10_index.csv',index=False)
        write_json(DATA/'cifar10.json',dict(dataset='cifar10',sha256=sha(DATA/'cifar10.npz'),
            source='Local Assignment5 image directories; sorted path index accompanies the immutable NPZ snapshot.',
            class_order=CLASSES,normalization='uint8 / 255.0',index_sha256=sha(DATA/'cifar10_index.csv')))
        print('Prepared cifar10',flush=True)


def checkpoint(dataset,name):
    if dataset=='mnist':
        if name.startswith('torch'):return ROOT/f'assets/models/a04/mnist_pytorch_{name.split("_")[1]}_fixed.pth'
        if name=='numpy_scratch':return ROOT/'assets/models/a04/mnist_scratch_fixed.npz'
        return ROOT/f'assets/models/a04/mnist_tensorflow_{name.split("_")[1]}_fixed.keras'
    return ROOT/f'assets/models/a05/{"fashion_mnist" if dataset=="fashion" else "cifar10"}_{name}.keras'


def infer_batches(dataset,name,x,batch_size=128):
    from demo.models import scratch_mnist, mnist_torch, keras_model
    import torch
    torch.set_num_threads(4)
    output=[]
    for start in range(0,len(x),batch_size):
        batch=x[start:start+batch_size].astype('float32')/255.
        if dataset=='mnist' and name.startswith('torch'):
            with torch.inference_mode():p=torch.softmax(mnist_torch(name)(torch.from_numpy(batch.transpose(0,3,1,2))),dim=1).numpy()
        elif name=='numpy_scratch':
            z=scratch_mnist(batch.transpose(0,3,1,2));p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        else:p=keras_model(str(checkpoint(dataset,name)))(batch,training=False).numpy()
        if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-4):raise ValueError('Invalid image-model output')
        output.append(p)
    return np.concatenate(output)


def figures(folder,x,y,pred,probabilities):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm=confusion_matrix(y,pred,labels=list(range(10)))
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set(xlabel='Predicted class',ylabel='Actual class',xticks=range(10),yticks=range(10),title='Full test confusion matrix')
    fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=150);plt.close(fig)
    wrong=np.flatnonzero(pred!=y)
    # Highest-confidence mistakes, deterministic and explicitly labeled.
    chosen=wrong[np.argsort(-probabilities[wrong].max(1))[:16]]
    fig,axes=plt.subplots(4,4,figsize=(8,8))
    for ax in axes.ravel():ax.axis('off')
    for ax,i in zip(axes.ravel(),chosen):
        ax.imshow(x[i].squeeze(),cmap='gray' if x.shape[-1]==1 else None)
        ax.set_title(f'#{i}: true={y[i]} pred={pred[i]}\np={probabilities[i].max():.3f}',fontsize=8)
    fig.tight_layout();fig.savefig(folder/'wrong_examples.png',dpi=150);plt.close(fig)
    pd.DataFrame(dict(test_index=chosen,actual=y[chosen],prediction=pred[chosen],score=probabilities[chosen].max(1))).to_csv(folder/'wrong_examples.csv',index=False)


def evaluate(dataset):
    from demo.config import CNN
    data=np.load(DATA/f'{dataset}.npz',allow_pickle=False);x,y=data['x_test'],data['y_test']
    dataset_hash=sha(DATA/f'{dataset}.npz');summary=[]
    for name,_ in CNN[dataset]['models']:
        model_path=checkpoint(dataset,name);model_hash=sha(model_path)
        signature=__import__('hashlib').sha256((dataset_hash+model_hash+sha(Path(__file__))).encode()).hexdigest()[:12]
        folder=ROOT/'runs/cnn'/dataset/name/signature
        if (folder/'complete.json').exists():
            check_run(folder);metrics=json.loads((folder/'metrics.json').read_text());print('Verified cached',dataset,name,flush=True)
        else:
            folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();print('Evaluating',dataset,name,len(y),'test images',flush=True)
            p=infer_batches(dataset,name,x);pred=p.argmax(1)
            metrics=dict(Accuracy=float(accuracy_score(y,pred)),Macro_F1=float(f1_score(y,pred,average='macro')),
                         test_samples=len(y),majority_baseline_class=int(np.bincount(data['y_train']).argmax()),
                         majority_baseline_accuracy=float(np.mean(y==np.bincount(data['y_train']).argmax())))
            records=pd.DataFrame(dict(test_index=np.arange(len(y)),actual=y,prediction=pred))
            for i in range(10):records[f'prob_{i}']=p[:,i]
            records.to_csv(folder/'predictions.csv',index=False)
            write_json(folder/'metrics.json',metrics)
            write_json(folder/'classification_report.json',classification_report(y,pred,output_dict=True,zero_division=0))
            write_json(folder/'config.json',dict(dataset=dataset,model=name,checkpoint=model_path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=model_hash,dataset_sha256=dataset_hash,normalization='uint8 / 255',
                operation='frozen-checkpoint full-test evaluation',selection='No checkpoint selected using these test results.',
                comparison_limit='Original training budgets differ. This does not isolate framework effects.'))
            figures(folder,x,y,pred,p)
            finish_run(folder,dict(dataset=dataset,model=name,test_samples=len(y),elapsed_seconds=round(time.perf_counter()-start,2)))
        summary.append(dict(dataset=dataset,model=name,run=folder.relative_to(ROOT).as_posix(),**metrics))
    write_json(ROOT/f'assets/data/cnn_verified_{dataset}.json',summary)
    return summary


def train(dataset,name,epochs):
    """New immutable run; existing demo checkpoints are never overwritten."""
    import os
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
    import torch
    from demo.models import MnistBasic,MnistImproved
    data=np.load(DATA/f'{dataset}.npz',allow_pickle=False)
    indexes=np.arange(len(data['y_train']));a,b=train_test_split(indexes,test_size=.2,stratify=data['y_train'],random_state=42)
    folder=ROOT/'runs/cnn_training'/dataset/name/time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'split.npz',train=a,validation=b)
    cfg=dict(dataset=dataset,model=name,seed=42,epochs=epochs,split='stratified 80/20 of provided train; original test held out',
             dataset_sha256=sha(DATA/f'{dataset}.npz'),checkpoint_selection='minimum validation cross-entropy',
             compare_frameworks='Architecture/initialization/optimizer parity is not asserted.')
    write_json(folder/'config.json',cfg)
    x=data['x_train'].astype('float32')/255.;y=data['y_train'];history=[]
    if name.startswith('torch'):
        torch.set_num_threads(4);torch.manual_seed(42);model=MnistBasic() if name=='torch_basic' else MnistImproved()
        optim=torch.optim.Adam(model.parameters(),lr=.001);criterion=torch.nn.CrossEntropyLoss()
        tx=torch.from_numpy(x.transpose(0,3,1,2));ty=torch.from_numpy(y);rng=np.random.default_rng(42);best=float('inf')
        for e in range(epochs):
            model.train();losses=[]
            for batch in np.array_split(rng.permutation(a),int(np.ceil(len(a)/128))):
                optim.zero_grad();loss=criterion(model(tx[batch]),ty[batch]);loss.backward();optim.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():val=float(criterion(model(tx[b]),ty[b]))
            history.append(dict(epoch=e+1,loss=float(np.mean(losses)),val_loss=val))
            if val<best:best=val;torch.save(model.state_dict(),folder/'checkpoint.pth')
    elif name=='numpy_scratch':
        from .manual_cnn import ManualMNISTCNN,softmax_ce
        model=ManualMNISTCNN(seed=42);nx=x.transpose(0,3,1,2);rng=np.random.default_rng(42);best=float('inf')
        for e in range(epochs):
            losses=[model.train_batch(nx[batch],y[batch],.005) for batch in np.array_split(rng.permutation(a),int(np.ceil(len(a)/64)))]
            vals=[softmax_ce(model.forward(nx[batch]),y[batch])[0] for batch in np.array_split(b,int(np.ceil(len(b)/128)))]
            val=float(np.mean(vals));history.append(dict(epoch=e+1,loss=float(np.mean(losses)),val_loss=val))
            if val<best:
                best=val;np.savez_compressed(folder/'checkpoint.npz',conv1_W=model.conv1.W,conv1_b=model.conv1.b,
                    conv2_W=model.conv2.W,conv2_b=model.conv2.b,fc_W=model.fc.W,fc_b=model.fc.b)
    else:
        from tensorflow import keras
        keras.utils.set_random_seed(42)
        original=keras.models.load_model(checkpoint(dataset,name),compile=False,safe_mode=True)
        model=keras.models.clone_model(original)
        model.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        h=model.fit(x[a],y[a],validation_data=(x[b],y[b]),epochs=epochs,batch_size=128,verbose=2,
            callbacks=[keras.callbacks.ModelCheckpoint(folder/'checkpoint.keras',monitor='val_loss',save_best_only=True)])
        history=[dict(epoch=i+1,**{key:float(value[i]) for key,value in h.history.items()}) for i in range(len(h.history['loss']))]
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    finish_run(folder,dict(**cfg,operation='training completed; run evaluate-trained before reporting test metrics'))
    print(folder)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--mnist',required=True);q.add_argument('--fashion',required=True);q.add_argument('--cifar',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--dataset',choices=['mnist','fashion','cifar10','all'],default='all')
    q=sub.add_parser('train');q.add_argument('--dataset',choices=['mnist','fashion','cifar10'],required=True);q.add_argument('--model',required=True);q.add_argument('--epochs',type=int,default=5)
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='train':
        from demo.config import CNN
        if args.model not in dict(CNN[args.dataset]['models']):p.error('Model does not belong to dataset')
        if not 1<=args.epochs<=100:p.error('epochs must be 1..100')
        train(args.dataset,args.model,args.epochs)
    else:
        for name in ['mnist','fashion','cifar10'] if args.dataset=='all' else [args.dataset]:evaluate(name)


if __name__=='__main__':main()
