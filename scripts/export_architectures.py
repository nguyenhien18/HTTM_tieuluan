"""Export shapes/parameter counts from actual model objects for report diagrams."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.common import write_json,sha
from experiments.images import checkpoint
from demo.config import CNN
from demo.models import keras_model,mnist_torch,rnn_torch


def main():
    rows=[]
    for task,config in CNN.items():
        for name,_ in config['models']:
            path=checkpoint(task,name)
            if name=='numpy_scratch':
                weights=np.load(path,allow_pickle=False)
                spec=dict(parameters=sum(v.size for v in weights.values()),
                    layers=[dict(layer='Conv2D + ReLU',output=[4,28,28]),dict(layer='Conv2D + ReLU',output=[8,28,28]),
                            dict(layer='MaxPool2D',output=[8,14,14]),dict(layer='Dense',output=[10])])
            elif name.startswith('torch'):
                model=mnist_torch(name);spec=dict(parameters=sum(p.numel() for p in model.parameters()),layers=[])
                x=torch.zeros(1,1,28,28)
                with torch.inference_mode():
                    for layer in model.net:
                        x=layer(x);spec['layers'].append(dict(layer=type(layer).__name__,output=list(x.shape),parameters=sum(p.numel() for p in layer.parameters())))
            else:
                model=keras_model(str(path));spec=dict(parameters=model.count_params(),layers=[dict(layer=type(l).__name__,name=l.name,output=str(l.output.shape),parameters=l.count_params()) for l in model.layers],keras_config=model.get_config())
            rows.append(dict(group='cnn',dataset=task,model=name,checkpoint=path.relative_to(ROOT).as_posix(),sha256=sha(path),**spec))
    for task in ['stock','customer']:
        prefix='AMZN' if task=='stock' else 'Customer'
        for framework,arch in [('Keras','SimpleRNN'),('PyTorch','SimpleRNN'),('Keras','LSTM'),('Keras','GRU')]:
            if framework=='PyTorch':
                model,metadata=rnn_torch(task);spec=dict(parameters=sum(p.numel() for p in model.parameters()),layers=str(model))
            else:
                model=keras_model(str(ROOT/f'assets/models/a06/{prefix}_{framework}_{arch}.keras'))
                spec=dict(parameters=model.count_params(),keras_config=model.get_config())
            rows.append(dict(group='sequence',task=task,framework=framework,architecture=arch,**spec))
    for registry in (ROOT/'assets/models/submission').glob('*/registry.json'):
        d=json.loads(registry.read_text());rows.append(dict(group='mlp',task=registry.parent.name,run=d['run'],**json.loads((ROOT/d['run']/'architecture.json').read_text())))
    write_json(ROOT/'assets/data/architecture_registry.json',rows)
    print('Exported',len(rows),'actual architecture descriptions')


if __name__=='__main__':main()
