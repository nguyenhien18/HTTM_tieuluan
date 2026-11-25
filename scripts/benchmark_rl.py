"""Train/evaluate a small multi-seed RL benchmark and retain each actual run."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.reinforcement import run_rl
from demo.config import ROOT
rows=[]
for seed in [42,43,44]:
 for task,beta in [('maze',None),('pricing',.15),('pricing',.4)]:
  payload=dict(seed=seed)
  if beta is not None:payload['beta']=beta
  d=run_rl(task,payload)
  row=dict(task=task,seed=seed,beta=beta,run_id=d['run_id'],metrics=d['metrics']);rows.append(row)
  print(json.dumps(row,ensure_ascii=False),flush=True)
summary=[]
for beta in [.15,.4]:
 subset=[r for r in rows if r['task']=='pricing' and r['beta']==beta]
 for policy in ['Q-learning','Agent luật','DP · biết mô hình']:
  values=[next(m['mean_profit'] for m in r['metrics'] if m['model']==policy) for r in subset]
  summary.append(dict(task='pricing',beta=beta,model=policy,seeds=len(values),mean_profit=float(np.mean(values)),sd_across_seeds=float(np.std(values,ddof=1))))
folder=ROOT/'runs/verification';folder.mkdir(parents=True,exist_ok=True)
(folder/'rl_benchmark.json').write_text(json.dumps(dict(rows=rows,summary=summary,note='Ba seed huấn luyện, đánh giá trên môi trường mô phỏng. Không phải số đo trên thị trường thực.'),ensure_ascii=False,indent=2),encoding='utf-8')
