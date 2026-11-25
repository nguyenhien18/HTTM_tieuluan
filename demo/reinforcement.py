"""Reproducible tabular Q-learning demos on two explicitly simulated environments."""
from collections import deque
from functools import lru_cache
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,time,threading
import numpy as np
from .models import InputError
from .config import ROOT

MAZE=np.array([
 [0,0,1,0,0,0,0,0],
 [1,0,1,0,1,1,1,0],
 [0,0,0,0,0,0,1,0],
 [0,1,1,1,1,0,1,0],
 [0,0,0,0,1,0,0,0],
 [1,1,0,0,1,1,1,0],
 [0,0,0,1,0,0,0,0],
 [0,1,0,0,0,1,1,0]],dtype=np.int8)
START=(0,0);GOAL=(7,7);ACTIONS=[(-1,0),(0,1),(1,0),(0,-1)]
_TRAIN_LOCK=threading.Lock()
IMPLEMENTATION_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def number(data,key,default,lo,hi,integer=False):
 value=data.get(key,default)
 if isinstance(value,bool):raise InputError(f'{key}: cần nhập số.')
 try:value=float(value)
 except (ValueError,TypeError,OverflowError):raise InputError(f'{key}: cần nhập số.')
 if not np.isfinite(value) or not lo<=value<=hi or (integer and not value.is_integer()):raise InputError(f'{key}: giá trị cần nằm trong [{lo}, {hi}]'+(' và là số nguyên.' if integer else '.'))
 return int(value) if integer else value

def configuration(task,data):
 if task not in ['maze','pricing']:raise InputError('Demo RL không hợp lệ.')
 if not isinstance(data,dict):raise InputError('Cấu hình cần là một đối tượng JSON.')
 return dict(task=task,seed=number(data,'seed',42,0,2**31-1,True),episodes=number(data,'episodes',2500 if task=='maze' else 6000,100,10000,True),alpha=number(data,'alpha',.15,.01,1),gamma=number(data,'gamma',.95,0,1),epsilon_start=number(data,'epsilon_start',.5,.01,1),epsilon_min=.02,epsilon_decay=.997 if task=='maze' else .999,beta=number(data,'beta',.15,.05,.5) if task=='pricing' else None)

def maze_step(state,action):
 candidate=tuple(state[i]+ACTIONS[action][i] for i in [0,1])
 if not (0<=candidate[0]<8 and 0<=candidate[1]<8) or MAZE[candidate]:return state,-8.,False
 return candidate,(30. if candidate==GOAL else -1.),candidate==GOAL

def shortest_path(start=START):
 queue=deque([start]);parent={start:None}
 while queue:
  state=queue.popleft()
  if state==GOAL:
   path=[]
   while state is not None:path.append(list(state));state=parent[state]
   return path[::-1]
  for action in range(4):
   nxt,_,_=maze_step(state,action)
   if nxt not in parent:parent[nxt]=state;queue.append(nxt)
 return []

def maze_rollout(q,rng,start=START,random=False,limit=180):
 state=start;path=[list(state)];reward=0.;bumps=0
 for _ in range(limit):
  action=int(rng.integers(4)) if random else int(np.argmax(q[state]))
  nxt,r,done=maze_step(state,action);bumps+=int(nxt==state);state=nxt;reward+=r;path.append(list(state))
  if done:break
 return dict(success=state==GOAL,steps=len(path)-1,reward=reward,bumps=bumps,path=path)

def train_maze(cfg):
 rng=np.random.default_rng(cfg['seed']);q=np.zeros((8,8,4));history=[];visits=np.zeros((8,8),dtype=int)
 for episode in range(cfg['episodes']):
  state=START;total=0.;eps=max(cfg['epsilon_min'],cfg['epsilon_start']*cfg['epsilon_decay']**episode)
  for step in range(180):
   action=int(rng.integers(4)) if rng.random()<eps else int(rng.choice(np.flatnonzero(q[state]==q[state].max())))
   nxt,r,done=maze_step(state,action);visits[state]+=1
   target=r+(0 if done else cfg['gamma']*float(q[nxt].max()))
   q[state+(action,)]+=cfg['alpha']*(target-q[state+(action,)])
   state=nxt;total+=r
   if done:break
  history.append(dict(episode=episode+1,reward=total,steps=step+1,epsilon=eps))
 before=q.copy();rng_eval=np.random.default_rng(cfg['seed']+100000)
 learned=[maze_rollout(q,rng_eval) for _ in range(100)]
 random=[maze_rollout(q,rng_eval,random=True) for _ in range(100)]
 assert np.array_equal(before,q)
 def stats(name,rows):return dict(model=name,episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),mean_steps=float(np.mean([r['steps'] for r in rows])),mean_reward=float(np.mean([r['reward'] for r in rows])),mean_bumps=float(np.mean([r['bumps'] for r in rows])))
 bfs=shortest_path();free=[tuple(x) for x in np.argwhere(MAZE==0) if tuple(x)!=GOAL]
 coverage=sum(maze_rollout(q,rng_eval,start=s)['success'] for s in free)
 data=dict(environment='maze',grid=MAZE.tolist(),start=list(START),goal=list(GOAL),history=history,path=learned[0]['path'],bfs_path=bfs,metrics=[stats('Q-learning · greedy',learned),stats('Ngẫu nhiên',random)],baseline_steps=len(bfs)-1,coverage=dict(success=coverage,total=len(free)),policy=np.argmax(q,axis=2).tolist(),values=np.max(q,axis=2).tolist(),visits=visits.tolist(),note='Mê cung mô phỏng 8×8. Đánh giá greedy với Q đã đóng băng; BFS là đường ngắn nhất khi biết bản đồ. Huấn luyện từ S cố định nên chưa bảo đảm đi tốt từ mọi ô.')
 return data,q

def competitor(demand,previous_a):return int(demand==1 and previous_a==1)
def conversion(demand,a,b,beta):return float(np.clip(.30+.25*demand+beta*a-.10*b,.02,.98))
def margin(action):return 10*(1-.20*action)-5

def pricing_dp(cfg,steps=20):
 # State = (time, demand, previous own action, previous opponent action).
 # Time is included so the finite-horizon problem is Markov.
 value=np.zeros((steps+1,2,2,2));policy=np.zeros((steps,2,2,2),dtype=np.int8)
 for t in range(steps-1,-1,-1):
  for d in range(2):
   for pa in range(2):
    for pb in range(2):
     b=competitor(d,pa);scores=[]
     for a in range(2):
      expected=(10 if d==0 else 20)*conversion(d,a,b,cfg['beta'])*margin(a)
      scores.append(expected+cfg['gamma']*.5*(value[t+1,0,a,b]+value[t+1,1,a,b]))
     policy[t,d,pa,pb]=int(np.argmax(scores));value[t,d,pa,pb]=max(scores)
 return policy,float(.5*(value[0,0,0,0]+value[0,1,0,0]))

def pricing_evaluate(q,cfg,n=500):
 steps=20;rng=np.random.default_rng(cfg['seed']+200000)
 demands=rng.integers(2,size=(n,steps));sales=rng.random((n,steps,20));dp,optimal=pricing_dp(cfg)
 policies={
  'Q-learning':lambda t,d,pa,pb:int(np.argmax(q[t,d,pa,pb])),
  'Agent luật':lambda t,d,pa,pb:int(d==1 and pb==1),
  'Luôn giảm giá':lambda t,d,pa,pb:1,
  'DP · biết mô hình':lambda t,d,pa,pb:int(dp[t,d,pa,pb])}
 before=q.copy();rows=[];curves=[];profits={}
 for name,policy in policies.items():
  rewards=np.zeros((n,steps));discounts=0
  for e in range(n):
   pa=pb=0
   for t in range(steps):
    d=int(demands[e,t]);a=policy(t,d,pa,pb);b=competitor(d,pa)
    count=10 if d==0 else 20;sold=int(np.sum(sales[e,t,:count]<conversion(d,a,b,cfg['beta'])))
    rewards[e,t]=sold*margin(a);discounts+=a;pa,pb=a,b
  mean=rewards.mean(1);discounted=(rewards*np.power(cfg['gamma'],np.arange(steps))).sum(1)
  profits[name]=mean
  rows.append(dict(model=name,episodes=n,mean_profit=float(mean.mean()),sd_profit=float(mean.std(ddof=1)),discount_rate=discounts/(n*steps),discounted_return=float(discounted.mean())))
  curves.append(dict(model=name,values=rewards.mean(0).tolist()))
 delta=profits['Q-learning']-profits['Agent luật'];se=float(delta.std(ddof=1)/np.sqrt(n))
 assert np.array_equal(before,q)
 return rows,curves,dict(episodes=n,paired_mean=float(delta.mean()),ci95=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],note='Khoảng xấp xỉ theo 500 episode đánh giá cho một model đã học; không đo biến thiên giữa các seed huấn luyện.'),optimal

def train_pricing(cfg):
 steps=20;q=np.zeros((steps,2,2,2,2));rng=np.random.default_rng(cfg['seed']);history=[]
 for e in range(cfg['episodes']):
  pa=pb=0;d=int(rng.integers(2));total=0.;eps=max(cfg['epsilon_min'],cfg['epsilon_start']*cfg['epsilon_decay']**e)
  discounts=0
  for t in range(steps):
   state=(t,d,pa,pb);a=int(rng.integers(2)) if rng.random()<eps else int(rng.choice(np.flatnonzero(q[state]==q[state].max())))
   b=competitor(d,pa);customers=10 if d==0 else 20
   reward=float(rng.binomial(customers,conversion(d,a,b,cfg['beta']))*margin(a));nd=int(rng.integers(2))
   target=reward+(cfg['gamma']*float(q[t+1,nd,a,b].max()) if t<steps-1 else 0.)
   q[state+(a,)]+=cfg['alpha']*(target-q[state+(a,)])
   total+=reward;discounts+=a;d,pa,pb=nd,a,b
  history.append(dict(episode=e+1,reward=total/steps,discount_rate=discounts/steps,epsilon=eps))
 rows,curves,paired,optimal=pricing_evaluate(q,cfg)
 states=[dict(step=t+1,demand=d,previous_a=pa,previous_b=pb,q_keep=float(q[t,d,pa,pb,0]),q_discount=float(q[t,d,pa,pb,1]),action=int(np.argmax(q[t,d,pa,pb]))) for t in [0,9,19] for d in range(2) for pa in range(2) for pb in range(2)]
 return dict(environment='pricing',history=history,metrics=rows,evaluation_curves=curves,paired=paired,dp_expected_discounted_return=optimal,q_rows=states,environment_spec=dict(price=10,cost=5,discount=.2,steps=20,beta=cfg['beta'],state='(bước, nhu cầu, hành động trước của mình, hành động trước của đối thủ)',competitor='Giảm giá khi nhu cầu cao và mình đã giảm ở bước trước.'),note='Thị trường mô phỏng, không phải dữ liệu khách hàng thực. Mỗi chính sách ở cùng vai người bán A, cùng nhu cầu và cùng số ngẫu nhiên doanh số (CRN). Q-learning dừng học khi đánh giá. DP tối ưu kỳ vọng tổng thưởng có chiết khấu theo gamma; model học có thể kém baseline.'),q

@lru_cache(maxsize=12)
def _train_cached(serialized):
 cfg=json.loads(serialized);start=time.perf_counter()
 run_id='rl-'+cfg['task']+'-'+hashlib.sha256((IMPLEMENTATION_SHA+serialized).encode()).hexdigest()[:12]
 folder=ROOT/'runs/rl'/run_id
 with _TRAIN_LOCK:
  saved=folder/'result.json'
  if saved.is_file() and (folder/'qtable.npz').is_file():
   try:
    data=json.loads(saved.read_text(encoding='utf-8'))
    if data['q_sha256']==hashlib.sha256((folder/'qtable.npz').read_bytes()).hexdigest():return data
   except (ValueError,KeyError,OSError):pass
  data,q=train_maze(cfg) if cfg['task']=='maze' else train_pricing(cfg)
  folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'qtable.npz',q=q)
  data.update(config=cfg,run_id=run_id,training_ms=round((time.perf_counter()-start)*1000,2),created_at=datetime.now(timezone.utc).isoformat(),q_sha256=hashlib.sha256((folder/'qtable.npz').read_bytes()).hexdigest(),implementation_sha256=IMPLEMENTATION_SHA,artifact=f'runs/rl/{run_id}/qtable.npz')
  saved.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
  return data

def run_rl(task,payload):
 cfg=configuration(task,payload)
 return copy.deepcopy(_train_cached(json.dumps(cfg,sort_keys=True)))
