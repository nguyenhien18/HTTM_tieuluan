from pathlib import Path
import json,sys,unittest
from unittest.mock import patch,Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
from demo.reinforcement import configuration,train_maze,shortest_path,maze_step,pricing_dp,pricing_evaluate
from demo.models import InputError,checked_probabilities
from demo.local_llm import generate

class CompletionTests(unittest.TestCase):
 def setUp(self):self.client=app.test_client()
 def test_malformed_inputs_are_client_errors(self):
  cases=[('/api/tabular',{'task':[]}),('/api/cnn',{'dataset':{},'model':[]}),('/api/domain-knowledge',{'domain':[],'question':'test'}),('/api/knowledge',{'question':'test','generative':'false'}),('/api/rl/maze',{'episodes':True}),('/api/rl/maze',{'alpha':'NaN'}),('/api/rl/pricing',{'episodes':10001}),('/api/agents',{'question':'a'*1501})]
  for path,body in cases:
   with self.subTest(path=path,body=body):self.assertEqual(self.client.post(path,json=body).status_code,400)
 def test_invalid_model_outputs_rejected(self):
  for values in [[float('nan'),1],[.1,.1],[-.1,1.1]]:
   with self.assertRaises(InputError):checked_probabilities(values)
 def test_maze_learning_reaches_bfs_and_uses_valid_moves(self):
  cfg=configuration('maze',{'episodes':2500});result,q=train_maze(cfg);path=result['path']
  self.assertEqual(path[-1],result['goal']);self.assertEqual(len(path),len(shortest_path()))
  for a,b in zip(path,path[1:]):self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]),1)
  self.assertEqual(result['metrics'][0]['success_rate'],1)
  self.assertLess(result['metrics'][1]['success_rate'],1)
 def test_maze_collision_penalty_and_bfs(self):
  state,r,done=maze_step((0,0),0);self.assertEqual(state,(0,0));self.assertLess(r,-1);self.assertFalse(done)
  self.assertTrue(shortest_path())
 def test_pricing_crn_equal_policies_and_q_frozen(self):
  cfg=configuration('pricing',{});q=np.zeros((20,2,2,2,2));before=q.copy()
  rows,curves,paired,opt=pricing_evaluate(q,cfg,n=40)
  self.assertEqual(rows[0]['mean_profit'],rows[1]['mean_profit'])
  self.assertEqual(paired['ci95'],[0,0]);np.testing.assert_array_equal(before,q)
 def test_pricing_dp_one_step_matches_manual_expectation(self):
  cfg=configuration('pricing',{'beta':.4});policy,value=pricing_dp(cfg,steps=1)
  self.assertAlmostEqual(value,39.0)
  self.assertEqual(policy[0,0,0,0],1)
 def test_rl_api_and_downloadable_artifacts(self):
  r=self.client.post('/api/rl/maze',json={'episodes':2500});self.assertEqual(r.status_code,200)
  d=r.get_json();self.assertTrue(d['q_sha256']);run=d['run_id']
  q=self.client.get(f'/api/rl/runs/{run}/qtable.npz');self.assertEqual(q.status_code,200);self.assertTrue(q.data.startswith(b'PK'));q.close()
  log=self.client.get(f'/api/rl/runs/{run}/history.csv');self.assertIn(b'episode',log.data)
  self.assertEqual(self.client.get('/api/rl/runs/invalid/qtable.npz').status_code,400)
 def test_llm_citations_must_reference_supplied_evidence(self):
  fake=Mock();fake.json.return_value={'choices':[{'message':{'content':json.dumps({'answer':'test','citations':['FAKE']})}}]}
  with patch('demo.local_llm.ensure_server'),patch('demo.local_llm.requests.post',return_value=fake),patch('demo.local_llm.artifact',return_value={}):
   with self.assertRaises(InputError):generate('test',[dict(id='K01',title='test',text='test')])
  fake.json.return_value={'choices':[{'message':{'content':json.dumps({'answer':'Valid','citations':['K01']})}}]}
  with patch('demo.local_llm.ensure_server'),patch('demo.local_llm.requests.post',return_value=fake),patch('demo.local_llm.artifact',return_value={}):
   d=generate('test',[dict(id='K01',title='test',text='test')]);self.assertEqual(d['citations'],['K01'])
 def test_health_and_model_runtime_status(self):
  d=self.client.get('/api/health').get_json();self.assertEqual(d['pages'],26)
  d=self.client.get('/api/llm').get_json();self.assertEqual(d['provider'],'llama.cpp');self.assertIn('installed',d)
 def test_financial_dates_and_statistics(self):
  d=self.client.post('/api/domain-knowledge',json=dict(domain='stock',question='Close ngày 05/12/2023')).get_json()
  self.assertEqual(d['sources'][0]['id'],'S20231205')
  d=self.client.post('/api/domain-knowledge',json=dict(domain='stock',question='Close trung bình từ 2023-12-01 đến 2023-12-05')).get_json()
  self.assertEqual(d['mode'],'statistics');self.assertEqual(d['sources'][0]['fields']['count'],3)
  self.assertAlmostEqual(d['sources'][0]['fields']['mean'],(147.029999+144.839996+146.880005)/3,places=4)
  self.assertEqual(self.client.post('/api/domain-knowledge',json=dict(domain='stock',question='Close ngày 2023-02-30')).status_code,400)
 def test_generation_scope_guard(self):
  d=self.client.post('/api/domain-knowledge',json=dict(domain='diabetes',question='Tôi nên tiêm bao nhiêu insulin?',generative=True)).get_json()
  self.assertEqual(d['status'],'out_of_scope');self.assertFalse(d['sources'])

if __name__=='__main__':unittest.main()
