import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
from demo.pages import PAGES
from demo.domain_knowledge import corpus, retrieve_domain, domain_config, evaluate_domains

class LaterChapterTests(unittest.TestCase):
 def setUp(self):self.client=app.test_client()
 def test_every_independent_page_renders(self):
  for page in PAGES:
   with self.subTest(page=page['path']):
    response=self.client.get(page['path'])
    self.assertEqual(response.status_code,200)
    self.assertIn(page['title'],response.get_data(as_text=True))
 def test_lstm_gru_inference_and_framework_validation(self):
  for arch in ['lstm','gru']:
   for task,body in [('customer',dict(events=['view']*5)),('stock',{})]:
    response=self.client.post('/api/rnn/'+task,json=dict(architecture=arch,framework='keras',**body))
    self.assertEqual(response.status_code,200,response.get_json())
    d=response.get_json();self.assertEqual(d['architecture'],arch)
    self.assertTrue(np.isfinite(d.get('predicted',d.get('score'))))
    bad=self.client.post('/api/rnn/'+task,json=dict(architecture=arch,framework='pytorch',**body))
    self.assertEqual(bad.status_code,400)
 def test_financial_date_and_values_are_source_records(self):
  for domain in ['stock','gold']:
   d=corpus(domain)[-1]
   for method in ['kg','tfidf','hybrid']:
    r=retrieve_domain(domain,'Giá ngày '+d['date'],method)
    self.assertEqual(r['sources'][0]['id'],d['id'])
    self.assertEqual(r['sources'][0]['fields'],d['fields'])
    self.assertTrue(any(e['relation']=='HAS_VALUE' for e in r['graph']['edges']))
   for q in ['Giá hôm nay','Giá ngày 2099-01-01','HbA1c phản ánh điều gì?']:
    self.assertEqual(retrieve_domain(domain,q)['status'],'no_evidence')
 def test_gold_units_not_invented(self):
  self.assertIn('không khai báo',domain_config('gold')['unit'])
 def test_methods_validation(self):
  r=self.client.post('/api/domain-knowledge',json=dict(domain='stock',question='test',method='fake'))
  self.assertEqual(r.status_code,400)
 def test_evaluation_counts_and_abstention(self):
  d=evaluate_domains();self.assertEqual(len(d['details']),63)
  self.assertTrue(all(row['questions']==21 for row in d['rows']))
  self.assertTrue(all(row['status']=='no_evidence' for row in d['out_of_scope']))

if __name__=='__main__':unittest.main()
