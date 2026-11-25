import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
from demo.config import DATA, TASKS, CNN, read_json
from demo.models import model_frame, pipeline, mlp, InputError
from demo.knowledge import retrieve

class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=app.test_client()
        cls.examples=read_json(DATA/'form_examples.json')

    def post(self,path,payload):
        response=self.client.post(path,json=payload)
        self.assertEqual(response.status_code,200,response.get_json())
        return response.get_json()

    def test_config_and_home(self):
        self.assertEqual(self.client.get('/').status_code,200)
        d=self.client.get('/api/config').get_json()
        self.assertEqual(len(d['gallery']),120)
        self.assertEqual(len(d['tasks']),5)

    def test_tabular_all_tasks_and_engines(self):
        for task in TASKS:
            for engine in ['ml','mlp'] if task in ['diabetes','house','ecommerce'] else ['ml']:
                with self.subTest(task=task,engine=engine):
                    d=self.post('/api/tabular',dict(task=task,engine=engine,input=self.examples[task][0]['input']))
                    self.assertEqual(len(d['results']),1)
                    self.assertTrue(np.isfinite(d['results'][0]['prediction']))
                    if d['results'][0]['score'] is not None:self.assertTrue(0<=d['results'][0]['score']<=1)

    def test_polish_city_consistent_with_both_encoders(self):
        values=self.examples['house'][1]['input']
        for engine in ['ml','mlp']:
            with self.subTest(engine=engine):
                row=model_frame('house',values,engine)
                pre=pipeline('house').steps[0][1] if engine=='ml' else mlp('house')[1]
                transformer=next(t for t in pre.transformers_ if t[0]=='cat')
                index=list(transformer[2]).index('city_name')
                enc=transformer[1].steps[-1][1]
                self.assertIn(row.city_name.iloc[0],enc.categories_[index])
        self.post('/api/tabular',dict(task='house',engine='compare',input=values))

    def test_invalid_tabular_inputs(self):
        for payload in [dict(task='diabetes',input={}),dict(task='unknown',input={}),dict(task='diabetes',input={**self.examples['diabetes'][0]['input'],'Age':45}),dict(task='diabetes',input={**self.examples['diabetes'][0]['input'],'BMI':float('inf')})]:
            self.assertEqual(self.client.post('/api/tabular',json=payload).status_code,400)

    def test_all_cnn_models_predict(self):
        gallery=read_json(DATA/'image_gallery.json')
        for dataset,config in CNN.items():
            sample=next(s for s in gallery if s['dataset']==dataset)
            for name,_ in config['models']:
                with self.subTest(dataset=dataset,model=name):
                    d=self.post('/api/cnn',dict(dataset=dataset,model=name,sample=sample['id']))
                    self.assertEqual(len(d['probabilities']),10)
                    self.assertAlmostEqual(sum(p['value'] for p in d['probabilities']),1,places=5)
                    self.assertIsNotNone(d['actual'])
                    self.assertTrue(d['processed'].startswith('data:image/png;base64,'))

    def test_invalid_image_and_model(self):
        for payload in [dict(dataset='mnist',model='fake',sample='mnist_000'),dict(dataset='mnist',model='torch_basic',sample='../secret'),dict(dataset='mnist',model='torch_basic',image='bad')]:
            self.assertEqual(self.client.post('/api/cnn',json=payload).status_code,400)

    def test_customer_both_frameworks(self):
        examples=read_json(DATA/'customer_examples.json')
        for framework in ['pytorch','keras']:
            d=self.post('/api/rnn/customer',dict(events=examples[0]['events'],example=0,framework=framework))
            self.assertEqual(d['actual'],examples[0]['actual'])
            self.assertEqual(len(d['probabilities']),3)
            self.assertAlmostEqual(sum(p['value'] for p in d['probabilities']),1,places=5)
        self.assertEqual(self.client.post('/api/rnn/customer',json=dict(events=['view']*4)).status_code,400)
        d=self.post('/api/rnn/customer',dict(events=['transaction']*5,example=0))
        self.assertIsNone(d['actual'])

    def test_stock_matches_saved_predictions(self):
        for fw,prefix in [('pytorch','PyTorch'),('keras','Keras')]:
            saved=pd.read_csv(DATA/f'a06_results/AMZN_{prefix}_predictions.csv')
            for i in [0,len(saved)-1]:
                row=saved.iloc[i]
                d=self.post('/api/rnn/stock',dict(date=str(row.Date)[:10],framework=fw))
                self.assertAlmostEqual(d['predicted'],row.RNN_Predicted_Close,places=3)
                self.assertAlmostEqual(d['actual'],row.Actual_Close,places=3)
        self.assertEqual(self.client.post('/api/rnn/stock',json=dict(date='2000-01-01')).status_code,400)

    def test_retrieval_type_disambiguation_and_no_evidence(self):
        self.assertEqual(retrieve('Nguyên nhân tiểu đường type 1 là gì?')[0]['id'],'K03')
        self.assertEqual(retrieve('Nguyên nhân tiểu đường type 2 là gì?')[0]['id'],'K04')
        self.assertEqual(retrieve('HbA1c phản ánh điều gì?')[0]['id'],'K05')
        self.assertEqual(retrieve('Bóng đá là gì?'),[])
        d=self.post('/api/knowledge',dict(question='xyzqwv'))
        self.assertEqual(d['status'],'no_evidence')

    def test_agent_routes_and_validation(self):
        d=self.post('/api/agents',dict(question='Phân loại hồ sơ và giải thích tiểu đường type 2 là gì',profile=self.examples['diabetes'][0]['input']))
        self.assertEqual(d['route'],'combined')
        self.assertIsNotNone(d['prediction'])
        self.assertTrue(d['sources'])
        self.assertEqual(d['trace'][-1]['sender'],'SynthesisAgent')
        d=self.post('/api/agents',dict(question='Dự đoán cho tôi',profile={'Age':9}))
        self.assertEqual(d['status'],'need_input')
        self.assertIsNone(d['prediction'])
        d=self.post('/api/agents',dict(question='Tăng liều thuốc như thế nào?'))
        self.assertEqual(d['route'],'out_of_scope')

if __name__=='__main__':unittest.main(verbosity=2)
