"""Checks for leakage, checkpoint linkage, manual derivatives and fault recovery."""
import json,unittest
from pathlib import Path
import numpy as np
import torch
from experiments.common import ROOT,check_run
from experiments.tabular import split_groups
from experiments.manual_cnn import Conv2DSameManual,MaxPool2x2Manual,softmax_ce
from demo.agents import run,verify_prediction,ToolOutputError


class ProtocolTests(unittest.TestCase):
    def test_group_split_keeps_identical_profiles_together(self):
        groups=np.repeat(np.arange(100),[3]*100)
        parts=split_groups(groups,42)
        self.assertEqual(sorted(np.concatenate(list(parts.values())).tolist()),list(range(len(groups))))
        sets=[set(groups[p]) for p in parts.values()]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])

    def test_submission_runs_are_complete_and_test_not_used_for_selection(self):
        for task in ['diabetes','house','ecommerce']:
            with self.subTest(task=task):
                registry=json.loads((ROOT/f'assets/models/submission/{task}/registry.json').read_text())
                folder=ROOT/registry['run'];check_run(folder)
                selection=json.loads((folder/'selection.json').read_text())
                self.assertIs(selection['test_used_for_selection'],False)
                verification=json.loads((folder/'verification.json').read_text())
                self.assertTrue(verification['passed']);self.assertEqual(verification['group_overlap'],0)

    def test_api_uses_verified_submission_checkpoint(self):
        from app import app
        examples=json.loads((ROOT/'assets/data/form_examples.json').read_text(encoding='utf-8'))
        for task in ['diabetes','house','ecommerce']:
            for engine in ['ml','mlp']:
                d=app.test_client().post('/api/tabular',json=dict(task=task,engine=engine,input=examples[task][0]['input'])).get_json()
                value=d['results'][0]
                self.assertIn('runs/tabular/',value['experiment_run'])
                self.assertTrue(all(p['sha256'] for p in value['artifacts']))

    def test_manual_convolution_matches_autograd_forward_and_gradients(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(2,2,5,5));dy=rng.normal(size=(2,3,5,5))
        conv=Conv2DSameManual(2,3);conv.W=conv.W.astype('float64');conv.b=conv.b.astype('float64')
        w=conv.W.copy();bias=conv.b.copy();out=conv.forward(x)
        tx=torch.tensor(x,requires_grad=True);tw=torch.tensor(w,requires_grad=True);tb=torch.tensor(bias,requires_grad=True)
        expected=torch.nn.functional.conv2d(tx,tw,tb,padding=1);expected.backward(torch.tensor(dy))
        np.testing.assert_allclose(out,expected.detach().numpy(),atol=1e-10)
        dx=conv.backward(dy,.1)
        np.testing.assert_allclose(dx,tx.grad.numpy(),atol=1e-10)
        np.testing.assert_allclose((w-conv.W)/.1,tw.grad.numpy(),atol=1e-10)
        np.testing.assert_allclose((bias-conv.b)/.1,tb.grad.numpy(),atol=1e-10)

    def test_manual_softmax_gradient_with_finite_difference(self):
        rng=np.random.default_rng(7);z=rng.normal(size=(3,4));y=np.array([0,2,3]);_,gradient=softmax_ce(z,y)
        numerical=np.zeros_like(z);eps=1e-6
        for index in np.ndindex(z.shape):
            a=z.copy();b=z.copy();a[index]+=eps;b[index]-=eps
            numerical[index]=(softmax_ce(a,y)[0]-softmax_ce(b,y)[0])/(2*eps)
        np.testing.assert_allclose(gradient,numerical,rtol=1e-5,atol=1e-8)

    def test_manual_pool_gradient_matches_autograd_without_ties(self):
        x=np.random.default_rng(11).normal(size=(2,2,4,4));pool=MaxPool2x2Manual();out=pool.forward(x)
        tx=torch.tensor(x,requires_grad=True);expected=torch.nn.functional.max_pool2d(tx,2);expected.sum().backward()
        np.testing.assert_allclose(out,expected.detach().numpy());np.testing.assert_allclose(pool.backward(np.ones_like(out)),tx.grad.numpy())

    def test_agent_recovers_once_and_stops_on_persistent_invalid_output(self):
        profile=json.loads((ROOT/'assets/data/form_examples.json').read_text(encoding='utf-8'))['diabetes'][0]['input']
        for fault in ['nan','range','raise']:
            d=run('Phân loại hồ sơ',profile,'diabetes',fault=fault)
            self.assertEqual(d['status'],'ok');self.assertEqual(d['attempts'],2);verify_prediction(d['prediction'],'diabetes')
        d=run('Phân loại hồ sơ',profile,'diabetes',fault='persistent_nan')
        self.assertEqual(d['status'],'tool_error');self.assertIsNone(d['prediction']);self.assertEqual(d['attempts'],2)

    def test_graph_retrieval_returns_real_source_linked_paths(self):
        from demo.domain_knowledge import retrieve_domain
        from demo.knowledge import EDGES
        value=retrieve_domain('diabetes','HbA1c phản ánh điều gì?','kg')
        hit=next(d for d in value['sources'] if d['id']=='K05')
        self.assertTrue(hit['graph_paths'])
        for path in hit['graph_paths']:
            self.assertTrue(any(edge[3]==path['source'] for edge in EDGES))
            self.assertEqual(len(path['nodes']),len(set(path['nodes'])))

    def test_rag_dev_test_questions_are_disjoint(self):
        data=json.loads((ROOT/'experiments/fixtures/rag_cases.json').read_text(encoding='utf-8'))
        self.assertFalse(set(r['question'] for r in data['dev'])&set(r['question'] for r in data['test']))

    def test_evidence_downloads_are_limited_to_results(self):
        from app import app
        client=app.test_client()
        for path in ['../app.py','app.py','data/raw/events.csv','C:/Windows/win.ini']:
            self.assertEqual(client.get('/api/evidence',query_string={'path':path}).status_code,400)
        registry=json.loads((ROOT/'assets/models/submission/diabetes/registry.json').read_text())
        response=client.get('/api/evidence',query_string={'path':registry['run']+'/metrics.json'})
        self.assertEqual(response.status_code,200);response.close()

    def test_generated_text_cannot_introduce_unknown_citation(self):
        from unittest.mock import patch,Mock
        from demo.local_llm import generate
        from demo.models import InputError
        fake=Mock();fake.json.return_value={'choices':[{'message':{'content':json.dumps({'answer':'Text [K99]','citations':['K01']})}}]}
        with patch('demo.local_llm.ensure_server'),patch('demo.local_llm.requests.post',return_value=fake):
            with self.assertRaises(InputError):generate('question',[dict(id='K01',title='source',text='source')])


if __name__=='__main__':unittest.main()
