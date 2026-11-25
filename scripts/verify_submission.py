"""Verify packaged data, frozen experiment artifacts and application regression tests.

Does not retrain. It reloads tabular checkpoints on TEST and checks full CNN/sequence
prediction artifacts. Use reproduce.py to rebuild experiment outputs.
"""
import argparse,json,sys,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from experiments.common import sha,write_json,check_run


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest():
    skip_parts={'__pycache__','.git','logs','work'}
    extensions={'.py','.js','.css','.html','.json','.csv','.npz','.pth','.keras','.joblib','.pkl','.md','.cmd','.txt','.gguf','.dll','.exe','.ipynb','.svg','.png'}
    entries={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in extensions:continue
        rel=p.relative_to(ROOT)
        if set(rel.parts)&skip_parts or rel.parts[:2]==('runs','verification'):continue
        entries[rel.as_posix()]=sha(p)
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--with-llm',action='store_true');args=parser.parse_args()
    command=[sys.executable,'-X','utf8',str(ROOT/'scripts/verify_demo.py')]
    if args.with_llm:command.append('--with-llm')
    subprocess.run(command,cwd=ROOT,check=True)
    tests=load(ROOT/'runs/verification/test_summary.json');assert tests['passed']
    for source in load(ROOT/'data/sources.json'):
        assert sha(ROOT/source['file'])==source['sha256'],source['file']
    for dataset in ['mnist','fashion','cifar10']:
        meta=load(ROOT/f'data/images/{dataset}.json');assert sha(ROOT/f'data/images/{dataset}.npz')==meta['sha256']
    from experiments.tabular import evaluate
    tabular=[]
    for task in ['diabetes','house','ecommerce']:
        registry=load(ROOT/f'assets/models/submission/{task}/registry.json');run=registry['run'];evaluate(run);check_run(ROOT/run)
        tabular.append(dict(task=task,run=run,split=load(ROOT/run/'split.json'),metrics=load(ROOT/run/'metrics.json'),selection=load(ROOT/run/'selection.json')))
    cnn=[]
    for dataset in ['mnist','fashion','cifar10']:
        source=np.load(ROOT/f'data/images/{dataset}.npz',allow_pickle=False);y=source['y_test']
        for row in load(ROOT/f'assets/data/cnn_verified_{dataset}.json'):
            folder=ROOT/row['run'];check_run(folder);cfg=load(folder/'config.json')
            assert sha(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256']
            records=pd.read_csv(folder/'predictions.csv');np.testing.assert_array_equal(records.actual,y)
            assert len(records)==10000
            assert row['majority_baseline_class']==int(np.bincount(source['y_train']).argmax())
            assert abs(row['majority_baseline_accuracy']-float(np.mean(y==row['majority_baseline_class'])))<1e-10
            probabilities=records[[f'prob_{i}' for i in range(10)]].to_numpy()
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-4)
            assert abs(accuracy_score(y,records.prediction)-row['Accuracy'])<1e-10
            assert abs(f1_score(y,records.prediction,average='macro')-row['Macro_F1'])<1e-10
            cnn.append(row)
    assert len(cnn)==13
    sequences=load(ROOT/'assets/data/sequences_verified.json');assert len(sequences)==8
    for row in sequences:
        folder=ROOT/row['run'];meta=check_run(folder);cfg=load(folder/'config.json')
        assert sha(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256']
        assert meta['full_test_matches_saved'] and meta['test_samples']==(338 if row['task']=='stock' else 30000)
    rag=load(ROOT/'assets/data/rag_verified.json');rag_folder=ROOT/rag['run'];rag_meta=check_run(rag_folder);rag_cfg=load(rag_folder/'config.json')
    assert rag_meta['generation_executed'] and rag_meta['test_cases']==22
    assert sha(ROOT/rag_cfg['case_file'])==rag_cfg['case_sha256']
    predictions=load(rag_folder/'generation_predictions.json');assert len(predictions)==22
    assert all(r['citation_ids_valid'] for r in predictions if r['mode']=='rag')
    agents=load(ROOT/'assets/data/agents_verified.json');check_run(ROOT/agents['run'])
    assert agents['routing_success']==1 and agents['bounded_persistent_failure']
    assert next(r for r in agents['faults'] if r['variant']=='verified')['success_rate']==1
    rl=load(ROOT/'runs/verification/rl_benchmark.json')
    assert len(rl['rows'])==9
    from demo.reinforcement import run_rl
    # A cached run still checks its source/config/Q-table identity.
    for row in rl['rows']:
        result=run_rl(row['task'],dict(seed=row['seed'],**({'beta':row['beta']} if row['beta'] is not None else {})))
        assert result['run_id']==row['run_id']
    training_checks=[]
    for path in sorted((ROOT/'runs/cnn_training').glob('*/*/*/complete.json')):
        folder=path.parent;check_run(folder);check_run(folder/'evaluation')
        cfg=load(folder/'config.json')
        training_checks.append(dict(run=folder.relative_to(ROOT).as_posix(),dataset=cfg['dataset'],model=cfg['model'],epochs=cfg['epochs'],
                                    purpose='Training/evaluation smoke run; original demo checkpoint retained.',metrics=load(folder/'evaluation/metrics.json')))
    assert len(training_checks)>=2
    entries=manifest();write_json(ROOT/'runs/verification/submission_manifest.json',dict(files=entries))
    report=dict(passed=True,verified_at=datetime.now(timezone.utc).isoformat(),tests=tests,
                manifest_files=len(entries),tabular=tabular,cnn=cnn,sequences=sequences,rag=rag,agents=agents,rl=rl,training_checks=training_checks,
                quality_scope='Passing means the defined integrity/protocol/functional checks passed. It does not imply every model or generated answer is correct.',
                limitations=['Tabular uses one training seed.','Original CNN framework training budgets differ.',
                             'RAG cases are authored; fixed fact checks are not full semantic faithfulness.',
                             'AMZN models lose to persistence on the saved split.',
                             'Gold original provider/unit and exact upstream e-commerce generation remain unverified.',
                             'RL environments are simulations; maze uses a fixed map.'])
    write_json(ROOT/'runs/verification/submission.json',report)
    print(json.dumps(dict(passed=True,tests=tests['tests'],manifest_files=len(entries),tabular_models=6,cnn_models=len(cnn),sequence_models=len(sequences),rag_generated=sum(r['mode']=='rag' for r in predictions),agent_cases=agents['routing_cases']),indent=2))


if __name__=='__main__':main()
