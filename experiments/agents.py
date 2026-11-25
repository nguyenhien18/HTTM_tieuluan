"""Controlled routing and tool-fault evaluation; measures orchestration, not diagnosis."""
import json,math,time
from pathlib import Path
import numpy as np
from .common import ROOT,sha,write_json,finish_run,check_run
from demo.agents import run,verify_prediction,ToolOutputError


def evaluate():
    examples=json.loads((ROOT/'assets/data/form_examples.json').read_text(encoding='utf-8'))
    cases=[]
    for domain in ['diabetes','house']:
        for i,example in enumerate(examples[domain]):
            cases.append(dict(id=f'{domain}_{i}',domain=domain,question='Phân loại hồ sơ tiểu đường' if domain=='diabetes' else 'Dự đoán giá nhà',profile=example['input'],expected='ok',category='valid'))
        base=examples[domain][0]['input']
        missing={k:v for k,v in base.items() if k!=('BMI' if domain=='diabetes' else 'area')}
        invalid={**base,('Age' if domain=='diabetes' else 'area'):99999}
        cases.extend([dict(id=domain+'_missing',domain=domain,question='Dự đoán hồ sơ',profile=missing,expected='need_input',category='invalid'),
                      dict(id=domain+'_invalid',domain=domain,question='Dự đoán hồ sơ',profile=invalid,expected='need_input',category='invalid')])
    for i,date in enumerate(['2023-12-05','2023-01-03']):cases.append(dict(id=f'stock_{i}',domain='stock',question=f'Dự đoán AMZN ngày {date}',profile=None,expected='ok',category='valid'))
    cases.extend([dict(id='stock_outside',domain='stock',question='Dự đoán AMZN ngày 2099-01-01',profile=None,expected='need_input',category='invalid'),
                  dict(id='scope',domain='diabetes',question='Tăng liều thuốc như thế nào?',profile=None,expected='out_of_scope',category='scope'),
                  dict(id='knowledge',domain='diabetes',question='HbA1c phản ánh điều gì?',profile=None,expected='ok',category='knowledge')])
    signature=__import__('hashlib').sha256((json.dumps(cases,sort_keys=True)+sha(Path(__file__))+sha(ROOT/'demo/agents.py')).encode()).hexdigest()[:12]
    folder=ROOT/'runs/agents'/signature
    if (folder/'complete.json').exists():check_run(folder);return folder
    folder.mkdir(parents=True,exist_ok=True);write_json(folder/'cases.json',cases);records=[]
    for case in cases:
        start=time.perf_counter();result=run(case['question'],case['profile'],case['domain'])
        records.append(dict(case_id=case['id'],category=case['category'],expected=case['expected'],passed=result['status']==case['expected'],latency_ms=(time.perf_counter()-start)*1000,response=result))
    # Same valid request, same model. Only the injected first-call output differs.
    faults=[]
    for case in [r for r in cases if r['category']=='valid']:
        for fault in ['raise','nan','range']:
            for variant in ['basic','verified']:
                value=run(case['question'],case['profile'],case['domain'],fault=fault,variant=variant)
                valid=False
                if value.get('prediction') is not None:
                    try:verify_prediction(value['prediction'],case['domain']);valid=True
                    except ToolOutputError:pass
                faults.append(dict(case_id=case['id'],fault=fault,variant=variant,success=valid,status=value['status'],attempts=value.get('attempts',0),trace=value['trace']))
    persistent=run(cases[0]['question'],cases[0]['profile'],cases[0]['domain'],fault='persistent_nan')
    assert persistent['status']=='tool_error' and persistent['prediction'] is None
    summary=[]
    for variant in ['basic','verified']:
        rows=[r for r in faults if r['variant']==variant]
        summary.append(dict(variant=variant,n=len(rows),success_rate=float(np.mean([r['success'] for r in rows]))))
    # JSON cannot represent NaN; only the status/trace of baseline faults is persisted.
    write_json(folder/'routing_predictions.json',records);write_json(folder/'fault_predictions.json',faults)
    metrics=dict(routing_cases=len(records),routing_success=float(np.mean([r['passed'] for r in records])),faults=summary,
                 bounded_persistent_failure=True,limits='Authored, controlled tool faults. Measures routing/recovery, not model accuracy or general language understanding.')
    write_json(folder/'metrics.json',metrics)
    finish_run(folder,dict(evaluation_completed=True,all_routing_cases_pass=all(r['passed'] for r in records)))
    write_json(ROOT/'assets/data/agents_verified.json',dict(run=folder.relative_to(ROOT).as_posix(),**metrics))
    print(json.dumps(metrics,indent=2));return folder


if __name__=='__main__':evaluate()
