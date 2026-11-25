"""Real local-model integration check; writes actual responses for inspection."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
from demo.knowledge import normalize
import re
from demo.config import ROOT
client=app.test_client();rows=[]
for path,payload in [('/api/knowledge',dict(question='HbA1c phản ánh điều gì?',generative=True)),('/api/domain-knowledge',dict(domain='stock',question='AMZN Close ngày 2023-12-05 là bao nhiêu?',method='hybrid',generative=True)),('/api/domain-knowledge',dict(domain='gold',question='Giá vàng ngày 2021-04-07 là bao nhiêu?',method='hybrid',generative=True))]:
 r=client.post(path,json=payload);d=r.get_json();print(json.dumps(dict(path=path,status=r.status_code,answer=d.get('answer'),error=d.get('error'),model=d.get('model'),ms=d.get('generation_ms')),ensure_ascii=False),flush=True)
 rows.append(dict(path=path,request=payload,http_status=r.status_code,response=d))
 if r.status_code!=200:break
out=ROOT/'runs/verification';out.mkdir(parents=True,exist_ok=True)
(out/'real_rag_checks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(rows)==3 and all(r['http_status']==200 and r['response'].get('mode')=='rag' for r in rows)
for row,expected in zip(rows,['K05','S20231205','G20210407']):
 assert expected in row['response']['citations'],row
 assert expected in [s['id'] for s in row['response']['sources']],row
text=normalize(rows[0]['response']['answer'])
assert 'ba thang' in text or '3 thang' in text,rows[0]
for row,expected in zip(rows[1:],[146.880005,1738.05]):
 numbers=[float(n) for n in re.findall(r'\d+(?:\.\d+)?',row['response']['answer'].replace(',','.'))]
 assert any(abs(n-expected)<.011 for n in numbers),row
