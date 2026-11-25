"""Three-domain retrieval with source records and inspectable graph evidence.

No generated answers are claimed in offline mode. TF-IDF is lexical retrieval,
KG uses entity matching, and hybrid uses reciprocal rank fusion.
"""
from functools import lru_cache
import re
from datetime import date
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import DATA
from .knowledge import DOCUMENTS, normalize
from .models import InputError

DOMAINS={
 'diabetes':dict(title='Tiểu đường',source='Kho kiến thức WHO/NIDDK và schema BRFSS',unit='Không áp dụng',note='Thông tin tổng quan từ tài liệu có nguồn; không dùng để chẩn đoán.'),
 'stock':dict(title='Cổ phiếu Amazon',source='AMZN.csv · dữ liệu A06',unit='Thang giá Close của CSV',note='250 phiên cuối trong CSV lịch sử. Đây là Amazon (AMZN), không phải FPT và không phải giá hiện tại.'),
 'gold':dict(title='Giá vàng',source='gold_price.csv · D:/RNN/data',unit='Đơn vị price trong CSV; nguồn không khai báo đơn vị',note='250 quan sát cuối có giá hợp lệ, không nội suy. Không tự quy đổi sang USD/oz hoặc giá vàng Việt Nam.')
}

@lru_cache(maxsize=3)
def corpus(domain):
 if not isinstance(domain,str) or domain not in DOMAINS:raise InputError('Miền kiến thức không hợp lệ.')
 if domain=='diabetes':
  return [{**d,'entities':d['tags']} for d in DOCUMENTS]
 filename='AMZN.csv' if domain=='stock' else 'gold_price.csv'
 df=pd.read_csv(DATA/filename);date='Date' if domain=='stock' else 'date';value='Close' if domain=='stock' else 'price'
 df[date]=pd.to_datetime(df[date],errors='coerce');df[value]=pd.to_numeric(df[value],errors='coerce')
 df=df.dropna(subset=[date,value]).sort_values(date).drop_duplicates(date,keep='last').tail(250)
 docs=[]
 for row in df.to_dict('records'):
  day=str(row[date].date());v=float(row[value]);is_stock=domain=='stock'
  title=f'AMZN · Close ngày {day}' if is_stock else f'Giá vàng · price ngày {day}'
  text=f'{title}: {v:.6g}. '
  fields={'Close':v} if is_stock else {'price':v}
  if is_stock:
   for col in ['Open','High','Low','Volume']:
    if col in row and pd.notna(row[col]):fields[col]=float(row[col])
   text+='; '.join(f'{k} = {val:.6g}' for k,val in fields.items())+'. '
  text+=DOMAINS[domain]['unit']+'. Dữ liệu lịch sử.'
  docs.append(dict(id=('S' if is_stock else 'G')+day.replace('-',''),title=title,text=text,source=DOMAINS[domain]['source'],url='',date=day,fields=fields,entities=[('amzn' if is_stock else 'gold'),day,*[k.lower() for k in fields]]))
 return docs

@lru_cache(maxsize=3)
def index(domain):
 docs=corpus(domain)
 v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True)
 m=v.fit_transform([normalize(d['title']+' '+d['text']) for d in docs])
 return v,m

def subgraph(docs):
 nodes={};edges=[]
 for d in docs:
  nodes[d['id']]=dict(id=d['id'],label=d['title'],kind='Document')
  for entity in d['entities']:
   key='entity:'+entity;nodes[key]=dict(id=key,label=entity,kind='Entity')
   edges.append(dict(source=d['id'],target=key,relation='MENTIONS',document=d['id']))
  for field,value in d.get('fields',{}).items():
   key=d['id']+':'+field;nodes[key]=dict(id=key,label=f'{field} = {value:.6g}',kind='Observation')
   edges.append(dict(source=d['id'],target=key,relation='HAS_VALUE',document=d['id']))
 return dict(nodes=list(nodes.values()),edges=edges)

def domain_config(domain):
 docs=corpus(domain)
 questions=(['HbA1c phản ánh điều gì?','Nguyên nhân tiểu đường type 1 là gì?','Phòng ngừa type 2 như thế nào?'] if domain=='diabetes' else [f'{"AMZN Close" if domain=="stock" else "Giá vàng price"} ngày {d["date"]} là bao nhiêu?' for d in [docs[-1],docs[-10],docs[-30]]])
 full=subgraph(docs)
 return dict(**DOMAINS[domain],domain=domain,documents=len(docs),nodes=len(full['nodes']),edges=len(full['edges']),examples=questions,first_date=docs[0].get('date'),last_date=docs[-1].get('date'))

def retrieve_domain(domain,question,method='hybrid'):
 if not isinstance(domain,str) or domain not in DOMAINS:raise InputError('Miền kiến thức không hợp lệ.')
 if not isinstance(question,str) or not question.strip() or len(question)>1500:raise InputError('Nhập câu hỏi từ 1 đến 1.500 ký tự.')
 if method not in ['tfidf','bm25','kg','hybrid']:raise InputError('Phương pháp không hợp lệ.')
 docs=corpus(domain);q=normalize(question);v,m=index(domain)
 if domain=='diabetes' and any(x in q for x in ['ke don','lieu thuoc','tiem bao nhieu','chan doan cho toi']):
  return dict(domain=domain,method=method,mode='retrieval',answer='Demo chỉ hỗ trợ kiến thức tổng quan; không xử lý chẩn đoán cá nhân hoặc liều thuốc.',sources=[],graph=subgraph([]),status='out_of_scope',note='Yêu cầu ngoài phạm vi demo.',config=domain_config(domain))
 scores=(m@v.transform([q]).T).toarray().ravel()
 entities=[sum(1 for e in d['entities'] if normalize(e) in q) for d in docs]
 # English schema fields have Vietnamese aliases, shared by every record.
 # A date narrows candidates before ranking; an absent date never fabricates a value.
 q=re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',lambda m:f'{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}',q)
 dates=list(dict.fromkeys(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',q)))
 for day in dates:
  try:date.fromisoformat(day)
  except ValueError:raise InputError('Ngày không hợp lệ: '+day)
 if domain!='diabetes' and len(dates)==2 and not any(x in q for x in ['hom nay','hien tai','moi nhat','hba1c','insulin','tieu duong']):
  if dates[0]>dates[1]:raise InputError('Ngày bắt đầu cần trước hoặc bằng ngày kết thúc.')
  records=[d for d in docs if dates[0]<=d['date']<=dates[1]]
  if records and docs[0]['date']<=dates[0]<=dates[1]<=docs[-1]['date']:
   field='Close' if domain=='stock' else 'price';values=np.array([d['fields'][field] for d in records])
   fields=dict(count=len(values),mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()))
   text=f'Thống kê {field} từ {dates[0]} đến {dates[1]}: {len(values)} quan sát, trung bình {fields["mean"]:.6g}, nhỏ nhất {fields["minimum"]:.6g}, lớn nhất {fields["maximum"]:.6g}. {DOMAINS[domain]["unit"]}.'
   source=dict(id='STAT_'+domain+'_'+dates[0].replace('-','')+'_'+dates[1].replace('-',''),title=f'Thống kê {field} theo khoảng ngày',text=text,source=DOMAINS[domain]['source'],url='',fields=fields,entities=[domain,field,dates[0],dates[1]],score=1.,entity_matches=2,record_ids=[d['id'] for d in records])
   return dict(domain=domain,method='csv_stats',mode='statistics',answer=text,sources=[source],graph=subgraph([source]),status='ok',note='Lọc và tính trực tiếp trên các bản ghi của corpus; không yêu cầu LLM tính số. Ngày không giao dịch hoặc giá thiếu không được nội suy.',config=domain_config(domain))
 candidates=[i for i,d in enumerate(docs) if not dates or d.get('date') in dates]
 outside=(any(x in q for x in ['hom nay','hien tai','moi nhat']) and domain!='diabetes')
 if domain=='diabetes' and any(x in q for x in ['gia vang','co phieu','amzn']):outside=True
 if domain!='diabetes' and any(x in q for x in ['hba1c','insulin','tieu duong']):outside=True
 if domain=='stock' and 'vang' in q:outside=True
 if domain=='gold' and any(x in q for x in ['amzn','co phieu']):outside=True
 if domain!='diabetes' and not dates:
  # Financial records need a date, not an arbitrary near-identical row.
  outside=True
 if outside:candidates=[]
 if 'type1' in q:candidates=[i for i in candidates if 'type2' not in docs[i]['entities']]
 if 'type2' in q:candidates=[i for i in candidates if 'type1' not in docs[i]['entities']]
 from .graph_retrieval import bm25,graph_paths
 bm_scores=bm25(q,docs);walks=graph_paths(domain,q,docs)
 lexical=sorted(candidates,key=lambda i:(-scores[i],i))
 bm_rank=sorted([i for i in candidates if bm_scores[i]>0],key=lambda i:(-bm_scores[i],i))
 kg=sorted([i for i in candidates if i in walks],key=lambda i:(-walks[i]['score'],i))
 ranks_a={i:r+1 for r,i in enumerate(lexical) if scores[i]>=.1}
 ranks_b={i:r+1 for r,i in enumerate(kg)}
 combined={i:1/(60+ranks_a[i]) if i in ranks_a else 0 for i in set(ranks_a)|set(ranks_b)}
 for i in ranks_b:combined[i]=combined.get(i,0)+1/(60+ranks_b[i])
 for r,i in enumerate(bm_rank,1):combined[i]=combined.get(i,0)+1/(60+r)
 order=(list(ranks_a) if method=='tfidf' else bm_rank if method=='bm25' else kg if method=='kg' else sorted(combined,key=lambda i:(-combined[i],i)))[:3]
 hits=[{**docs[i],'score':float(scores[i] if method=='tfidf' else bm_scores[i] if method=='bm25' else walks[i]['score'] if method=='kg' else combined[i]),'tfidf_score':float(scores[i]),'bm25_score':float(bm_scores[i]),'entity_matches':entities[i],'graph_paths':walks.get(i,{}).get('paths',[])} for i in order]
 text=('\n\n'.join(f'[{d["id"]}] {d["text"]}' for d in hits) if hits else ('Hãy dùng ngày YYYY-MM-DD có trong corpus lịch sử. Demo không có giá hiện tại.' if domain!='diabetes' else 'Không có bằng chứng phù hợp trong miền đã chọn.'))
 return dict(domain=domain,method=method,mode='retrieval',answer=text,sources=hits,graph=subgraph(hits),status='ok' if hits else 'no_evidence',note='Trả lời trích xuất từ bằng chứng. TF-IDF là truy xuất từ vựng; chế độ này không chạy LLM.',config=domain_config(domain))

def evaluate_domains():
 # Small authored regression suite; not an independent benchmark of general QA.
 questions=[]
 for q,target in [('HbA1c phản ánh điều gì?','K05'),('Nguyên nhân type1 insulin','K03'),('Nguyên nhân type2 insulin','K04'),('Triệu chứng khát nhiều','K02'),('Biến chứng tiểu đường','K09'),('BRFSS Age nhóm tuổi','K10'),('Phòng ngừa type2 vận động','K07')]:questions.append(('diabetes',q,target))
 for domain in ['stock','gold']:
  docs=corpus(domain)
  for i in [-1,-5,-20,-40,-80,-120,-200]:
   d=docs[i];questions.append((domain,f'{"AMZN Close" if domain=="stock" else "Giá vàng price"} ngày {d["date"]} là bao nhiêu?',d['id']))
 rows=[];details=[]
 for method in ['kg','tfidf','hybrid']:
  reciprocal=[];hit=[]
  for domain,q,target in questions:
   found=retrieve_domain(domain,q,method)['sources'];ids=[d['id'] for d in found];rank=ids.index(target)+1 if target in ids else None
   hit.append(int(rank is not None));reciprocal.append(1/rank if rank else 0)
   details.append(dict(method=method,domain=domain,question=q,expected=target,retrieved=ids,rank=rank))
  rows.append(dict(method=method,questions=len(questions),hit_at_3=float(np.mean(hit)),mrr_at_3=float(np.mean(reciprocal))))
 outside=[]
 for domain,q in [('stock','AMZN hôm nay bao nhiêu?'),('gold','Giá vàng ngày 2099-01-01'),('diabetes','Giá vàng hôm nay')]:
  r=retrieve_domain(domain,q);outside.append(dict(domain=domain,question=q,status=r['status']))
 return dict(rows=rows,details=details,out_of_scope=outside,note='21 câu kiểm tra tự soạn (7 mỗi miền), có câu hỏi ngày trùng khóa bản ghi. Dùng kiểm tra tính đúng của pipeline; không đại diện hiệu quả hỏi đáp tổng quát và không đánh giá LLM.')
