"""Small curated, source-linked KB. Extractive mode works offline; generation is explicit."""
from functools import lru_cache
import os, re, unicodedata
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import InputError, predict_tabular, validate

WHO='https://www.who.int/news-room/fact-sheets/detail/diabetes'
SYM='https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes'
TEST='https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis'
PREV='https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-type-2-diabetes'
DOCUMENTS=[
 dict(id='K01',title='Tiểu đường là gì?',text='Tiểu đường là bệnh mạn tính liên quan đến thiếu insulin hoặc cơ thể sử dụng insulin không hiệu quả. Đường huyết cao kéo dài có thể làm tổn thương nhiều cơ quan.',source='WHO · Diabetes',url=WHO,tags=['diabetes','insulin']),
 dict(id='K02',title='Triệu chứng thường gặp',text='Những dấu hiệu thường gặp gồm khát nhiều, tiểu nhiều, mệt mỏi, nhìn mờ, đói nhiều hoặc vết thương lâu lành. Một số người mắc type 2 có biểu hiện nhẹ hoặc khó nhận biết.',source='NIDDK · Symptoms & Causes',url=SYM,tags=['diabetes','symptoms','thirst','urination']),
 dict(id='K03',title='Nguyên nhân tiểu đường type 1',text='Ở type 1, hệ miễn dịch phá hủy các tế bào beta sản xuất insulin. Các yếu tố di truyền và môi trường có thể tham gia vào quá trình này.',source='NIDDK · Symptoms & Causes',url=SYM,tags=['type1','insulin']),
 dict(id='K04',title='Nguyên nhân tiểu đường type 2',text='Type 2 liên quan đến đề kháng insulin và việc tụy không tạo đủ insulin để duy trì đường huyết phù hợp. Hoạt động thể chất ít, thừa cân và tiền sử gia đình là những yếu tố liên quan.',source='NIDDK · Symptoms & Causes',url=SYM,tags=['type2','insulin']),
 dict(id='K05',title='HbA1c phản ánh điều gì?',text='Xét nghiệm A1C, còn gọi là HbA1c, phản ánh mức đường huyết trung bình trong khoảng ba tháng trước. Một số tình trạng sức khỏe có thể ảnh hưởng độ chính xác của xét nghiệm.',source='NIDDK · Tests & Diagnosis',url=TEST,tags=['hba1c','tests']),
 dict(id='K06',title='Xét nghiệm và chẩn đoán',text='Chẩn đoán tiểu đường dựa vào xét nghiệm máu và đánh giá của nhân viên y tế. Các phương pháp gồm A1C, đường huyết lúc đói và một số xét nghiệm khác. Điểm của mô hình học máy không thay thế quy trình này.',source='NIDDK · Tests & Diagnosis',url=TEST,tags=['tests','diabetes']),
 dict(id='K07',title='Phòng ngừa hoặc trì hoãn type 2',text='Lối sống phù hợp, kiểm soát cân nặng và vận động đều đặn có thể giúp phòng ngừa hoặc trì hoãn type 2. Thay đổi cụ thể cần phù hợp với tình trạng sức khỏe của từng người.',source='NIDDK · Preventing Type 2',url=PREV,tags=['prevention','type2']),
 dict(id='K08',title='Ăn uống và hoạt động thể chất',text='Các biện pháp được đề cập trong phòng ngừa type 2 gồm ăn uống lành mạnh, điều chỉnh lượng năng lượng và tăng vận động. Nên trao đổi với chuyên gia y tế về thay đổi phù hợp.',source='NIDDK · Preventing Type 2',url=PREV,tags=['prevention','activity']),
 dict(id='K09',title='Biến chứng có thể gặp',text='Tiểu đường có thể ảnh hưởng tim mạch, thận, mắt và thần kinh. Theo dõi và quản lý bệnh giúp hạn chế hậu quả. Kho kiến thức demo chỉ cung cấp thông tin tổng quan.',source='WHO · Diabetes',url=WHO,tags=['complications','diabetes']),
 dict(id='K10',title='Điểm mô hình và nhóm tuổi BRFSS',text='Mô hình trong demo học từ 21 đặc trưng khảo sát BRFSS. Age là mã nhóm tuổi từ 1 đến 13, không phải số tuổi trực tiếp. Điểm phân loại chưa được hiệu chỉnh thành xác suất lâm sàng.',source='Schema dữ liệu A02/A03',url='https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset',tags=['model','brfss'])
]
NODES=[('diabetes','Tiểu đường','disease'),('type1','Type 1','disease'),('type2','Type 2','disease'),('insulin','Insulin','concept'),('symptoms','Triệu chứng','concept'),('thirst','Khát nhiều','symptom'),('urination','Tiểu nhiều','symptom'),('tests','Xét nghiệm','test'),('hba1c','HbA1c','test'),('prevention','Phòng ngừa','concept'),('activity','Vận động','concept'),('complications','Biến chứng','concept'),('model','Mô hình MLP','model'),('brfss','Khảo sát BRFSS','data')]
EDGES=[('diabetes','type1','gồm loại','K03'),('diabetes','type2','gồm loại','K04'),('diabetes','insulin','liên quan','K01'),('diabetes','symptoms','có thể biểu hiện','K02'),('symptoms','thirst','gồm','K02'),('symptoms','urination','gồm','K02'),('type1','insulin','thiếu sản xuất','K03'),('type2','insulin','sử dụng kém hiệu quả','K04'),('diabetes','tests','đánh giá bằng','K06'),('tests','hba1c','gồm','K05'),('type2','prevention','có thể trì hoãn','K07'),('prevention','activity','bao gồm','K08'),('diabetes','complications','có thể dẫn tới','K09'),('model','brfss','học từ','K10')]

def normalize(text):
    text=str(text).lower().replace('đ','d')
    text=''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))
    text=re.sub(r'(type|loai|tuy[pb])\s*([12])',r'type\2',text)
    text=text.replace('dai thao duong','tieu duong')
    text=text.replace('a1c','hba1c') if 'hba1c' not in text else text
    return text

@lru_cache(maxsize=1)
def index():
    vectorizer=TfidfVectorizer(ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',sublinear_tf=True,stop_words=['la','gi','the','nao','co','khong','toi','cho','cua','ve','duoc','voi','cac','nhung','va','thi','bao','nhieu','trong','khi','nhu','sao','lam','can','mot','hay','de','nguoi','nhung','nhung','nay','do'])
    matrix=vectorizer.fit_transform([normalize(d['title']+' '+d['text']) for d in DOCUMENTS])
    return vectorizer,matrix

def retrieve(question,k=3):
    if not isinstance(question,str) or not question.strip():raise InputError('Bạn hãy nhập câu hỏi.')
    if len(question)>1500:raise InputError('Câu hỏi cần dưới 1.500 ký tự.')
    query=normalize(question);v,m=index();scores=(m@v.transform([query]).T).toarray().ravel()
    if 'type1' in query:scores=np.array([s if 'type2' not in d['tags'] else 0 for s,d in zip(scores,DOCUMENTS)])
    if 'type2' in query:scores=np.array([s if 'type1' not in d['tags'] else 0 for s,d in zip(scores,DOCUMENTS)])
    order=np.argsort(-scores,kind='stable')[:k]
    return [{**DOCUMENTS[i],'score':float(scores[i])} for i in order if scores[i]>=.10]

def graph(highlight=None):
    ids=set(highlight or [])
    tags={tag for d in DOCUMENTS if d['id'] in ids for tag in d['tags']}
    return dict(nodes=[dict(id=i,label=l,kind=k,highlight=i in tags) for i,l,k in NODES],edges=[dict(source=s,target=t,label=l,document=d,highlight=d in ids) for s,t,l,d in EDGES])

def ollama_status():
    # Backwards-compatible endpoint name; generation now uses the project-local runtime.
    from .local_llm import status
    return status()


def answer(question,generative=False,model=None):
    hits=retrieve(question)
    if not hits:return dict(mode='extractive',answer='Kho kiến thức hiện chưa có bằng chứng phù hợp cho câu hỏi này.',sources=[],graph=graph(),status='no_evidence')
    query=normalize(question)
    if any(x in query for x in ['ke don','lieu thuoc','tang lieu','giam lieu','tiem bao nhieu','chan doan cho toi']):
        return dict(mode='extractive',status='out_of_scope',answer='Demo hỗ trợ học kiến thức tổng quan, không xử lý chẩn đoán cá nhân hoặc liều thuốc.',sources=[],graph=graph())
    generated={}
    if generative:
        from .local_llm import generate
        generated=generate(question,hits)
        text=generated.pop('answer');mode='rag'
    else:text='\n\n'.join(f"{d['text']} [{d['id']}]" for d in hits[:2]);mode='extractive'
    return dict(mode=mode,answer=text,sources=hits,graph=graph([d['id'] for d in hits]),status='ok',**generated)


def run_agents(question,profile=None,domain=None):
    from .agents import run
    return run(question,profile,domain)
