"""Rule-based router, typed tools, output verifier and one bounded retry.

Fault hooks are private evaluation inputs and are never accepted by the web API.
"""
import math,re
from datetime import datetime
from .models import InputError,validate,predict_tabular,predict_stock
from .knowledge import normalize,answer


class ToolOutputError(ValueError):pass


def verify_prediction(value,domain):
    if not isinstance(value,dict):raise ToolOutputError('Tool output must be an object')
    if domain=='stock':
        for key in ['predicted','actual','baseline']:
            v=value.get(key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ToolOutputError('Invalid '+key)
        if not isinstance(value.get('artifact'),dict):raise ToolOutputError('Missing checkpoint identity')
    else:
        p=value.get('prediction');s=value.get('score')
        if isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p):raise ToolOutputError('Invalid prediction')
        if domain=='house':
            if p<=0:raise ToolOutputError('Nonpositive house price')
        elif p not in [0,1] or isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or not 0<=s<=1:
            raise ToolOutputError('Invalid class/score')
        if not value.get('artifacts'):raise ToolOutputError('Missing checkpoint identity')
    return value


def run(question,profile=None,domain=None,*,fault=None,variant='verified'):
    if not isinstance(question,str) or not question.strip() or len(question)>1500:raise InputError('Nhập yêu cầu từ 1 đến 1.500 ký tự.')
    if domain not in [None,'diabetes','house','stock','gold']:raise InputError('Miền Agent không hợp lệ.')
    trace=[]
    def event(sender,receiver,action,detail,status='ok'):
        trace.append(dict(step=len(trace)+1,sender=sender,receiver=receiver,action=action,detail=detail,status=status))
    q=normalize(question)
    if any(t in q for t in ['ke don','lieu thuoc','tang lieu','giam lieu','uong bao nhieu','tiem bao nhieu','co nen mua','co nen ban']):
        event('RouterAgent','User','Giới hạn phạm vi','Yêu cầu kê đơn/liều thuốc hoặc khuyến nghị mua bán.','out_of_scope')
        return dict(route='out_of_scope',status='out_of_scope',answer='Demo không kê đơn, chỉnh liều hoặc khuyến nghị mua bán. Hỗ trợ kiến thức có nguồn và minh họa mô hình.',trace=trace,sources=[],prediction=None)
    inferred=('house' if any(t in q for t in ['gia nha','can nha','can ho','dien tich']) else
              'stock' if any(t in q for t in ['amzn','amazon','co phieu']) else
              'gold' if 'vang' in q else 'diabetes')
    selected=domain or inferred
    wants_prediction=profile is not None or any(t in q for t in ['du doan','phan loai','nguy co cua toi','ho so','du bao'])
    wants_knowledge=not wants_prediction or any(t in q for t in ['giai thich','tai sao','kien thuc','la gi','hba1c','trieu chung','phong ngua'])
    route='combined' if wants_prediction and wants_knowledge else 'prediction' if wants_prediction else 'knowledge'
    event('User','RouterAgent','Chọn tuyến',f'{selected} / {route}')
    prediction=None;tool=None;attempts=0;checked=None
    if wants_prediction:
        try:
            if selected in ['diabetes','house']:checked=validate(selected,profile or {})
            elif selected=='stock':
                if profile is not None and not isinstance(profile,dict):raise InputError('Hồ sơ cần là một đối tượng JSON.')
                dates=re.findall(r'\b\d{4}-\d{2}-\d{2}\b',question)
                checked=dict(date=(profile or {}).get('date',dates[0] if dates else None))
                if not checked['date']:raise InputError('Cần ngày YYYY-MM-DD trong tập test AMZN lịch sử.')
                try:datetime.strptime(checked['date'],'%Y-%m-%d')
                except (ValueError,TypeError):raise InputError('Ngày cần có dạng YYYY-MM-DD hợp lệ.')
                from .models import stock_data
                data=stock_data();valid=set(data.Date.iloc[int(len(data)*.85):].dt.strftime('%Y-%m-%d'))
                if checked['date'] not in valid:raise InputError('Ngày nằm ngoài tập test AMZN lịch sử.')
            else:raise InputError('Miền giá vàng chỉ hỗ trợ tra cứu lịch sử có nguồn.')
        except InputError as exc:
            event('RouterAgent','ValidationAgent','Kiểm tra đầu vào',str(exc),'need_input')
            return dict(domain=selected,route=route,status='need_input',answer=str(exc),trace=trace,sources=[],prediction=None)
        event('RouterAgent','ValidationAgent','Kiểm tra đầu vào','Đúng schema, đơn vị và miền dữ liệu.')
        tool={'diabetes':'diabetes_predict','house':'house_predict','stock':'stock_predict'}[selected]
        for attempt in range(1,3 if variant=='verified' else 2):
            attempts=attempt
            try:
                event('ValidationAgent','SpecialistAgent','Gọi công cụ',f'{tool}, lần {attempt}')
                if fault=='raise' and attempt==1:raise TimeoutError('Injected evaluation timeout')
                if selected=='stock':value=predict_stock(checked['date'],'keras','gru')
                else:value=predict_tabular(selected,'mlp' if selected=='diabetes' else 'ml',checked)['results'][0]
                if attempt==1 and fault in ['nan','range','persistent_nan']:
                    value={**value,('predicted' if selected=='stock' else 'prediction'):float('nan') if fault!='range' else -1}
                if fault=='persistent_nan':value={**value,('predicted' if selected=='stock' else 'prediction'):float('nan')}
                if variant=='verified':verify_prediction(value,selected)
                prediction=value
                event('SpecialistAgent','VerifierAgent','Kiểm tra đầu ra','Giá trị hữu hạn, miền hợp lệ, có danh tính checkpoint.' if variant=='verified' else 'Biến thể đối chứng không có kiểm định đầu ra.')
                break
            except (ToolOutputError,InputError,TimeoutError,RuntimeError,ValueError) as exc:
                event('SpecialistAgent','VerifierAgent','Lỗi công cụ',str(exc),'tool_error')
                if variant=='verified' and attempt==1:
                    event('VerifierAgent','SpecialistAgent','Thử lại công cụ','Giữ nguyên tham số đã kiểm tra; tối đa một lần thử lại.','retry')
        if prediction is None:
            return dict(domain=selected,route=route,status='tool_error',answer='Công cụ chưa trả kết quả hợp lệ sau số lần thử cho phép.',trace=trace,sources=[],prediction=None,tool=tool,attempts=attempts)
    if wants_knowledge:
        if selected=='diabetes':evidence=answer(question)
        elif selected in ['stock','gold']:
            from .domain_knowledge import retrieve_domain
            evidence=retrieve_domain(selected,question,'hybrid')
        else:evidence=dict(answer='Miền giá nhà hỗ trợ mô hình dự đoán từ hồ sơ; chưa có kho văn bản để giải thích nguyên nhân của từng dự đoán.',sources=[],status='no_evidence')
        event('RouterAgent','KnowledgeAgent','Truy xuất bằng chứng',f'{len(evidence["sources"])} nguồn; {evidence.get("status","ok")}')
    else:evidence=dict(answer='',sources=[],status='ok')
    parts=[]
    if prediction:
        if selected=='stock':parts.append(f'GRU dự đoán Close lịch sử ngày {prediction["date"]}: {prediction["predicted"]:.4f}; giá thực {prediction["actual"]:.4f}; baseline {prediction["baseline"]:.4f}.')
        else:parts.append(f'Công cụ trả {prediction["label"]}.'+(f' Điểm lớp dương {prediction["score"]:.3f}; chưa hiệu chuẩn thành xác suất chẩn đoán.' if prediction['score'] is not None else ''))
    if evidence['answer']:parts.append(evidence['answer'])
    event('VerifierAgent','SynthesisAgent','Giữ kết quả và nguồn','Tổng hợp theo mẫu; không tự tạo số liệu hoặc suy nguyên nhân từ điểm model.')
    event('SynthesisAgent','User','Trả phản hồi','Kèm trace và artifact của công cụ.')
    return dict(domain=selected,route=route,status='ok' if prediction else evidence.get('status','ok'),answer='\n\n'.join(parts),
                trace=trace,sources=evidence['sources'],prediction=prediction,tool=tool,attempts=attempts,mode='rule_based_verified',variant=variant)
