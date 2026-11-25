from pathlib import Path
import os, time, json, logging, sys, uuid, re, csv
from io import StringIO
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template, send_from_directory, Response
from werkzeug.exceptions import HTTPException
from demo.config import ROOT, ASSETS, DATA, TASKS, CNN, read_json
from demo.models import InputError, predict_tabular, predict_image, predict_customer, predict_stock, stock_data
from demo.knowledge import DOCUMENTS, graph, answer, ollama_status, run_agents
from demo.pages import PAGES, GROUPS
from demo.domain_knowledge import domain_config, retrieve_domain, evaluate_domains
from demo.reinforcement import run_rl
from functools import partial

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=7*1024*1024
app.json.ensure_ascii=False

def payload():
    data=request.get_json(silent=True)
    if not isinstance(data,dict):raise InputError('Nội dung gửi lên cần là JSON hợp lệ.')
    for name in ['generative','drawing']:
        if name in data and not isinstance(data[name],bool):raise InputError(name+': cần là true hoặc false.')
    return data

def result(data,start):
    data.setdefault('run_id',uuid.uuid4().hex)
    data.setdefault('status','ok')
    data.update(latency_ms=round((time.perf_counter()-start)*1000,2),timestamp=datetime.now(timezone.utc).isoformat())
    return jsonify(data)

@app.get('/')
def home():return render_template('index.html',page=None,pages=PAGES,groups=GROUPS)

def demo_page(spec):
    return render_template('pages/'+spec['id']+'.html',page=spec,pages=PAGES,groups=GROUPS)

for page_spec in PAGES:
    app.add_url_rule(page_spec['path'],endpoint='demo_'+page_spec['id'],view_func=partial(demo_page,page_spec))

@app.get('/api/health')
def health():return jsonify(status='ok',app='HTTM Demo Studio',version='3.0',pages=len(PAGES))

@app.get('/api/config')
def config():
    from demo.submission import current_metrics
    stock=stock_data();test=stock.iloc[int(len(stock)*.85):]
    return jsonify(tasks=TASKS,cnn={k:{**v,'models':[dict(id=i,label=l) for i,l in v['models']]} for k,v in CNN.items()},examples=read_json(DATA/'form_examples.json'),gallery=read_json(DATA/'image_gallery.json'),customer_examples=read_json(DATA/'customer_examples.json'),stock_dates=test.Date.dt.strftime('%Y-%m-%d').tolist(),metrics=current_metrics(read_json(DATA/'experiment_metrics.json')),knowledge=DOCUMENTS,graph=graph())

@app.get('/api/submission')
def submission_status():
    path=ROOT/'runs/verification/submission.json'
    return jsonify(read_json(path) if path.is_file() else dict(passed=False,status='not_verified',note='Chạy KIEM_TRA_NOP_BAI.cmd để tạo kết quả kiểm chứng.'))

@app.get('/api/evidence')
def evidence_file():
    relative=request.args.get('path','')
    if not relative or '\\' in relative or '..' in Path(relative).parts:raise InputError('Đường dẫn kết quả không hợp lệ.')
    path=(ROOT/relative).resolve()
    try:path.relative_to((ROOT/'runs').resolve())
    except ValueError:raise InputError('Chỉ tải được kết quả thực nghiệm đã công khai.')
    if path.suffix not in ['.json','.csv','.png','.svg'] or not path.is_file():raise InputError('Tệp kết quả không tồn tại hoặc không được hỗ trợ.')
    return send_from_directory(path.parent,path.name,as_attachment=request.args.get('download')=='1')

@app.get('/samples/<path:filename>')
def sample(filename):return send_from_directory(ASSETS/'samples',filename)

@app.post('/api/tabular')
def tabular():
    start=time.perf_counter();d=payload();return result(predict_tabular(d.get('task'),d.get('engine','ml'),d.get('input')),start)

@app.post('/api/cnn')
def cnn():
    start=time.perf_counter();d=payload();return result(predict_image(d.get('dataset'),d.get('model'),d.get('sample'),d.get('image'),bool(d.get('drawing',False))),start)

@app.post('/api/rnn/customer')
def customer():
    start=time.perf_counter();d=payload();return result(predict_customer(d.get('events'),d.get('framework','pytorch'),d.get('example'),d.get('architecture','rnn')),start)

@app.post('/api/rnn/stock')
def stock():
    start=time.perf_counter();d=payload();return result(predict_stock(d.get('date'),d.get('framework','pytorch'),d.get('architecture','rnn')),start)

@app.get('/api/ollama')
@app.get('/api/llm')
def ollama():return jsonify(ollama_status())

@app.post('/api/rl/<task>')
def reinforcement(task):
    start=time.perf_counter();return result(run_rl(task,payload()),start)

@app.get('/api/rl/runs/<run_id>/<filename>')
def rl_artifact(run_id,filename):
    if not re.fullmatch(r'rl-(maze|pricing)-[a-f0-9]{12}',run_id) or filename not in ['qtable.npz','result.json','history.csv']:raise InputError('Artifact không hợp lệ.')
    folder=ROOT/'runs/rl'/run_id
    if not (folder/'result.json').is_file():raise InputError('Chưa có kết quả cho cấu hình này.')
    if filename=='history.csv':
        rows=read_json(folder/'result.json')['history'];out=StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return Response('\ufeff'+out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':f'attachment; filename="{run_id}-history.csv"'})
    return send_from_directory(folder,filename,as_attachment=True)

@app.get('/api/domains/<domain>')
def domains(domain):return jsonify(domain_config(domain))

@app.post('/api/domain-knowledge')
def domain_knowledge():
    start=time.perf_counter();d=payload();data=retrieve_domain(d.get('domain'),d.get('question'),d.get('method','hybrid'))
    if d.get('generative') is True and data['sources']:
        from demo.local_llm import generate
        data.update(generate(d['question'],data['sources']),mode='rag',note='Câu trả lời do Qwen cục bộ sinh từ bằng chứng. Kiểm tra nguồn khi đọc kết quả.')
    return result(data,start)

@app.post('/api/experiments')
def experiments():
    start=time.perf_counter();data=evaluate_domains()
    data['agents']=[dict(question=q,**run_agents(q)) for q in ['HbA1c phản ánh điều gì?','Dự đoán nguy cơ của tôi','Tôi nên tiêm bao nhiêu insulin?']]
    return result(data,start)

@app.post('/api/knowledge')
def knowledge():
    start=time.perf_counter();d=payload();return result(answer(d.get('question'),bool(d.get('generative',False)),d.get('model')),start)

@app.post('/api/agents')
def agents():
    start=time.perf_counter();d=payload();return result(run_agents(d.get('question'),d.get('profile'),d.get('domain')),start)

@app.errorhandler(InputError)
def bad_input(exc):return jsonify(error=str(exc)),400

@app.errorhandler(413)
def too_big(exc):return jsonify(error='Ảnh hoặc yêu cầu quá lớn. Hãy chọn ảnh dưới 5 MB.'),413

@app.errorhandler(Exception)
def failed(exc):
    if isinstance(exc,HTTPException):return jsonify(error=exc.description),exc.code
    app.logger.exception('Prediction request failed')
    return jsonify(error='Không chạy được mô hình. Xem Terminal để biết chi tiết và kiểm tra các file trong assets/models.'),500

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    port=int(os.environ.get('HTTM_PORT','8501'))
    print(f'\nHTTM Demo Studio: http://127.0.0.1:{port}\nNhấn Ctrl+C để dừng.\n',flush=True)
    (ROOT/'demo.pid').write_text(str(os.getpid()),encoding='ascii')
    try:app.run(host='127.0.0.1',port=port,debug=False,threaded=True)
    finally:
        if (ROOT/'demo.pid').exists() and (ROOT/'demo.pid').read_text().strip()==str(os.getpid()):(ROOT/'demo.pid').unlink()
