"""Double-click launcher. Keeps inference local and opens the default browser."""
from pathlib import Path
import json, os, sys, threading, time, urllib.request, webbrowser, socket

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
port=int(os.environ.get('HTTM_PORT','8501'))
url=f'http://localhost:{port}'

def already_running():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as r:
            return json.load(r).get('app')=='HTTM Demo Studio'
    except Exception:return False

def open_when_ready():
    for _ in range(40):
        if already_running():
            if '--no-browser' not in sys.argv:webbrowser.open(url)
            return
        time.sleep(.5)

if already_running():
    print('Demo đang chạy tại '+url)
    if '--no-browser' not in sys.argv:webbrowser.open(url)
else:
    try:
        from app import app
        print('\nHTTM DEMO STUDIO\n'+url+'\nGiữ cửa sổ này mở trong khi trình diễn. Ctrl+C để dừng.\n',flush=True)
        threading.Thread(target=open_when_ready,daemon=True).start()
        (ROOT/'demo.pid').write_text(str(os.getpid()),encoding='ascii')
        try:app.run(host='127.0.0.1',port=port,debug=False,threaded=True)
        finally:
            if (ROOT/'demo.pid').exists() and (ROOT/'demo.pid').read_text().strip()==str(os.getpid()):(ROOT/'demo.pid').unlink()
    except ModuleNotFoundError as exc:
        print(f'Thiếu thư viện: {exc.name}\nChạy: python -m pip install -r requirements.txt')
        sys.exit(1)
