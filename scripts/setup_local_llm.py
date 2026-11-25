"""Download pinned official CPU runtime/model into this project; verify SHA-256."""
from pathlib import Path
import hashlib,json,sys,zipfile
import requests
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=dict(name='llama.cpp b12284 CPU x64',url='https://github.com/ggml-org/llama.cpp/releases/download/b12284/llama-b12284-bin-win-cpu-x64.zip',sha256='6b3d2c7034b0c0bac65aa0524ddf01ba9c2b622ce7e32675976df642627defa9')
MODEL=dict(name='Qwen2.5-1.5B-Instruct Q4_K_M',url='https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf',sha256='6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def download(info,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and sha(path)==info['sha256']:
  print('Verified cached',path.name,flush=True);return
 part=path.with_suffix(path.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
 r=requests.get(info['url'],headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,60));r.raise_for_status()
 if r.status_code!=206:offset=0
 mode='ab' if offset else 'wb';total=int(r.headers.get('Content-Length',0))+offset;done=offset;last=done
 print('Downloading',path.name,'bytes',total,flush=True)
 with part.open(mode) as f:
  for chunk in r.iter_content(1024*1024):
   if not chunk:continue
   f.write(chunk);done+=len(chunk)
   if done-last>=64*1024*1024:print(path.name,round(done/1024**2),'MB /',round(total/1024**2),'MB',flush=True);last=done
 if sha(part)!=info['sha256']:raise RuntimeError('SHA-256 mismatch: '+path.name)
 part.replace(path);print('SHA-256 verified',path.name,flush=True)
def main():
 archive=ROOT/'runtime/downloads/llama-b12284-bin-win-cpu-x64.zip';download(RUNTIME,archive)
 target=(ROOT/'runtime/llama').resolve();target.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for entry in z.infolist():
   out=(target/entry.filename).resolve()
   if not out.is_relative_to(target):raise RuntimeError('Unsafe archive path')
  z.extractall(target)
 model=ROOT/'assets/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf';download(MODEL,model)
 (ROOT/'runtime/llm_source.json').write_text(json.dumps(dict(runtime=RUNTIME,model=MODEL),indent=2),encoding='utf-8')
 print('Ready: local CPU generation; no external inference API',flush=True)
if __name__=='__main__':main()
