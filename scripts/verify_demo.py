"""Run regression checks plus asset integrity; optionally test actual local generation."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,unittest,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--with-llm',action='store_true');args=parser.parse_args()
 start=time.perf_counter();manifest=json.loads((ROOT/'assets/manifest.json').read_text(encoding='utf-8'));bad=[]
 for entry in manifest['files']:
  p=ROOT/entry['file']
  if not p.is_file() or sha(p)!=entry['sha256']:bad.append(entry['file'])
 suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
 r=unittest.TextTestRunner(verbosity=2).run(suite)
 llm=None
 if args.with_llm:llm=subprocess.run([sys.executable,'-u','-X','utf8',str(ROOT/'scripts/check_rag.py')],cwd=ROOT).returncode
 out=dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),asset_files=len(manifest['files']),asset_mismatches=bad,real_llm_exit_code=llm,elapsed_seconds=round(time.perf_counter()-start,2),passed=r.wasSuccessful() and not bad and (llm in [None,0]))
 folder=ROOT/'runs/verification';folder.mkdir(parents=True,exist_ok=True);(folder/'test_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps(out,indent=2),flush=True);return 0 if out['passed'] else 1
if __name__=='__main__':sys.exit(main())
