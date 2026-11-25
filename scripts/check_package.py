"""Compare the handover manifest without rewriting it."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.common import sha


def main():
    path=ROOT/'runs/verification/submission_manifest.json'
    if not path.exists():raise SystemExit('Missing handover manifest. Run verify_submission.py to build it.')
    files=json.loads(path.read_text(encoding='utf-8'))['files']
    bad=[name for name,digest in files.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    print(json.dumps(dict(files=len(files),mismatches=bad,passed=not bad),indent=2))
    return 1 if bad else 0


if __name__=='__main__':sys.exit(main())
