from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')


def versions():
    names = ['numpy', 'pandas', 'scikit-learn', 'torch', 'tensorflow', 'matplotlib']
    result = {}
    for name in names:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return result


def finish_run(folder, metadata):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name not in ['complete.json', 'verification.json']}
    write_json(folder/'complete.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                          versions=versions(), files=files, **metadata))


def check_run(folder):
    folder = Path(folder)
    d = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    bad = [name for name, digest in d['files'].items()
           if not (folder/name).is_file() or sha(folder/name) != digest]
    if bad: raise ValueError('Run artifacts changed: '+', '.join(bad))
    return d
