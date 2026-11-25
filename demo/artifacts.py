"""Small, cached provenance descriptors for artifacts actually used in inference."""
from functools import lru_cache
from pathlib import Path
import hashlib
from .config import ROOT

@lru_cache(maxsize=96)
def _descriptor(path,mtime,size):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return dict(file=p.relative_to(ROOT).as_posix(),sha256=h.hexdigest(),bytes=size)

def artifact(path):
 p=Path(path).resolve();stat=p.stat()
 return dict(_descriptor(str(p),stat.st_mtime_ns,stat.st_size))
