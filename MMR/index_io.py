from __future__ import annotations
import hashlib, os, pickle, struct
from dataclasses import dataclass

INDEX_MAGIC=b'MMRIDX02'; INDEX_VERSION=2; _HEADER=struct.Struct('>8sIQ32s')

@dataclass(slots=True)
class PersistedIndex:
    road:object
    store:object
    mmr_index:object
    skeleton:object
    l1_index:object  # v2: EdgeBlockIndex, name retained for compatibility
    l0_index:object
    theta:int
    trajectory_files:tuple[str,...]

def save_index(index:PersistedIndex,file_path:str)->int:
    path=os.path.abspath(file_path); os.makedirs(os.path.dirname(path),exist_ok=True)
    payload=pickle.dumps(index,protocol=5); digest=hashlib.sha256(payload).digest(); tmp=path+'.tmp'
    with open(tmp,'wb') as f:
        f.write(_HEADER.pack(INDEX_MAGIC,INDEX_VERSION,len(payload),digest)); f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); return os.path.getsize(path)

def load_index(file_path:str)->PersistedIndex:
    with open(file_path,'rb') as f:
        header=f.read(_HEADER.size)
        if len(header)!=_HEADER.size: raise ValueError('Incomplete index header')
        magic,version,n,expected=_HEADER.unpack(header)
        if magic!=INDEX_MAGIC or version!=INDEX_VERSION:
            raise ValueError('This project uses index format v2. Rebuild old v1 indexes with build_index.py.')
        payload=f.read(n)
        if len(payload)!=n or f.read(1): raise ValueError('Corrupted index file')
    if hashlib.sha256(payload).digest()!=expected: raise ValueError('Index SHA-256 verification failed')
    obj=pickle.loads(payload)
    if not isinstance(obj,PersistedIndex): raise ValueError('Invalid index payload')
    if obj.skeleton.root_index<0 or not isinstance(obj.l0_index.root_hash,bytes) or len(obj.l0_index.root_hash)!=32: raise ValueError('Invalid authenticated index')
    return obj
