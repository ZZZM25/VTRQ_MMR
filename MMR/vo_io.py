from __future__ import annotations
import os, struct
from dataclasses import dataclass
from composite_vo import (
    CompositeVO,TOKEN_L0_PRUNED,TOKEN_L0_LEAF,TOKEN_L0_INTERNAL,
    TOKEN_EDGE_OPAQUE,TOKEN_EDGE_EXPANDED,TOKEN_MMR_PRUNED,TOKEN_MMR_RESULT_SLOT,TOKEN_MMR_INTERNAL,TOKEN_MMR_ROOT,
)

VO_MAGIC=b'CMVO0002'; VS_MAGIC=b'VSET0002'; ROOT_MAGIC=b'ROOT0002'; VERSION=2
_HEADER=struct.Struct('>8sII'); _TOKEN_TYPE=struct.Struct('>B')
_L0_PRUNED=struct.Struct('>32sddddII')
_L0_LEAF=struct.Struct('>ddddIII')
_L0_INTERNAL=struct.Struct('>ddddII')
_EDGE_OPAQUE=struct.Struct('>IddddIII32s')
_EDGE_EXPANDED=struct.Struct('>IddddIII')
_MMR_PRUNED=struct.Struct('>32sII')
_MMR_ROOT=struct.Struct('>I')
_VS_ENTRY=struct.Struct('>I32sII'); _ROOT=struct.Struct('>32s')

@dataclass(slots=True)
class VerificationEntryRecord:
    eid:int; trajectory_id:bytes; start:int; end:int

def _write_token(f,token):
    t=token[0]; f.write(_TOKEN_TYPE.pack(t))
    if t==TOKEN_L0_PRUNED: f.write(_L0_PRUNED.pack(*token[1:])); return
    if t==TOKEN_L0_LEAF: f.write(_L0_LEAF.pack(*token[1:])); return
    if t==TOKEN_L0_INTERNAL: f.write(_L0_INTERNAL.pack(*token[1:])); return
    if t==TOKEN_EDGE_OPAQUE: f.write(_EDGE_OPAQUE.pack(*token[1:])); return
    if t==TOKEN_EDGE_EXPANDED: f.write(_EDGE_EXPANDED.pack(*token[1:])); return
    if t==TOKEN_MMR_PRUNED: f.write(_MMR_PRUNED.pack(*token[1:])); return
    if t in (TOKEN_MMR_RESULT_SLOT,TOKEN_MMR_INTERNAL): return
    if t==TOKEN_MMR_ROOT: f.write(_MMR_ROOT.pack(token[1])); return
    raise ValueError(f'Unknown VO token type: {t}')

def save_composite_vo(vo:CompositeVO,file_path:str)->int:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)),exist_ok=True)
    with open(file_path,'wb') as f:
        f.write(_HEADER.pack(VO_MAGIC,VERSION,len(vo.tokens)))
        for token in vo.tokens: _write_token(f,token)
    return os.path.getsize(file_path)

def _read_exact(f,n):
    b=f.read(n)
    if len(b)!=n: raise ValueError('File ended prematurely')
    return b

def load_composite_vo_tokens(file_path:str)->list[tuple]:
    with open(file_path,'rb') as f:
        magic,version,count=_HEADER.unpack(_read_exact(f,_HEADER.size))
        if magic!=VO_MAGIC or version!=VERSION: raise ValueError(f'Unsupported VO format/version: {magic!r} v{version}')
        out=[]
        for _ in range(count):
            t=_TOKEN_TYPE.unpack(_read_exact(f,1))[0]
            if t==TOKEN_L0_PRUNED: vals=_L0_PRUNED.unpack(_read_exact(f,_L0_PRUNED.size)); out.append((t,*vals))
            elif t==TOKEN_L0_LEAF: vals=_L0_LEAF.unpack(_read_exact(f,_L0_LEAF.size)); out.append((t,*vals))
            elif t==TOKEN_L0_INTERNAL: vals=_L0_INTERNAL.unpack(_read_exact(f,_L0_INTERNAL.size)); out.append((t,*vals))
            elif t==TOKEN_EDGE_OPAQUE: vals=_EDGE_OPAQUE.unpack(_read_exact(f,_EDGE_OPAQUE.size)); out.append((t,*vals))
            elif t==TOKEN_EDGE_EXPANDED: vals=_EDGE_EXPANDED.unpack(_read_exact(f,_EDGE_EXPANDED.size)); out.append((t,*vals))
            elif t==TOKEN_MMR_PRUNED: vals=_MMR_PRUNED.unpack(_read_exact(f,_MMR_PRUNED.size)); out.append((t,*vals))
            elif t in (TOKEN_MMR_RESULT_SLOT,TOKEN_MMR_INTERNAL): out.append((t,))
            elif t==TOKEN_MMR_ROOT: out.append((t,_MMR_ROOT.unpack(_read_exact(f,_MMR_ROOT.size))[0]))
            else: raise ValueError(f'Unknown VO token type: {t}')
        if f.read(1): raise ValueError('Extra data at end of VO file')
        return out

def save_verification_set(verification_set,file_path:str)->int:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)),exist_ok=True)
    with open(file_path,'wb') as f:
        f.write(_HEADER.pack(VS_MAGIC,VERSION,len(verification_set)))
        for e in verification_set:
            if hasattr(e,'eid'): eid,tid,start,end=e.eid,e.trajectory_id,e.start,e.end
            else: eid,tid,start,end=e
            if len(tid)!=32: raise ValueError('trajectory_id must be 32 bytes')
            f.write(_VS_ENTRY.pack(eid,tid,start,end))
    return os.path.getsize(file_path)

def load_verification_set(file_path:str)->list[VerificationEntryRecord]:
    with open(file_path,'rb') as f:
        magic,version,count=_HEADER.unpack(_read_exact(f,_HEADER.size))
        if magic!=VS_MAGIC or version!=VERSION: raise ValueError('Unsupported Verification Set format/version')
        out=[]
        for _ in range(count):
            eid,tid,start,end=_VS_ENTRY.unpack(_read_exact(f,_VS_ENTRY.size)); out.append(VerificationEntryRecord(eid,tid,start,end))
        if f.read(1): raise ValueError('Extra data at end of Verification Set file')
        return out

def save_trusted_root(root_hash:bytes,file_path:str)->int:
    if len(root_hash)!=32: raise ValueError('root_S must be 32 bytes')
    os.makedirs(os.path.dirname(os.path.abspath(file_path)),exist_ok=True)
    with open(file_path,'wb') as f: f.write(_HEADER.pack(ROOT_MAGIC,VERSION,1)); f.write(_ROOT.pack(root_hash))
    return os.path.getsize(file_path)

def load_trusted_root(file_path:str)->bytes:
    with open(file_path,'rb') as f:
        magic,version,count=_HEADER.unpack(_read_exact(f,_HEADER.size))
        if magic!=ROOT_MAGIC or version!=VERSION or count!=1: raise ValueError('Unsupported root format/version')
        root=_ROOT.unpack(_read_exact(f,_ROOT.size))[0]
        if f.read(1): raise ValueError('Extra data at end of root file')
        return root
