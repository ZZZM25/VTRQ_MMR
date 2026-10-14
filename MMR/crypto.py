from __future__ import annotations
import hashlib
import struct

_TRAJECTORY_LEAF_STRUCT = struct.Struct('>III')
_MMR_LEAF_STRUCT = struct.Struct('>32sII')
_MMR_CHILD_STRUCT = struct.Struct('>32sII')
_MMR_INTERNAL_STRUCT = struct.Struct('>32sII32sII')

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def hash_trajectory_leaf(eid:int,start:int,end:int)->bytes:
    return sha256(_TRAJECTORY_LEAF_STRUCT.pack(eid,start,end))

def hash_merkle_internal(left_hash:bytes,right_hash:bytes)->bytes:
    h=hashlib.sha256(); h.update(left_hash); h.update(right_hash); return h.digest()

def hash_mmr_leaf(trajectory_id:bytes,start:int,end:int)->bytes:
    if len(trajectory_id)!=32: raise ValueError('trajectory_id must be 32 bytes')
    return sha256(_MMR_LEAF_STRUCT.pack(trajectory_id,start,end))

def hash_mmr_internal(left_hash:bytes,left_min_start:int,left_max_end:int,right_hash:bytes,right_min_start:int,right_max_end:int)->bytes:
    """Paper-aligned τ-MMR internal hash: H(hash0 || τ0 || hash1 || τ1).

    Hot-path implementation: serialize the exact same byte string with one
    Struct.pack call and hash it once.  This changes only implementation cost;
    the authenticated message and resulting digest are unchanged.
    """
    if len(left_hash)!=32 or len(right_hash)!=32:
        raise ValueError('MMR child hash must be 32 bytes')
    return hashlib.sha256(
        _MMR_INTERNAL_STRUCT.pack(
            left_hash, left_min_start, left_max_end,
            right_hash, right_min_start, right_max_end,
        )
    ).digest()

def time_overlap(start:int|float,end:int|float,query_start:int|float,query_end:int|float)->bool:
    return not (end<query_start or start>query_end)
