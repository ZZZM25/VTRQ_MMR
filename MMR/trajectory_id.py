from __future__ import annotations
import hashlib, sys
from array import array

def calculate_trajectory_id(eids:list[int], trajectory_start:int, trajectory_end:int)->bytes:
    if not eids: raise ValueError('Trajectory path cannot be empty')
    if trajectory_start>trajectory_end: raise ValueError('trajectory_start cannot be greater than trajectory_end')
    values=array('I',eids); values.append(trajectory_start); values.append(trajectory_end)
    if values.itemsize!=4: raise RuntimeError('unsigned int is not 4 bytes on the current platform')
    if sys.byteorder=='little': values.byteswap()
    return hashlib.sha256(values).digest()

def trajectory_id_from_segments(eids:list[int],starts:list[int],ends:list[int])->bytes:
    if not eids or len(eids)!=len(starts) or len(eids)!=len(ends): raise ValueError('Invalid trajectory segment data')
    return calculate_trajectory_id(eids,starts[0],ends[-1])
