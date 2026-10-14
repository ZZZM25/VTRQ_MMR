from __future__ import annotations
import hashlib, struct
from dataclasses import dataclass

from crypto import hash_mmr_leaf, hash_mmr_internal, time_overlap
from l1_merkle import hash_edge_record, hash_edge_block_records
from l0_authenticated import hash_l0_leaf, hash_l0_internal, encode_child_phi
from query_utils import UINT32_MAX, normalize_rectangle, rectangle_overlap, geometry_intersects_query, is_empty_time_range
from composite_vo import (
    TOKEN_L0_PRUNED,TOKEN_L0_LEAF,TOKEN_L0_INTERNAL,
    TOKEN_EDGE_OPAQUE,TOKEN_EDGE_EXPANDED,
    TOKEN_MMR_PRUNED,TOKEN_MMR_RESULT_SLOT,TOKEN_MMR_INTERNAL,TOKEN_MMR_ROOT,
)

HASH_SIZE=32; _U32=struct.Struct('>I'); _U32_2=struct.Struct('>II'); _sha256=hashlib.sha256
STACK_MMR_NODE=1; STACK_MMR_ROOT=2; STACK_EDGE_RECORD=3; STACK_L0_NODE=4

@dataclass(slots=True)
class VerificationReport:
    success:bool
    reconstructed_root:bytes
    trusted_root:bytes
    token_count:int
    verification_entry_count:int
    consumed_verification_entries:int
    candidate_ids:list[bytes]

@dataclass(slots=True)
class StackItem:
    kind:int
    node_hash:bytes
    min_start:int=UINT32_MAX
    max_end:int=0
    k:int=0
    edge_eid:int=-1
    min_lon:float=0.0
    min_lat:float=0.0
    max_lon:float=0.0
    max_lat:float=0.0

class VerificationError(RuntimeError): pass

def validate_hash(v,name):
    if not isinstance(v,bytes) or len(v)!=32: raise VerificationError(f'{name} is not a valid 32-byte hash')

def calculate_empty_mmr_root()->bytes: return _sha256(_U32.pack(0)).digest()
EMPTY_MMR_ROOT=calculate_empty_mmr_root()

def calculate_mmr_root_from_peaks(k:int,peaks:list[StackItem])->bytes:
    h=_sha256(); h.update(_U32.pack(k))
    for p in peaks: h.update(p.node_hash); h.update(_U32_2.pack(p.min_start,p.max_end))
    return h.digest()

def l0_can_prune(min_lon,min_lat,max_lon,max_lat,min_start,max_end,qminlon,qminlat,qmaxlon,qmaxlat,qstart,qend)->bool:
    if not rectangle_overlap(min_lon,min_lat,max_lon,max_lat,qminlon,qminlat,qmaxlon,qmaxlat): return True
    if is_empty_time_range(min_start,max_end): return True
    return not time_overlap(min_start,max_end,qstart,qend)

def _merge_edge_eid(a:int,b:int)->int:
    if a>=0 and b>=0 and a!=b: raise VerificationError('Different eid values appeared inside one MMR')
    return a if a>=0 else b

def _phi(x:StackItem)->bytes:
    return encode_child_phi(x.node_hash,x.min_lon,x.min_lat,x.max_lon,x.max_lat,x.min_start,x.max_end)

def verify_composite_vo(tokens,verification_set,road=None,trusted_root:bytes=b'',
                        query_min_lon:float=0,query_min_lat:float=0,query_max_lon:float=0,query_max_lat:float=0,
                        query_start:int=0,query_end:int=0)->VerificationReport:
    validate_hash(trusted_root,'trusted_root')
    if query_start>query_end: raise VerificationError('query_start > query_end')
    qminlon,qminlat,qmaxlon,qmaxlat=normalize_rectangle(query_min_lon,query_min_lat,query_max_lon,query_max_lat)
    if not tokens: raise VerificationError('empty VO')
    if tokens[-1][0] not in (TOKEN_L0_LEAF,TOKEN_L0_INTERNAL):
        raise VerificationError('SEM-tree root must be opened; a pruned root is not accepted')

    stack=[]; vi=0
    for pos,token in enumerate(tokens):
        t=token[0]
        if t==TOKEN_MMR_PRUNED:
            _,node_hash,mn,mx=token; validate_hash(node_hash,'MMR pruned hash')
            if mn>mx: raise VerificationError('MMR_PRUNED invalid tau')
            if time_overlap(mn,mx,query_start,query_end): raise VerificationError('MMR_PRUNED overlaps query time')
            stack.append(StackItem(STACK_MMR_NODE,node_hash,mn,mx)); continue

        if t==TOKEN_MMR_RESULT_SLOT:
            if vi>=len(verification_set): raise VerificationError('Insufficient Verification Set entries')
            e=verification_set[vi]; vi+=1
            if len(e.trajectory_id)!=32 or e.start>e.end: raise VerificationError('Invalid Verification Entry')
            if not time_overlap(e.start,e.end,query_start,query_end): raise VerificationError('Verification Entry does not overlap query time')
            stack.append(StackItem(STACK_MMR_NODE,hash_mmr_leaf(e.trajectory_id,e.start,e.end),e.start,e.end,edge_eid=int(e.eid))); continue

        if t==TOKEN_MMR_INTERNAL:
            if len(stack)<2: raise VerificationError('MMR_INTERNAL stack underflow')
            right=stack.pop(); left=stack.pop()
            if left.kind!=STACK_MMR_NODE or right.kind!=STACK_MMR_NODE: raise VerificationError('MMR_INTERNAL child type error')
            mn=min(left.min_start,right.min_start); mx=max(left.max_end,right.max_end)
            digest=hash_mmr_internal(left.node_hash,left.min_start,left.max_end,right.node_hash,right.min_start,right.max_end)
            stack.append(StackItem(STACK_MMR_NODE,digest,mn,mx,edge_eid=_merge_edge_eid(left.edge_eid,right.edge_eid))); continue

        if t==TOKEN_MMR_ROOT:
            _,k=token
            if k<=0: raise VerificationError('expanded MMR k must be >0')
            pc=int(k).bit_count()
            if len(stack)<pc: raise VerificationError('MMR peak stack underflow')
            peaks=[stack.pop() for _ in range(pc)][::-1]
            if any(p.kind!=STACK_MMR_NODE for p in peaks): raise VerificationError('invalid MMR peak type')
            eid=-1
            for p in peaks: eid=_merge_edge_eid(eid,p.edge_eid)
            stack.append(StackItem(STACK_MMR_ROOT,calculate_mmr_root_from_peaks(k,peaks),min(p.min_start for p in peaks),max(p.max_end for p in peaks),k=k,edge_eid=eid)); continue

        if t==TOKEN_EDGE_OPAQUE:
            _,eid,u_lon,u_lat,v_lon,v_lat,k,mn,mx,root_e=token; validate_hash(root_e,'root_e')
            if k==0:
                if mn!=UINT32_MAX or mx!=0 or root_e!=EMPTY_MMR_ROOT: raise VerificationError(f'invalid empty edge record eid={eid}')
            elif mn==UINT32_MAX or mn>mx: raise VerificationError(f'invalid edge tau eid={eid}')
            spatial=geometry_intersects_query(u_lon,u_lat,v_lon,v_lat,qminlon,qminlat,qmaxlon,qmaxlat)
            if spatial and k>0 and time_overlap(mn,mx,query_start,query_end): raise VerificationError(f'eid={eid} should have expanded its MMR')
            rec_hash=hash_edge_record(eid,u_lon,u_lat,v_lon,v_lat,k,mn,mx,root_e)
            stack.append(StackItem(STACK_EDGE_RECORD,rec_hash,mn,mx,k=k,edge_eid=eid,min_lon=u_lon,min_lat=u_lat,max_lon=v_lon,max_lat=v_lat)); continue

        if t==TOKEN_EDGE_EXPANDED:
            _,eid,u_lon,u_lat,v_lon,v_lat,k,mn,mx=token
            if not stack: raise VerificationError('missing MMR_ROOT before EDGE_EXPANDED')
            m=stack.pop()
            if m.kind!=STACK_MMR_ROOT or m.k!=k or m.min_start!=mn or m.max_end!=mx: raise VerificationError(f'MMR state mismatch for eid={eid}')
            if m.edge_eid>=0 and m.edge_eid!=eid: raise VerificationError('Verification Set eid does not match edge record')
            if not geometry_intersects_query(u_lon,u_lat,v_lon,v_lat,qminlon,qminlat,qmaxlon,qmaxlat): raise VerificationError(f'expanded eid={eid} misses query geometry')
            if not time_overlap(mn,mx,query_start,query_end): raise VerificationError(f'expanded eid={eid} misses query time')
            rec_hash=hash_edge_record(eid,u_lon,u_lat,v_lon,v_lat,k,mn,mx,m.node_hash)
            stack.append(StackItem(STACK_EDGE_RECORD,rec_hash,mn,mx,k=k,edge_eid=eid,min_lon=u_lon,min_lat=u_lat,max_lon=v_lon,max_lat=v_lat)); continue

        if t==TOKEN_L0_PRUNED:
            _,node_hash,min_lon,min_lat,max_lon,max_lat,mn,mx=token; validate_hash(node_hash,'L0 pruned hash')
            if not l0_can_prune(min_lon,min_lat,max_lon,max_lat,mn,mx,qminlon,qminlat,qmaxlon,qmaxlat,query_start,query_end): raise VerificationError('L0_PRUNED does not satisfy pruning condition')
            stack.append(StackItem(STACK_L0_NODE,node_hash,mn,mx,min_lon=min_lon,min_lat=min_lat,max_lon=max_lon,max_lat=max_lat)); continue

        if t==TOKEN_L0_LEAF:
            _,min_lon,min_lat,max_lon,max_lat,mn,mx,count=token
            if count<0 or len(stack)<count: raise VerificationError('L0_LEAF edge stack underflow')
            records=[stack.pop() for _ in range(count)][::-1]
            if any(r.kind!=STACK_EDGE_RECORD for r in records): raise VerificationError('L0_LEAF contains non-edge item')
            eids=[r.edge_eid for r in records]
            if eids!=sorted(eids) or len(eids)!=len(set(eids)): raise VerificationError('E_h must be strictly eid-sorted')
            nonempty=[r for r in records if r.k>0]
            exp_mn=min((r.min_start for r in nonempty),default=UINT32_MAX); exp_mx=max((r.max_end for r in nonempty),default=0)
            if mn!=exp_mn or mx!=exp_mx: raise VerificationError('L0 leaf tau does not equal union of edge tau values')
            block=hash_edge_block_records([r.node_hash for r in records])
            digest=hash_l0_leaf(min_lon,min_lat,max_lon,max_lat,mn,mx,edge_block_hash=block)
            stack.append(StackItem(STACK_L0_NODE,digest,mn,mx,min_lon=min_lon,min_lat=min_lat,max_lon=max_lon,max_lat=max_lat)); continue

        if t==TOKEN_L0_INTERNAL:
            _,min_lon,min_lat,max_lon,max_lat,mn,mx=token
            if len(stack)<3: raise VerificationError('L0_INTERNAL stack underflow')
            right=stack.pop(); mid=stack.pop(); left=stack.pop()
            if any(x.kind!=STACK_L0_NODE for x in (left,mid,right)): raise VerificationError('L0_INTERNAL child type error')
            exp_mn=min(left.min_start,mid.min_start,right.min_start); exp_mx=max(left.max_end,mid.max_end,right.max_end)
            if mn!=exp_mn or mx!=exp_mx: raise VerificationError('L0_INTERNAL tau aggregation mismatch')
            digest=hash_l0_internal(min_lon,min_lat,max_lon,max_lat,mn,mx,left_phi=_phi(left),mid_phi=_phi(mid),right_phi=_phi(right))
            stack.append(StackItem(STACK_L0_NODE,digest,mn,mx,min_lon=min_lon,min_lat=min_lat,max_lon=max_lon,max_lat=max_lat)); continue

        raise VerificationError(f'Unknown token type {t} at position {pos}')

    if vi!=len(verification_set): raise VerificationError(f'Verification Set not fully consumed: {vi}/{len(verification_set)}')
    if len(stack)!=1 or stack[0].kind!=STACK_L0_NODE: raise VerificationError('VO did not reconstruct exactly one SEM-tree root')
    root=stack[0].node_hash
    if root!=trusted_root: raise VerificationError(f'root_S verification failed: reconstructed={root.hex()} trusted={trusted_root.hex()}')
    candidates=sorted({e.trajectory_id for e in verification_set})
    return VerificationReport(True,root,trusted_root,len(tokens),len(verification_set),vi,candidates)
