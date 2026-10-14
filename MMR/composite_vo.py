from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

from road_network import RoadNetwork
from edge_entries import EdgeEntryStore
from edge_mmr import EdgeMMRIndex, get_peak_heights, get_peak_node_indices
from spatial_skeleton import SpatialSkeleton, NODE_LEAF
from l1_merkle import L1MerkleIndex, edge_record_tuple
from l0_authenticated import L0AuthenticatedIndex
from crypto import time_overlap
from query_utils import normalize_rectangle, rectangle_overlap, edge_intersects_query, is_empty_time_range

# Paper-aligned VO v2 token types.
TOKEN_L0_PRUNED = 1
TOKEN_L0_LEAF = 3
TOKEN_L0_INTERNAL = 4
TOKEN_EDGE_OPAQUE = 10
TOKEN_EDGE_EXPANDED = 11
TOKEN_MMR_PRUNED = 20
TOKEN_MMR_RESULT_SLOT = 22
TOKEN_MMR_INTERNAL = 23
TOKEN_MMR_ROOT = 24

# Compatibility aliases used by older scripts.
TOKEN_L0_PRUNED_LEAF = TOKEN_L0_PRUNED
TOKEN_L0_PRUNED_INTERNAL = TOKEN_L0_PRUNED
TOKEN_L1_EDGE_OPAQUE = TOKEN_EDGE_OPAQUE
TOKEN_L1_EDGE_EXPANDED = TOKEN_EDGE_EXPANDED
TOKEN_L1_INTERNAL = 12  # not emitted in v2
TOKEN_MMR_PRUNED_LEAF = TOKEN_MMR_PRUNED
TOKEN_MMR_PRUNED_INTERNAL = TOKEN_MMR_PRUNED

TOKEN_NAMES={
    TOKEN_L0_PRUNED:'L0_PRUNED',TOKEN_L0_LEAF:'L0_LEAF',TOKEN_L0_INTERNAL:'L0_INTERNAL',
    TOKEN_EDGE_OPAQUE:'EDGE_OPAQUE',TOKEN_EDGE_EXPANDED:'EDGE_EXPANDED',
    TOKEN_MMR_PRUNED:'MMR_PRUNED',TOKEN_MMR_RESULT_SLOT:'MMR_RESULT_SLOT',
    TOKEN_MMR_INTERNAL:'MMR_INTERNAL',TOKEN_MMR_ROOT:'MMR_ROOT',
}

@dataclass(slots=True)
class VerificationEntry:
    eid:int
    trajectory_id:bytes
    start:int
    end:int

@dataclass(slots=True)
class CompositeVO:
    tokens:list[tuple]
    verification_set:list[VerificationEntry]
    candidate_ids:list[bytes]

class CompositeVOBuilder:
    __slots__=(
        'road','store','mmr_index','skeleton','l1_index','l0_index',
        'query_min_lon','query_min_lat','query_max_lon','query_max_lat','query_start','query_end',
        'tokens','verification_set'
    )
    def __init__(self,road:RoadNetwork,store:EdgeEntryStore,mmr_index:EdgeMMRIndex,
                 skeleton:SpatialSkeleton,l1_index:L1MerkleIndex,l0_index:L0AuthenticatedIndex,
                 query_min_lon:float,query_min_lat:float,query_max_lon:float,query_max_lat:float,
                 query_start:int,query_end:int):
        if query_start>query_end: raise ValueError('query_start > query_end')
        query_min_lon,query_min_lat,query_max_lon,query_max_lat=normalize_rectangle(query_min_lon,query_min_lat,query_max_lon,query_max_lat)
        self.road=road; self.store=store; self.mmr_index=mmr_index; self.skeleton=skeleton; self.l1_index=l1_index; self.l0_index=l0_index
        self.query_min_lon=query_min_lon; self.query_min_lat=query_min_lat; self.query_max_lon=query_max_lon; self.query_max_lat=query_max_lat
        self.query_start=query_start; self.query_end=query_end
        self.tokens=[]; self.verification_set=[]

    def _l0_should_prune(self,node:int)->bool:
        if not rectangle_overlap(self.skeleton.min_lon[node],self.skeleton.min_lat[node],self.skeleton.max_lon[node],self.skeleton.max_lat[node],
                                 self.query_min_lon,self.query_min_lat,self.query_max_lon,self.query_max_lat): return True
        mn=int(self.l0_index.node_min_start[node]); mx=int(self.l0_index.node_max_end[node])
        return is_empty_time_range(mn,mx) or not time_overlap(mn,mx,self.query_start,self.query_end)

    def _emit_pruned_l0(self,node:int)->None:
        self.tokens.append((TOKEN_L0_PRUNED,self.l0_index.get_node_hash(node),
                            self.skeleton.min_lon[node],self.skeleton.min_lat[node],self.skeleton.max_lon[node],self.skeleton.max_lat[node],
                            int(self.l0_index.node_min_start[node]),int(self.l0_index.node_max_end[node])))

    def _emit_mmr_subtree(self,eid:int,node:int,height:int,leaf_start:int)->None:
        mn=int(self.mmr_index.node_min_start[node]); mx=int(self.mmr_index.node_max_end[node])
        if not time_overlap(mn,mx,self.query_start,self.query_end):
            self.tokens.append((TOKEN_MMR_PRUNED,self.mmr_index.get_node_hash(node),mn,mx))
            return
        if height==0:
            trajectory_id,start,end=self.store.get_entry(eid=eid,index=leaf_start)
            self.verification_set.append(VerificationEntry(eid,trajectory_id,start,end))
            self.tokens.append((TOKEN_MMR_RESULT_SLOT,))
            return
        left=node-(1<<height); right=node-1; ch=height-1; left_count=1<<ch
        self._emit_mmr_subtree(eid,left,ch,leaf_start)
        self._emit_mmr_subtree(eid,right,ch,leaf_start+left_count)
        # Open internal nodes carry no parent tau; verifier derives it from children.
        self.tokens.append((TOKEN_MMR_INTERNAL,))

    def _emit_edge_mmr(self,eid:int)->None:
        k=int(self.mmr_index.edge_k[eid])
        if k<=0: raise RuntimeError(f'eid={eid} has empty MMR')
        offset=int(self.mmr_index.edge_node_offset[eid]); leaf_start=0
        for height,node in zip(get_peak_heights(k),get_peak_node_indices(offset,k)):
            self._emit_mmr_subtree(eid,node,height,leaf_start); leaf_start += 1<<height
        self.tokens.append((TOKEN_MMR_ROOT,k))

    def _emit_edge_record(self,eid:int)->None:
        rec=edge_record_tuple(eid,self.mmr_index,self.road)
        _,u_lon,u_lat,v_lon,v_lat,k,mn,mx,root_e=rec
        spatial_match=edge_intersects_query(eid,self.road,self.query_min_lon,self.query_min_lat,self.query_max_lon,self.query_max_lat)
        if (not spatial_match) or k==0 or not time_overlap(mn,mx,self.query_start,self.query_end):
            self.tokens.append((TOKEN_EDGE_OPAQUE,*rec))
            return
        self._emit_edge_mmr(eid)
        # root_e is reconstructed from the preceding MMR proof.
        self.tokens.append((TOKEN_EDGE_EXPANDED,eid,u_lon,u_lat,v_lon,v_lat,k,mn,mx))

    def _leaf_eids(self,node:int)->list[int]:
        off=int(self.skeleton.leaf_edge_offset[node]); count=int(self.skeleton.leaf_edge_count[node])
        eids=[int(self.skeleton.leaf_edge_ids[off+i]) for i in range(count)]
        if eids!=sorted(eids): eids.sort()
        return eids

    def _emit_l0_leaf(self,node:int)->None:
        eids=self._leaf_eids(node)
        for eid in eids: self._emit_edge_record(eid)
        self.tokens.append((TOKEN_L0_LEAF,
                            self.skeleton.min_lon[node],self.skeleton.min_lat[node],self.skeleton.max_lon[node],self.skeleton.max_lat[node],
                            int(self.l0_index.node_min_start[node]),int(self.l0_index.node_max_end[node]),len(eids)))

    def _emit_l0_node(self,node:int,force_open:bool=False)->None:
        if (not force_open) and self._l0_should_prune(node):
            self._emit_pruned_l0(node); return
        if self.skeleton.node_type[node]==NODE_LEAF:
            self._emit_l0_leaf(node); return
        left=int(self.skeleton.left_child[node]); mid=int(self.skeleton.mid_child[node]); right=int(self.skeleton.right_child[node])
        self._emit_l0_node(left); self._emit_l0_node(mid); self._emit_l0_node(right)
        self.tokens.append((TOKEN_L0_INTERNAL,
                            self.skeleton.min_lon[node],self.skeleton.min_lat[node],self.skeleton.max_lon[node],self.skeleton.max_lat[node],
                            int(self.l0_index.node_min_start[node]),int(self.l0_index.node_max_end[node])))

    def build(self)->CompositeVO:
        self.tokens.clear(); self.verification_set.clear()
        # The SEM-tree root is always opened, even if the query misses everything.
        self._emit_l0_node(int(self.skeleton.root_index),force_open=True)
        candidates=sorted({e.trajectory_id for e in self.verification_set})
        return CompositeVO(list(self.tokens),list(self.verification_set),candidates)

def print_vo_statistics(vo:CompositeVO)->None:
    c=Counter(t[0] for t in vo.tokens)
    print('Total number of VO tokens:',len(vo.tokens)); print('Verification Set count:',len(vo.verification_set)); print('Candidate trajectory_id count:',len(vo.candidate_ids))
    for t in sorted(c): print(TOKEN_NAMES.get(t,str(t)),':',c[t])

def print_verification_set(vo:CompositeVO,show_count:int=10)->None:
    for i,e in enumerate(vo.verification_set[:show_count]): print(i,e.eid,e.start,e.end,e.trajectory_id.hex())
