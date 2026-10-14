from __future__ import annotations
import hashlib, struct
from array import array
from spatial_skeleton import SpatialSkeleton, NODE_LEAF, NODE_INTERNAL
from l1_merkle import L1MerkleIndex
from query_utils import UINT32_MAX

HASH_SIZE=32
_L0_META_STRUCT=struct.Struct('>ddddII')

def _meta(min_lon,min_lat,max_lon,max_lat,min_start,max_end)->bytes:
    return _L0_META_STRUCT.pack(min_lon,min_lat,max_lon,max_lat,min_start,max_end)

def hash_l0_leaf(min_lon,min_lat,max_lon,max_lat,min_start,max_end,l1_root=None,edge_block_hash=None)->bytes:
    """SEM-tree leaf: H(MBR || tau || H_e(E_h)).
    l1_root is accepted as a compatibility alias for edge_block_hash.
    """
    block=edge_block_hash if edge_block_hash is not None else l1_root
    if block is None or len(block)!=32: raise ValueError('edge block hash must be 32 bytes')
    h=hashlib.sha256(); h.update(_meta(min_lon,min_lat,max_lon,max_lat,min_start,max_end)); h.update(block); return h.digest()

def encode_child_phi(child_hash,min_lon,min_lat,max_lon,max_lat,min_start,max_end)->bytes:
    """phi_i = child.Hs || child.MBR || child.tau."""
    if len(child_hash)!=32: raise ValueError('child hash must be 32 bytes')
    return child_hash + _meta(min_lon,min_lat,max_lon,max_lat,min_start,max_end)

def hash_l0_internal(min_lon,min_lat,max_lon,max_lat,min_start,max_end,
                     left_hash=None,mid_hash=None,right_hash=None,
                     left_phi=None,mid_phi=None,right_phi=None,
                     left_meta=None,mid_meta=None,right_meta=None)->bytes:
    """SEM-tree internal: H(MBR || tau || phi0 || phil || phi1).

    New callers should pass left_phi/mid_phi/right_phi. Compatibility callers may
    pass child hashes together with left_meta/mid_meta/right_meta tuples.
    """
    if left_phi is None:
        if left_meta is None or mid_meta is None or right_meta is None:
            raise ValueError('child metadata is required by the paper-aligned SEM-tree hash')
        left_phi=encode_child_phi(left_hash,*left_meta)
        mid_phi=encode_child_phi(mid_hash,*mid_meta)
        right_phi=encode_child_phi(right_hash,*right_meta)
    h=hashlib.sha256(); h.update(_meta(min_lon,min_lat,max_lon,max_lat,min_start,max_end)); h.update(left_phi); h.update(mid_phi); h.update(right_phi); return h.digest()

def merge_three_time_ranges(a0,a1,b0,b1,c0,c1):
    return min(a0,b0,c0), max(a1,b1,c1)

class L0AuthenticatedIndex:
    __slots__=('node_hashes','node_min_start','node_max_end','root_index','root_hash')
    def __init__(self,node_count:int):
        self.node_hashes=bytearray(HASH_SIZE*node_count)
        self.node_min_start=array('I',[UINT32_MAX]*node_count)
        self.node_max_end=array('I',[0]*node_count)
        self.root_index=-1; self.root_hash=bytes(HASH_SIZE)
    def set_node_hash(self,node_index:int,node_hash:bytes):
        if len(node_hash)!=32: raise ValueError('node hash must be 32 bytes')
        p=node_index*32; self.node_hashes[p:p+32]=node_hash
    def get_node_hash(self,node_index:int)->bytes:
        p=node_index*32; return bytes(self.node_hashes[p:p+32])

def _child_phi(skeleton,index,child:int)->bytes:
    return encode_child_phi(
        index.get_node_hash(child),
        skeleton.min_lon[child],skeleton.min_lat[child],skeleton.max_lon[child],skeleton.max_lat[child],
        index.node_min_start[child],index.node_max_end[child],
    )

def build_authenticated_l0(skeleton:SpatialSkeleton,l1_index:L1MerkleIndex)->L0AuthenticatedIndex:
    n=skeleton.node_count
    out=L0AuthenticatedIndex(n)
    for i in range(n):
        if skeleton.node_type[i]==NODE_LEAF:
            mn=int(l1_index.l0_min_start[i]); mx=int(l1_index.l0_max_end[i]); block=l1_index.get_l1_root(i)
            digest=hash_l0_leaf(skeleton.min_lon[i],skeleton.min_lat[i],skeleton.max_lon[i],skeleton.max_lat[i],mn,mx,edge_block_hash=block)
        elif skeleton.node_type[i]==NODE_INTERNAL:
            left=int(skeleton.left_child[i]); mid=int(skeleton.mid_child[i]); right=int(skeleton.right_child[i])
            if min(left,mid,right)<0 or max(left,mid,right)>=i: raise RuntimeError(f'invalid postorder at L0 node={i}')
            mn,mx=merge_three_time_ranges(out.node_min_start[left],out.node_max_end[left],out.node_min_start[mid],out.node_max_end[mid],out.node_min_start[right],out.node_max_end[right])
            digest=hash_l0_internal(
                skeleton.min_lon[i],skeleton.min_lat[i],skeleton.max_lon[i],skeleton.max_lat[i],mn,mx,
                left_phi=_child_phi(skeleton,out,left),mid_phi=_child_phi(skeleton,out,mid),right_phi=_child_phi(skeleton,out,right),
            )
        else:
            raise RuntimeError(f'Unknown L0 node type: node={i}')
        out.node_min_start[i]=mn; out.node_max_end[i]=mx; out.set_node_hash(i,digest)
    root=int(skeleton.root_index)
    if root<0 or root>=n: raise RuntimeError('Invalid L0 root index')
    out.root_index=root; out.root_hash=out.get_node_hash(root); return out

def validate_authenticated_l0(skeleton,l1_index,l0_index)->None:
    rebuilt=build_authenticated_l0(skeleton,l1_index)
    if rebuilt.root_hash!=l0_index.root_hash: raise RuntimeError('L0 root mismatch')
