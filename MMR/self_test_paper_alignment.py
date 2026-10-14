from pathlib import Path
import hashlib, tempfile
from road_network import RoadNetwork
from edge_entries import EdgeEntryStore
from edge_mmr import build_all_edge_mmrs
from spatial_skeleton import SpatialSkeletonBuilder
from l1_merkle import build_all_l1_trees
from l0_authenticated import build_authenticated_l0
from composite_vo import CompositeVOBuilder
from verifier import verify_composite_vo
from vo_io import save_composite_vo,load_composite_vo_tokens,save_verification_set,load_verification_set,save_trusted_root,load_trusted_root
from index_io import PersistedIndex,save_index,load_index

def main():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        (d/'nodes.txt').write_text('\n'.join(['0 0 0','1 1 0','2 2 0','3 3 0','4 4 0','5 5 0']),encoding='utf-8')
        (d/'edges.txt').write_text('\n'.join(['0 0 1','1 1 2','2 2 3','3 3 4','4 4 5']),encoding='utf-8')
        road=RoadNetwork(); road.load(str(d/'nodes.txt'),str(d/'edges.txt'))
        store=EdgeEntryStore(road.max_eid)
        for eid in range(5):
            store.append(eid,hashlib.sha256(f't{eid}'.encode()).digest(),100+10*eid,150+10*eid)
        mmr=build_all_edge_mmrs(store,road)
        sk=SpatialSkeletonBuilder(road=road,theta=2).build()
        blocks=build_all_l1_trees(sk,mmr,road)
        l0=build_authenticated_l0(sk,blocks)
        vo=CompositeVOBuilder(road,store,mmr,sk,blocks,l0,1.2,-.5,3.2,.5,105,190).build()
        vof=d/'vo.dat'; vsf=d/'vs.dat'; rf=d/'root.dat'
        save_composite_vo(vo,str(vof)); save_verification_set(vo.verification_set,str(vsf)); save_trusted_root(l0.root_hash,str(rf))
        tokens=load_composite_vo_tokens(str(vof)); vs=load_verification_set(str(vsf)); root=load_trusted_root(str(rf))
        report=verify_composite_vo(tokens,vs,None,root,1.2,-.5,3.2,.5,105,190)
        assert report.success and report.reconstructed_root==l0.root_hash
        idx=PersistedIndex(road,store,mmr,sk,blocks,l0,2,tuple())
        ip=d/'index.dat'; save_index(idx,str(ip)); loaded=load_index(str(ip)); assert loaded.l0_index.root_hash==l0.root_hash
        print('PAPER ALIGNMENT SELF-TEST PASS')
        print('root_S =',l0.root_hash.hex())
        print('tokens =',len(tokens),'verification_entries =',len(vs))

if __name__=='__main__': main()
