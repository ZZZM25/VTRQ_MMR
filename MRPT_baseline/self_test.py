from __future__ import annotations

from baseline.road_network import RoadNetwork
from baseline.entry_store import PackedEdgeEntryStore
from baseline.models import TrajectorySegment
from baseline.trajectory_catalog import PackedTrajectoryCatalog,fine_filter_candidates
from baseline.spatial_tree import BinarySpatialTreeBuilder
from baseline.authenticated_index import BaselineIndex,build_edge_commitments,authenticate_spatial_tree
from baseline.server import BaselineServerQuery
from baseline.verifier import verify_vo
from baseline.trajectory_id import compute_trajectory_id


def tiny_road():
    r=RoadNetwork(); r.max_nid=5; r.node_lon=[0,1,2,0,1,2]; r.node_lat=[0,0,0,1,1,1]; r.node_present=[True]*6
    edges=[(0,0,1),(1,1,2),(2,3,4),(3,4,5),(4,1,4)]
    r.max_eid=4; r.edge_u=[-1]*5; r.edge_v=[-1]*5; r.edge_present=[False]*5; r.pair_to_eids={}
    for eid,u,v in edges: r.edge_u[eid]=u; r.edge_v[eid]=v; r.edge_present[eid]=True; r.pair_to_eids.setdefault(r._canonical_pair(u,v),[]).append(eid)
    return r


def main():
    # Ordered insertion must work even when arrivals are out of time order.
    probe = PackedEdgeEntryStore(0)
    probe.insert_by_start(0, b"a" * 32, 30, 31)
    probe.insert_by_start(0, b"b" * 32, 10, 11)
    probe.insert_by_start(0, b"c" * 32, 20, 21)
    probe.insert_by_start(0, b"d" * 32, 20, 22)
    assert list(probe.starts[0]) == [10, 20, 20, 30]
    assert [x[0] for x in probe.iter_entries(0)] == [b"b" * 32, b"c" * 32, b"d" * 32, b"a" * 32]

    r=tiny_road(); s=PackedEdgeEntryStore(r.max_eid); c=PackedTrajectoryCatalog()
    specs=[([0,1],10,30,[TrajectorySegment(0,10,1,20),TrajectorySegment(1,20,2,30)]),([2,3],100,130,[TrajectorySegment(3,100,4,115),TrajectorySegment(4,115,5,130)]),([4],15,25,[TrajectorySegment(1,15,4,25)])]
    for path,st,en,segs in specs:
        tid=compute_trajectory_id(path,st,en); c.add(tid,segs)
        for eid,seg in zip(path,segs): s.insert_by_start(eid,tid,seg.start_time,seg.end_time)
    s.validate_start_order(); er,eh,ss=build_edge_commitments(r,s); tree=BinarySpatialTreeBuilder(r,theta=2).build(); root=authenticate_spatial_tree(tree,eh); idx=BaselineIndex(r,s,tree,er,eh,ss,root,2,())
    q=(-0.1,-0.1,1.1,0.6,14,22); resp=BaselineServerQuery(idx,*q).build(); rep=verify_vo(resp.vo,r,root,*q,resp.candidate_ids); final=fine_filter_candidates(rep.candidate_ids,c,r,*q)
    assert rep.success and final
    print("SELF TEST PASS",len(resp.candidate_ids),len(resp.vo.verification_set),root.hex())

if __name__=="__main__": main()
