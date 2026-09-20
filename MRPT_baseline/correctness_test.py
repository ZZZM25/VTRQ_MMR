from __future__ import annotations

import argparse

from baseline.config import load_config,resolve_path
from baseline.persistence import load_index
from baseline.server import BaselineServerQuery
from baseline.verifier import verify_vo
from baseline.query_utils import edge_intersects_query
from baseline.crypto import time_overlap


def brute(index,q):
    out=set(); road=index.road; store=index.store
    for eid in range(road.max_eid+1):
        if not road.edge_exists(eid): continue
        if not edge_intersects_query(eid,road,q[0],q[1],q[2],q[3]): continue
        for tid,start,end in store.iter_entries(eid):
            if time_overlap(start,end,q[4],q[5]): out.add(tid)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--min-lon",type=float,required=True); ap.add_argument("--min-lat",type=float,required=True); ap.add_argument("--max-lon",type=float,required=True); ap.add_argument("--max-lat",type=float,required=True); ap.add_argument("--start",type=int,required=True); ap.add_argument("--end",type=int,required=True); a=ap.parse_args()
    cfg=load_config(a.config); index=load_index(resolve_path(cfg,cfg["index_file"])); q=(a.min_lon,a.min_lat,a.max_lon,a.max_lat,a.start,a.end)
    resp=BaselineServerQuery(index,*q).build(); report=verify_vo(resp.vo,index.road,index.root_hash,*q,resp.candidate_ids); brute_ids=brute(index,q)
    assert set(resp.candidate_ids)==brute_ids==set(report.candidate_ids)
    print("PASS: spatial-tree + list query == brute force; VO verification PASS")
    print("candidates:",len(brute_ids),"verification entries:",len(resp.vo.verification_set))

if __name__=="__main__": main()
