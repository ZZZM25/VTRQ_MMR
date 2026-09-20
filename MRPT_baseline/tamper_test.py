from __future__ import annotations

import argparse,copy

from baseline.config import load_config,resolve_path
from baseline.persistence import load_index
from baseline.server import BaselineServerQuery
from baseline.verifier import verify_vo,VerificationError
from baseline.models import VerificationEntry


def expect_fail(label,fn):
    try: fn()
    except VerificationError as e: print("PASS",label,"->",e); return
    raise RuntimeError("tamper was not detected: "+label)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--min-lon",type=float,required=True); ap.add_argument("--min-lat",type=float,required=True); ap.add_argument("--max-lon",type=float,required=True); ap.add_argument("--max-lat",type=float,required=True); ap.add_argument("--start",type=int,required=True); ap.add_argument("--end",type=int,required=True); a=ap.parse_args()
    cfg=load_config(a.config); index=load_index(resolve_path(cfg,cfg["index_file"])); q=(a.min_lon,a.min_lat,a.max_lon,a.max_lat,a.start,a.end)
    resp=BaselineServerQuery(index,*q).build(); verify_vo(resp.vo,index.road,index.root_hash,*q,resp.candidate_ids)
    if resp.vo.verification_set:
        bad=copy.deepcopy(resp.vo); e=bad.verification_set[0]; bad.verification_set[0]=VerificationEntry(e.eid,e.trajectory_id,e.start,e.end+1)
        expect_fail("modify entry timestamp",lambda:verify_vo(bad,index.road,index.root_hash,*q,resp.candidate_ids))
        bad=copy.deepcopy(resp.vo); bad.verification_set.pop()
        expect_fail("remove verification entry",lambda:verify_vo(bad,index.road,index.root_hash,*q,resp.candidate_ids))
    if resp.vo.tokens:
        bad=copy.deepcopy(resp.vo); t=list(bad.tokens[-1]);
        if len(t)>=5 and isinstance(t[1],float): t[1]+=1e-6
        bad.tokens[-1]=tuple(t)
        expect_fail("modify spatial MBR",lambda:verify_vo(bad,index.road,index.root_hash,*q,resp.candidate_ids))
    print("tamper tests completed")

if __name__=="__main__": main()
