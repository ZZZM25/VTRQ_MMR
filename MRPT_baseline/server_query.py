from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from baseline.config import load_config,resolve_path
from baseline.persistence import load_index
from baseline.server import BaselineServerQuery
from baseline.vo_io import save_vo
from baseline.response_io import save_response_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    for name in ("min_lon","min_lat","max_lon","max_lat"): ap.add_argument("--"+name.replace("_","-"),dest=name,type=float,required=True)
    ap.add_argument("--start",type=int,required=True); ap.add_argument("--end",type=int,required=True); ap.add_argument("--out",default="query_output")
    a=ap.parse_args(); cfg=load_config(a.config)
    index=load_index(resolve_path(cfg,cfg["index_file"]))
    t0=perf_counter(); resp=BaselineServerQuery(index,a.min_lon,a.min_lat,a.max_lon,a.max_lat,a.start,a.end).build(); elapsed=perf_counter()-t0
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); vo_file=(out/"baseline_vo.bin").resolve(); response_file=(out/"response.json").resolve()
    vo_size=save_vo(resp.vo,vo_file)
    query={"min_lon":a.min_lon,"min_lat":a.min_lat,"max_lon":a.max_lon,"max_lat":a.max_lat,"start":a.start,"end":a.end}
    save_response_json(response_file,vo_file,query,resp.candidate_ids,resp.stats,elapsed)
    print(f"Server Query + VO Time: {elapsed:.9f} s")
    print(f"candidate trajectories: {len(resp.candidate_ids)}")
    print(f"verification entries (sorted prefix + boundary witnesses): {len(resp.vo.verification_set)}")
    print(f"VO size: {vo_size/1024/1024:.6f} MiB")
    print(f"response: {response_file}")

if __name__=="__main__": main()
