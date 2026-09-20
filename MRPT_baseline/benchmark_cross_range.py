from __future__ import annotations

import argparse,csv,statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from baseline.config import load_config,resolve_path
from baseline.persistence import load_index,load_catalog
from baseline.server import BaselineServerQuery
from baseline.verifier import verify_vo
from baseline.trajectory_catalog import fine_filter_candidates
from baseline.vo_io import serialize_vo
from baseline.benchmark_utils import SPATIAL_SIDES,TEMPORAL_RANGES,make_square,load_setup,write_matrix


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--setup",default=None,help="Use Ours selected setup CSV for a fair comparison"); a=ap.parse_args()
    cfg=load_config(a.config); bcfg=cfg.get("benchmark",{})
    index=load_index(resolve_path(cfg,cfg["index_file"])); catalog=load_catalog(resolve_path(cfg,cfg["catalog_file"]))
    setup=a.setup or bcfg.get("setup_file")
    if setup:
        p=Path(setup); p=p if p.is_absolute() else resolve_path(cfg,setup)
        center_lon,center_lat,base_start=load_setup(p,bcfg.get("base_query_start"))
    else:
        if "center_lon" not in bcfg or "center_lat" not in bcfg:
            raise ValueError("benchmark needs --setup (recommended) or benchmark.center_lon/center_lat in config")
        center_lon=float(bcfg["center_lon"]); center_lat=float(bcfg["center_lat"]); base_start=bcfg.get("base_query_start")
    if base_start is None: raise ValueError("base_query_start missing")
    repeats=int(bcfg.get("repeats",30)); warmups=int(bcfg.get("warmup_repeats",1))
    out=resolve_path(cfg,bcfg.get("output_dir","../benchmark_output/baseline")); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for side in SPATIAL_SIDES:
        qminx,qminy,qmaxx,qmaxy=make_square(center_lon,center_lat,side)
        for label,seconds in TEMPORAL_RANGES:
            qs=int(base_start); qe=qs+seconds
            for _ in range(warmups):
                r=BaselineServerQuery(index,qminx,qminy,qmaxx,qmaxy,qs,qe).build(); rep=verify_vo(r.vo,index.road,index.root_hash,qminx,qminy,qmaxx,qmaxy,qs,qe,r.candidate_ids); fine_filter_candidates(rep.candidate_ids,catalog,index.road,qminx,qminy,qmaxx,qmaxy,qs,qe)
            st=[]; vt=[]; ft=[]; last=None; last_rep=None; last_final=None
            for _ in range(repeats):
                t0=perf_counter(); r=BaselineServerQuery(index,qminx,qminy,qmaxx,qmaxy,qs,qe).build(); st.append(perf_counter()-t0)
                t0=perf_counter(); rep=verify_vo(r.vo,index.road,index.root_hash,qminx,qminy,qmaxx,qmaxy,qs,qe,r.candidate_ids); vt.append(perf_counter()-t0)
                t0=perf_counter(); final=fine_filter_candidates(rep.candidate_ids,catalog,index.road,qminx,qminy,qmaxx,qmaxy,qs,qe); ft.append(perf_counter()-t0)
                last,last_rep,last_final=r,rep,final
            vo_size=len(serialize_vo(last.vo))
            row={
                "spatial_side_km":side,"temporal_label":label,"temporal_seconds":seconds,
                "query_min_lon":qminx,"query_min_lat":qminy,"query_max_lon":qmaxx,"query_max_lat":qmaxy,"query_start":qs,"query_end":qe,
                "server_query_time_s":statistics.mean(st),"server_query_time_std_s":statistics.stdev(st) if len(st)>1 else 0,
                "fine_filtering_time_s":statistics.mean(ft),"fine_filtering_time_std_s":statistics.stdev(ft) if len(ft)>1 else 0,
                "query_time_s":statistics.mean([x+y for x,y in zip(st,ft)]),
                "verification_time_s":statistics.mean(vt),"verification_time_std_s":statistics.stdev(vt) if len(vt)>1 else 0,
                "vo_size_bytes":vo_size,"vo_size_mb":vo_size/1_000_000,
                "verification_set_count":len(last.vo.verification_set),"candidate_trajectory_count":len(last_rep.candidate_ids),"final_trajectory_count":len(last_final),
                "visited_spatial_nodes":last.stats.visited_spatial_nodes,"pruned_spatial_nodes":last.stats.pruned_spatial_nodes,
                "checked_cross_edges":last.stats.checked_cross_edges,"checked_leaf_edges":last.stats.checked_leaf_edges,
                "spatial_matched_edges":last.stats.spatial_matched_edges,"scanned_entries":last.stats.scanned_entries,"boundary_witness_entries":last.stats.boundary_witness_entries,
            }
            rows.append(row); print(f"{side:.0f}km x {label}: Query={row['query_time_s']:.6f}s Verify={row['verification_time_s']:.6f}s VO={row['vo_size_mb']:.3f}MB scanned={row['scanned_entries']}")
    result=out/"cross_range_results.csv"
    with open(result,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for name,field in [
        ("query_time_matrix_s.csv","query_time_s"),("server_query_time_matrix_s.csv","server_query_time_s"),("fine_filtering_time_matrix_s.csv","fine_filtering_time_s"),("verification_time_matrix_s.csv","verification_time_s"),("vo_size_matrix_mb.csv","vo_size_mb"),("candidate_count_matrix.csv","candidate_trajectory_count"),("final_result_count_matrix.csv","final_trajectory_count"),("scanned_entries_matrix.csv","scanned_entries"),("checked_cross_edges_matrix.csv","checked_cross_edges")]:
        write_matrix(out/name,rows,field)
    print("saved:",result)

if __name__=="__main__": main()
