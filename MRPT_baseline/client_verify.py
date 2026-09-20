from __future__ import annotations

import argparse
from time import perf_counter

from baseline.config import load_config,resolve_path
from baseline.road_network import RoadNetwork
from baseline.persistence import load_catalog
from baseline.vo_io import load_vo
from baseline.response_io import load_response_json
from baseline.verifier import verify_vo
from baseline.trajectory_catalog import fine_filter_candidates


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--response",required=True); a=ap.parse_args()
    cfg=load_config(a.config); data=load_response_json(a.response); q=data["query"]
    road=RoadNetwork(); road.load(str(resolve_path(cfg,cfg["node_file"])),str(resolve_path(cfg,cfg["edge_file"])))
    trusted=bytes.fromhex(resolve_path(cfg,cfg["trusted_root_file"]).read_text(encoding="ascii").strip())
    vo=load_vo(data["vo_file"])
    t0=perf_counter(); report=verify_vo(vo,road,trusted,q["min_lon"],q["min_lat"],q["max_lon"],q["max_lat"],q["start"],q["end"],data["candidate_ids"]); vt=perf_counter()-t0
    catalog=load_catalog(resolve_path(cfg,cfg["catalog_file"]))
    t0=perf_counter(); final=fine_filter_candidates(report.candidate_ids,catalog,road,q["min_lon"],q["min_lat"],q["max_lon"],q["max_lat"],q["start"],q["end"]); ft=perf_counter()-t0
    print("verification success:",report.success)
    print("reconstructed root:",report.reconstructed_root.hex())
    print("candidate trajectories:",len(report.candidate_ids))
    print("final trajectories:",len(final))
    print(f"Verification Time: {vt:.9f} s")
    print(f"Fine Filtering Time: {ft:.9f} s")

if __name__=="__main__": main()
