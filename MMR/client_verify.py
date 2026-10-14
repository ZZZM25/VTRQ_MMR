from __future__ import annotations
from time import perf_counter
from road_network import RoadNetwork
from vo_io import load_composite_vo_tokens,load_verification_set,load_trusted_root
from verifier import verify_composite_vo
from trajectory_catalog import load_trajectory_catalog,fine_filter_candidates

NODE_FILE=r'E:\MMR_Trajectory_range\xian_nodes.txt'; EDGE_FILE=r'E:\MMR_Trajectory_range\xian_edges.txt'
VO_FILE='query_output/xian_composite_vo.dat'; VS_FILE='query_output/xian_verification_set.dat'; ROOT_FILE='query_output/xian_root_S.dat'; TRAJECTORY_CSV='catalog_output/xian_trajectory_catalog.csv'
QUERY_MIN_LON=104.0400; QUERY_MIN_LAT=30.6550; QUERY_MAX_LON=104.0900; QUERY_MAX_LAT=30.7050
QUERY_START=1538415951; QUERY_END=1538628400

def main():
    tokens=load_composite_vo_tokens(VO_FILE); vs=load_verification_set(VS_FILE); root=load_trusted_root(ROOT_FILE)
    t0=perf_counter(); report=verify_composite_vo(tokens=tokens,verification_set=vs,road=None,trusted_root=root,
        query_min_lon=QUERY_MIN_LON,query_min_lat=QUERY_MIN_LAT,query_max_lon=QUERY_MAX_LON,query_max_lat=QUERY_MAX_LAT,query_start=QUERY_START,query_end=QUERY_END)
    vt=perf_counter()-t0
    print('Verification successful:',report.success); print('Verification Time:',f'{vt:.9f} s'); print('Candidates:',len(report.candidate_ids))
    # Road data is no longer needed for authentication verification. It is still used by the unchanged fine-filter stage.
    road=RoadNetwork(); road.load(node_file=NODE_FILE,edge_file=EDGE_FILE)
    catalog=load_trajectory_catalog(TRAJECTORY_CSV)
    t0=perf_counter(); final_ids=fine_filter_candidates(candidate_ids=report.candidate_ids,catalog=catalog,road=road,
        query_min_lon=QUERY_MIN_LON,query_min_lat=QUERY_MIN_LAT,query_max_lon=QUERY_MAX_LON,query_max_lat=QUERY_MAX_LAT,query_start=QUERY_START,query_end=QUERY_END)
    print('Fine Filtering Time:',f'{perf_counter()-t0:.9f} s'); print('Final results:',len(final_ids))

if __name__=='__main__': main()
