from __future__ import annotations
import os
from time import perf_counter
from composite_vo import CompositeVOBuilder
from index_io import load_index
from vo_io import save_composite_vo,save_verification_set,save_trusted_root

INDEX_FILE='index_output/xian_trajectory_index.dat'
QUERY_MIN_LON=104.0400; QUERY_MIN_LAT=30.6550; QUERY_MAX_LON=104.0900; QUERY_MAX_LAT=30.7050
QUERY_START=1538415951; QUERY_END=1538628400
OUTPUT_DIR='query_output'; VO_FILE=os.path.join(OUTPUT_DIR,'xian_composite_vo.dat'); VS_FILE=os.path.join(OUTPUT_DIR,'xian_verification_set.dat'); ROOT_FILE=os.path.join(OUTPUT_DIR,'xian_root_S.dat')

def main():
    idx=load_index(INDEX_FILE); os.makedirs(OUTPUT_DIR,exist_ok=True)
    t0=perf_counter()
    vo=CompositeVOBuilder(idx.road,idx.store,idx.mmr_index,idx.skeleton,idx.l1_index,idx.l0_index,
                          QUERY_MIN_LON,QUERY_MIN_LAT,QUERY_MAX_LON,QUERY_MAX_LAT,QUERY_START,QUERY_END).build()
    query_time=perf_counter()-t0
    vo_size=save_composite_vo(vo,VO_FILE); vs_size=save_verification_set(vo.verification_set,VS_FILE); save_trusted_root(idx.l0_index.root_hash,ROOT_FILE)
    print('VO Token count:',len(vo.tokens)); print('Verification Set count:',len(vo.verification_set)); print('Candidate count:',len(vo.candidate_ids))
    print('Query Time:',f'{query_time:.9f} s'); print('Server Response size:',vo_size+vs_size,'bytes')

if __name__=='__main__': main()
