from __future__ import annotations
import os
from pathlib import Path
from time import perf_counter
from road_network import RoadNetwork
from edge_entries import build_edge_entries_from_files
from edge_mmr import build_all_edge_mmrs
from spatial_skeleton import SpatialSkeletonBuilder
from l1_merkle import build_all_l1_trees
from l0_authenticated import build_authenticated_l0
from index_io import PersistedIndex, save_index

# Edit these paths for your local dataset.
NODE_FILE=r'E:\MMR_Trajectory_range\xian_nodes.txt'
EDGE_FILE=r'E:\MMR_Trajectory_range\xian_edges.txt'
TRAJECTORY_DIR=Path(r'E:\Graph-Diffusion-Planning-main\loader\preprocess\mm\sets_data\real2\trajectories')
TRAJECTORY_GLOB='traj_mapped_xian_xian10-*.json'
THETA=64
INDEX_FILE='index_output/xian_trajectory_index.dat'

# # Edit these paths for your local dataset.
# NODE_FILE=r'E:\MMR_Trajectory_range\chengdu_nodes.txt'
# EDGE_FILE=r'E:\MMR_Trajectory_range\chengdu_edges.txt'
# TRAJECTORY_DIR=Path(r'E:\Graph-Diffusion-Planning-main\chengdu-tra-json')
# TRAJECTORY_GLOB='traj-*.json'
# THETA=64
# INDEX_FILE='index_output/chengdu_trajectory_index.dat'


def main():
    files=sorted(TRAJECTORY_DIR.glob(TRAJECTORY_GLOB),key=lambda p:int(p.stem.rsplit('-',1)[1]))
    files=[str(p) for p in files]
    if not files: raise RuntimeError('No trajectory files found; edit paths at top of build_index.py')
    road=RoadNetwork(); road.load(node_file=NODE_FILE,edge_file=EDGE_FILE)
    store=build_edge_entries_from_files(trajectory_files=files,road=road)
    t0=perf_counter()
    mmr=build_all_edge_mmrs(store=store,road=road)
    skeleton=SpatialSkeletonBuilder(road=road,theta=THETA).build()
    edge_blocks=build_all_l1_trees(skeleton=skeleton,mmr_index=mmr,road=road)
    l0=build_authenticated_l0(skeleton=skeleton,l1_index=edge_blocks)
    elapsed=perf_counter()-t0
    idx=PersistedIndex(road,store,mmr,skeleton,edge_blocks,l0,THETA,tuple(files))
    size=save_index(idx,INDEX_FILE)
    print('Index Construction Time:',f'{elapsed:.9f} s')
    print('root_S:',l0.root_hash.hex())
    print('Index file:',INDEX_FILE,'size=',size,'bytes')
    print('Layer note: l1_index now stores eid-sorted E_h edge blocks, not binary Merkle trees.')

if __name__=='__main__': main()
