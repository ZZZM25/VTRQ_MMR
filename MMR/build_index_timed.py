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


# ============================================================
# Dataset
# ============================================================

NODE_FILE = r'E:\MMR_Trajectory_range\chengdu_nodes.txt'
EDGE_FILE = r'E:\MMR_Trajectory_range\chengdu_edges.txt'
TRAJECTORY_DIR = Path(r'E:\Graph-Diffusion-Planning-main\chengdu-tra-json')
TRAJECTORY_GLOB = 'traj-*.json'

THETA = 64

INDEX_FILE = 'index_output/chengdu_trajectory_index.dat'


def main():
    files = sorted(
        TRAJECTORY_DIR.glob(TRAJECTORY_GLOB),
        key=lambda p: int(p.stem.rsplit('-', 1)[1]),
    )
    files = [str(p) for p in files]

    if not files:
        raise RuntimeError(
            'No trajectory files found; edit paths at top of build_index.py'
        )

    # --------------------------------------------------------
    # Excluded from Index Construction Time:
    # road loading
    # --------------------------------------------------------
    t0 = perf_counter()
    road = RoadNetwork()
    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )
    road_load_time = perf_counter() - t0

    # --------------------------------------------------------
    # Excluded from Index Construction Time:
    # trajectory parsing / EdgeEntryStore preparation
    # --------------------------------------------------------
    t0 = perf_counter()
    store = build_edge_entries_from_files(
        trajectory_files=files,
        road=road,
    )
    processing_time = perf_counter() - t0

    # ========================================================
    # Timed ADS construction
    # Same timing scope as the original build_index.py
    # ========================================================
    construction_t0 = perf_counter()

    # 1. tau-MMR
    t0 = perf_counter()
    mmr = build_all_edge_mmrs(
        store=store,
        road=road,
    )
    mmr_time = perf_counter() - t0

    # 2. Spatial Skeleton
    t0 = perf_counter()
    skeleton = SpatialSkeletonBuilder(
        road=road,
        theta=THETA,
    ).build()
    skeleton_time = perf_counter() - t0

    # 3. E_h edge blocks
    # Function name remains build_all_l1_trees only for compatibility.
    t0 = perf_counter()
    edge_blocks = build_all_l1_trees(
        skeleton=skeleton,
        mmr_index=mmr,
        road=road,
    )
    edge_block_time = perf_counter() - t0

    # 4. Authenticated SEM
    t0 = perf_counter()
    l0 = build_authenticated_l0(
        skeleton=skeleton,
        l1_index=edge_blocks,
    )
    sem_auth_time = perf_counter() - t0

    elapsed = perf_counter() - construction_t0

    # --------------------------------------------------------
    # Excluded from Index Construction Time:
    # serialization
    # --------------------------------------------------------
    idx = PersistedIndex(
        road,
        store,
        mmr,
        skeleton,
        edge_blocks,
        l0,
        THETA,
        tuple(files),
    )

    t0 = perf_counter()
    size = save_index(
        idx,
        INDEX_FILE,
    )
    serialization_time = perf_counter() - t0

    print()
    print("=" * 64)
    print("===== Build Timing Breakdown =====")
    print(f"Road Load Time (excluded):            {road_load_time:.9f} s")
    print(f"Processing Time (excluded):           {processing_time:.9f} s")
    print("-" * 64)
    print(f"tau-MMR Time:                         {mmr_time:.9f} s")
    print(f"Spatial Skeleton Time:                {skeleton_time:.9f} s")
    print(f"E_h Edge Block Time:                  {edge_block_time:.9f} s")
    print(f"SEM Authentication Time:              {sem_auth_time:.9f} s")
    print("-" * 64)
    print(f"Index Construction Time:              {elapsed:.9f} s")
    print(f"Serialization Time (excluded):        {serialization_time:.9f} s")

    if elapsed > 0:
        print()
        print("===== Construction Percentage =====")
        print(f"tau-MMR:             {mmr_time / elapsed * 100:8.3f}%")
        print(f"Spatial Skeleton:    {skeleton_time / elapsed * 100:8.3f}%")
        print(f"E_h Edge Block:      {edge_block_time / elapsed * 100:8.3f}%")
        print(f"SEM Authentication: {sem_auth_time / elapsed * 100:8.3f}%")

    print()
    print('root_S:', l0.root_hash.hex())
    print('Index file:', INDEX_FILE, 'size=', size, 'bytes')
    print(
        'Layer note: l1_index now stores eid-sorted E_h edge blocks, '
        'not binary Merkle trees.'
    )


if __name__ == '__main__':
    main()
