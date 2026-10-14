from __future__ import annotations

import os
from time import perf_counter


# ============================================================
# 0. Beijing CSV Adapter
#
# Must be installed before importing edge_entries.
# ============================================================

from beijing_csv_adapter import (
    install_beijing_adapter,
)

install_beijing_adapter()


# ============================================================
# All modules below continue to use the original project code.
# ============================================================

from road_network import RoadNetwork

import edge_entries

from edge_mmr import (
    build_all_edge_mmrs,
)

from spatial_skeleton import (
    SpatialSkeletonBuilder,
)

from l1_merkle import (
    build_all_l1_trees,
)

from l0_authenticated import (
    build_authenticated_l0,
)

from index_io import (
    PersistedIndex,
    save_index,
)

# Defensive rebinding:
# If edge_entries.py uses
# from trajectory_parser import ...
# ensure that it uses the Beijing version here.
from beijing_csv_adapter import (
    load_beijing_trajectory_csv,
    parse_beijing_trajectory,
)

if hasattr(
    edge_entries,
    "load_trajectory_json",
):
    edge_entries.load_trajectory_json = (
        load_beijing_trajectory_csv
    )

if hasattr(
    edge_entries,
    "parse_trajectory",
):
    edge_entries.parse_trajectory = (
        parse_beijing_trajectory
    )


def main():

    # ========================================================
    # Three preprocessed Beijing files
    # ========================================================

    NODE_FILE = (
        r"E:\MMR_Trajectory_range"
        r"\beijing_preprocessed"
        r"\beijing_nodes.txt"
    )

    EDGE_FILE = (
        r"E:\MMR_Trajectory_range"
        r"\beijing_preprocessed"
        r"\beijing_edges.txt"
    )

    TRAJECTORY_FILE = (
        r"E:\MMR_Trajectory_range"
        r"\beijing_preprocessed"
        r"\beijing_trajectories.csv"
    )

    TRAJECTORY_FILES = [
        TRAJECTORY_FILE,
    ]

    THETA = 64

    INDEX_DIR = (
        "index_output"
    )

    INDEX_FILE = os.path.join(
        INDEX_DIR,
        "beijing_trajectory_index.dat",
    )

    os.makedirs(
        INDEX_DIR,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load road network
    #
    # Not included in Index Construction Time.
    # ========================================================

    print(
        "===== Beijing Index Build ====="
    )

    print()

    print(
        "===== Load Beijing Road Network ====="
    )

    road = RoadNetwork()

    road.load(
        node_file=
        NODE_FILE,

        edge_file=
        EDGE_FILE,
    )

    print(
        "Road network loaded"
    )

    print(
        "max_eid:",
        road.max_eid,
    )

    # ========================================================
    # 2. Beijing CSV -> Entry Store
    #
    # Continue to call the original project function:
    #
    #   build_edge_entries_from_files()
    #
    # The only difference is that the loader/parser has been
    # replaced with the Beijing CSV adapter.
    #
    # Not included in Index Construction Time.
    # ========================================================

    print()

    print(
        "===== Beijing CSV Trajectory Preprocessing / Entry Construction ====="
    )

    store = (
        edge_entries
        .build_edge_entries_from_files(
            trajectory_files=
            TRAJECTORY_FILES,

            road=
            road,
        )
    )

    print(
        "Entry preparation completed"
    )

    # ========================================================
    # 3. Index Construction Time
    #
    # Uses exactly the same measurement definition as
    # Chengdu/Xi'an:
    #
    # Included:
    #   τ-MMR
    #   L0 Spatial Skeleton
    #   L1 Merkle
    #   L0 Authentication
    #
    # Excluded:
    #   Road network file loading
    #   CSV trajectory loading
    #   Entry preprocessing
    #   Index saving
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "===== Index Construction ====="
    )

    index_t0 = (
        perf_counter()
    )

    # --------------------------------------------------------
    # Edge τ-MMR
    # --------------------------------------------------------

    print(
        "Building Edge τ-MMR ..."
    )

    mmr_index = (
        build_all_edge_mmrs(
            store=
            store,

            road=
            road,
        )
    )

    # --------------------------------------------------------
    # L0 Spatial Skeleton
    # --------------------------------------------------------

    print(
        "Building L0 Spatial Skeleton ..."
    )

    skeleton = (
        SpatialSkeletonBuilder(
            road=
            road,

            theta=
            THETA,
        ).build()
    )

    # --------------------------------------------------------
    # L1 Merkle
    # --------------------------------------------------------

    print(
        "Building L1 Merkle ..."
    )

    l1_index = (
        build_all_l1_trees(
            skeleton=
            skeleton,

            mmr_index=
            mmr_index,

            road=
            road,
        )
    )

    # --------------------------------------------------------
    # Authenticated L0
    # --------------------------------------------------------

    print(
        "Building Authenticated L0 ..."
    )

    l0_index = (
        build_authenticated_l0(
            skeleton=
            skeleton,

            l1_index=
            l1_index,
        )
    )

    index_construction_time = (
        perf_counter()
        -
        index_t0
    )

    trusted_root = (
        l0_index.root_hash
    )

    print()

    print(
        "Index Construction Time:",
        f"{index_construction_time:.9f} s",
    )

    print()

    print(
        "root_S:"
    )

    print(
        trusted_root.hex()
    )

    # ========================================================
    # 4. Persisted object
    # ========================================================

    persisted_index = (
        PersistedIndex(

            road=
            road,

            store=
            store,

            mmr_index=
            mmr_index,

            skeleton=
            skeleton,

            l1_index=
            l1_index,

            l0_index=
            l0_index,

            theta=
            THETA,

            trajectory_files=
            tuple(
                TRAJECTORY_FILES
            ),
        )
    )

    # ========================================================
    # 5. Save
    #
    # Not included in Index Construction Time.
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "===== Save Complete Beijing Index ====="
    )

    save_t0 = (
        perf_counter()
    )

    index_size = (
        save_index(
            index=
            persisted_index,

            file_path=
            INDEX_FILE,
        )
    )

    save_time = (
        perf_counter()
        -
        save_t0
    )

    print()

    print(
        "Index file:"
    )

    print(
        INDEX_FILE
    )

    print()

    print(
        "Index file size:",
        index_size,
        "bytes",
    )

    print(
        "Index file size:",
        f"{index_size / 1024.0 / 1024.0:.3f} MiB",
    )

    print()

    print(
        "Index save time:",
        f"{save_time:.9f} s",
    )

    print(
        "(Not included in Index Construction Time)"
    )

    print()

    print(
        "build_beijing_index.py completed successfully"
    )


if __name__ == "__main__":
    main()