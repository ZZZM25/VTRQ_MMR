from __future__ import annotations

import os
from time import perf_counter


from road_network import RoadNetwork

from edge_entries import (
    build_edge_entries_from_files,
)

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


def main():

    # ========================================================
    # Data
    # ========================================================

    NODE_FILE = (
        r"E:\MMR_Trajectory_range\xian_nodes.txt"
    )

    EDGE_FILE = (
        r"E:\MMR_Trajectory_range\xian_edges.txt"
    )

    from pathlib import Path

    TRAJECTORY_DIR = Path(
        r"E:\Graph-Diffusion-Planning-main\loader\preprocess\mm\sets_data\real2\trajectories"
    )

    def trajectory_sort_key(
            path: Path,
    ):
        # traj-10-1.json
        stem = path.stem

        parts = stem.split("-")

        month = int(
            parts[1]
        )

        day = int(
            parts[2]
        )

        return (
            month,
            day,
        )

    TRAJECTORY_FILES = sorted(
        TRAJECTORY_DIR.glob(
            "traj_mapped_xian_xian10-*.json"
        ),
        key=lambda p: int(
            p.stem.rsplit("-", 1)[1]
        ),
    )

    TRAJECTORY_FILES = [
        str(path)
        for path in TRAJECTORY_FILES
    ]

    if not TRAJECTORY_FILES:
        raise RuntimeError(
            "No trajectory files found"
        )

    print(
        "Number of trajectory files for one month:",
        len(
            TRAJECTORY_FILES
        ),
    )

    print(
        "First file:",
        TRAJECTORY_FILES[0],
    )

    print(
        "Last file:",
        TRAJECTORY_FILES[-1],
    )

    THETA = 64

    INDEX_DIR = (
        "index_output"
    )

    INDEX_FILE = os.path.join(
        INDEX_DIR,
        "xian_trajectory_index.dat",
    )

    os.makedirs(
        INDEX_DIR,
        exist_ok=True,
    )

    # ========================================================
    # 1. Raw data loading / preprocessing
    #
    # Not included in Index Construction Time.
    # ========================================================

    print(
        "===== Load Road Network ====="
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

    print()

    print(
        "===== Trajectory Preprocessing / Entry Construction ====="
    )

    store = (
        build_edge_entries_from_files(
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
    # 2. Index Construction Time
    #
    # According to the unified experimental definition:
    #
    # Included:
    #   τ-MMR
    #   L0 Spatial Skeleton
    #   L1 Merkle
    #   L0 Authentication
    #
    # Excluded:
    #   File loading
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
        - index_t0
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
    # 3. Build the complete persisted index object
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
    # 4. Save index
    #
    # Save time is explicitly excluded from
    # Index Construction Time.
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "===== Save Complete Index ====="
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
        - save_t0
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
        "build_index.py completed successfully"
    )


if __name__ == "__main__":
    main()