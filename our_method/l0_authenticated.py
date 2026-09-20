from __future__ import annotations

import hashlib
import struct

from array import array
from time import perf_counter

from road_network import RoadNetwork

from edge_entries import (
    build_edge_entries_from_files,
    UINT32_MAX,
)

from edge_mmr import (
    build_all_edge_mmrs,
)

from spatial_skeleton import (
    SpatialSkeleton,
    SpatialSkeletonBuilder,
    NODE_LEAF,
    NODE_INTERNAL,
)

from l1_merkle import (
    L1MerkleIndex,
    build_all_l1_trees,
    validate_l1,
)


# ============================================================
# Basic Parameters
# ============================================================

HASH_SIZE = 32

_sha256 = hashlib.sha256


# ============================================================
# L0 Node Metadata Encoding
#
# MBR:
#
#     min_lon     float64
#     min_lat     float64
#     max_lon     float64
#     max_lat     float64
#
# Time:
#
#     min_start   uint32
#     max_end     uint32
#
#
# Total size:
#
# 8 * 4 + 4 * 2
# =
# 40 bytes
#
# Use fixed big-endian binary encoding,
# without using strings.
# ============================================================

_L0_META_STRUCT = struct.Struct(
    ">ddddII"
)


# ============================================================
# L0 Leaf Hash
# ============================================================

def hash_l0_leaf(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    min_start: int,
    max_end: int,
    l1_root: bytes,
) -> bytes:
    """
    L0 Leaf:

        H(
            MBR
            ||
            min_start
            ||
            max_end
            ||
            L1root
        )
    """

    h = _sha256()

    h.update(
        _L0_META_STRUCT.pack(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            min_start,
            max_end,
        )
    )

    h.update(
        l1_root
    )

    return h.digest()


# ============================================================
# L0 Internal Hash
# ============================================================

def hash_l0_internal(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    min_start: int,
    max_end: int,
    left_hash: bytes,
    mid_hash: bytes,
    right_hash: bytes,
) -> bytes:
    """
    L0 Internal:

        H(
            MBR
            ||
            min_start
            ||
            max_end
            ||
            h_left
            ||
            h_mid
            ||
            h_right
        )

    Child order is fixed:

        left
        mid
        right

    Must not be changed.
    """

    h = _sha256()

    h.update(
        _L0_META_STRUCT.pack(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            min_start,
            max_end,
        )
    )

    h.update(
        left_hash
    )

    h.update(
        mid_hash
    )

    h.update(
        right_hash
    )

    return h.digest()


# ============================================================
# Merge Time Ranges of Three Children
# ============================================================

def merge_three_time_ranges(
    left_min_start: int,
    left_max_end: int,
    mid_min_start: int,
    mid_max_end: int,
    right_min_start: int,
    right_max_end: int,
) -> tuple[int, int]:
    """
    Empty-node time range:

        min_start = UINT32_MAX
        max_end   = 0

    Therefore, direct min/max is sufficient.

    If all three children are empty:

        min_start = UINT32_MAX
        max_end   = 0
    """

    min_start = min(
        left_min_start,
        mid_min_start,
        right_min_start,
    )

    max_end = max(
        left_max_end,
        mid_max_end,
        right_max_end,
    )

    return (
        min_start,
        max_end,
    )


# ============================================================
# L0 Authentication Index
# ============================================================

class L0AuthenticatedIndex:
    """
    Store the fully authenticated L0.

    Each L0 node finally stores:

        min_start
        max_end
        hash

    The spatial MBR and child structure remain stored in:

        SpatialSkeleton

    They are not duplicated here.


    ----------------------------------------------------------
    Why can this be computed sequentially?

    spatial_skeleton.py already guarantees during construction:

        left
        mid
        right
        parent

    are stored in postorder.

    Therefore, when computing a parent,
    all three children have already been computed.
    ----------------------------------------------------------
    """

    __slots__ = (
        "node_hashes",
        "node_min_start",
        "node_max_end",

        "root_index",
        "root_hash",
    )

    def __init__(
        self,
        node_count: int,
    ):

        # ====================================================
        # Each node uses a fixed 32-byte hash
        # ====================================================

        self.node_hashes = bytearray(
            HASH_SIZE
            * node_count
        )

        # ====================================================
        # Time range of each L0 node
        # ====================================================

        self.node_min_start = array(
            "I",
            [UINT32_MAX] * node_count,
        )

        self.node_max_end = array(
            "I",
            [0] * node_count,
        )

        self.root_index = -1

        self.root_hash = bytes(
            HASH_SIZE
        )

    # ========================================================
    # Set Node Hash
    # ========================================================

    def set_node_hash(
        self,
        node_index: int,
        node_hash: bytes,
    ) -> None:

        offset = (
            node_index
            * HASH_SIZE
        )

        self.node_hashes[
            offset:
            offset + HASH_SIZE
        ] = node_hash

    # ========================================================
    # Get Node Hash
    # ========================================================

    def get_node_hash(
        self,
        node_index: int,
    ) -> bytes:

        offset = (
            node_index
            * HASH_SIZE
        )

        return bytes(
            self.node_hashes[
                offset:
                offset + HASH_SIZE
            ]
        )


# ============================================================
# Build Authenticated L0
# ============================================================

def build_authenticated_l0(
    skeleton: SpatialSkeleton,
    l1_index: L1MerkleIndex,
) -> L0AuthenticatedIndex:
    """
    Scan all L0 nodes from beginning to end.

    Leaf:
        Obtain from L1:
            min_start
            max_end
            L1root

        Compute:
            H(
                MBR
                || time
                || L1root
            )


    Internal:
        Obtain from the three children:
            min_start
            max_end
            child hashes

        Compute:
            H(
                MBR
                || time
                || left
                || mid
                || right
            )
    """

    node_count = (
        skeleton.node_count
    )

    index = (
        L0AuthenticatedIndex(
            node_count=node_count
        )
    )

    node_type = (
        skeleton.node_type
    )

    left_child = (
        skeleton.left_child
    )

    mid_child = (
        skeleton.mid_child
    )

    right_child = (
        skeleton.right_child
    )

    min_lon_array = (
        skeleton.min_lon
    )

    min_lat_array = (
        skeleton.min_lat
    )

    max_lon_array = (
        skeleton.max_lon
    )

    max_lat_array = (
        skeleton.max_lat
    )

    node_min_start = (
        index.node_min_start
    )

    node_max_end = (
        index.node_max_end
    )

    # ========================================================
    # Postorder sequence:
    #
    # 0 → 1 → 2 → ... → root
    # ========================================================

    for node_index in range(
        node_count
    ):

        # ====================================================
        # L0 Leaf
        # ====================================================

        if (
            node_type[node_index]
            == NODE_LEAF
        ):

            min_start = (
                l1_index.l0_min_start[
                    node_index
                ]
            )

            max_end = (
                l1_index.l0_max_end[
                    node_index
                ]
            )

            l1_root = (
                l1_index.get_l1_root(
                    node_index
                )
            )

            node_hash = (
                hash_l0_leaf(
                    min_lon=
                    min_lon_array[
                        node_index
                    ],

                    min_lat=
                    min_lat_array[
                        node_index
                    ],

                    max_lon=
                    max_lon_array[
                        node_index
                    ],

                    max_lat=
                    max_lat_array[
                        node_index
                    ],

                    min_start=
                    min_start,

                    max_end=
                    max_end,

                    l1_root=
                    l1_root,
                )
            )

            node_min_start[
                node_index
            ] = min_start

            node_max_end[
                node_index
            ] = max_end

            index.set_node_hash(
                node_index,
                node_hash,
            )

        # ====================================================
        # L0 Internal
        # ====================================================

        elif (
            node_type[node_index]
            == NODE_INTERNAL
        ):

            left = (
                left_child[
                    node_index
                ]
            )

            mid = (
                mid_child[
                    node_index
                ]
            )

            right = (
                right_child[
                    node_index
                ]
            )

            # =================================================
            # Postorder Structure Check
            #
            # All three children must already exist.
            # =================================================

            if (
                left < 0
                or
                mid < 0
                or
                right < 0
            ):

                raise RuntimeError(
                    f"L0 Internal node={node_index} "
                    f"invalid child index"
                )

            if (
                left >= node_index
                or
                mid >= node_index
                or
                right >= node_index
            ):

                raise RuntimeError(
                    f"L0 node={node_index} "
                    f"invalid postorder structure"
                )

            # =================================================
            # Merge Time Ranges of Three Children
            # =================================================

            (
                min_start,
                max_end,
            ) = (
                merge_three_time_ranges(
                    node_min_start[
                        left
                    ],
                    node_max_end[
                        left
                    ],

                    node_min_start[
                        mid
                    ],
                    node_max_end[
                        mid
                    ],

                    node_min_start[
                        right
                    ],
                    node_max_end[
                        right
                    ],
                )
            )

            # =================================================
            # Hashes of the Three Children
            # =================================================

            left_hash = (
                index.get_node_hash(
                    left
                )
            )

            mid_hash = (
                index.get_node_hash(
                    mid
                )
            )

            right_hash = (
                index.get_node_hash(
                    right
                )
            )

            # =================================================
            # Internal Hash
            # =================================================

            node_hash = (
                hash_l0_internal(
                    min_lon=
                    min_lon_array[
                        node_index
                    ],

                    min_lat=
                    min_lat_array[
                        node_index
                    ],

                    max_lon=
                    max_lon_array[
                        node_index
                    ],

                    max_lat=
                    max_lat_array[
                        node_index
                    ],

                    min_start=
                    min_start,

                    max_end=
                    max_end,

                    left_hash=
                    left_hash,

                    mid_hash=
                    mid_hash,

                    right_hash=
                    right_hash,
                )
            )

            node_min_start[
                node_index
            ] = min_start

            node_max_end[
                node_index
            ] = max_end

            index.set_node_hash(
                node_index,
                node_hash,
            )

        else:

            raise RuntimeError(
                f"Unknown L0 node type: "
                f"node={node_index}"
            )

    # ========================================================
    # Root
    # ========================================================

    root_index = (
        skeleton.root_index
    )

    if (
        root_index < 0
        or
        root_index >= node_count
    ):

        raise RuntimeError(
            "Invalid L0 root index"
        )

    index.root_index = (
        root_index
    )

    index.root_hash = (
        index.get_node_hash(
            root_index
        )
    )

    return index


# ============================================================
# Validate the Entire L0
# ============================================================

def validate_authenticated_l0(
    skeleton: SpatialSkeleton,
    l1_index: L1MerkleIndex,
    l0_index: L0AuthenticatedIndex,
) -> None:
    """
    Fully recheck:

        Leaf Hash
        Internal time range
        Internal Hash
        Root
    """

    for node_index in range(
        skeleton.node_count
    ):

        # ====================================================
        # Leaf
        # ====================================================

        if (
            skeleton.node_type[
                node_index
            ]
            == NODE_LEAF
        ):

            expected_min_start = (
                l1_index.l0_min_start[
                    node_index
                ]
            )

            expected_max_end = (
                l1_index.l0_max_end[
                    node_index
                ]
            )

            if (
                l0_index.node_min_start[
                    node_index
                ]
                != expected_min_start
            ):

                raise RuntimeError(
                    f"L0 Leaf node={node_index} "
                    f"incorrect min_start"
                )

            if (
                l0_index.node_max_end[
                    node_index
                ]
                != expected_max_end
            ):

                raise RuntimeError(
                    f"L0 Leaf node={node_index} "
                    f"incorrect max_end"
                )

            expected_hash = (
                hash_l0_leaf(
                    skeleton.min_lon[
                        node_index
                    ],

                    skeleton.min_lat[
                        node_index
                    ],

                    skeleton.max_lon[
                        node_index
                    ],

                    skeleton.max_lat[
                        node_index
                    ],

                    expected_min_start,
                    expected_max_end,

                    l1_index.get_l1_root(
                        node_index
                    ),
                )
            )

        # ====================================================
        # Internal
        # ====================================================

        else:

            left = (
                skeleton.left_child[
                    node_index
                ]
            )

            mid = (
                skeleton.mid_child[
                    node_index
                ]
            )

            right = (
                skeleton.right_child[
                    node_index
                ]
            )

            (
                expected_min_start,
                expected_max_end,
            ) = (
                merge_three_time_ranges(
                    l0_index.node_min_start[
                        left
                    ],
                    l0_index.node_max_end[
                        left
                    ],

                    l0_index.node_min_start[
                        mid
                    ],
                    l0_index.node_max_end[
                        mid
                    ],

                    l0_index.node_min_start[
                        right
                    ],
                    l0_index.node_max_end[
                        right
                    ],
                )
            )

            if (
                l0_index.node_min_start[
                    node_index
                ]
                != expected_min_start
            ):

                raise RuntimeError(
                    f"L0 Internal node={node_index} "
                    f"incorrect min_start"
                )

            if (
                l0_index.node_max_end[
                    node_index
                ]
                != expected_max_end
            ):

                raise RuntimeError(
                    f"L0 Internal node={node_index} "
                    f"incorrect max_end"
                )

            expected_hash = (
                hash_l0_internal(
                    skeleton.min_lon[
                        node_index
                    ],

                    skeleton.min_lat[
                        node_index
                    ],

                    skeleton.max_lon[
                        node_index
                    ],

                    skeleton.max_lat[
                        node_index
                    ],

                    expected_min_start,
                    expected_max_end,

                    l0_index.get_node_hash(
                        left
                    ),

                    l0_index.get_node_hash(
                        mid
                    ),

                    l0_index.get_node_hash(
                        right
                    ),
                )
            )

        actual_hash = (
            l0_index.get_node_hash(
                node_index
            )
        )

        if (
            actual_hash
            != expected_hash
        ):

            raise RuntimeError(
                f"L0 node={node_index} "
                f"hash verification failed"
            )

    # ========================================================
    # Root Check
    # ========================================================

    if (
        l0_index.root_index
        != skeleton.root_index
    ):

        raise RuntimeError(
            "L0 root index mismatch"
        )

    if (
        l0_index.root_hash
        != l0_index.get_node_hash(
            skeleton.root_index
        )
    ):

        raise RuntimeError(
            "Invalid root_S"
        )


# ============================================================
# Print One L0 Leaf
# ============================================================

def print_l0_leaf(
    node_index: int,
    skeleton: SpatialSkeleton,
    l1_index: L1MerkleIndex,
    l0_index: L0AuthenticatedIndex,
) -> None:

    print(
        "L0 Leaf node:",
        node_index,
    )

    print(
        "MBR:",
        (
            skeleton.min_lon[
                node_index
            ],

            skeleton.min_lat[
                node_index
            ],

            skeleton.max_lon[
                node_index
            ],

            skeleton.max_lat[
                node_index
            ],
        ),
    )

    print(
        "Road edge count:",
        skeleton.leaf_edge_count[
            node_index
        ],
    )

    print(
        "min_start:",
        l0_index.node_min_start[
            node_index
        ],
    )

    print(
        "max_end:",
        l0_index.node_max_end[
            node_index
        ],
    )

    print(
        "L1 root:"
    )

    print(
        l1_index.get_l1_root(
            node_index
        ).hex()
    )

    print(
        "L0 Leaf hash:"
    )

    print(
        l0_index.get_node_hash(
            node_index
        ).hex()
    )


# ============================================================
# Run Directly with the PyCharm Green Triangle
# ============================================================

def main():

    # ========================================================
    # Road Network
    # ========================================================

    NODE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_nodes.txt"
    )

    EDGE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_edges.txt"
    )

    # ========================================================
    # Trajectory Files
    # ========================================================

    TRAJECTORY_FILES = [
        r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-1.json",

        # Additional files:
        #
        # r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-2.json",
        # r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-3.json",
    ]

    THETA = 64

    total_start = (
        perf_counter()
    )

    # ========================================================
    # 1. Road Network
    # ========================================================

    print(
        "===== 1. Load Road Network ====="
    )

    road = RoadNetwork()

    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )

    print(
        "Node count:",
        road.node_count,
    )

    print(
        "Road edge count:",
        road.edge_count,
    )

    # ========================================================
    # 2. Edge Entry
    # ========================================================

    print()

    print(
        "===== 2. Build Edge Entries ====="
    )

    entry_start = (
        perf_counter()
    )

    store = (
        build_edge_entries_from_files(
            trajectory_files=
            TRAJECTORY_FILES,

            road=
            road,
        )
    )

    entry_elapsed = (
        perf_counter()
        - entry_start
    )

    print(
        "Total Entry count:",
        store.total_entries,
    )

    # ========================================================
    # 3. Edge τ-MMR
    # ========================================================

    print()

    print(
        "===== 3. Build Edge τ-MMR ====="
    )

    mmr_start = (
        perf_counter()
    )

    mmr_index = (
        build_all_edge_mmrs(
            store=store,
            road=road,
        )
    )

    mmr_elapsed = (
        perf_counter()
        - mmr_start
    )

    print(
        "Total MMR nodes:",
        mmr_index.total_nodes,
    )

    print(
        "MMR construction time:",
        f"{mmr_elapsed:.6f} s",
    )

    # ========================================================
    # 4. L0 Spatial Skeleton
    # ========================================================

    print()

    print(
        "===== 4. Build L0 Spatial Skeleton ====="
    )

    spatial_start = (
        perf_counter()
    )

    skeleton_builder = (
        SpatialSkeletonBuilder(
            road=road,
            theta=THETA,
        )
    )

    skeleton = (
        skeleton_builder.build()
    )

    spatial_elapsed = (
        perf_counter()
        - spatial_start
    )

    print(
        "L0 node count:",
        skeleton.node_count,
    )

    print(
        "L0 Leaf count:",
        skeleton.leaf_count,
    )

    print(
        "L0 spatial skeleton time:",
        f"{spatial_elapsed:.6f} s",
    )

    # ========================================================
    # 5. L1
    # ========================================================

    print()

    print(
        "===== 5. Build L1 Merkle Tree ====="
    )

    l1_start = (
        perf_counter()
    )

    l1_index = (
        build_all_l1_trees(
            skeleton=skeleton,
            mmr_index=mmr_index,
            road=road,
        )
    )

    l1_elapsed = (
        perf_counter()
        - l1_start
    )

    validate_l1(
        l1_index=
        l1_index,

        skeleton=
        skeleton,

        road=
        road,
    )

    print(
        "L1 Tree count:",
        l1_index.tree_count,
    )

    print(
        "L1 Leaf count:",
        l1_index.total_leaves,
    )

    print(
        "L1 Node count:",
        l1_index.total_nodes,
    )

    print(
        "L1 construction time:",
        f"{l1_elapsed:.6f} s",
    )

    # ========================================================
    # 6. Authenticated L0
    # ========================================================

    print()

    print(
        "===== 6. Compute Authenticated L0 ====="
    )

    l0_start = (
        perf_counter()
    )

    l0_index = (
        build_authenticated_l0(
            skeleton=skeleton,
            l1_index=l1_index,
        )
    )

    l0_elapsed = (
        perf_counter()
        - l0_start
    )

    # ========================================================
    # Full Validation
    # ========================================================

    print(
        "Checking L0 Hash..."
    )

    validate_authenticated_l0(
        skeleton=skeleton,
        l1_index=l1_index,
        l0_index=l0_index,
    )

    print(
        "L0 Hash validation passed"
    )

    # ========================================================
    # Root_S
    # ========================================================

    root = (
        l0_index.root_index
    )

    print()

    print(
        "========================================"
    )

    print(
        "===== Global Authenticated Root root_S ====="
    )

    print(
        "root index:",
        root,
    )

    print(
        "root MBR:",
        (
            skeleton.min_lon[root],
            skeleton.min_lat[root],
            skeleton.max_lon[root],
            skeleton.max_lat[root],
        ),
    )

    print(
        "root min_start:",
        l0_index.node_min_start[
            root
        ],
    )

    print(
        "root max_end:",
        l0_index.node_max_end[
            root
        ],
    )

    print(
        "root_S:"
    )

    print(
        l0_index.root_hash.hex()
    )

    # ========================================================
    # Print the First Non-Empty Leaf
    # ========================================================

    test_leaf = -1

    for node_index in range(
        skeleton.node_count
    ):

        if (
            skeleton.node_type[
                node_index
            ]
            != NODE_LEAF
        ):

            continue

        if (
            skeleton.leaf_edge_count[
                node_index
            ]
            > 0
        ):

            test_leaf = (
                node_index
            )

            break

    print()

    print(
        "========================================"
    )

    print(
        "===== L0 Leaf Example ====="
    )

    print_l0_leaf(
        node_index=
        test_leaf,

        skeleton=
        skeleton,

        l1_index=
        l1_index,

        l0_index=
        l0_index,
    )

    # ========================================================
    # Timing Statistics
    # ========================================================

    total_elapsed = (
        perf_counter()
        - total_start
    )

    print()

    print(
        "========================================"
    )

    print(
        "===== Construction Time Statistics ====="
    )

    print(
        "Entry:",
        f"{entry_elapsed:.6f} s",
    )

    print(
        "MMR:",
        f"{mmr_elapsed:.6f} s",
    )

    print(
        "L0 Spatial:",
        f"{spatial_elapsed:.6f} s",
    )

    print(
        "L1:",
        f"{l1_elapsed:.6f} s",
    )

    print(
        "L0 Authentication:",
        f"{l0_elapsed:.6f} s",
    )

    print(
        "Total time:",
        f"{total_elapsed:.6f} s",
    )

    print()

    print(
        "l0_authenticated.py completed successfully"
    )


if __name__ == "__main__":
    main()
