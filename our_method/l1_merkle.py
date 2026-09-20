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
    EdgeMMRIndex,
    build_all_edge_mmrs,
)

from spatial_skeleton import (
    SpatialSkeleton,
    SpatialSkeletonBuilder,
    NODE_LEAF,
)


# ============================================================
# Basic Parameters
# ============================================================

HASH_SIZE = 32


# ============================================================
# L1 Leaf Encoding
#
# eid         uint32
# k_e         uint32
# min_start   uint32
# max_end     uint32
# root_e      32 bytes
#
# Total:
#
# 4 + 4 + 4 + 4 + 32
# =
# 48 bytes
# ============================================================

_L1_LEAF_STRUCT = struct.Struct(
    ">IIII32s"
)

_sha256 = hashlib.sha256


# ============================================================
# L1 Hash
# ============================================================

def hash_l1_leaf(
    eid: int,
    k_e: int,
    min_start: int,
    max_end: int,
    root_e: bytes,
) -> bytes:
    """
    L1 Leaf:

        H(
            eid
            ||
            k_e
            ||
            min_start
            ||
            max_end
            ||
            root_e
        )
    """

    return _sha256(
        _L1_LEAF_STRUCT.pack(
            eid,
            k_e,
            min_start,
            max_end,
            root_e,
        )
    ).digest()


def hash_l1_internal(
    left_hash: bytes,
    right_hash: bytes,
) -> bytes:
    """
    Standard L1 Merkle Internal:

        H(
            left_hash
            ||
            right_hash
        )
    """

    h = _sha256()

    h.update(
        left_hash
    )

    h.update(
        right_hash
    )

    return h.digest()


# ============================================================
# Empty L1 Root
# ============================================================

EMPTY_L1_ROOT = (
    _sha256(b"").digest()
)


# ============================================================
# Merkle Split
# ============================================================

def largest_power_of_two_less_than(
    n: int,
) -> int:
    """
    Return the largest power of two strictly less than n.

    Used to build a deterministic Merkle Tree.

    For example:

        n = 8
        split = 4

        n = 7
        split = 4

        n = 6
        split = 4

        n = 5
        split = 4

        n = 4
        split = 2

        n = 3
        split = 2
    """

    if n < 2:

        raise ValueError(
            "n must be >= 2"
        )

    return (
        1
        << (
            (n - 1).bit_length()
            - 1
        )
    )


# ============================================================
# L1 Merkle Index
# ============================================================

class L1MerkleIndex:
    """
    Store the L1 Merkle Trees for all L0 Leaves.


    ==========================================================
    Global L1 Nodes
    ==========================================================

    All L1 Tree nodes are stored in unified contiguous arrays:

        node_hashes
        node_left
        node_right
        node_eid


    Leaf：

        node_eid = eid
        left     = -1
        right    = -1


    Internal：

        node_eid = -1
        left     = left child node index
        right    = right child node index


    ==========================================================
    Stored for Each L0 Leaf
    ==========================================================

        l0_l1_root_node

        l0_l1_root_hash

        l0_min_start

        l0_max_end


    These are used directly in the next step to compute:

        L0 Leaf Hash
    """

    __slots__ = (
        "node_hashes",
        "node_left",
        "node_right",
        "node_eid",

        "l0_l1_root_node",
        "l0_l1_root_hashes",

        "l0_min_start",
        "l0_max_end",

        "eid_to_l0_leaf",
        "eid_to_l1_leaf_node",

        "total_nodes",
        "total_leaves",
        "tree_count",
    )

    def __init__(
        self,
        l0_node_count: int,
        max_eid: int,
    ):

        # ====================================================
        # Global L1 Nodes
        # ====================================================

        self.node_hashes = (
            bytearray()
        )

        self.node_left = (
            array("i")
        )

        self.node_right = (
            array("i")
        )

        self.node_eid = (
            array("i")
        )

        # ====================================================
        # L1 Root for Each L0 Node
        #
        # Used only by L0 Leaves.
        #
        # Internal nodes remain:
        #
        # root_node = -1
        # ====================================================

        self.l0_l1_root_node = array(
            "i",
            [-1] * l0_node_count,
        )

        # Reserve a fixed 32 bytes for each L0 node
        self.l0_l1_root_hashes = bytearray(
            EMPTY_L1_ROOT
            * l0_node_count
        )

        # ====================================================
        # Time Range of Each L0 Leaf
        #
        # Derived from all edge τ-MMRs in the Leaf:
        #
        # min_start =
        #     min(edge.min_start)
        #
        # max_end =
        #     max(edge.max_end)
        #
        # When there are no trajectories:
        #
        # min_start = UINT32_MAX
        # max_end   = 0
        # ====================================================

        self.l0_min_start = array(
            "I",
            [UINT32_MAX] * l0_node_count,
        )

        self.l0_max_end = array(
            "I",
            [0] * l0_node_count,
        )

        # ====================================================
        # Reverse Index
        #
        # Very useful for later queries:
        #
        # eid
        # ↓
        # Which L0 Leaf it belongs to
        #
        # eid
        # ↓
        # Which L1 Leaf Node it maps to
        # ====================================================

        self.eid_to_l0_leaf = array(
            "i",
            [-1] * (max_eid + 1),
        )

        self.eid_to_l1_leaf_node = array(
            "i",
            [-1] * (max_eid + 1),
        )

        self.total_nodes = 0
        self.total_leaves = 0
        self.tree_count = 0

    # ========================================================
    # Get L1 Node Hash
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

    # ========================================================
    # Get the L1 Root of an L0 Leaf
    # ========================================================

    def get_l1_root(
        self,
        l0_node_index: int,
    ) -> bytes:

        offset = (
            l0_node_index
            * HASH_SIZE
        )

        return bytes(
            self.l0_l1_root_hashes[
                offset:
                offset + HASH_SIZE
            ]
        )

    # ========================================================
    # Set the L1 Root of an L0 Leaf
    # ========================================================

    def set_l1_root(
        self,
        l0_node_index: int,
        root_hash: bytes,
    ) -> None:

        offset = (
            l0_node_index
            * HASH_SIZE
        )

        self.l0_l1_root_hashes[
            offset:
            offset + HASH_SIZE
        ] = root_hash


# ============================================================
# Add an L1 Leaf Node
# ============================================================

def append_l1_leaf_node(
    index: L1MerkleIndex,
    eid: int,
    l0_leaf_index: int,
    mmr_index: EdgeMMRIndex,
) -> tuple[
    int,
    bytes,
    int,
    int,
]:
    """
    Build:

        L1 Leaf

    Returns:

        node_index
        hash
        min_start
        max_end
    """

    (
        k_e,
        min_start,
        max_end,
        root_e,
    ) = (
        mmr_index.get_edge_state(
            eid
        )
    )

    # ========================================================
    # L1 Leaf Hash
    # ========================================================

    leaf_hash = (
        hash_l1_leaf(
            eid=eid,
            k_e=k_e,
            min_start=min_start,
            max_end=max_end,
            root_e=root_e,
        )
    )

    node_index = len(
        index.node_left
    )

    index.node_hashes.extend(
        leaf_hash
    )

    index.node_left.append(
        -1
    )

    index.node_right.append(
        -1
    )

    index.node_eid.append(
        eid
    )

    # ========================================================
    # Build the eid Reverse Mapping
    # ========================================================

    if (
        index.eid_to_l0_leaf[eid]
        != -1
    ):

        raise RuntimeError(
            f"eid={eid} "
            f"appears in multiple L0 Leaves"
        )

    index.eid_to_l0_leaf[
        eid
    ] = l0_leaf_index

    index.eid_to_l1_leaf_node[
        eid
    ] = node_index

    index.total_leaves += 1

    return (
        node_index,
        leaf_hash,
        min_start,
        max_end,
    )


# ============================================================
# Add an L1 Internal Node
# ============================================================

def append_l1_internal_node(
    index: L1MerkleIndex,
    left_node: int,
    right_node: int,
    left_hash: bytes,
    right_hash: bytes,
) -> tuple[
    int,
    bytes,
]:

    parent_hash = (
        hash_l1_internal(
            left_hash,
            right_hash,
        )
    )

    node_index = len(
        index.node_left
    )

    index.node_hashes.extend(
        parent_hash
    )

    index.node_left.append(
        left_node
    )

    index.node_right.append(
        right_node
    )

    index.node_eid.append(
        -1
    )

    return (
        node_index,
        parent_hash,
    )


# ============================================================
# Merge Time Ranges
# ============================================================

def merge_time_range(
    left_min_start: int,
    left_max_end: int,
    right_min_start: int,
    right_max_end: int,
) -> tuple[
    int,
    int,
]:
    """
    Empty time range definition:

        min_start = UINT32_MAX
        max_end   = 0

    Therefore, direct use of the following is sufficient:

        min()
        max()
    """

    if (
        left_min_start
        <= right_min_start
    ):

        min_start = (
            left_min_start
        )

    else:

        min_start = (
            right_min_start
        )

    if (
        left_max_end
        >= right_max_end
    ):

        max_end = (
            left_max_end
        )

    else:

        max_end = (
            right_max_end
        )

    return (
        min_start,
        max_end,
    )


# ============================================================
# Recursively Build an L1 Subtree
# ============================================================

def build_l1_subtree(
    index: L1MerkleIndex,
    mmr_index: EdgeMMRIndex,
    edge_ids: array,
    edge_offset: int,
    start: int,
    count: int,
    l0_leaf_index: int,
) -> tuple[
    int,
    bytes,
    int,
    int,
]:
    """
    Build the Merkle Tree using a deterministic RFC6962-style split.


    Returns:

        root node index
        root hash
        subtree min_start
        subtree max_end
    """

    # ========================================================
    # One Leaf
    # ========================================================

    if count == 1:

        eid = edge_ids[
            edge_offset + start
        ]

        return (
            append_l1_leaf_node(
                index=index,
                eid=eid,
                l0_leaf_index=l0_leaf_index,
                mmr_index=mmr_index,
            )
        )

    # ========================================================
    # Find the Number of Leaves in the Left Subtree
    #
    # Largest 2^x < count
    # ========================================================

    left_count = (
        largest_power_of_two_less_than(
            count
        )
    )

    right_count = (
        count
        - left_count
    )

    # ========================================================
    # Left
    # ========================================================

    (
        left_node,
        left_hash,
        left_min_start,
        left_max_end,
    ) = (
        build_l1_subtree(
            index=index,
            mmr_index=mmr_index,
            edge_ids=edge_ids,
            edge_offset=edge_offset,
            start=start,
            count=left_count,
            l0_leaf_index=l0_leaf_index,
        )
    )

    # ========================================================
    # Right
    # ========================================================

    (
        right_node,
        right_hash,
        right_min_start,
        right_max_end,
    ) = (
        build_l1_subtree(
            index=index,
            mmr_index=mmr_index,
            edge_ids=edge_ids,
            edge_offset=edge_offset,
            start=start + left_count,
            count=right_count,
            l0_leaf_index=l0_leaf_index,
        )
    )

    # ========================================================
    # Parent
    # ========================================================

    (
        parent_node,
        parent_hash,
    ) = (
        append_l1_internal_node(
            index=index,
            left_node=left_node,
            right_node=right_node,
            left_hash=left_hash,
            right_hash=right_hash,
        )
    )

    # ========================================================
    # The Time Range Is Used Only for the Final L0 Leaf
    #
    # Note:
    #
    # The L1 Internal Hash itself does not contain the time range.
    #
    # Time is already committed by the edge state in each L1 Leaf.
    # ========================================================

    (
        parent_min_start,
        parent_max_end,
    ) = (
        merge_time_range(
            left_min_start,
            left_max_end,
            right_min_start,
            right_max_end,
        )
    )

    return (
        parent_node,
        parent_hash,
        parent_min_start,
        parent_max_end,
    )


# ============================================================
# Build L1 for One L0 Leaf
# ============================================================

def build_single_l1_tree(
    l0_leaf_index: int,
    skeleton: SpatialSkeleton,
    mmr_index: EdgeMMRIndex,
    index: L1MerkleIndex,
) -> None:

    edge_offset = (
        skeleton.leaf_edge_offset[
            l0_leaf_index
        ]
    )

    edge_count = (
        skeleton.leaf_edge_count[
            l0_leaf_index
        ]
    )

    # ========================================================
    # Empty L0 Leaf
    # ========================================================

    if edge_count == 0:

        index.l0_l1_root_node[
            l0_leaf_index
        ] = -1

        index.set_l1_root(
            l0_leaf_index,
            EMPTY_L1_ROOT,
        )

        index.l0_min_start[
            l0_leaf_index
        ] = UINT32_MAX

        index.l0_max_end[
            l0_leaf_index
        ] = 0

        index.tree_count += 1

        return

    # ========================================================
    # Build L1 Tree
    #
    # skeleton.leaf_edge_ids is already sorted by eid.
    # ========================================================

    (
        root_node,
        root_hash,
        min_start,
        max_end,
    ) = (
        build_l1_subtree(
            index=index,
            mmr_index=mmr_index,
            edge_ids=skeleton.leaf_edge_ids,
            edge_offset=edge_offset,
            start=0,
            count=edge_count,
            l0_leaf_index=l0_leaf_index,
        )
    )

    # ========================================================
    # Save to L0 Leaf
    # ========================================================

    index.l0_l1_root_node[
        l0_leaf_index
    ] = root_node

    index.set_l1_root(
        l0_leaf_index,
        root_hash,
    )

    index.l0_min_start[
        l0_leaf_index
    ] = min_start

    index.l0_max_end[
        l0_leaf_index
    ] = max_end

    index.tree_count += 1


# ============================================================
# Build All L1 Trees
# ============================================================

def build_all_l1_trees(
    skeleton: SpatialSkeleton,
    mmr_index: EdgeMMRIndex,
    road: RoadNetwork,
) -> L1MerkleIndex:

    index = L1MerkleIndex(
        l0_node_count=
        skeleton.node_count,

        max_eid=
        road.max_eid,
    )

    node_type = (
        skeleton.node_type
    )

    # ========================================================
    # L0 nodes are already stored in postorder.
    #
    # Iterate over all L0 Leaves here,
    # and build one independent L1 tree for each Leaf.
    # ========================================================

    for l0_node_index in range(
        skeleton.node_count
    ):

        if (
            node_type[l0_node_index]
            != NODE_LEAF
        ):

            continue

        build_single_l1_tree(
            l0_leaf_index=
            l0_node_index,

            skeleton=
            skeleton,

            mmr_index=
            mmr_index,

            index=
            index,
        )

    index.total_nodes = len(
        index.node_left
    )

    return index


# ============================================================
# L1 Integrity Check
# ============================================================

def validate_l1(
    l1_index: L1MerkleIndex,
    skeleton: SpatialSkeleton,
    road: RoadNetwork,
) -> None:
    """
    Check:

        Every real road edge

        Must:

        1. Belong to one L0 Leaf
        2. Map to one L1 Leaf
    """

    for eid in range(
        road.max_eid + 1
    ):

        if not road.edge_exists(
            eid
        ):

            continue

        l0_leaf = (
            l1_index.eid_to_l0_leaf[
                eid
            ]
        )

        l1_leaf = (
            l1_index.eid_to_l1_leaf_node[
                eid
            ]
        )

        if l0_leaf < 0:

            raise RuntimeError(
                f"eid={eid} "
                f"has no L0 Leaf mapping"
            )

        if l1_leaf < 0:

            raise RuntimeError(
                f"eid={eid} "
                f"has no L1 Leaf mapping"
            )

        if (
            skeleton.node_type[
                l0_leaf
            ]
            != NODE_LEAF
        ):

            raise RuntimeError(
                f"eid={eid} "
                f"maps to a non-Leaf L0 node"
            )

        if (
            l1_index.node_eid[
                l1_leaf
            ]
            != eid
        ):

            raise RuntimeError(
                f"eid={eid} "
                f"has an incorrect L1 Leaf mapping"
            )


# ============================================================
# Print the L1 of an L0 Leaf
# ============================================================

def print_l1_leaf_information(
    l0_leaf_index: int,
    skeleton: SpatialSkeleton,
    mmr_index: EdgeMMRIndex,
    l1_index: L1MerkleIndex,
    show_count: int = 5,
) -> None:

    edge_offset = (
        skeleton.leaf_edge_offset[
            l0_leaf_index
        ]
    )

    edge_count = (
        skeleton.leaf_edge_count[
            l0_leaf_index
        ]
    )

    print(
        "L0 Leaf node:",
        l0_leaf_index,
    )

    print(
        "Road edge count:",
        edge_count,
    )

    print(
        "L0 Leaf min_start:",
        l1_index.l0_min_start[
            l0_leaf_index
        ],
    )

    print(
        "L0 Leaf max_end:",
        l1_index.l0_max_end[
            l0_leaf_index
        ],
    )

    print(
        "L1 root:"
    )

    print(
        l1_index.get_l1_root(
            l0_leaf_index
        ).hex()
    )

    print()

    print(
        "First few L1 Leaves:"
    )

    count = min(
        edge_count,
        show_count,
    )

    for i in range(
        count
    ):

        eid = (
            skeleton.leaf_edge_ids[
                edge_offset + i
            ]
        )

        (
            k_e,
            min_start,
            max_end,
            root_e,
        ) = (
            mmr_index.get_edge_state(
                eid
            )
        )

        l1_node = (
            l1_index.eid_to_l1_leaf_node[
                eid
            ]
        )

        print()

        print(
            f"eid = {eid}"
        )

        print(
            "  k_e =",
            k_e,
        )

        print(
            "  min_start =",
            min_start,
        )

        print(
            "  max_end =",
            max_end,
        )

        print(
            "  root_e =",
            root_e.hex(),
        )

        print(
            "  L1 leaf node =",
            l1_node,
        )

        print(
            "  L1 leaf hash =",
            l1_index.get_node_hash(
                l1_node
            ).hex(),
        )


# ============================================================
# Run Directly with the PyCharm Green Triangle
# ============================================================

def main():

    # ========================================================
    # Road Network Files
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

        # Add more files here if available:
        #
        # r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-2.json",
        # r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-3.json",
    ]

    THETA = 64

    # ========================================================
    # 1. Road Network
    # ========================================================

    print(
        "Loading road network..."
    )

    road = RoadNetwork()

    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )

    print(
        "Road network loaded"
    )

    # ========================================================
    # 2. Edge Entry
    # ========================================================

    print()

    print(
        "===== Build Edge Entries ====="
    )

    store = (
        build_edge_entries_from_files(
            trajectory_files=
            TRAJECTORY_FILES,

            road=
            road,
        )
    )

    # ========================================================
    # 3. Edge τ-MMR
    # ========================================================

    print()

    print(
        "===== Build Edge τ-MMR ====="
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
        "MMR node count:",
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
        "===== Build L0 Spatial Skeleton ====="
    )

    spatial_start = (
        perf_counter()
    )

    builder = (
        SpatialSkeletonBuilder(
            road=road,
            theta=THETA,
        )
    )

    skeleton = (
        builder.build()
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
        "L0 construction time:",
        f"{spatial_elapsed:.6f} s",
    )

    # ========================================================
    # 5. L1
    # ========================================================

    print()

    print(
        "===== Start Building All L1 Merkle Trees ====="
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

    # ========================================================
    # Check
    # ========================================================

    print()

    print(
        "Checking L1 integrity..."
    )

    validate_l1(
        l1_index=l1_index,
        skeleton=skeleton,
        road=road,
    )

    print(
        "L1 integrity check passed"
    )

    # ========================================================
    # Statistics
    # ========================================================

    print()

    print(
        "===== L1 Construction Completed ====="
    )

    print(
        "L1 Tree count:",
        l1_index.tree_count,
    )

    print(
        "Total L1 Leaf count:",
        l1_index.total_leaves,
    )

    print(
        "Total L1 Node count:",
        l1_index.total_nodes,
    )

    print(
        "Total road edge count:",
        road.edge_count,
    )

    print(
        "Construction time:",
        f"{l1_elapsed:.6f} s",
    )

    # ========================================================
    # Theoretical Check
    #
    # The total number of L1 Leaves should be:
    #
    # 8831
    #
    # Because each road edge enters exactly one L1.
    # ========================================================

    print()

    print(
        "===== L1 Leaf Count Check ====="
    )

    print(
        "L1 Leaf count:",
        l1_index.total_leaves,
    )

    print(
        "Real road edge count:",
        road.edge_count,
    )

    if (
        l1_index.total_leaves
        != road.edge_count
    ):

        raise RuntimeError(
            "L1 Leaf count does not match the road edge count"
        )

    print(
        "Counts match"
    )

    # ========================================================
    # Find the First Non-Empty L0 Leaf for Testing
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

    if test_leaf < 0:

        raise RuntimeError(
            "No non-empty L0 Leaf found"
        )

    print()

    print(
        "========================================"
    )

    print(
        "===== L1 Tree Example ====="
    )

    print_l1_leaf_information(
        l0_leaf_index=
        test_leaf,

        skeleton=
        skeleton,

        mmr_index=
        mmr_index,

        l1_index=
        l1_index,

        show_count=5,
    )

    # ========================================================
    # Edge 1985 Mapping Test
    # ========================================================

    test_eid = 1985

    print()

    print(
        "========================================"
    )

    print(
        f"===== Edge {test_eid} L1 Mapping ====="
    )

    l0_leaf = (
        l1_index.eid_to_l0_leaf[
            test_eid
        ]
    )

    l1_leaf = (
        l1_index.eid_to_l1_leaf_node[
            test_eid
        ]
    )

    print(
        "L0 Leaf:",
        l0_leaf,
    )

    print(
        "Corresponding L1 Leaf Node:",
        l1_leaf,
    )

    print(
        "L1 Leaf Hash:"
    )

    print(
        l1_index.get_node_hash(
            l1_leaf
        ).hex()
    )

    print()

    print(
        "l1_merkle.py completed successfully"
    )


if __name__ == "__main__":
    main()
