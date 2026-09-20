from __future__ import annotations

import hashlib
import struct

from array import array
from time import perf_counter

from crypto import (
    hash_mmr_leaf,
    hash_mmr_internal,
)

from road_network import RoadNetwork

from edge_entries import (
    EdgeEntryStore,
    build_edge_entries_from_files,
    _ENTRY_STRUCT,
    ENTRY_SIZE,
    UINT32_MAX,
)


# ============================================================
# Basic Parameters
# ============================================================

HASH_SIZE = 32

_U32 = struct.Struct(">I")
_U32_2 = struct.Struct(">II")

_sha256 = hashlib.sha256


# ============================================================
# MMR Peak Heights
# ============================================================

def get_peak_heights(
    k: int,
) -> list[int]:
    """
    Get the heights of all Peaks from the number of MMR leaves k.

    For example:

        k = 13

        13 = 8 + 4 + 1

    Therefore:

        peak heights = [3, 2, 0]

    Because:

        8 = 2^3
        4 = 2^2
        1 = 2^0
    """

    if k <= 0:
        return []

    heights: list[int] = []

    height = (
        k.bit_length() - 1
    )

    while height >= 0:

        if (
            k
            & (1 << height)
        ):
            heights.append(
                height
            )

        height -= 1

    return heights


# ============================================================
# Calculate Total MMR Node Count from k
# ============================================================

def mmr_node_count(
    k: int,
) -> int:
    """
    When the MMR has k leaves:

        node_count
        =
        2*k - popcount(k)

    For example:

        k = 7
        binary = 111
        popcount = 3

        node_count
        =
        14 - 3
        =
        11
    """

    if k <= 0:
        return 0

    return (
        2 * k
        - k.bit_count()
    )


# ============================================================
# Derive Peak Root Node Positions from Edge Offset and k
# ============================================================

def get_peak_node_indices(
    node_offset: int,
    k: int,
) -> list[int]:
    """
    MMR nodes are stored contiguously in postorder.

    Therefore, given:

        node_offset
        k

    all Peak root node indices can be derived.

    No additional Peak pointers need to be stored.
    """

    heights = (
        get_peak_heights(k)
    )

    peak_indices: list[int] = []

    cursor = node_offset

    for height in heights:

        # A complete binary tree of height h:
        #
        # node count =
        #
        # 2^(h+1) - 1

        subtree_node_count = (
            (1 << (height + 1))
            - 1
        )

        cursor += (
            subtree_node_count
        )

        peak_indices.append(
            cursor - 1
        )

    return peak_indices


# ============================================================
# Edge MMR Index
# ============================================================

class EdgeMMRIndex:
    """
    τ-MMRs for all road edges.

    To reduce Python objects:

    Do not create:

        MMRNode()
        MMRLeaf()
        left object
        right object

    Instead, store all MMR nodes contiguously in one unified area.


    Each node stores:

        hash
        min_start
        max_end


    node_hashes：

        [node0 hash][node1 hash][node2 hash]...

    Each hash is fixed at 32 bytes.


    For leaf nodes:

        min_start = start
        max_end   = end


    For internal nodes:

        min_start =
            min(left.min_start,
                right.min_start)

        max_end =
            max(left.max_end,
                right.max_end)
    """

    __slots__ = (
        "node_hashes",
        "node_min_start",
        "node_max_end",
        "edge_node_offset",
        "edge_node_count",
        "edge_k",
        "edge_min_start",
        "edge_max_end",
        "edge_root_hashes",
        "total_nodes",
        "non_empty_edges",
    )

    def __init__(
        self,
        max_eid: int,
    ):

        edge_size = (
            max_eid + 1
        )

        # ====================================================
        # Store all MMR nodes in one global contiguous area
        # ====================================================

        self.node_hashes = (
            bytearray()
        )

        self.node_min_start = (
            array("I")
        )

        self.node_max_end = (
            array("I")
        )

        # ====================================================
        # Position of each edge in the global node array
        # ====================================================

        self.edge_node_offset = array(
            "Q",
            [0] * edge_size,
        )

        self.edge_node_count = array(
            "I",
            [0] * edge_size,
        )

        # ====================================================
        # MMR state of each edge
        #
        # k_e
        # min_start
        # max_end
        # root_e
        # ====================================================

        self.edge_k = array(
            "I",
            [0] * edge_size,
        )

        self.edge_min_start = array(
            "I",
            [UINT32_MAX] * edge_size,
        )

        self.edge_max_end = array(
            "I",
            [0] * edge_size,
        )

        # ====================================================
        # Empty MMR root
        #
        # root_empty =
        #
        # SHA256(k=0)
        # ====================================================

        empty_root = (
            _sha256(
                _U32.pack(0)
            ).digest()
        )

        # Reserve a 32-byte root for each eid
        self.edge_root_hashes = bytearray(
            empty_root
            * edge_size
        )

        self.total_nodes = 0

        self.non_empty_edges = 0

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

    # ========================================================
    # Get Edge Root
    # ========================================================

    def get_edge_root(
        self,
        eid: int,
    ) -> bytes:

        offset = (
            eid
            * HASH_SIZE
        )

        return bytes(
            self.edge_root_hashes[
                offset:
                offset + HASH_SIZE
            ]
        )

    # ========================================================
    # Set Edge Root
    # ========================================================

    def set_edge_root(
        self,
        eid: int,
        root: bytes,
    ) -> None:

        offset = (
            eid
            * HASH_SIZE
        )

        self.edge_root_hashes[
            offset:
            offset + HASH_SIZE
        ] = root

    # ========================================================
    # Get Edge MMR State
    # ========================================================

    def get_edge_state(
        self,
        eid: int,
    ) -> tuple[
        int,
        int,
        int,
        bytes,
    ]:
        """
        Returns:

            k_e
            min_start
            max_end
            root_e
        """

        return (
            self.edge_k[eid],
            self.edge_min_start[eid],
            self.edge_max_end[eid],
            self.get_edge_root(eid),
        )


# ============================================================
# MMR Root
# ============================================================

def calculate_mmr_root(
    index: EdgeMMRIndex,
    k: int,
    peak_indices: list[int],
) -> bytes:
    """
    MMR root definition:

        H(
            k
            ||
            peak1_hash
            || peak1_min_start
            || peak1_max_end
            ||
            peak2_hash
            || peak2_min_start
            || peak2_max_end
            ||
            ...
        )

    Peaks are strictly ordered from left to right.
    """

    h = _sha256()

    # k
    h.update(
        _U32.pack(k)
    )

    node_hashes = (
        index.node_hashes
    )

    node_min_start = (
        index.node_min_start
    )

    node_max_end = (
        index.node_max_end
    )

    for node_index in peak_indices:

        hash_offset = (
            node_index
            * HASH_SIZE
        )

        # Peak hash
        h.update(
            node_hashes[
                hash_offset:
                hash_offset + HASH_SIZE
            ]
        )

        # Time coverage range of the Peak
        h.update(
            _U32_2.pack(
                node_min_start[
                    node_index
                ],
                node_max_end[
                    node_index
                ],
            )
        )

    return h.digest()


# ============================================================
# Build the MMR for a Single Road Edge
# ============================================================

def build_single_edge_mmr(
    eid: int,
    store: EdgeEntryStore,
    index: EdgeMMRIndex,
) -> None:
    """
    Build the τ-MMR for a single road edge.

    Process:

        Edge Entry
            ↓
        time sorting
            ↓
        MMR Leaf
            ↓
        merge equal heights
            ↓
        Peaks
            ↓
        root_e
    """

    buffer = (
        store.edge_buffers[eid]
    )

    if buffer is None:
        return

    k = (
        store.entry_counts[eid]
    )

    if k == 0:
        return

    # ========================================================
    # 1. Read Entries
    #
    # Process only one edge at a time.
    #
    # Therefore, even if the entire dataset is large,
    # the extra memory used for sorting mainly depends on:
    #
    #     the number of Entries on the current edge
    #
    # rather than the total number of Entries.
    # ========================================================

    records = list(
        _ENTRY_STRUCT.iter_unpack(
            buffer
        )
    )

    # record:
    #
    # (
    #     trajectory_id,
    #     start,
    #     end
    # )

    # ========================================================
    # 2. Sort by Time During Initial Batch Construction
    #
    # Primary key:
    #     start
    #
    # Secondary key:
    #     end
    #
    # Tertiary key:
    #     trajectory_id
    #
    # Use trajectory_id as the final tie-breaker,
    # ensuring a fully deterministic reconstruction order.
    # ========================================================

    records.sort(
        key=lambda record: (
            record[1],
            record[2],
            record[0],
        )
    )

    # ========================================================
    # 3. Rebuild the Sorted Contiguous Entry Buffer
    #
    # Queries later use this order directly.
    # ========================================================

    sorted_buffer = bytearray(
        k * ENTRY_SIZE
    )

    # ========================================================
    # Starting node position of the current road edge
    # ========================================================

    node_offset = len(
        index.node_min_start
    )

    index.edge_node_offset[
        eid
    ] = node_offset

    # ========================================================
    # Peak Stack
    #
    # Store only:
    #
    #     node index
    #     height
    #
    # Do not create tree-node objects.
    # ========================================================

    peak_nodes: list[int] = []

    peak_heights: list[int] = []

    append_peak_node = (
        peak_nodes.append
    )

    append_peak_height = (
        peak_heights.append
    )

    pop_peak_node = (
        peak_nodes.pop
    )

    pop_peak_height = (
        peak_heights.pop
    )

    # ========================================================
    # Local binding for frequently used operations
    # ========================================================

    node_hashes = (
        index.node_hashes
    )

    node_min_start = (
        index.node_min_start
    )

    node_max_end = (
        index.node_max_end
    )

    hash_leaf = (
        hash_mmr_leaf
    )

    hash_internal = (
        hash_mmr_internal
    )

    pack_entry_into = (
        _ENTRY_STRUCT.pack_into
    )

    # ========================================================
    # 4. Build the MMR in Sorted Entry Order
    # ========================================================

    for entry_index in range(k):

        (
            trajectory_id,
            start,
            end,
        ) = records[
            entry_index
        ]

        # ----------------------------------------------------
        # Save the sorted Entry
        # ----------------------------------------------------

        pack_entry_into(
            sorted_buffer,
            entry_index * ENTRY_SIZE,
            trajectory_id,
            start,
            end,
        )

        # ----------------------------------------------------
        # MMR Leaf
        #
        # H(
        #   trajectory_id
        #   || start
        #   || end
        # )
        # ----------------------------------------------------

        current_hash = (
            hash_leaf(
                trajectory_id,
                start,
                end,
            )
        )

        # Leaf time range:
        #
        # min_start = start
        # max_end   = end

        current_min_start = (
            start
        )

        current_max_end = (
            end
        )

        # New leaf height = 0
        current_height = 0

        # ----------------------------------------------------
        # Append the leaf to the global contiguous node area
        # ----------------------------------------------------

        current_node_index = len(
            node_min_start
        )

        node_hashes.extend(
            current_hash
        )

        node_min_start.append(
            current_min_start
        )

        node_max_end.append(
            current_max_end
        )

        # ====================================================
        # MMR Core:
        #
        # If the two rightmost Peaks have the same height,
        # keep merging upward.
        # ====================================================

        while (
            peak_heights
            and
            peak_heights[-1]
            == current_height
        ):

            # The left child is the previous Peak
            left_node_index = (
                pop_peak_node()
            )

            pop_peak_height()

            # The right child is current_node_index
            right_node_index = (
                current_node_index
            )

            # ------------------------------------------------
            # Time range of the internal node
            # ------------------------------------------------

            left_min_start = (
                node_min_start[
                    left_node_index
                ]
            )

            right_min_start = (
                node_min_start[
                    right_node_index
                ]
            )

            if (
                left_min_start
                <= right_min_start
            ):
                parent_min_start = (
                    left_min_start
                )
            else:
                parent_min_start = (
                    right_min_start
                )

            left_max_end = (
                node_max_end[
                    left_node_index
                ]
            )

            right_max_end = (
                node_max_end[
                    right_node_index
                ]
            )

            if (
                left_max_end
                >= right_max_end
            ):
                parent_max_end = (
                    left_max_end
                )
            else:
                parent_max_end = (
                    right_max_end
                )

            # ------------------------------------------------
            # Get left and right child hashes
            # ------------------------------------------------

            left_hash_offset = (
                left_node_index
                * HASH_SIZE
            )

            right_hash_offset = (
                right_node_index
                * HASH_SIZE
            )

            left_hash = bytes(
                node_hashes[
                    left_hash_offset:
                    left_hash_offset
                    + HASH_SIZE
                ]
            )

            right_hash = bytes(
                node_hashes[
                    right_hash_offset:
                    right_hash_offset
                    + HASH_SIZE
                ]
            )

            # ------------------------------------------------
            # Parent Hash
            #
            # H(
            #   min_start
            #   || max_end
            #   || left_hash
            #   || right_hash
            # )
            # ------------------------------------------------

            parent_hash = (
                hash_internal(
                    parent_min_start,
                    parent_max_end,
                    left_hash,
                    right_hash,
                )
            )

            # ------------------------------------------------
            # Append the Parent to the global node area
            # ------------------------------------------------

            current_node_index = len(
                node_min_start
            )

            node_hashes.extend(
                parent_hash
            )

            node_min_start.append(
                parent_min_start
            )

            node_max_end.append(
                parent_max_end
            )

            current_height += 1

        # ====================================================
        # The current subtree becomes a new Peak
        # ====================================================

        append_peak_node(
            current_node_index
        )

        append_peak_height(
            current_height
        )

    # ========================================================
    # Replace the Original Buffer with the Sorted Buffer
    #
    # From this point on:
    #
    # store.edge_buffers[eid]
    #
    # contains Entries sorted by time.
    # ========================================================

    store.edge_buffers[
        eid
    ] = sorted_buffer

    # The temporary records list is no longer needed
    del records

    # ========================================================
    # 5. Node Count
    # ========================================================

    node_count = (
        len(node_min_start)
        - node_offset
    )

    index.edge_node_count[
        eid
    ] = node_count

    # ========================================================
    # Theoretical Node Count Check
    #
    # For k leaves:
    #
    # nodes =
    # 2*k - popcount(k)
    # ========================================================

    expected_node_count = (
        mmr_node_count(k)
    )

    if (
        node_count
        != expected_node_count
    ):
        raise RuntimeError(
            f"Incorrect MMR node count for eid={eid}: "
            f"actual={node_count}, "
            f"expected={expected_node_count}"
        )

    # ========================================================
    # 6. Edge MMR Root
    # ========================================================

    root_e = (
        calculate_mmr_root(
            index=index,
            k=k,
            peak_indices=peak_nodes,
        )
    )

    # ========================================================
    # Save Edge State
    # ========================================================

    index.edge_k[
        eid
    ] = k

    index.edge_min_start[
        eid
    ] = store.min_start[
        eid
    ]

    index.edge_max_end[
        eid
    ] = store.max_end[
        eid
    ]

    index.set_edge_root(
        eid,
        root_e,
    )

    index.non_empty_edges += 1


# ============================================================
# Build MMRs for All Road Edges
# ============================================================

def build_all_edge_mmrs(
    store: EdgeEntryStore,
    road: RoadNetwork,
) -> EdgeMMRIndex:
    """
    Build a τ-MMR for every road edge that has trajectories.
    """

    index = EdgeMMRIndex(
        max_eid=road.max_eid
    )

    edge_buffers = (
        store.edge_buffers
    )

    # ========================================================
    # Process in eid order.
    #
    # Maintain a strictly deterministic order within each edge.
    # ========================================================

    for eid in range(
        len(edge_buffers)
    ):

        if (
            edge_buffers[eid]
            is None
        ):
            continue

        build_single_edge_mmr(
            eid=eid,
            store=store,
            index=index,
        )

    index.total_nodes = len(
        index.node_min_start
    )

    return index


# ============================================================
# Print Peaks for One Edge
# ============================================================

def print_edge_peaks(
    eid: int,
    index: EdgeMMRIndex,
) -> None:

    k = (
        index.edge_k[eid]
    )

    node_offset = (
        index.edge_node_offset[
            eid
        ]
    )

    heights = (
        get_peak_heights(k)
    )

    peak_indices = (
        get_peak_node_indices(
            node_offset=node_offset,
            k=k,
        )
    )

    print(
        "Peak count:",
        len(peak_indices),
    )

    print(
        "Peak heights:",
        heights,
    )

    for i in range(
        len(peak_indices)
    ):

        node_index = (
            peak_indices[i]
        )

        print()

        print(
            f"Peak {i}:"
        )

        print(
            "  height =",
            heights[i],
        )

        print(
            "  node_index =",
            node_index,
        )

        print(
            "  min_start =",
            index.node_min_start[
                node_index
            ],
        )

        print(
            "  max_end =",
            index.node_max_end[
                node_index
            ],
        )

        print(
            "  hash =",
            index.get_node_hash(
                node_index
            ).hex(),
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
    #
    # Add as many files as available.
    # The listed order is the initial insertion order.
    # ========================================================

    TRAJECTORY_FILES = [
        r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-1.json",

        # Uncomment additional files if available:
        #
        # r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-2.json",
        # r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json\traj-10-3.json",
    ]

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
    # 2. Entry Bucketing
    # ========================================================

    print()

    print(
        "===== Build Edge Entries ====="
    )

    store = (
        build_edge_entries_from_files(
            trajectory_files=TRAJECTORY_FILES,
            road=road,
        )
    )

    print()

    print(
        "Total Entry count:",
        store.total_entries,
    )

    print(
        "Road edges with trajectories:",
        store.non_empty_edges,
    )

    # ========================================================
    # 3. Build MMR
    # ========================================================

    print()

    print(
        "===== Start Building Edge τ-MMR ====="
    )

    start_time = (
        perf_counter()
    )

    mmr_index = (
        build_all_edge_mmrs(
            store=store,
            road=road,
        )
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    print()

    print(
        "===== Edge τ-MMR Construction Completed ====="
    )

    print(
        "Number of MMRs with trajectories:",
        mmr_index.non_empty_edges,
    )

    print(
        "Total MMR node count:",
        mmr_index.total_nodes,
    )

    print(
        "Construction time:",
        f"{elapsed:.6f} s",
    )

    # ========================================================
    # 4. Test Edge 1985
    # ========================================================

    test_eid = 1985

    print()

    print(
        "========================================"
    )

    print(
        f"===== Edge {test_eid} MMR ====="
    )

    (
        k_e,
        min_start,
        max_end,
        root_e,
    ) = mmr_index.get_edge_state(
        test_eid
    )

    print(
        "k_e:",
        k_e,
    )

    print(
        "min_start:",
        min_start,
    )

    print(
        "max_end:",
        max_end,
    )

    print(
        "root_e:"
    )

    print(
        root_e.hex()
    )

    # ========================================================
    # Sorted Entries
    # ========================================================

    print()

    print(
        "===== Sorted Entries ====="
    )

    show_count = min(
        10,
        store.get_entry_count(
            test_eid
        ),
    )

    for i in range(
        show_count
    ):

        (
            trajectory_id,
            start,
            end,
        ) = store.get_entry(
            eid=test_eid,
            index=i,
        )

        print(
            f"{i}: "
            f"start={start}, "
            f"end={end}, "
            f"trajectory_id="
            f"{trajectory_id.hex()[:16]}..."
        )

    # ========================================================
    # Peak
    # ========================================================

    print()

    print(
        "===== Peaks ====="
    )

    print_edge_peaks(
        eid=test_eid,
        index=mmr_index,
    )

    # ========================================================
    # Theoretical Node Count
    # ========================================================

    print()

    print(
        "===== Node Count Check ====="
    )

    actual_nodes = (
        mmr_index.edge_node_count[
            test_eid
        ]
    )

    expected_nodes = (
        mmr_node_count(
            k_e
        )
    )

    print(
        "Actual node count:",
        actual_nodes,
    )

    print(
        "Expected node count:",
        expected_nodes,
    )

    print()

    print(
        "edge_mmr.py completed successfully"
    )


if __name__ == "__main__":
    main()
