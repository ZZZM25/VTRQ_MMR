from __future__ import annotations

import hashlib
import struct

from array import array
from operator import itemgetter
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
_MMR_INTERNAL_STRUCT = struct.Struct(">32sII32sII")

_sha256 = hashlib.sha256
_ENTRY_SORT_KEY = itemgetter(1, 2, 0)


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
    sort_records: bool = True,
) -> None:
    """Build one paper-aligned tau-MMR with a low-overhead hot loop.

    The authenticated structure and byte-level hash formulas are unchanged.
    This implementation only removes redundant Python work:

    1. An entry is packed once; the same bytes are appended to the sorted
       buffer and hashed as the MMR leaf.
    2. The peak stack keeps only live subtree (hash, min_start, max_end).
       Peak node indices/heights are not pushed/popped on every merge.
    3. Binary carry from entry_index tells us exactly how many equal-height
       peaks must be merged.
    4. Internal-node input is packed once as
       hash0 || tau0 || hash1 || tau1 and hashed once.

    Nodes are still appended to the global arrays in exactly the same
    postorder, so node offsets/counts, roots, proofs and serialization remain
    compatible with the paper-aligned implementation.
    """

    buffer = store.edge_buffers[eid]
    if buffer is None:
        return

    k = store.entry_counts[eid]
    if k == 0:
        return

    # Deterministic per-edge order is unchanged.
    records = list(_ENTRY_STRUCT.iter_unpack(buffer))
    if sort_records:
        records.sort(key=_ENTRY_SORT_KEY)

    # Grow by extend instead of preallocating + pack_into.  Each entry is
    # serialized only once and the same bytes feed both storage and SHA-256.
    sorted_buffer = bytearray()

    node_hashes = index.node_hashes
    node_min_start = index.node_min_start
    node_max_end = index.node_max_end

    node_offset = len(node_min_start)
    index.edge_node_offset[eid] = node_offset

    # Peaks remain in left-to-right order.  Heights are implicit in the
    # binary representation of the number of already processed leaves.
    peak_hashes: list[bytes] = []
    peak_min_start: list[int] = []
    peak_max_end: list[int] = []

    append_peak_hash = peak_hashes.append
    append_peak_min = peak_min_start.append
    append_peak_max = peak_max_end.append
    pop_peak_hash = peak_hashes.pop
    pop_peak_min = peak_min_start.pop
    pop_peak_max = peak_max_end.pop

    append_node_hash = node_hashes.extend
    append_node_min = node_min_start.append
    append_node_max = node_max_end.append
    append_sorted_entry = sorted_buffer.extend

    pack_entry = _ENTRY_STRUCT.pack
    pack_internal = _MMR_INTERNAL_STRUCT.pack
    sha256 = _sha256

    for entry_index in range(k):
        trajectory_id, start, end = records[entry_index]

        # Leaf bytes are exactly trajectory_id || start || end.
        entry_bytes = pack_entry(
            trajectory_id,
            start,
            end,
        )
        append_sorted_entry(entry_bytes)

        current_hash = sha256(entry_bytes).digest()
        current_min_start = start
        current_max_end = end

        append_node_hash(current_hash)
        append_node_min(current_min_start)
        append_node_max(current_max_end)

        # Adding leaf entry_index is a binary +1 operation on the number of
        # existing leaves.  Every trailing 1 in entry_index causes one MMR
        # merge, exactly matching the equal-height peak-stack algorithm.
        carry = entry_index

        while carry & 1:
            left_hash = pop_peak_hash()
            left_min_start = pop_peak_min()
            left_max_end = pop_peak_max()

            # The append-order experiment intentionally retains file/trajectory
            # arrival order, so the left child is not guaranteed to have the
            # smaller timestamp.  The authenticated tau range is the union of
            # both children; for sorted builds this is equivalent to the old
            # left-child shortcut.
            parent_min_start = min(left_min_start, current_min_start)

            if left_max_end >= current_max_end:
                parent_max_end = left_max_end
            else:
                parent_max_end = current_max_end

            # Paper formula, byte-for-byte:
            # H(hash0 || tau0 || hash1 || tau1)
            current_hash = sha256(
                pack_internal(
                    left_hash,
                    left_min_start,
                    left_max_end,
                    current_hash,
                    current_min_start,
                    current_max_end,
                )
            ).digest()

            current_min_start = parent_min_start
            current_max_end = parent_max_end

            append_node_hash(current_hash)
            append_node_min(current_min_start)
            append_node_max(current_max_end)

            carry >>= 1

        append_peak_hash(current_hash)
        append_peak_min(current_min_start)
        append_peak_max(current_max_end)

    store.edge_buffers[eid] = sorted_buffer
    del records

    node_count = len(node_min_start) - node_offset
    index.edge_node_count[eid] = node_count

    expected_node_count = mmr_node_count(k)
    if node_count != expected_node_count:
        raise RuntimeError(
            f"Incorrect MMR node count for eid={eid}: "
            f"actual={node_count}, expected={expected_node_count}"
        )

    # Peak indices are deterministic from (node_offset, k); deriving them
    # once here is much cheaper than maintaining a node-index stack for every
    # leaf and every internal merge.
    peak_indices = get_peak_node_indices(
        node_offset=node_offset,
        k=k,
    )

    root_e = calculate_mmr_root(
        index=index,
        k=k,
        peak_indices=peak_indices,
    )

    index.edge_k[eid] = k
    index.edge_min_start[eid] = store.min_start[eid]
    index.edge_max_end[eid] = store.max_end[eid]
    index.set_edge_root(eid, root_e)
    index.non_empty_edges += 1


# ============================================================
# Build MMRs for All Road Edges
# ============================================================

def build_all_edge_mmrs(
    store: EdgeEntryStore,
    road: RoadNetwork,
    sort_records: bool = True,
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
            sort_records=sort_records,
        )

    index.total_nodes = len(
        index.node_min_start
    )

    return index


def append_single_edge_mmr(
    eid: int,
    store: EdgeEntryStore,
    index: EdgeMMRIndex,
    delta_records: list[tuple[bytes, int, int]],
) -> dict[str, int]:
    """Append arrival-order leaves and write the updated MMR block.

    Historical nodes are copied byte-for-byte into a new contiguous block so
    the existing postorder addressing rule remains valid. Only new leaves and
    carry merges are hashed. ``store`` must already contain base + delta entries
    in arrival order; ``delta_records`` must be in that same order.
    """
    if not delta_records:
        return {"copied_historical_mmr_nodes": 0, "hashed_new_mmr_nodes": 0}
    old_k = int(index.edge_k[eid])
    old_offset = int(index.edge_node_offset[eid])
    old_count = int(index.edge_node_count[eid])
    if old_count != mmr_node_count(old_k):
        raise RuntimeError(f"eid={eid}: invalid historical MMR node count")

    peaks: list[tuple[int, bytes, int, int]] = []
    if old_k:
        heights = get_peak_heights(old_k)
        nodes = get_peak_node_indices(old_offset, old_k)
        for height, node in zip(heights, nodes):
            peaks.append((int(height), index.get_node_hash(node),
                          int(index.node_min_start[node]), int(index.node_max_end[node])))

    new_offset = len(index.node_min_start)
    if old_count:
        begin = old_offset * HASH_SIZE
        end = begin + old_count * HASH_SIZE
        index.node_hashes.extend(index.node_hashes[begin:end])
        index.node_min_start.extend(index.node_min_start[old_offset:old_offset + old_count])
        index.node_max_end.extend(index.node_max_end[old_offset:old_offset + old_count])

    from crypto import hash_mmr_leaf, hash_mmr_internal
    import struct
    h_count = 0
    new_k = old_k
    new_min = int(index.edge_min_start[eid])
    new_max = int(index.edge_max_end[eid])
    for tid, start, end in delta_records:
        start, end = int(start), int(end)
        current = (0, hash_mmr_leaf(tid, start, end), start, end)
        index.node_hashes.extend(current[1])
        index.node_min_start.append(start)
        index.node_max_end.append(end)
        h_count += 1
        new_k += 1
        new_min = start if new_k == 1 else min(new_min, start)
        new_max = max(new_max, end)
        carry = new_k - 1
        while peaks and peaks[-1][0] == current[0] and (carry & 1):
            left = peaks.pop()
            current = (
                left[0] + 1,
                hash_mmr_internal(left[1], left[2], left[3], current[1], current[2], current[3]),
                min(left[2], current[2]), max(left[3], current[3]),
            )
            index.node_hashes.extend(current[1])
            index.node_min_start.append(current[2])
            index.node_max_end.append(current[3])
            h_count += 1
            carry >>= 1
        peaks.append(current)

    index.edge_node_offset[eid] = new_offset
    index.edge_node_count[eid] = len(index.node_min_start) - new_offset
    index.edge_k[eid] = new_k
    index.edge_min_start[eid] = new_min
    index.edge_max_end[eid] = new_max
    peak_indices = get_peak_node_indices(new_offset, new_k)
    index.set_edge_root(eid, calculate_mmr_root(index, new_k, peak_indices))
    return {"copied_historical_mmr_nodes": old_count, "hashed_new_mmr_nodes": h_count}


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
