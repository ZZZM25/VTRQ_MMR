from __future__ import annotations

import struct
from time import perf_counter

from road_network import RoadNetwork

from trajectory_parser import (
    load_trajectory_json,
    parse_trajectory,
)

from trajectory_id import (
    calculate_trajectory_id,
)


# ============================================================
# Entry Binary Format
#
# trajectory_id : 32 bytes
# start         : uint32 4 bytes
# end           : uint32 4 bytes
#
# Total Entry size:
#
#     32 + 4 + 4 = 40 bytes
# ============================================================

_ENTRY_STRUCT = struct.Struct(">32sII")

ENTRY_SIZE = _ENTRY_STRUCT.size

UINT32_MAX = 0xFFFFFFFF


# ============================================================
# EdgeEntryStore
# ============================================================

class EdgeEntryStore:
    """
    Store all entries on road edges.

    For each road edge:

        edge_buffers[eid]

    Corresponds to:

        Entry1
        Entry2
        Entry3
        ...

    Each Entry:

        (
            trajectory_id,
            start,
            end
        )

    Fixed at 40 bytes.
    """

    __slots__ = (
        "edge_buffers",
        "entry_counts",
        "min_start",
        "max_end",
        "total_entries",
        "non_empty_edges",
        "total_trajectories",
        "processed_files",
    )

    def __init__(
        self,
        max_eid: int,
    ):

        size = max_eid + 1

        # ====================================================
        # Data buffer for each edge
        #
        # None when no trajectory passes through the edge.
        # Create the bytearray only on the first insertion.
        # ====================================================

        self.edge_buffers: list[
            bytearray | None
        ] = [None] * size

        # ====================================================
        # Number of entries on each road edge
        #
        # When building the MMR later:
        #
        # k_e = entry_counts[eid]
        # ====================================================

        self.entry_counts: list[int] = (
            [0] * size
        )

        # ====================================================
        # Overall time range of each edge
        # ====================================================

        self.min_start: list[int] = (
            [UINT32_MAX] * size
        )

        self.max_end: list[int] = (
            [0] * size
        )

        # Total number of entries
        self.total_entries = 0

        # Number of road edges with at least one entry
        self.non_empty_edges = 0

        # Total number of processed trajectories
        self.total_trajectories = 0

        # Number of processed files
        self.processed_files = 0

    # ========================================================
    # Insert Entry
    # ========================================================

    def append(
        self,
        eid: int,
        trajectory_id: bytes,
        start: int,
        end: int,
    ) -> None:
        """
        Insert into the specified road edge:

            Entry(
                trajectory_id,
                start,
                end
            )
        """

        buffer = self.edge_buffers[eid]

        # ====================================================
        # First trajectory passing through this road edge
        # ====================================================

        if buffer is None:

            buffer = bytearray()

            self.edge_buffers[eid] = (
                buffer
            )

            self.non_empty_edges += 1

        # ====================================================
        # Write directly as 40-byte binary data
        # ====================================================

        buffer.extend(
            _ENTRY_STRUCT.pack(
                trajectory_id,
                start,
                end,
            )
        )

        # ====================================================
        # Entry count
        # ====================================================

        self.entry_counts[eid] += 1

        # ====================================================
        # Update the overall time range of the road edge
        #
        # min_start:
        #     Minimum start among all entries
        #
        # max_end:
        #     Maximum end among all entries
        # ====================================================

        if start < self.min_start[eid]:

            self.min_start[eid] = (
                start
            )

        if end > self.max_end[eid]:

            self.max_end[eid] = (
                end
            )

        self.total_entries += 1

    # ========================================================
    # Entry count
    # ========================================================

    def get_entry_count(
        self,
        eid: int,
    ) -> int:

        return self.entry_counts[eid]

    # ========================================================
    # Overall time range of a road edge
    # ========================================================

    def get_time_range(
        self,
        eid: int,
    ) -> tuple[int, int] | None:

        if self.entry_counts[eid] == 0:
            return None

        return (
            self.min_start[eid],
            self.max_end[eid],
        )

    # ========================================================
    # Read a specified Entry
    #
    # Mainly used for testing.
    # During formal MMR construction, the binary buffer is scanned directly.
    # ========================================================

    def get_entry(
        self,
        eid: int,
        index: int,
    ) -> tuple[
        bytes,
        int,
        int,
    ]:

        buffer = self.edge_buffers[eid]

        if buffer is None:

            raise IndexError(
                f"eid={eid} has no Entry"
            )

        count = self.entry_counts[eid]

        if (
            index < 0
            or
            index >= count
        ):
            raise IndexError(
                f"Entry index out of range: "
                f"{index}"
            )

        offset = (
            index * ENTRY_SIZE
        )

        return _ENTRY_STRUCT.unpack_from(
            buffer,
            offset,
        )


# ============================================================
# Process a Trajectory File
# ============================================================

def append_trajectory_file(
    filename: str,
    road: RoadNetwork,
    store: EdgeEntryStore,
) -> tuple[
    int,
    int,
]:
    """
    Append all trajectories from one trajectory JSON file
    to the existing EdgeEntryStore.

    Note:

        Do not create a new store.

    Therefore:

        File 1
        File 2
        File 3

    All are appended into the same set of road-edge buckets.

    Returns:

        trajectory_count
        entry_count
    """

    # ========================================================
    # Read one file
    # ========================================================

    trajectories = (
        load_trajectory_json(
            filename
        )
    )

    trajectory_count = len(
        trajectories
    )

    file_entry_count = 0

    # Local binding for frequently used functions
    parse = parse_trajectory

    calculate_id = (
        calculate_trajectory_id
    )

    append_entry = store.append

    # ========================================================
    # Process trajectories in the current file one by one
    # ========================================================

    for trajectory_index in range(
        trajectory_count
    ):

        trajectory = (
            trajectories[
                trajectory_index
            ]
        )

        # ====================================================
        # 1.
        # Raw trajectory
        #
        # →
        #
        # EID path
        # start list
        # end list
        # ====================================================

        eids, starts, ends = (
            parse(
                trajectory=trajectory,
                trajectory_index=trajectory_index,
                road=road,
            )
        )

        segment_count = len(eids)

        if segment_count == 0:
            continue

        # ====================================================
        # 2.
        # Full trajectory time range
        # ====================================================

        trajectory_start = (
            starts[0]
        )

        trajectory_end = (
            ends[-1]
        )

        # ====================================================
        # 3.
        # Calculate trajectory_id
        #
        # trajectory_id =
        #
        # SHA256(
        #     eid1
        #     || eid2
        #     || ...
        #     || eidN
        #     || trajectory_start
        #     || trajectory_end
        # )
        # ====================================================

        trajectory_id = (
            calculate_id(
                eids=eids,
                trajectory_start=trajectory_start,
                trajectory_end=trajectory_end,
            )
        )

        # ====================================================
        # 4.
        # Split one trajectory into multiple entries
        #
        # Insert each road segment into
        # its corresponding road edge by EID.
        # ====================================================

        for i in range(
            segment_count
        ):

            append_entry(
                eid=eids[i],
                trajectory_id=trajectory_id,
                start=starts[i],
                end=ends[i],
            )

        file_entry_count += (
            segment_count
        )

    # ========================================================
    # Update global statistics
    # ========================================================

    store.total_trajectories += (
        trajectory_count
    )

    store.processed_files += 1

    # ========================================================
    # trajectories are no longer used after this function returns.
    #
    # Reload the next file separately.
    # Therefore, all JSON files are not kept in memory at the same time.
    # ========================================================

    return (
        trajectory_count,
        file_entry_count,
    )


# ============================================================
# Process Multiple Trajectory Files
# ============================================================

def build_edge_entries_from_files(
    trajectory_files: list[str],
    road: RoadNetwork,
) -> EdgeEntryStore:
    """
    Bucket multiple trajectory files into one unified store.

    Files are processed strictly in the order
    given in trajectory_files.

    For example:

        [
            "traj-10-1.json",
            "traj-10-2.json",
            "traj-10-3.json",
        ]

    The Entry insertion order is:

        10-1
        ↓
        10-2
        ↓
        10-3

    No automatic sorting is performed.

    This makes the MMR Entry order fully deterministic.
    """

    if not trajectory_files:

        raise ValueError(
            "trajectory_files cannot be empty"
        )

    # ========================================================
    # All files share one store
    # ========================================================

    store = EdgeEntryStore(
        max_eid=road.max_eid
    )

    file_count = len(
        trajectory_files
    )

    # ========================================================
    # Process the next file only after the current file is finished
    # ========================================================

    for file_index in range(
        file_count
    ):

        filename = (
            trajectory_files[
                file_index
            ]
        )

        print()

        print(
            "----------------------------------------"
        )

        print(
            f"Processing trajectory file "
            f"{file_index + 1}/{file_count}"
        )

        print(
            "File:",
            filename,
        )

        start_time = perf_counter()

        (
            trajectory_count,
            entry_count,
        ) = append_trajectory_file(
            filename=filename,
            road=road,
            store=store,
        )

        elapsed = (
            perf_counter()
            - start_time
        )

        print(
            "Trajectory count:",
            trajectory_count,
        )

        print(
            "Entry count:",
            entry_count,
        )

        print(
            "Processing time:",
            f"{elapsed:.6f} s",
        )

        print(
            "Cumulative trajectories:",
            store.total_trajectories,
        )

        print(
            "Cumulative entries:",
            store.total_entries,
        )

    return store


# ============================================================
# Run Directly with the PyCharm Green Triangle
# ============================================================

def main():

    # ========================================================
    # Road network files
    # ========================================================

    NODE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_nodes.txt"
    )

    EDGE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_edges.txt"
    )

    # ========================================================
    # Trajectory files
    #
    # Add as many files as you have.
    #
    # Insertion order strictly follows the list below.
    #
    # If only some files are currently available,
    # keep only the files that actually exist.
    # ========================================================

    TRAJECTORY_FILES = [
        "E:\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-1.json",
        # "traj-10-2.json",
        # "traj-10-3.json",
        # "traj-10-4.json",
        # "traj-10-5.json",
        # "traj-10-6.json",
        # "traj-10-7.json",
    ]

    # ========================================================
    # Load road network
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

    print(
        "Number of road edges:",
        road.edge_count,
    )

    # ========================================================
    # Bucket all trajectory files into one unified store
    # ========================================================

    print()

    print(
        "===== Start Processing All Trajectory Files ====="
    )

    total_start_time = (
        perf_counter()
    )

    store = (
        build_edge_entries_from_files(
            trajectory_files=TRAJECTORY_FILES,
            road=road,
        )
    )

    total_elapsed = (
        perf_counter()
        - total_start_time
    )

    # ========================================================
    # Final statistics
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "===== All Edge Entries Built ====="
    )

    print(
        "Number of processed files:",
        store.processed_files,
    )

    print(
        "Total number of trajectories:",
        store.total_trajectories,
    )

    print(
        "Total number of entries:",
        store.total_entries,
    )

    print(
        "Number of road edges with trajectories:",
        store.non_empty_edges,
    )

    print(
        "Total processing time:",
        f"{total_elapsed:.6f} s",
    )

    # ========================================================
    # Data size
    # ========================================================

    total_bytes = (
        store.total_entries
        * ENTRY_SIZE
    )

    print()

    print(
        "===== Entry Data Size ====="
    )

    print(
        "Each Entry:",
        ENTRY_SIZE,
        "bytes",
    )

    print(
        "Total Entry size:",
        total_bytes,
        "bytes",
    )

    print(
        "Total Entry size:",
        f"{total_bytes / 1024 / 1024:.3f} MiB",
    )

    # ========================================================
    # Test road edge 1985
    # ========================================================

    test_eid = 1985

    print()

    print(
        f"===== Edge {test_eid} Test ====="
    )

    entry_count = (
        store.get_entry_count(
            test_eid
        )
    )

    print(
        "Entry count:",
        entry_count,
    )

    print(
        "Overall time range:",
        store.get_time_range(
            test_eid
        ),
    )

    # ========================================================
    # Print the first 5 entries
    # ========================================================

    show_count = min(
        5,
        entry_count,
    )

    print()

    print(
        "First few Entries:"
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

        print()

        print(
            f"Entry {i}"
        )

        print(
            "trajectory_id:",
            trajectory_id.hex(),
        )

        print(
            "start:",
            start,
        )

        print(
            "end:",
            end,
        )

    print()

    print(
        "edge_entries.py completed successfully"
    )


if __name__ == "__main__":
    main()
