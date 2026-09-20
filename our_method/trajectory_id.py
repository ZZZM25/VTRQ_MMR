from __future__ import annotations

import hashlib
import sys

from array import array
from time import perf_counter

from road_network import RoadNetwork

from trajectory_parser import (
    load_trajectory_json,
    parse_trajectory,
)


# ============================================================
# Local SHA-256 Binding
# ============================================================

_sha256 = hashlib.sha256


# ============================================================
# Calculate trajectory_id
# ============================================================

def calculate_trajectory_id(
    eids: list[int],
    trajectory_start: int,
    trajectory_end: int,
) -> bytes:
    """
    trajectory_id definition:

        SHA256(
            eid1
            || eid2
            || ...
            || eidN
            || trajectory_start
            || trajectory_end
        )

    All eid values and timestamps are encoded uniformly as uint32,
    with each value occupying 4 bytes.

    Note:

    1. The eid order must be preserved.
    2. Repeated eid values must be preserved.
    3. Do not sort.
    4. Do not deduplicate.
    """

    if not eids:
        raise ValueError(
            "Trajectory path cannot be empty"
        )

    if trajectory_start > trajectory_end:
        raise ValueError(
            "trajectory_start cannot be greater than trajectory_end"
        )

    # ========================================================
    # Use array('I')
    #
    # Convert the entire eid path into contiguous uint32 memory
    # in one batch.
    #
    # Compared with looping:
    #
    # struct.pack(...)
    # struct.pack(...)
    # struct.pack(...)
    #
    # this reduces a large number of Python-level function calls.
    # ========================================================

    values = array(
        "I",
        eids,
    )

    # Start and end times of the entire trajectory
    values.append(
        trajectory_start
    )

    values.append(
        trajectory_end
    )

    # On Python / Windows, unsigned int is normally 4 bytes.
    # Perform a safety check here.
    if values.itemsize != 4:
        raise RuntimeError(
            "unsigned int is not 4 bytes on the current platform"
        )

    # ========================================================
    # Hash encoding is standardized to big-endian.
    #
    # Windows/x86 is generally little-endian.
    # array.byteswap() performs the conversion in C,
    # which is much faster than converting values one by one
    # in a Python loop.
    # ========================================================

    if sys.byteorder == "little":
        values.byteswap()

    # ========================================================
    # One SHA-256 operation
    # ========================================================

    return _sha256(
        values
    ).digest()


# ============================================================
# Calculate trajectory_id Directly from Parsed Trajectory
# ============================================================

def trajectory_id_from_segments(
    eids: list[int],
    starts: list[int],
    ends: list[int],
) -> bytes:
    """
    Calculate trajectory_id from parsed trajectory data.

    For the entire trajectory:

        trajectory_start = start of the first segment
        trajectory_end   = end of the last segment
    """

    count = len(
        eids
    )

    if (
        count == 0
        or
        count != len(starts)
        or
        count != len(ends)
    ):
        raise ValueError(
            "Invalid trajectory segment data"
        )

    trajectory_start = (
        starts[0]
    )

    trajectory_end = (
        ends[-1]
    )

    return calculate_trajectory_id(
        eids=eids,
        trajectory_start=trajectory_start,
        trajectory_end=trajectory_end,
    )


# ============================================================
# Run Directly with the PyCharm Green Triangle
# ============================================================

def main():
    NODE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_nodes.txt"
    )

    EDGE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_edges.txt"
    )

    # Currently using 10-1
    TRAJECTORY_FILE = (
        "E:\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-1.json"
    )

    # ========================================================
    # Load Road Network
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
    # Load Trajectories
    # ========================================================

    print()

    print(
        "Loading trajectories..."
    )

    trajectories = (
        load_trajectory_json(
            TRAJECTORY_FILE
        )
    )

    print(
        "Trajectory count:",
        len(trajectories),
    )

    # ========================================================
    # First Trajectory
    # ========================================================

    print()

    print(
        "===== First Trajectory ====="
    )

    eids, starts, ends = (
        parse_trajectory(
            trajectory=trajectories[0],
            trajectory_index=0,
            road=road,
        )
    )

    trajectory_start = (
        starts[0]
    )

    trajectory_end = (
        ends[-1]
    )

    trajectory_id = (
        calculate_trajectory_id(
            eids=eids,
            trajectory_start=trajectory_start,
            trajectory_end=trajectory_end,
        )
    )

    print(
        "Road segment count:",
        len(eids),
    )

    print(
        "eid path:"
    )

    print(
        eids
    )

    print(
        "trajectory_start:",
        trajectory_start,
    )

    print(
        "trajectory_end:",
        trajectory_end,
    )

    print()

    print(
        "trajectory_id:"
    )

    print(
        trajectory_id.hex()
    )

    print(
        "trajectory_id length:",
        len(trajectory_id),
        "bytes",
    )

    # ========================================================
    # Test All Trajectories
    # ========================================================

    print()

    print(
        "===== Calculate All trajectory_id Values ====="
    )

    start_time = (
        perf_counter()
    )

    trajectory_count = len(
        trajectories
    )

    trajectory_ids = [
        b""
    ] * trajectory_count

    total_segments = 0

    for trajectory_index in range(
        trajectory_count
    ):

        trajectory = (
            trajectories[
                trajectory_index
            ]
        )

        eids, starts, ends = (
            parse_trajectory(
                trajectory=trajectory,
                trajectory_index=trajectory_index,
                road=road,
            )
        )

        total_segments += len(
            eids
        )

        trajectory_ids[
            trajectory_index
        ] = (
            trajectory_id_from_segments(
                eids=eids,
                starts=starts,
                ends=ends,
            )
        )

    elapsed = (
        perf_counter()
        - start_time
    )

    # ========================================================
    # Check for Duplicate trajectory_id Values
    # ========================================================

    unique_count = len(
        set(
            trajectory_ids
        )
    )

    duplicate_count = (
        trajectory_count
        - unique_count
    )

    print()

    print(
        "===== Statistics ====="
    )

    print(
        "Trajectory count:",
        trajectory_count,
    )

    print(
        "Total road segment count:",
        total_segments,
    )

    print(
        "trajectory_id count:",
        len(
            trajectory_ids
        ),
    )

    print(
        "Unique trajectory_id count:",
        unique_count,
    )

    print(
        "Duplicate trajectory_id count:",
        duplicate_count,
    )

    print(
        "Calculation time:",
        f"{elapsed:.6f} s",
    )

    if trajectory_count:

        print(
            "Average per trajectory:",
            f"{elapsed / trajectory_count:.9f} s",
        )

    print()

    print(
        "trajectory_id.py completed successfully"
    )


if __name__ == "__main__":
    main()

