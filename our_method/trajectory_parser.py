from __future__ import annotations

import json
from time import perf_counter

from road_network import RoadNetwork


# ============================================================
# JSON Parser
#
# If orjson is already installed in the environment:
#     automatically use orjson for better performance.
#
# If not:
#     automatically use Python's built-in json module.
#
# The program will not fail if orjson is unavailable.
# ============================================================

try:
    import orjson

    _HAS_ORJSON = True

except ImportError:

    orjson = None
    _HAS_ORJSON = False


# ============================================================
# Load Trajectory JSON
# ============================================================

def load_trajectory_json(
    filename: str,
):
    """
    Load the complete trajectory JSON.

    Preferred:
        orjson

    Fallback:
        Python json

    Returns:
        top-level trajectory list
    """

    if _HAS_ORJSON:

        # orjson reads binary data directly
        # avoiding the UTF-8 text decoding step
        with open(
            filename,
            "rb",
            buffering=8 * 1024 * 1024,
        ) as f:

            data = orjson.loads(
                f.read()
            )

    else:

        with open(
            filename,
            "r",
            encoding="utf-8",
            buffering=8 * 1024 * 1024,
        ) as f:

            data = json.load(
                f
            )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "The top level of the trajectory JSON must be a list"
        )

    return data


# ============================================================
# Parse One Trajectory
# ============================================================

def parse_trajectory(
    trajectory,
    trajectory_index: int,
    road: RoadNetwork,
) -> tuple[
    list[int],
    list[int],
    list[int],
]:
    """
    Parse one trajectory.

    Use only:

        trajectory[1]
            road node pairs

        trajectory[2]
            GPS point group corresponding to each road segment

    trajectory[0]
        Not needed currently.

    trajectory[3]
        Completely ignored as required.


    Return three lists of equal length:

        eids
        starts
        ends

    For example:

        eids   = [100, 200, 300]

        starts = [1000, 1005, 1012]

        ends   = [1005, 1012, 1020]
    """

    # ========================================================
    # Basic Format Check
    # ========================================================

    if len(
        trajectory
    ) < 3:

        raise ValueError(
            f"Trajectory {trajectory_index} "
            f"contains fewer than 3 parts"
        )

    # Second part:
    #
    # [
    #     [u1, v1],
    #     [u2, v2],
    #     ...
    # ]

    edge_pairs = (
        trajectory[1]
    )

    # Third part:
    #
    # [
    #     GPS group 1,
    #     GPS group 2,
    #     ...
    # ]

    gps_groups = (
        trajectory[2]
    )

    segment_count = len(
        edge_pairs
    )

    # ========================================================
    # Road Segment Count Must Match GPS Group Count
    # ========================================================

    if (
        segment_count
        != len(gps_groups)
    ):

        raise ValueError(
            f"Trajectory {trajectory_index} "
            f"road segment count does not match GPS group count: "
            f"{segment_count} vs "
            f"{len(gps_groups)}"
        )

    # ========================================================
    # Empty Trajectory
    # ========================================================

    if segment_count == 0:

        return (
            [],
            [],
            [],
        )

    # ========================================================
    # Preallocation
    #
    # More suitable than repeated append operations when the length is already known.
    # ========================================================

    eids = [
        0
    ] * segment_count

    starts = [
        0
    ] * segment_count

    ends = [
        0
    ] * segment_count

    # ========================================================
    # Local Binding for Frequently Used Variables
    #
    # Reduce repeated attribute lookups in the Python loop:
    #
    # road.edge_lookup
    # road.edge_key
    #
    #
    # ========================================================

    edge_lookup = (
        road.edge_lookup
    )

    # ========================================================
    # First Road Segment
    # ========================================================

    first_group = (
        gps_groups[0]
    )

    if not first_group:

        raise ValueError(
            f"Trajectory {trajectory_index} "
            f"GPS group 0 is empty"
        )

    # Rule:
    #
    # First edge:
    #
    # start = first GPS timestamp of the first group
    # end   = last GPS timestamp of the first group

    first_start = int(
        first_group[0][0]
    )

    first_end = int(
        first_group[-1][0]
    )

    if first_start > first_end:

        raise ValueError(
            f"Trajectory {trajectory_index} "
            f"segment 0 has invalid time: "
            f"{first_start} > {first_end}"
        )

    # ========================================================
    # Find eid for the First Edge
    # ========================================================

    first_pair = (
        edge_pairs[0]
    )

    if len(
        first_pair
    ) != 2:

        raise ValueError(
            f"Trajectory {trajectory_index} "
            f"road node pair 0 has invalid format"
        )

    u = int(
        first_pair[0]
    )

    v = int(
        first_pair[1]
    )

    # Undirected edge:
    #
    # min(u,v), max(u,v)
    #
    # Compute the packed key directly here,
    # avoiding repeated function calls inside the loop.

    if u > v:
        u, v = v, u

    key = (
        (u << 32)
        | v
    )

    eid = edge_lookup.get(
        key
    )

    if eid is None:

        raise ValueError(
            f"Trajectory {trajectory_index} "
            f"cannot find road edge for segment 0: "
            f"({first_pair[0]}, "
            f"{first_pair[1]})"
        )

    eids[0] = eid
    starts[0] = first_start
    ends[0] = first_end

    # Last Timestamp of the Previous GPS Group
    previous_end = (
        first_end
    )

    # ========================================================
    # Start from the Second Road Segment
    # ========================================================

    for i in range(
        1,
        segment_count,
    ):

        # ----------------------------------------------------
        # GPS group
        # ----------------------------------------------------

        group = (
            gps_groups[i]
        )

        if not group:

            raise ValueError(
                f"Trajectory {trajectory_index} "
                f"GPS group {i} is empty"
            )

        # ================================================
        # Core Continuous-Time Rule
        #
        # start =
        #     last timestamp of the previous GPS group
        #
        # end =
        #     last timestamp of the current GPS group
        # ================================================

        start = (
            previous_end
        )

        end = int(
            group[-1][0]
        )

        if start > end:

            raise ValueError(
                f"Trajectory {trajectory_index} "
                f"segment {i} has invalid time: "
                f"{start} > {end}"
            )

        # The current end becomes the start of the next edge
        previous_end = (
            end
        )

        # ----------------------------------------------------
        # Node Pair
        # ----------------------------------------------------

        pair = (
            edge_pairs[i]
        )

        if len(
            pair
        ) != 2:

            raise ValueError(
                f"Trajectory {trajectory_index} "
                f"road node pair {i} has invalid format"
            )

        u = int(
            pair[0]
        )

        v = int(
            pair[1]
        )

        # ----------------------------------------------------
        # Undirected Road Edge
        # ----------------------------------------------------

        if u > v:
            u, v = v, u

        key = (
            (u << 32)
            | v
        )

        eid = edge_lookup.get(
            key
        )

        if eid is None:

            raise ValueError(
                f"Trajectory {trajectory_index} "
                f"cannot find road edge for segment {i}: "
                f"({pair[0]}, {pair[1]})"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        eids[i] = (
            eid
        )

        starts[i] = (
            start
        )

        ends[i] = (
            end
        )

    return (
        eids,
        starts,
        ends,
    )


# ============================================================
# Run with the PyCharm Green Triangle
# ============================================================

def main():
    NODE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_nodes.txt"
    )

    EDGE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_edges.txt"
    )

    TRAJECTORY_FILE = (
        "E:\\Graph-Diffusion-Planning-main\\chengdu-tra-json\\traj-10-1.json"
    )

    # ========================================================
    # Load Road Network
    # ========================================================

    print(
        "Loading road network..."
    )

    road = (
        RoadNetwork()
    )

    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )

    print(
        "Road network loaded"
    )

    # ========================================================
    # Load Trajectory JSON
    # ========================================================

    print()

    print(
        "Loading trajectory JSON..."
    )

    start_time = (
        perf_counter()
    )

    trajectories = (
        load_trajectory_json(
            TRAJECTORY_FILE
        )
    )

    load_time = (
        perf_counter()
        - start_time
    )

    print(
        "Trajectory count:",
        len(
            trajectories
        ),
    )

    print(
        "JSON parser:",
        (
            "orjson"
            if _HAS_ORJSON
            else "json"
        ),
    )

    print(
        "JSON loading time:",
        f"{load_time:.6f} s",
    )

    # ========================================================
    # Test the First Trajectory
    # ========================================================

    print()

    print(
        "===== First Trajectory Test ====="
    )

    start_time = (
        perf_counter()
    )

    eids, starts, ends = (
        parse_trajectory(
            trajectory=trajectories[0],
            trajectory_index=0,
            road=road,
        )
    )

    parse_time = (
        perf_counter()
        - start_time
    )

    print(
        "Road segment count:",
        len(
            eids
        ),
    )

    print(
        "Parsing time:",
        f"{parse_time:.6f} s",
    )

    # ========================================================
    # Print the First 10 Segments
    # ========================================================

    print()

    print(
        "===== First 10 Road Segments ====="
    )

    show_count = min(
        10,
        len(
            eids
        ),
    )

    for i in range(
        show_count
    ):

        print(
            f"{i}: "
            f"eid={eids[i]}, "
            f"start={starts[i]}, "
            f"end={ends[i]}"
        )

    # ========================================================
    # Validate Time Continuity
    # ========================================================

    print()

    print(
        "===== Time Continuity Check ====="
    )

    continuous = True

    for i in range(
        1,
        len(eids),
    ):

        if (
            starts[i]
            != ends[i - 1]
        ):

            continuous = False

            print(
                f"Segment {i} is not continuous: "
                f"previous segment end="
                f"{ends[i - 1]}, "
                f"current start="
                f"{starts[i]}"
            )

            break

    print(
        "Continuous:",
        continuous,
    )

    if not continuous:

        raise RuntimeError(
            "Trajectory time continuity test failed"
        )

    print()

    print(
        "trajectory_parser.py completed successfully"
    )


if __name__ == "__main__":
    main()
