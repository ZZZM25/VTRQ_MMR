from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _set_large_csv_field_limit() -> None:

    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _canonical_pair(
    u: int,
    v: int,
) -> tuple[int, int]:
    if u <= v:
        return u, v
    return v, u


_ROAD_PAIR_CACHE: dict[
    int,
    dict[tuple[int, int], int],
] = {}


def _build_edge_lookup(
    road,
) -> dict[tuple[int, int], int]:

    cache_key = id(road)

    cached = _ROAD_PAIR_CACHE.get(
        cache_key
    )

    if cached is not None:
        return cached

    pair_to_eid: dict[
        tuple[int, int],
        int,
    ] = {}

    for eid in range(
        int(road.max_eid) + 1
    ):
        if not road.edge_exists(eid):
            continue

        u, v = road.get_edge_nodes(
            eid
        )

        u = int(u)
        v = int(v)

        pair = _canonical_pair(
            u,
            v,
        )

        # The Beijing dataset should theoretically have no duplicate
        # undirected edges after preprocessing.
        # If duplicates still appear here, raise an error immediately
        # to avoid ambiguity in trajectory mapping.
        old = pair_to_eid.get(
            pair
        )

        if old is not None:
            raise ValueError(
                "The normalized Beijing edge file still contains "
                "duplicate undirected edges: "
                f"{pair}, EID={old} and {eid}"
            )

        pair_to_eid[
            pair
        ] = eid

    if not pair_to_eid:
        raise RuntimeError(
            "No valid edges found in the road network"
        )

    _ROAD_PAIR_CACHE[
        cache_key
    ] = pair_to_eid

    return pair_to_eid


def load_beijing_trajectory_csv(
    file_path: str | Path,
) -> list[dict[str, Any]]:

    _set_large_csv_field_limit()

    path = Path(
        file_path
    )

    trajectories: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        fields = set(
            reader.fieldnames
            or
            []
        )

        if "points_data" not in fields:
            raise ValueError(
                "The Beijing trajectory CSV is missing "
                "the points_data field"
            )

        if (
            "source_traj_id" not in fields
            and
            "traj_id" not in fields
        ):
            raise ValueError(
                "The Beijing trajectory CSV is missing the "
                "source_traj_id/traj_id field"
            )

        for row_no, row in enumerate(
            reader,
            2,
        ):
            source_traj_id = (
                row.get(
                    "source_traj_id"
                )
                or
                row.get(
                    "traj_id"
                )
                or
                ""
            ).strip()

            try:
                points = json.loads(
                    row["points_data"]
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse points_data JSON at row "
                    f"{row_no} in the Beijing trajectory CSV"
                ) from exc

            if (
                not isinstance(
                    points,
                    list,
                )
                or
                len(points) < 2
            ):
                # Fewer than two points cannot form a road segment.
                continue

            trajectories.append(
                {
                    "source_traj_id":
                    source_traj_id,

                    "points":
                    points,

                    "row_no":
                    row_no,
                }
            )

    print(
        "Beijing CSV loaded successfully, "
        "number of parseable trajectories:",
        len(trajectories),
    )

    return trajectories


# ============================================================
# Parser
#
# The function signature is intentionally flexible:
#
#   parse_trajectory(trajectory, road)
#   parse_trajectory(trajectory, trajectory_index, road)
#
# Both legacy calling conventions are supported.
#
# Returns:
#
#   (eids, starts, ends)
#
# This remains consistent with the existing project pipeline.
# ============================================================

def parse_beijing_trajectory(
    trajectory,
    *args,
    **kwargs,
):
    road = kwargs.get(
        "road"
    )

    if road is None:
        road = kwargs.get(
            "road_network"
        )

    # Support positional arguments:
    #
    # parse_trajectory(traj, road)
    # parse_trajectory(traj, idx, road)
    if road is None:
        for value in reversed(
            args
        ):
            if (
                hasattr(
                    value,
                    "max_eid",
                )
                and
                hasattr(
                    value,
                    "edge_exists",
                )
                and
                hasattr(
                    value,
                    "get_edge_nodes",
                )
            ):
                road = value
                break

    if road is None:
        raise TypeError(
            "parse_beijing_trajectory could not obtain "
            "a RoadNetwork instance"
        )

    if not isinstance(
        trajectory,
        dict,
    ):
        raise TypeError(
            "Beijing adapter expects trajectory to be a dict"
        )

    points = trajectory.get(
        "points"
    )

    row_no = trajectory.get(
        "row_no",
        "?",
    )

    if (
        not isinstance(
            points,
            list,
        )
        or
        len(points) < 2
    ):
        return (
            [],
            [],
            [],
        )

    pair_to_eid = (
        _build_edge_lookup(
            road
        )
    )

    eids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    previous_timestamp = None

    # First, verify that timestamps never go backward.
    normalized_points = []

    for point_index, point in enumerate(
        points
    ):
        if (
            not isinstance(
                point,
                (list, tuple),
            )
            or
            len(point) < 4
        ):
            raise ValueError(
                f"Invalid format for point {point_index} "
                f"at row {row_no} in the Beijing trajectory CSV"
            )

        node_id = int(
            point[0]
        )

        timestamp = int(
            point[3]
        )

        if (
            previous_timestamp
            is not None
            and
            timestamp
            <
            previous_timestamp
        ):
            raise ValueError(
                f"Timestamp goes backward at row {row_no} "
                f"in the Beijing trajectory CSV: "
                f"{previous_timestamp} -> {timestamp}"
            )

        normalized_points.append(
            (
                node_id,
                timestamp,
            )
        )

        previous_timestamp = (
            timestamp
        )

    # --------------------------------------------------------
    # Adjacent nodes -> EID + exact time interval
    # --------------------------------------------------------

    for i in range(
        len(normalized_points) - 1
    ):
        u, start = (
            normalized_points[i]
        )

        v, end = (
            normalized_points[i + 1]
        )

        # Repeated consecutive occurrences of the same node
        # indicate a stay.
        # This is not a road traversal, so no Entry is generated.
        if u == v:
            continue

        pair = _canonical_pair(
            u,
            v,
        )

        eid = pair_to_eid.get(
            pair
        )

        if eid is None:
            raise KeyError(
                f"Beijing trajectory CSV row {row_no}: "
                f"no road was found in the road network "
                f"for adjacent nodes {u} -> {v}"
            )

        if end < start:
            raise ValueError(
                f"Beijing trajectory CSV row {row_no}: "
                f"road traversal timestamps are reversed: "
                f"{start}>{end}"
            )

        eids.append(
            int(eid)
        )

        starts.append(
            int(start)
        )

        ends.append(
            int(end)
        )

    return (
        eids,
        starts,
        ends,
    )


# ============================================================
# Install the adapter into the existing trajectory_parser module.
#
# This must be called before importing edge_entries.
# ============================================================

def install_beijing_adapter():
    import trajectory_parser

    trajectory_parser.load_trajectory_json = (
        load_beijing_trajectory_csv
    )

    trajectory_parser.parse_trajectory = (
        parse_beijing_trajectory
    )

    print(
        "Beijing CSV adapter installed"
    )