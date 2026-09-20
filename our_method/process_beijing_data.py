from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path


# ============================================================
# Beijing Unified Preprocessing
#
# Functions:
# 1. Original node_id -> contiguous internal_node_id
# 2. Original edges -> contiguous internal_eid
# 3. Merge duplicate edges with the same undirected endpoints
# 4. Convert node_id in trajectory points_data to internal_node_id
# 5. Verify that every non-stationary trajectory step maps to an edge in the normalized road network
#
# Only 3 final files are generated:
#   beijing_nodes.txt
#   beijing_edges.txt
#   beijing_trajectories.csv
#
# Node/edge mapping files are not saved.
# ============================================================


# ============================================================
# You only need to confirm the three raw file paths here
#
# The filenames below follow the uploaded files by default.
# If your local filenames/directories differ, modify only these three items.
# ============================================================

RAW_NODE_FILE = Path(
    r"E:\MMR_Trajectory_range\beijing_nodes.txt"
)

RAW_EDGE_FILE = Path(
    r"E:\MMR_Trajectory_range\beijing_directed_edges_25w.csv"
)

RAW_TRAJECTORY_FILE = Path(
    r"E:\MMR_Trajectory_range\beijing_trajectories.csv"
)


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = Path(
    r"E:\MMR_Trajectory_range\beijing_preprocessed"
)

OUTPUT_NODE_FILE = (
    OUTPUT_DIR
    / "beijing_nodes.txt"
)

OUTPUT_EDGE_FILE = (
    OUTPUT_DIR
    / "beijing_edges.txt"
)

OUTPUT_TRAJECTORY_FILE = (
    OUTPUT_DIR
    / "beijing_trajectories.csv"
)


UINT32_MAX = (1 << 32) - 1


def set_large_csv_field_limit() -> None:
    """
    points_data can be very long, so increase the CSV single-field size limit.
    """

    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(
                limit
            )
            return
        except OverflowError:
            limit = limit // 10


def canonical_pair(
    u: int,
    v: int,
) -> tuple[int, int]:

    if u <= v:
        return u, v

    return v, u


def load_and_write_nodes():
    """
    Raw nodes:
        original_node_id lon lat

    Output:
        internal_node_id lon lat

    internal_node_id:
        0,1,2,... contiguous IDs

    Returns:
        original_node_id -> internal_node_id
    """

    original_to_internal: dict[
        int,
        int,
    ] = {}

    node_count = 0

    temp_file = OUTPUT_NODE_FILE.with_suffix(
        ".txt.tmp"
    )

    try:
        with RAW_NODE_FILE.open(
            "r",
            encoding="utf-8-sig",
        ) as fin, temp_file.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as fout:

            for line_no, line in enumerate(
                fin,
                1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                parts = stripped.split()

                if len(parts) < 3:
                    raise ValueError(
                        f"Node file line {line_no} has fewer than 3 columns: "
                        f"{stripped}"
                    )

                original_node_id = int(
                    parts[0]
                )

                lon_text = parts[1]
                lat_text = parts[2]

                # Validate longitude and latitude.
                float(lon_text)
                float(lat_text)

                if (
                    original_node_id
                    in
                    original_to_internal
                ):
                    raise ValueError(
                        "Duplicate original_node_id: "
                        f"{original_node_id}"
                    )

                internal_node_id = (
                    node_count
                )

                if (
                    internal_node_id
                    >
                    UINT32_MAX
                ):
                    raise OverflowError(
                        "internal_node_id "
                        "exceeds uint32"
                    )

                original_to_internal[
                    original_node_id
                ] = internal_node_id

                fout.write(
                    f"{internal_node_id} "
                    f"{lon_text} "
                    f"{lat_text}\n"
                )

                node_count += 1

        if node_count == 0:
            raise RuntimeError(
                "Node file is empty"
            )

        os.replace(
            temp_file,
            OUTPUT_NODE_FILE,
        )

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    return (
        original_to_internal,
        node_count,
    )


def load_and_write_edges(
    node_id_map: dict[int, int],
):
    """
    Raw edge CSV:
        edge_id,start_node,end_node,...

    Use only the first three columns:
        edge_id
        start_node
        end_node

    Output:
        internal_eid internal_u internal_v

    Note:
    The current system uses undirected canonical(u,v) for road lookup.
    Therefore, only one internal_eid is kept for each undirected endpoint pair.

    Returns:
        valid_edge_pairs:
            {(u,v), ...}

        raw_edge_count
        unique_edge_count
        duplicate_edge_count
    """

    valid_edge_pairs: set[
        tuple[int, int]
    ] = set()

    raw_edge_count = 0
    unique_edge_count = 0
    duplicate_edge_count = 0

    seen_original_edge_ids: set[int] = set()

    temp_file = OUTPUT_EDGE_FILE.with_suffix(
        ".txt.tmp"
    )

    try:
        with RAW_EDGE_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as fin, temp_file.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as fout:

            reader = csv.DictReader(
                fin
            )

            required = {
                "edge_id",
                "start_node",
                "end_node",
            }

            fields = set(
                reader.fieldnames
                or
                []
            )

            missing = (
                required
                -
                fields
            )

            if missing:
                raise ValueError(
                    "Edge CSV is missing fields: "
                    +
                    ", ".join(
                        sorted(
                            missing
                        )
                    )
                )

            for row_no, row in enumerate(
                reader,
                2,
            ):
                raw_edge_count += 1

                original_edge_id = int(
                    row["edge_id"]
                )

                original_u = int(
                    row["start_node"]
                )

                original_v = int(
                    row["end_node"]
                )

                if (
                    original_edge_id
                    in
                    seen_original_edge_ids
                ):
                    raise ValueError(
                        "Duplicate original_edge_id: "
                        f"{original_edge_id}"
                    )

                seen_original_edge_ids.add(
                    original_edge_id
                )

                if original_u not in node_id_map:
                    raise KeyError(
                        f"Edge row {row_no}: "
                        f"start_node={original_u} "
                        "is not present in the node file"
                    )

                if original_v not in node_id_map:
                    raise KeyError(
                        f"Edge row {row_no}: "
                        f"end_node={original_v} "
                        "is not present in the node file"
                    )

                internal_u = (
                    node_id_map[
                        original_u
                    ]
                )

                internal_v = (
                    node_id_map[
                        original_v
                    ]
                )

                pair = canonical_pair(
                    internal_u,
                    internal_v,
                )

                if pair in valid_edge_pairs:
                    duplicate_edge_count += 1
                    continue

                internal_eid = (
                    unique_edge_count
                )

                if internal_eid > UINT32_MAX:
                    raise OverflowError(
                        "internal_eid "
                        "exceeds uint32"
                    )

                valid_edge_pairs.add(
                    pair
                )

                # ------------------------------------------------
                # Output is also normalized to undirected canonical form:
                #
                #     eid min(u,v) max(u,v)
                #
                # Therefore, both directions will not appear simultaneously:
                #     u v
                #     v u
                #
                # Keep each undirected endpoint pair only once.
                # ------------------------------------------------

                canonical_u, canonical_v = pair

                fout.write(
                    f"{internal_eid} "
                    f"{canonical_u} "
                    f"{canonical_v}\n"
                )

                unique_edge_count += 1

        if unique_edge_count == 0:
            raise RuntimeError(
                "Edge file is empty"
            )

        # valid_edge_pairs is itself a set,
        # so all undirected edges are naturally guaranteed to be unique.
        if len(valid_edge_pairs) != unique_edge_count:
            raise AssertionError(
                "Undirected edge deduplication check failed"
            )

        os.replace(
            temp_file,
            OUTPUT_EDGE_FILE,
        )

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    return (
        valid_edge_pairs,
        raw_edge_count,
        unique_edge_count,
        duplicate_edge_count,
    )


def convert_trajectory_point(
    point,
    node_id_map: dict[int, int],
    row_no: int,
    point_index: int,
):
    """
    Raw Beijing point:

        [
            original_node_id,
            lon,
            lat,
            timestamp
        ]

    Output:

        [
            internal_node_id,
            lon,
            lat,
            timestamp
        ]
    """

    if (
        not isinstance(
            point,
            (list, tuple),
        )
        or
        len(point) < 4
    ):
        raise ValueError(
            f"Trajectory CSV row {row_no} "
            f"has invalid format at point {point_index}: "
            f"{point!r}"
        )

    original_node_id = int(
        point[0]
    )

    if original_node_id not in node_id_map:
        raise KeyError(
            f"Trajectory CSV row {row_no} "
            f"point {point_index}: "
            f"node_id={original_node_id} "
            "is not present in the node file"
        )

    internal_node_id = (
        node_id_map[
            original_node_id
        ]
    )

    lon = float(
        point[1]
    )

    lat = float(
        point[2]
    )

    timestamp = int(
        point[3]
    )

    return [
        internal_node_id,
        lon,
        lat,
        timestamp,
    ]


def convert_and_write_trajectories(
    node_id_map: dict[int, int],
    valid_edge_pairs: set[
        tuple[int, int]
    ],
):
    """
    Raw trajectory CSV:

        traj_id,points_data

    points_data:
        [
            [original_node_id, lon, lat, timestamp],
            ...
        ]

    Output trajectory CSV:

        source_traj_id,points_data

    points_data:
        [
            [internal_node_id, lon, lat, timestamp],
            ...
        ]

    All node_id values are converted into the unified new ID space.

    Also verify:
    - timestamps do not go backward
    - every non-stationary adjacent node pair exists in the normalized road network
    """

    trajectory_count = 0
    point_count = 0
    movement_step_count = 0
    stationary_step_count = 0

    # --------------------------------------------------------
    # The raw Beijing trajectory CSV contains fully duplicated rows.
    #
    # Use source traj_id as the original-record identity for input cleaning:
    # - First occurrence: keep it
    # - Same source traj_id appears again with identical points_data: skip it
    # - Same source traj_id appears again with different points_data: raise an error
    #
    # This is only input deduplication and does not change the system's trajectory_id definition.
    # --------------------------------------------------------
    seen_source_trajectories: dict[str, str] = {}
    duplicate_trajectory_rows = 0

    min_timestamp = None
    max_timestamp = None

    temp_file = OUTPUT_TRAJECTORY_FILE.with_suffix(
        ".csv.tmp"
    )

    try:
        with RAW_TRAJECTORY_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as fin, temp_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as fout:

            reader = csv.DictReader(
                fin
            )

            required = {
                "traj_id",
                "points_data",
            }

            fields = set(
                reader.fieldnames
                or
                []
            )

            missing = (
                required
                -
                fields
            )

            if missing:
                raise ValueError(
                    "Trajectory CSV is missing fields: "
                    +
                    ", ".join(
                        sorted(
                            missing
                        )
                    )
                )

            writer = csv.writer(
                fout
            )

            writer.writerow(
                [
                    "source_traj_id",
                    "points_data",
                ]
            )

            for row_no, row in enumerate(
                reader,
                2,
            ):
                source_traj_id = (
                    row["traj_id"].strip()
                )

                if not source_traj_id:
                    raise ValueError(
                        f"Trajectory CSV row {row_no} "
                        "traj_id is empty"
                    )

                raw_points = json.loads(
                    row["points_data"]
                )

                if (
                    not isinstance(
                        raw_points,
                        list,
                    )
                    or
                    len(raw_points) == 0
                ):
                    raise ValueError(
                        f"Trajectory CSV row {row_no} "
                        "points_data is empty or is not an array"
                    )

                # Normalize points_data and compute a digest to verify duplicate rows
                # and confirm whether the data is truly identical.
                points_digest = hashlib.sha256(
                    json.dumps(
                        raw_points,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()

                previous_digest = (
                    seen_source_trajectories.get(
                        source_traj_id
                    )
                )

                if previous_digest is not None:
                    if previous_digest != points_digest:
                        raise ValueError(
                            f"The same source traj_id has different trajectory contents: "
                            f"row={row_no}, "
                            f"source_traj_id={source_traj_id}"
                        )

                    duplicate_trajectory_rows += 1
                    continue

                seen_source_trajectories[
                    source_traj_id
                ] = points_digest

                new_points = []

                previous_node = None
                previous_time = None

                for point_index, point in enumerate(
                    raw_points
                ):
                    new_point = (
                        convert_trajectory_point(
                            point,
                            node_id_map,
                            row_no,
                            point_index,
                        )
                    )

                    (
                        internal_node_id,
                        _lon,
                        _lat,
                        timestamp,
                    ) = new_point

                    if (
                        previous_time is not None
                        and
                        timestamp
                        <
                        previous_time
                    ):
                        raise ValueError(
                            f"Trajectory CSV row {row_no} "
                            f"timestamp goes backward: "
                            f"{previous_time} -> {timestamp}"
                        )

                    if (
                        previous_node is not None
                    ):
                        if (
                            previous_node
                            ==
                            internal_node_id
                        ):
                            # Staying at the same node does not correspond to road traversal.
                            stationary_step_count += 1

                        else:
                            pair = canonical_pair(
                                previous_node,
                                internal_node_id,
                            )

                            if (
                                pair
                                not in
                                valid_edge_pairs
                            ):
                                raise KeyError(
                                    f"Trajectory CSV row {row_no}: "
                                    "Adjacent nodes cannot be mapped to an edge in the normalized road network: "
                                    f"{previous_node} -> "
                                    f"{internal_node_id}"
                                )

                            movement_step_count += 1

                    previous_node = (
                        internal_node_id
                    )

                    previous_time = (
                        timestamp
                    )

                    if (
                        min_timestamp is None
                        or
                        timestamp
                        <
                        min_timestamp
                    ):
                        min_timestamp = (
                            timestamp
                        )

                    if (
                        max_timestamp is None
                        or
                        timestamp
                        >
                        max_timestamp
                    ):
                        max_timestamp = (
                            timestamp
                        )

                    new_points.append(
                        new_point
                    )

                    point_count += 1

                writer.writerow(
                    [
                        source_traj_id,
                        json.dumps(
                            new_points,
                            separators=(
                                ",",
                                ":",
                            ),
                            ensure_ascii=False,
                        ),
                    ]
                )

                trajectory_count += 1

                if (
                    trajectory_count
                    %
                    1000
                    ==
                    0
                ):
                    print(
                        "Converted trajectories:",
                        trajectory_count,
                    )

        if trajectory_count == 0:
            raise RuntimeError(
                "Trajectory file is empty"
            )

        os.replace(
            temp_file,
            OUTPUT_TRAJECTORY_FILE,
        )

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    return (
        trajectory_count,
        point_count,
        movement_step_count,
        stationary_step_count,
        duplicate_trajectory_rows,
        min_timestamp,
        max_timestamp,
    )


def main():
    set_large_csv_field_limit()

    print(
        "===== Beijing Unified Preprocessing [UNDIRECTED + TRAJECTORY DEDUP] ====="
    )

    print()
    print(
        "Raw node file:"
    )
    print(
        RAW_NODE_FILE
    )

    print()
    print(
        "Raw edge file:"
    )
    print(
        RAW_EDGE_FILE
    )

    print()
    print(
        "Raw trajectory file:"
    )
    print(
        RAW_TRAJECTORY_FILE
    )

    for file_path in (
        RAW_NODE_FILE,
        RAW_EDGE_FILE,
        RAW_TRAJECTORY_FILE,
    ):
        if not file_path.exists():
            raise FileNotFoundError(
                file_path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Nodes
    # ========================================================

    print()
    print(
        "===== Step 1/3: Nodes ====="
    )

    (
        node_id_map,
        node_count,
    ) = load_and_write_nodes()

    print(
        "Node count:",
        node_count,
    )

    print(
        "New node_id range:",
        f"0 ~ {node_count - 1}",
    )

    # ========================================================
    # 2. Edges
    # ========================================================

    print()
    print(
        "===== Step 2/3: Edges ====="
    )

    (
        valid_edge_pairs,
        raw_edge_count,
        unique_edge_count,
        duplicate_edge_count,
    ) = load_and_write_edges(
        node_id_map
    )

    print(
        "Raw edge count:",
        raw_edge_count,
    )

    print(
        "Merged duplicate undirected edges:",
        duplicate_edge_count,
    )

    print(
        "New road count:",
        unique_edge_count,
    )

    print(
        "New EID range:",
        f"0 ~ {unique_edge_count - 1}",
    )

    # ========================================================
    # 3. Trajectories
    # ========================================================

    print()
    print(
        "===== Step 3/3: Trajectories ====="
    )

    (
        trajectory_count,
        point_count,
        movement_step_count,
        stationary_step_count,
        duplicate_trajectory_rows,
        min_timestamp,
        max_timestamp,
    ) = convert_and_write_trajectories(
        node_id_map,
        valid_edge_pairs,
    )

    print()
    print(
        "===== Validation Summary ====="
    )

    print(
        "Trajectory count after deduplication:",
        trajectory_count,
    )

    print(
        "Skipped fully duplicated trajectory rows:",
        duplicate_trajectory_rows,
    )

    print(
        "Unique source_traj_id count:",
        trajectory_count,
    )

    print(
        "Trajectory point count:",
        point_count,
    )

    print(
        "Road movement step count:",
        movement_step_count,
    )

    print(
        "Same-node stationary step count:",
        stationary_step_count,
    )

    print(
        "Minimum timestamp:",
        min_timestamp,
    )

    print(
        "Maximum timestamp:",
        max_timestamp,
    )

    print()
    print(
        "All non-stationary adjacent trajectory nodes were successfully mapped to road edges."
    )

    print()
    print(
        "===== Final Output ====="
    )

    print(
        OUTPUT_NODE_FILE
    )

    print(
        OUTPUT_EDGE_FILE
    )

    print(
        OUTPUT_TRAJECTORY_FILE
    )

    print()
    print(
        "Beijing unified preprocessing completed successfully."
    )


if __name__ == "__main__":
    main()
