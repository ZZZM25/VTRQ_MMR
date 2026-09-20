from __future__ import annotations

import csv
import hashlib
import inspect
import json
import struct
from pathlib import Path

from road_network import RoadNetwork
from trajectory_parser import (
    load_trajectory_json,
    parse_trajectory,
)

# ============================================================
# Configuration
# ============================================================

NODE_FILE = (
    r"E:\MMR_Trajectory_range\xian_nodes.txt"
)

EDGE_FILE = (
    r"E:\MMR_Trajectory_range\xian_edges.txt"
)

TRAJECTORY_DIR = Path(
    r"E:\Graph-Diffusion-Planning-main\loader\preprocess\mm\sets_data\real2\trajectories"
)

# Two months of data
MONTHS = (10,)

OUTPUT_DIR = Path("catalog_output")
CSV_FILE = OUTPUT_DIR / "xian_trajectory_catalog.csv"

_U32 = struct.Struct(">I")


# ============================================================
# trajectory_id
#
# Keep consistent with the current project definition:
#
# SHA256(
#     eid_path
#     || trajectory_start
#     || trajectory_end
# )
#
# All integers use uint32 big-endian encoding.
# Preserve both EID order and duplicates.
# ============================================================

def compute_trajectory_id(
        eid_path: list[int],
        trajectory_start: int,
        trajectory_end: int,
) -> bytes:
    h = hashlib.sha256()

    for eid in eid_path:
        h.update(
            _U32.pack(
                int(eid)
            )
        )

    h.update(
        _U32.pack(
            int(trajectory_start)
        )
    )

    h.update(
        _U32.pack(
            int(trajectory_end)
        )
    )

    return h.digest()


# ============================================================
# File sorting
# ============================================================

def trajectory_sort_key(
        path: Path,
) -> tuple[int, int]:
    # traj-10-1.json
    parts = (
        path.stem
        .split("-")
    )

    if len(parts) < 3:
        raise ValueError(
            f"Failed to parse trajectory filename: {path.name}"
        )

    return (
        int(parts[1]),
        int(parts[2]),
    )


# def collect_trajectory_files() -> list[Path]:
#     files: list[Path] = []
#
#     for month in MONTHS:
#         files.extend(
#             TRAJECTORY_DIR.glob(
#                 f"traj-{month}-*.json"
#             )
#         )
#
#     files = sorted(
#         files,
#         key=trajectory_sort_key,
#     )
#
#     if not files:
#         raise FileNotFoundError(
#             "No trajectory JSON files found"
#         )
#
#     return files

def collect_trajectory_files() -> list[Path]:
    files = sorted(
        TRAJECTORY_DIR.glob(
            "traj_mapped_xian_xian10-*.json"
        ),
        key=lambda p: int(
            p.stem.rsplit("-", 1)[1]
        ),
    )

    if not files:
        raise FileNotFoundError(
            "No trajectory JSON files found"
        )

    return files
# ============================================================
# Compatibility wrapper for the existing trajectory_parser.parse_trajectory
#
# Purpose:
# Never re-parse start/end values for the CSV,
# directly reuse the trajectory_parser.py used by the main index.
#
# Supports common signatures:
#
# parse_trajectory(trajectory, road)
# parse_trajectory(trajectory, trajectory_index, road)
#
# Also supports automatic binding by parameter name.
# ============================================================

def call_project_parser(
        raw_trajectory,
        trajectory_index: int,
        road: RoadNetwork,
):
    signature = inspect.signature(
        parse_trajectory
    )

    parameters = list(
        signature.parameters.values()
    )

    kwargs = {}

    for parameter in parameters:

        name = (
            parameter.name
            .lower()
        )

        if name in {
            "trajectory",
            "traj",
            "raw_trajectory",
            "trajectory_data",
        }:
            kwargs[
                parameter.name
            ] = raw_trajectory

        elif name in {
            "trajectory_index",
            "traj_index",
            "index",
            "idx",
        }:
            kwargs[
                parameter.name
            ] = trajectory_index

        elif name in {
            "road",
            "road_network",
            "network",
        }:
            kwargs[
                parameter.name
            ] = road

    # If all required parameters can be matched by name, prefer keyword invocation.
    required_names = [
        p.name
        for p in parameters
        if (
                p.default
                is inspect._empty
                and
                p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
        )
    ]

    if all(
            name in kwargs
            for name in required_names
    ):
        return parse_trajectory(
            **kwargs
        )

    # Then fall back to positional-argument compatibility.
    positional = [
        p
        for p in parameters
        if p.kind
           in (
               inspect.Parameter.POSITIONAL_ONLY,
               inspect.Parameter.POSITIONAL_OR_KEYWORD,
           )
    ]

    if len(positional) == 2:
        return parse_trajectory(
            raw_trajectory,
            road,
        )

    if len(positional) == 3:
        return parse_trajectory(
            raw_trajectory,
            trajectory_index,
            road,
        )

    raise TypeError(
        "Unable to automatically match parameters for trajectory_parser.parse_trajectory."
        f" Current signature: {signature}"
    )


# ============================================================
# Unified extraction:
# eids / starts / ends
#
# Supports:
# 1. tuple/list: (eids, starts, ends, ...)
# 2. Object attributes: .eids .starts .ends
# ============================================================

def unpack_parser_result(
        result,
) -> tuple[
    list[int],
    list[int],
    list[int],
]:
    if isinstance(
            result,
            (tuple, list),
    ):

        if len(result) < 3:
            raise ValueError(
                "parse_trajectory returned fewer than 3 items, "
                "unable to obtain eids/starts/ends"
            )

        eids = result[0]
        starts = result[1]
        ends = result[2]

    elif all(
            hasattr(
                result,
                name,
            )
            for name in (
                    "eids",
                    "starts",
                    "ends",
            )
    ):

        eids = result.eids
        starts = result.starts
        ends = result.ends

    else:
        raise TypeError(
            "Unable to recognize the parse_trajectory return value. "
            "Expected (eids, starts, ends) or an object with "
            ".eids/.starts/.ends attributes."
        )

    eids = [
        int(x)
        for x in eids
    ]

    starts = [
        int(x)
        for x in starts
    ]

    ends = [
        int(x)
        for x in ends
    ]

    if not (
            len(eids)
            ==
            len(starts)
            ==
            len(ends)
    ):
        raise ValueError(
            "The eids/starts/ends lengths returned by "
            "parse_trajectory are inconsistent"
        )

    if not eids:
        raise ValueError(
            "parse_trajectory returned an empty trajectory"
        )

    return (
        eids,
        starts,
        ends,
    )


# ============================================================
# Original direction
#
# Time:
#   100% from trajectory_parser.py
#
# Direction:
#   From the node pairs in the original trajectory[1],
#   without converting to undirected form or sorting u/v.
# ============================================================

def extract_directed_segments(
        raw_trajectory,
        eids: list[int],
        starts: list[int],
        ends: list[int],
) -> list[list[int]]:
    if len(raw_trajectory) < 2:
        raise ValueError(
            "Invalid raw trajectory structure; unable to obtain node pairs"
        )

    node_pairs = (
        raw_trajectory[1]
    )

    if len(node_pairs) != len(eids):
        raise ValueError(
            "The number of raw node pairs does not match "
            "the number of EIDs returned by trajectory_parser: "
            f"{len(node_pairs)} != {len(eids)}"
        )

    segments: list[
        list[int]
    ] = []

    for i, node_pair in enumerate(
            node_pairs
    ):

        if len(node_pair) < 2:
            raise ValueError(
                f"Invalid node pair format at segment {i}"
            )

        actual_start_node = int(
            node_pair[0]
        )

        actual_end_node = int(
            node_pair[1]
        )

        start = int(
            starts[i]
        )

        end = int(
            ends[i]
        )

        if start > end:
            raise ValueError(
                f"Reversed timestamps at segment {i}: "
                f"{start}>{end}"
            )

        # Note:
        # Do not sort nodes here under any circumstances.
        # Preserve the actual movement direction of the trajectory.
        segments.append(
            [
                actual_start_node,
                start,
                actual_end_node,
                end,
            ]
        )

    return segments


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== Build Trajectory Catalog CSV ====="
    )

    # --------------------------------------------------------
    # Use the same RoadNetwork as the main system.
    # --------------------------------------------------------

    road = RoadNetwork()

    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )

    trajectory_files = (
        collect_trajectory_files()
    )

    print(
        "Number of trajectory files:",
        len(trajectory_files),
    )

    print(
        "First file:",
        trajectory_files[0],
    )

    print(
        "Last file:",
        trajectory_files[-1],
    )

    print()
    print(
        "parse_trajectory signature:"
    )

    print(
        inspect.signature(
            parse_trajectory
        )
    )

    total_trajectories = 0
    total_segments = 0

    seen_ids: set[
        bytes
    ] = set()

    with open(
            CSV_FILE,
            "w",
            encoding="utf-8",
            newline="",
    ) as csv_f:

        writer = csv.writer(
            csv_f
        )

        writer.writerow(
            [
                "trajectory_id",
                "segment_count",
                "segments",
            ]
        )

        global_trajectory_index = 0

        for file_index, path in enumerate(
                trajectory_files,
                1,
        ):

            print(
                f"[{file_index}/{len(trajectory_files)}] "
                f"{path.name}"
            )

            # ------------------------------------------------
            # Do not call json.load directly anymore.
            # Directly reuse the existing trajectory loader in the project.
            # ------------------------------------------------

            trajectories = (
                load_trajectory_json(
                    str(path)
                )
            )

            for raw_trajectory in trajectories:

                parser_result = (
                    call_project_parser(
                        raw_trajectory=
                        raw_trajectory,

                        trajectory_index=
                        global_trajectory_index,

                        road=
                        road,
                    )
                )

                (
                    eids,
                    starts,
                    ends,
                ) = unpack_parser_result(
                    parser_result
                )

                # --------------------------------------------
                # start/end come entirely from trajectory_parser.
                # --------------------------------------------

                segments = (
                    extract_directed_segments(
                        raw_trajectory=
                        raw_trajectory,

                        eids=
                        eids,

                        starts=
                        starts,

                        ends=
                        ends,
                    )
                )

                trajectory_start = (
                    starts[0]
                )

                trajectory_end = (
                    ends[-1]
                )

                trajectory_id = (
                    compute_trajectory_id(
                        eid_path=
                        eids,

                        trajectory_start=
                        trajectory_start,

                        trajectory_end=
                        trajectory_end,
                    )
                )

                if trajectory_id in seen_ids:
                    raise ValueError(
                        "Duplicate trajectory_id detected: "
                        + trajectory_id.hex()
                    )

                seen_ids.add(
                    trajectory_id
                )

                writer.writerow(
                    [
                        trajectory_id.hex(),
                        len(segments),
                        json.dumps(
                            segments,
                            separators=(
                                ",",
                                ":",
                            ),
                        ),
                    ]
                )

                total_trajectories += 1

                total_segments += (
                    len(
                        segments
                    )
                )

                global_trajectory_index += 1

    print()
    print(
        "========================================"
    )

    print(
        "CSV generation completed"
    )

    print(
        "File:",
        CSV_FILE,
    )

    print(
        "Number of trajectories:",
        total_trajectories,
    )

    print(
        "Number of road segments:",
        total_segments,
    )

    print(
        "CSV size:",
        f"{CSV_FILE.stat().st_size / 1024 / 1024:.3f} MiB",
    )

    print()
    print(
        "Segment format:"
    )

    print(
        "[actual_start_node, start, "
        "actual_end_node, end]"
    )

    print()
    print(
        "start/end come from trajectory_parser.py; "
        "node directions come from the original node_pair."
    )

    print()
    print(
        "build_trajectory_catalog.py completed successfully"
    )


if __name__ == "__main__":
    main()
