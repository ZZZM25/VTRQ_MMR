from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass

from query_utils import (
    normalize_rectangle,
    segment_intersects_rectangle,
)


@dataclass(
    slots=True,
    frozen=True,
)
class TrajectorySegment:
    start_node: int
    start_time: int
    end_node: int
    end_time: int


def _raise_csv_field_limit() -> None:

    limit = (
        sys.maxsize
    )

    while True:

        try:
            csv.field_size_limit(
                limit
            )

            return

        except OverflowError:

            limit //= 10


def load_trajectory_catalog(
    csv_file: str,
):
    """
    Preload the CSV into:
        trajectory_id -> segments

    The file reading and JSON parsing time in this step
    is not included in Fine Filtering Time.
    """

    _raise_csv_field_limit()

    catalog = {}

    with open(
        csv_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = (
            csv.DictReader(
                f
            )
        )

        required = {
            "trajectory_id",
            "segment_count",
            "segments",
        }

        if (
            reader.fieldnames is None
            or
            set(
                reader.fieldnames
            )
            != required
        ):

            raise ValueError(
                "Invalid trajectory_catalog.csv header"
            )

        for row_number, row in enumerate(
            reader,
            2,
        ):

            trajectory_id = (
                bytes.fromhex(
                    row[
                        "trajectory_id"
                    ]
                )
            )

            if len(
                trajectory_id
            ) != 32:

                raise ValueError(
                    f"CSV row {row_number} "
                    "trajectory_id is not 32 bytes"
                )

            raw_segments = (
                json.loads(
                    row[
                        "segments"
                    ]
                )
            )

            expected_count = int(
                row[
                    "segment_count"
                ]
            )

            if (
                len(
                    raw_segments
                )
                !=
                expected_count
            ):

                raise ValueError(
                    f"CSV row {row_number} "
                    "segment_count does not match"
                )

            segments = []

            for segment_index, raw in enumerate(
                raw_segments
            ):

                if len(raw) != 4:

                    raise ValueError(
                        f"CSV row {row_number} "
                        f"segment {segment_index} has invalid format"
                    )

                segment = (
                    TrajectorySegment(
                        start_node=
                        int(
                            raw[0]
                        ),

                        start_time=
                        int(
                            raw[1]
                        ),

                        end_node=
                        int(
                            raw[2]
                        ),

                        end_time=
                        int(
                            raw[3]
                        ),
                    )
                )

                if (
                    segment.start_time
                    >
                    segment.end_time
                ):

                    raise ValueError(
                        f"CSV row {row_number} "
                        f"segment {segment_index} has reversed time order"
                    )

                segments.append(
                    segment
                )

            if trajectory_id in catalog:

                raise ValueError(
                    "Duplicate trajectory_id found in CSV: "
                    +
                    trajectory_id.hex()
                )

            catalog[
                trajectory_id
            ] = tuple(
                segments
            )

    return catalog


def _interpolate(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    alpha: float,
) -> tuple[
    float,
    float,
]:

    return (
        x1
        +
        (
            x2
            -
            x1
        )
        *
        alpha,

        y1
        +
        (
            y2
            -
            y1
        )
        *
        alpha,
    )


def trajectory_matches_exact_query(
    segments,
    road,
    query_min_lon: float,
    query_min_lat: float,
    query_max_lon: float,
    query_max_lat: float,
    query_start: int,
    query_end: int,
) -> bool:
    """
    Client-side fine spatiotemporal filtering.

    For each directed Segment:

        start_node @ start_time
              ↓
        end_node   @ end_time

    First compute the overlap between the Segment time and the Query time,
    then perform linear interpolation along the actual movement direction,
    obtain the actual subsegment traveled during the Query time interval,
    and finally determine whether that subsegment intersects the spatial Query.
    """

    if query_start > query_end:

        raise ValueError(
            "query_start > query_end"
        )

    (
        query_min_lon,
        query_min_lat,
        query_max_lon,
        query_max_lat,
    ) = normalize_rectangle(
        query_min_lon,
        query_min_lat,
        query_max_lon,
        query_max_lat,
    )

    for segment in segments:

        # ----------------------------------------------------
        # Time Overlap
        # ----------------------------------------------------

        overlap_start = max(
            segment.start_time,
            query_start,
        )

        overlap_end = min(
            segment.end_time,
            query_end,
        )

        if (
            overlap_start
            >
            overlap_end
        ):
            continue

        # ----------------------------------------------------
        # Note:
        # start_node/end_node preserve the actual direction of the original trajectory.
        # Do not apply min/max and do not convert to an undirected representation.
        # ----------------------------------------------------

        start_node = (
            segment.start_node
        )

        end_node = (
            segment.end_node
        )

        x1 = (
            road.node_lon[
                start_node
            ]
        )

        y1 = (
            road.node_lat[
                start_node
            ]
        )

        x2 = (
            road.node_lon[
                end_node
            ]
        )

        y2 = (
            road.node_lat[
                end_node
            ]
        )

        duration = (
            segment.end_time
            -
            segment.start_time
        )

        # ----------------------------------------------------
        # Zero-Duration Segment:
        # Velocity interpolation is not possible, so evaluate using the start position.
        # ----------------------------------------------------

        if duration == 0:

            if (
                query_min_lon
                <=
                x1
                <=
                query_max_lon
                and
                query_min_lat
                <=
                y1
                <=
                query_max_lat
            ):

                return True

            continue

        # ----------------------------------------------------
        # Directed Linear Interpolation
        #
        # alpha=0 -> actual start node
        # alpha=1 -> actual end node
        # ----------------------------------------------------

        alpha_start = (
            overlap_start
            -
            segment.start_time
        ) / duration

        alpha_end = (
            overlap_end
            -
            segment.start_time
        ) / duration

        sx, sy = _interpolate(
            x1,
            y1,
            x2,
            y2,
            alpha_start,
        )

        ex, ey = _interpolate(
            x1,
            y1,
            x2,
            y2,
            alpha_end,
        )

        # ----------------------------------------------------
        # Check only the spatial subsegment actually traversed within the Query time range.
        # ----------------------------------------------------

        if segment_intersects_rectangle(
            sx,
            sy,
            ex,
            ey,
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
        ):

            return True

    return False


def fine_filter_candidates(
    candidate_ids,
    catalog,
    road,
    query_min_lon: float,
    query_min_lat: float,
    query_max_lon: float,
    query_max_lat: float,
    query_start: int,
    query_end: int,
) -> list[bytes]:
    """
    Call this after the CSV has already been preloaded.

    Fine Filtering Time only needs to cover this function.

    Includes:
      1. candidate trajectory_id -> dict lookup
      2. directed linear interpolation
      3. fine spatiotemporal intersection check

    Excludes:
      CSV reading
      CSV JSON parsing
      road network file loading
    """

    result: list[
        bytes
    ] = []

    for trajectory_id in candidate_ids:

        try:

            segments = catalog[
                trajectory_id
            ]

        except KeyError as e:

            raise KeyError(
                "Candidate trajectory_id "
                "does not exist in trajectory_catalog.csv: "
                +
                trajectory_id.hex()
            ) from e

        if trajectory_matches_exact_query(
            segments=
            segments,

            road=
            road,

            query_min_lon=
            query_min_lon,

            query_min_lat=
            query_min_lat,

            query_max_lon=
            query_max_lon,

            query_max_lat=
            query_max_lat,

            query_start=
            query_start,

            query_end=
            query_end,
        ):

            result.append(
                trajectory_id
            )

    return result
