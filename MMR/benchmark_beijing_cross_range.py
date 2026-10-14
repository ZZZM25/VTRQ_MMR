from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from beijing_csv_adapter import (
    parse_beijing_trajectory,
)
from composite_vo import CompositeVOBuilder
from index_io import load_index
from query_utils import (
    edge_intersects_query,
    segment_intersects_rectangle,
)
from trajectory_catalog import (
    TrajectorySegment,
    fine_filter_candidates,
)
from verifier import verify_composite_vo
from vo_io import save_composite_vo


# ============================================================
# Beijing 5×6 Cross Range Benchmark
# ============================================================

VERSION = "BEIJING_AUTO_DENSE_V1"

INDEX_FILE = (
    "index_output/"
    "beijing_trajectory_index.dat"
)

TRAJECTORY_CSV = (
    "beijing_preprocessed/"
    "beijing_trajectories.csv"
)

OUTPUT_DIR = Path(
    "benchmark_output_beijing"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "beijing_cross_range_results.csv"
)

QUERY_TIME_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_query_time_matrix_s.csv"
)

COARSE_QUERY_TIME_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_coarse_query_time_matrix_s.csv"
)

FINE_FILTERING_TIME_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_fine_filtering_time_matrix_s.csv"
)

VERIFICATION_TIME_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_verification_time_matrix_s.csv"
)

VO_SIZE_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_vo_size_matrix_mb.csv"
)

CANDIDATE_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_candidate_count_matrix.csv"
)

FINAL_RESULT_MATRIX_FILE = (
    OUTPUT_DIR
    / "beijing_final_result_count_matrix.csv"
)

DENSE_REGION_FILE = (
    OUTPUT_DIR
    / "beijing_selected_experiment_setup.csv"
)

TEMP_VO_FILE = (
    OUTPUT_DIR
    / "_temp_vo_size.dat"
)


# ============================================================
# User-specified lower bounds for the Beijing spatial range
#
# The maximum 5km×5km query box must satisfy:
#
#   query_min_lon >= 116.3
#   query_min_lat >= 39.965
#
# No artificial upper bound is specified;
# The query box only needs to lie completely within the actual Beijing road-network MBR.
# ============================================================

MIN_ALLOWED_LON = 0
MIN_ALLOWED_LAT = 0


# ============================================================
# Same spatial and temporal ranges as Chengdu / Xi'an
# ============================================================

SPATIAL_SIDE_KM = [
    # 1.0,
    2.0,
    # 3.0,
    # 4.0,
    # 5.0,
]


TEMPORAL_RANGES = [
    # ("1min", 60),
    # ("30min", 30 * 60),
    ("1h", 60 * 60),
    # ("4h", 4 * 60 * 60),
    # ("12h", 12 * 60 * 60),
    # ("24h", 24 * 60 * 60),

]



# ============================================================
# 30 formal runs + 1 warmup for each group
# ============================================================

REPEATS = 30
WARMUP_REPEATS = 1


# ============================================================
# Automatic spatial-selection parameters
#
# Beijing covers a much larger area than the previous small Xi'an ROI,
# so the entire city is no longer exhaustively scanned at 0.25km intervals.
#
# First aggregate road midpoints on a 0.5km grid using the number of Entries,
# then compute exact 1km Entry weights for the top-ranked candidate centers.
# ============================================================

SPATIAL_GRID_KM = 0.5
SPATIAL_BUCKET_SHORTLIST = 50
CENTER_REFINE_COUNT = 20


# ============================================================
# Automatic temporal-selection parameters
#
# The Beijing dataset may span a very long time, so unlike Xi'an,
# we cannot scan the entire range from the minimum to maximum time every 30 minutes.
#
# Here we first use:
#   real trajectory segments within the 1km spatial box
# to build 1-minute candidate time buckets,
# then run the actual
# Composite VO + Verification + Fine Filtering Probe。
# ============================================================

TIME_BUCKET_SHORTLIST = 20
TIME_PROBE_OFFSETS_SECONDS = (
    -30,
    0,
    30,
)

TIME_FINE_RADIUS_SECONDS = (
    30 * 60
)

TIME_FINE_STEP_SECONDS = 60

MAX_EXPERIMENT_WINDOW_SECONDS = (
    24 * 60 * 60
)


_U32 = struct.Struct(">I")

BEIJING_TZ = timezone(
    timedelta(hours=8)
)


@dataclass(slots=True)
class ResultRow:
    spatial_side_km: float
    temporal_label: str
    temporal_seconds: int

    query_min_lon: float
    query_min_lat: float
    query_max_lon: float
    query_max_lat: float

    query_start: int
    query_end: int

    coarse_query_time_s: float
    coarse_query_time_std_s: float

    fine_filtering_time_s: float
    fine_filtering_time_std_s: float

    # Previously specified by the user:
    # Query Time = Coarse Query + Fine Filtering
    query_time_s: float
    query_time_std_s: float

    verification_time_s: float
    verification_time_std_s: float

    vo_size_bytes: int
    vo_size_mb: float

    verification_set_count: int
    candidate_trajectory_count: int
    final_trajectory_count: int


# ============================================================
# Basic utilities
# ============================================================

def _raise_csv_field_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(
                limit
            )
            return
        except OverflowError:
            limit //= 10


def km_to_lat_degree(
    km: float,
) -> float:
    return km / 111.32


def km_to_lon_degree(
    km: float,
    latitude: float,
) -> float:
    denominator = (
        111.32
        *
        math.cos(
            math.radians(
                latitude
            )
        )
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid latitude"
        )

    return km / denominator


def make_square(
    center_lon: float,
    center_lat: float,
    side_km: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    half_km = side_km / 2.0

    half_lat = km_to_lat_degree(
        half_km
    )

    half_lon = km_to_lon_degree(
        half_km,
        center_lat,
    )

    return (
        center_lon - half_lon,
        center_lat - half_lat,
        center_lon + half_lon,
        center_lat + half_lat,
    )


def format_beijing_time(
    timestamp: int,
) -> str:
    try:
        return (
            datetime.fromtimestamp(
                int(timestamp),
                tz=BEIJING_TZ,
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S %z"
            )
        )
    except Exception:
        return str(timestamp)


def center_is_valid(
    index,
    center_lon: float,
    center_lat: float,
) -> bool:
    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        max(SPATIAL_SIDE_KM),
    )

    # User-specified lower bounds
    if min_lon < MIN_ALLOWED_LON:
        return False

    if min_lat < MIN_ALLOWED_LAT:
        return False

    # Actual road-network MBR
    skeleton = index.skeleton
    root = skeleton.root_index

    if min_lon < skeleton.min_lon[root]:
        return False

    if min_lat < skeleton.min_lat[root]:
        return False

    if max_lon > skeleton.max_lon[root]:
        return False

    if max_lat > skeleton.max_lat[root]:
        return False

    return True


def clamp_center_to_valid_region(
    index,
    lon: float,
    lat: float,
) -> tuple[
    float,
    float,
] | None:
    """
    Slightly move the candidate center back into a valid region that can contain a 5km query box.

    Used only when generating candidate centers from road midpoints near the boundary.
    """
    skeleton = index.skeleton
    root = skeleton.root_index

    max_side = max(
        SPATIAL_SIDE_KM
    )

    half_km = max_side / 2.0

    half_lat = km_to_lat_degree(
        half_km
    )

    lat_lower = max(
        MIN_ALLOWED_LAT
        + half_lat,

        skeleton.min_lat[root]
        + half_lat,
    )

    lat_upper = (
        skeleton.max_lat[root]
        - half_lat
    )

    if lat_lower > lat_upper:
        return None

    center_lat = min(
        max(
            lat,
            lat_lower,
        ),
        lat_upper,
    )

    half_lon = km_to_lon_degree(
        half_km,
        center_lat,
    )

    lon_lower = max(
        MIN_ALLOWED_LON
        + half_lon,

        skeleton.min_lon[root]
        + half_lon,
    )

    lon_upper = (
        skeleton.max_lon[root]
        - half_lon
    )

    if lon_lower > lon_upper:
        return None

    center_lon = min(
        max(
            lon,
            lon_lower,
        ),
        lon_upper,
    )

    if not center_is_valid(
        index,
        center_lon,
        center_lat,
    ):
        return None

    return (
        center_lon,
        center_lat,
    )


# ============================================================
# trajectory_id
#
# Keep consistent with the current project:
#
# SHA256(
#     eid_path
#     || trajectory_start
#     || trajectory_end
# )
#
# uint32 big-endian
# Preserve both EID order and duplicates
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
            int(
                trajectory_start
            )
        )
    )

    h.update(
        _U32.pack(
            int(
                trajectory_end
            )
        )
    )

    return h.digest()


# ============================================================
# Beijing Catalog
#
# A separate catalog CSV is no longer required.
#
# Directly from:
#   beijing_trajectories.csv
#
# Build in memory:
#   trajectory_id -> tuple[TrajectorySegment,...]
#
# This stage occurs before the formal benchmark,
# and is not included in Query Time / Fine Filtering Time.
# ============================================================

def build_beijing_catalog(
    index,
):
    _raise_csv_field_limit()

    path = Path(
        TRAJECTORY_CSV
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    catalog = {}

    total_raw = 0
    total_kept = 0
    total_segments = 0
    total_stationary = 0

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
                "beijing_trajectories.csv "
                "is missing points_data"
            )

        for row_no, row in enumerate(
            reader,
            2,
        ):
            total_raw += 1

            points = json.loads(
                row["points_data"]
            )

            if (
                not isinstance(
                    points,
                    list,
                )
                or
                len(points) < 2
            ):
                continue

            parser_input = {
                "points": points,
                "row_no": row_no,
            }

            (
                eids,
                starts,
                ends,
            ) = parse_beijing_trajectory(
                parser_input,
                index.road,
            )

            if not eids:
                continue

            directed_segments = []

            for i in range(
                len(points) - 1
            ):
                p1 = points[i]
                p2 = points[i + 1]

                u = int(
                    p1[0]
                )

                v = int(
                    p2[0]
                )

                start = int(
                    p1[3]
                )

                end = int(
                    p2[3]
                )

                if u == v:
                    total_stationary += 1
                    continue

                directed_segments.append(
                    TrajectorySegment(
                        start_node=u,
                        start_time=start,
                        end_node=v,
                        end_time=end,
                    )
                )

            if (
                len(directed_segments)
                !=
                len(eids)
            ):
                raise RuntimeError(
                    "The number of directed segments in the Beijing Catalog "
                    "does not match the parser EID count: "
                    f"{len(directed_segments)} "
                    f"!= {len(eids)}"
                )

            trajectory_start = int(
                starts[0]
            )

            trajectory_end = int(
                ends[-1]
            )

            trajectory_id = (
                compute_trajectory_id(
                    eid_path=[
                        int(x)
                        for x in eids
                    ],
                    trajectory_start=
                    trajectory_start,
                    trajectory_end=
                    trajectory_end,
                )
            )

            if trajectory_id in catalog:
                raise ValueError(
                    "Duplicate trajectory_id detected: "
                    +
                    trajectory_id.hex()
                )

            catalog[
                trajectory_id
            ] = tuple(
                directed_segments
            )

            total_kept += 1
            total_segments += (
                len(
                    directed_segments
                )
            )

            if (
                total_raw % 1000
                ==
                0
            ):
                print(
                    "Catalog parsing progress:",
                    total_raw,
                )

    if not catalog:
        raise RuntimeError(
            "Beijing Catalog is empty"
        )

    print()
    print(
        "Beijing Catalog build completed"
    )

    print(
        "Number of raw trajectories:",
        total_raw,
    )

    print(
        "Number of valid trajectories:",
        total_kept,
    )

    print(
        "Number of directed road segments:",
        total_segments,
    )

    print(
        "Skipped same-node stationary segments:",
        total_stationary,
    )

    return catalog


def get_catalog_time_range(
    catalog,
) -> tuple[
    int,
    int,
]:
    min_time = None
    max_time = None

    for segments in catalog.values():
        for seg in segments:
            start = int(
                seg.start_time
            )

            end = int(
                seg.end_time
            )

            if (
                min_time is None
                or
                start < min_time
            ):
                min_time = start

            if (
                max_time is None
                or
                end > max_time
            ):
                max_time = end

    if (
        min_time is None
        or
        max_time is None
    ):
        raise RuntimeError(
            "trajectory catalog is empty"
        )

    return (
        min_time,
        max_time,
    )


# ============================================================
# Composite VO / Verification
# ============================================================

def build_vo(
    index,
    query_min_lon: float,
    query_min_lat: float,
    query_max_lon: float,
    query_max_lat: float,
    query_start: int,
    query_end: int,
):
    return CompositeVOBuilder(
        road=index.road,
        store=index.store,
        mmr_index=index.mmr_index,
        skeleton=index.skeleton,
        l1_index=index.l1_index,
        l0_index=index.l0_index,

        query_min_lon=query_min_lon,
        query_min_lat=query_min_lat,
        query_max_lon=query_max_lon,
        query_max_lat=query_max_lat,

        query_start=query_start,
        query_end=query_end,
    ).build()


def verify_vo(
    index,
    vo,
    query_min_lon: float,
    query_min_lat: float,
    query_max_lon: float,
    query_max_lat: float,
    query_start: int,
    query_end: int,
):
    return verify_composite_vo(
        tokens=vo.tokens,
        verification_set=
        vo.verification_set,

        road=index.road,
        trusted_root=
        index.l0_index.root_hash,

        query_min_lon=
        query_min_lon,
        query_min_lat=
        query_min_lat,
        query_max_lon=
        query_max_lon,
        query_max_lat=
        query_max_lat,

        query_start=query_start,
        query_end=query_end,
    )


def probe_real_count(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    query_start: int,
    probe_seconds: int = 60,
) -> tuple[
    int,
    int,
]:
    """
    Actually executes:
        Composite VO
        Verification
        Fine Filtering

    However, the entire Probe stage occurs before the formal Benchmark,
    and is not included in any formal timing.
    """
    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        1.0,
    )

    query_end = (
        query_start
        +
        probe_seconds
    )

    vo = build_vo(
        index,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        query_start,
        query_end,
    )

    report = verify_vo(
        index,
        vo,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        query_start,
        query_end,
    )

    if not report.success:
        raise RuntimeError(
            "Automatic region-selection Probe verification failed"
        )

    final_ids = (
        fine_filter_candidates(
            candidate_ids=
            report.candidate_ids,

            catalog=catalog,
            road=index.road,

            query_min_lon=
            min_lon,
            query_min_lat=
            min_lat,
            query_max_lon=
            max_lon,
            query_max_lat=
            max_lat,

            query_start=
            query_start,
            query_end=
            query_end,
        )
    )

    return (
        len(final_ids),
        len(
            report.candidate_ids
        ),
    )


# ============================================================
# Spatial-density selection
# ============================================================

def spatial_entry_weight(
    index,
    center_lon: float,
    center_lat: float,
) -> int:
    """
    Total number of Entries on roads that actually intersect the 1km×1km box.
    """
    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        1.0,
    )

    road = index.road
    store = index.store

    total = 0

    for eid in range(
        road.max_eid + 1
    ):
        if not road.edge_exists(
            eid
        ):
            continue

        count = int(
            store.entry_counts[eid]
        )

        if count <= 0:
            continue

        if edge_intersects_query(
            eid,
            road,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        ):
            total += count

    return total


def generate_spatial_shortlist(
    index,
):
    """
    Beijing covers a large area, so no exhaustive 0.25km grid scan is performed over the entire city.

    Step 1:
      Aggregate road midpoints with Entries on an approximately 0.5km grid,
      where weight = number of road Entries.

    Step 2:
      Select the top 50 grid centers.

    Step 3:
      Call spatial_entry_weight() again for these 50 centers
      to count Entries on roads that actually intersect the 1km spatial box.

    Return candidate centers ranked by exact Entry weight.
    """
    road = index.road
    store = index.store

    skeleton = index.skeleton
    root = skeleton.root_index

    ref_lat = (
        max(
            MIN_ALLOWED_LAT,
            skeleton.min_lat[root],
        )
        +
        skeleton.max_lat[root]
    ) / 2.0

    cos_ref = math.cos(
        math.radians(
            ref_lat
        )
    )

    if cos_ref <= 0:
        raise RuntimeError(
            "Invalid Beijing reference latitude"
        )

    buckets = defaultdict(
        lambda: [
            0,
            0.0,
            0.0,
        ]
    )

    for eid in range(
        road.max_eid + 1
    ):
        if not road.edge_exists(
            eid
        ):
            continue

        count = int(
            store.entry_counts[eid]
        )

        if count <= 0:
            continue

        u, v = (
            road.get_edge_nodes(
                eid
            )
        )

        x1 = float(
            road.node_lon[u]
        )

        y1 = float(
            road.node_lat[u]
        )

        x2 = float(
            road.node_lon[v]
        )

        y2 = float(
            road.node_lat[v]
        )

        mid_lon = (
            x1 + x2
        ) / 2.0

        mid_lat = (
            y1 + y2
        ) / 2.0

        # Roads outside the user-specified lower bounds are excluded from region selection.
        if mid_lon < MIN_ALLOWED_LON:
            continue

        if mid_lat < MIN_ALLOWED_LAT:
            continue

        adjusted = (
            clamp_center_to_valid_region(
                index,
                mid_lon,
                mid_lat,
            )
        )

        if adjusted is None:
            continue

        center_lon, center_lat = (
            adjusted
        )

        x_km = (
            center_lon
            *
            111.32
            *
            cos_ref
        )

        y_km = (
            center_lat
            *
            111.32
        )

        bx = int(
            math.floor(
                x_km
                /
                SPATIAL_GRID_KM
            )
        )

        by = int(
            math.floor(
                y_km
                /
                SPATIAL_GRID_KM
            )
        )

        bucket = buckets[
            (
                bx,
                by,
            )
        ]

        bucket[0] += count
        bucket[1] += (
            center_lon
            *
            count
        )
        bucket[2] += (
            center_lat
            *
            count
        )

    if not buckets:
        raise RuntimeError(
            "After applying the user-specified longitude/latitude lower bounds, "
            "no roads with Entries were found"
        )

    approx_ranked = []

    for (
        _bucket_key,
        values,
    ) in buckets.items():

        weight = int(
            values[0]
        )

        if weight <= 0:
            continue

        center_lon = (
            values[1]
            /
            weight
        )

        center_lat = (
            values[2]
            /
            weight
        )

        adjusted = (
            clamp_center_to_valid_region(
                index,
                center_lon,
                center_lat,
            )
        )

        if adjusted is None:
            continue

        center_lon, center_lat = (
            adjusted
        )

        approx_ranked.append(
            (
                weight,
                center_lon,
                center_lat,
            )
        )

    approx_ranked.sort(
        reverse=True
    )

    approx_ranked = (
        approx_ranked[
            :SPATIAL_BUCKET_SHORTLIST
        ]
    )

    print()
    print(
        "===== Spatial Density Preselection ====="
    )

    print(
        "Spatial constraints:",
        f"query_min_lon >= {MIN_ALLOWED_LON}",
        f"query_min_lat >= {MIN_ALLOWED_LAT}",
    )

    print(
        "Road Entry grid candidates:",
        len(
            approx_ranked
        ),
    )

    exact_ranked = []

    for (
        approx_weight,
        center_lon,
        center_lat,
    ) in approx_ranked:

        exact_weight = (
            spatial_entry_weight(
                index,
                center_lon,
                center_lat,
            )
        )

        exact_ranked.append(
            (
                exact_weight,
                approx_weight,
                center_lon,
                center_lat,
            )
        )

    exact_ranked.sort(
        reverse=True
    )

    if not exact_ranked:
        raise RuntimeError(
            "No valid spatial center found"
        )

    for i, (
        exact_weight,
        approx_weight,
        center_lon,
        center_lat,
    ) in enumerate(
        exact_ranked[:10],
        1,
    ):
        print(
            f"{i:02d}  "
            f"center="
            f"({center_lon:.6f}, "
            f"{center_lat:.6f})  "
            f"exact_entry_weight="
            f"{exact_weight}  "
            f"grid_weight="
            f"{approx_weight}"
        )

    return [
        (
            center_lon,
            center_lat,
            exact_weight,
        )
        for (
            exact_weight,
            _approx_weight,
            center_lon,
            center_lat,
        )
        in exact_ranked
    ]


# ============================================================
# Temporal-density preselection
# ============================================================

def build_dense_time_shortlist(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    catalog_min_time: int,
    catalog_max_time: int,
):
    """
    Within the current 1km spatial box, count which 1-minute start times
    are covered by the largest number of distinct trajectories.

    Note:
    This is used only for high-density temporal preselection, not as the final result.
    The final comparison still uses the actual Composite VO + Fine Filtering Probe.
    """
    latest_start = (
        catalog_max_time
        -
        MAX_EXPERIMENT_WINDOW_SECONDS
    )

    if latest_start < catalog_min_time:
        raise RuntimeError(
            "The data time span is less than 24h"
        )

    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        1.0,
    )

    road = index.road

    bucket_counts = defaultdict(
        int
    )

    spatial_trajectory_count = 0

    for segments in catalog.values():

        trajectory_buckets = set()

        has_spatial_segment = False

        for seg in segments:
            u = int(
                seg.start_node
            )

            v = int(
                seg.end_node
            )

            x1 = float(
                road.node_lon[u]
            )

            y1 = float(
                road.node_lat[u]
            )

            x2 = float(
                road.node_lon[v]
            )

            y2 = float(
                road.node_lat[v]
            )

            if not segment_intersects_rectangle(
                x1,
                y1,
                x2,
                y2,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            ):
                continue

            has_spatial_segment = True

            start = int(
                seg.start_time
            )

            end = int(
                seg.end_time
            )

            # A 60-second query window [b,b+60]
            # intersects [start,end] when
            # b ranges from start-60 to end.
            bucket_start = (
                (
                    start - 60
                )
                //
                60
            ) * 60

            while bucket_start <= end:

                if (
                    bucket_start
                    >=
                    catalog_min_time
                    and
                    bucket_start
                    <=
                    latest_start
                ):
                    trajectory_buckets.add(
                        bucket_start
                    )

                bucket_start += 60

        if has_spatial_segment:
            spatial_trajectory_count += 1

        for bucket_start in (
            trajectory_buckets
        ):
            bucket_counts[
                bucket_start
            ] += 1

    if not bucket_counts:
        raise RuntimeError(
            "No trajectories were found around the current 1km center "
            "that can be used for temporal selection"
        )

    ranked = sorted(
        bucket_counts.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
        reverse=True,
    )

    ranked = ranked[
        :TIME_BUCKET_SHORTLIST
    ]

    print()
    print(
        "===== Dense Time Preselection ====="
    )

    print(
        "Trajectories covered by the current 1km spatial box:",
        spatial_trajectory_count,
    )

    print(
        "Catalog min:",
        catalog_min_time,
        format_beijing_time(
            catalog_min_time
        ),
    )

    print(
        "Catalog max:",
        catalog_max_time,
        format_beijing_time(
            catalog_max_time
        ),
    )

    print(
        "Top 1-minute time buckets:"
    )

    for i, (
        bucket_start,
        count,
    ) in enumerate(
        ranked[:10],
        1,
    ):
        print(
            f"{i:02d}  "
            f"start={bucket_start}  "
            f"Beijing="
            f"{format_beijing_time(bucket_start)}  "
            f"spatial_traj_bucket_count="
            f"{count}"
        )

    return [
        bucket_start
        for (
            bucket_start,
            _count,
        )
        in ranked
    ]


def choose_dense_time(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    catalog_min_time: int,
    catalog_max_time: int,
):
    latest_start = (
        catalog_max_time
        -
        MAX_EXPERIMENT_WINDOW_SECONDS
    )

    shortlist = (
        build_dense_time_shortlist(
            index,
            catalog,
            center_lon,
            center_lat,
            catalog_min_time,
            catalog_max_time,
        )
    )

    candidate_starts = set()

    for bucket_start in shortlist:
        for offset in (
            TIME_PROBE_OFFSETS_SECONDS
        ):
            start = (
                bucket_start
                +
                offset
            )

            if (
                start
                <
                catalog_min_time
            ):
                continue

            if start > latest_start:
                continue

            candidate_starts.add(
                start
            )

    if not candidate_starts:
        raise RuntimeError(
            "High-density temporal candidate set is empty"
        )

    print()
    print(
        "===== Dense Time Real Probe ====="
    )

    print(
        "Number of actual Probe candidate start times:",
        len(
            candidate_starts
        ),
    )

    best = None

    for start in sorted(
        candidate_starts
    ):
        (
            final_count,
            candidate_count,
        ) = probe_real_count(
            index,
            catalog,
            center_lon,
            center_lat,
            start,
            60,
        )

        score = (
            final_count,
            candidate_count,
        )

        if (
            best is None
            or
            score > best[0]
        ):
            best = (
                score,
                start,
                final_count,
                candidate_count,
            )

            print(
                "New best:",
                f"start={start}",
                f"Beijing="
                f"{format_beijing_time(start)}",
                f"final_1min="
                f"{final_count}",
                f"candidate_1min="
                f"{candidate_count}",
            )

    if best is None:
        raise RuntimeError(
            "High-density temporal selection failed"
        )

    (
        _score,
        coarse_best_start,
        _final_count,
        _candidate_count,
    ) = best

    # --------------------------------------------------------
    # Refine within ±30min around the best point at 1-minute intervals.
    # --------------------------------------------------------

    fine_start = max(
        catalog_min_time,
        coarse_best_start
        -
        TIME_FINE_RADIUS_SECONDS,
    )

    fine_end = min(
        latest_start,
        coarse_best_start
        +
        TIME_FINE_RADIUS_SECONDS,
    )

    current = fine_start

    while current <= fine_end:
        (
            final_count,
            candidate_count,
        ) = probe_real_count(
            index,
            catalog,
            center_lon,
            center_lat,
            current,
            60,
        )

        score = (
            final_count,
            candidate_count,
        )

        if score > best[0]:
            best = (
                score,
                current,
                final_count,
                candidate_count,
            )

        current += (
            TIME_FINE_STEP_SECONDS
        )

    (
        _,
        best_start,
        final_count,
        candidate_count,
    ) = best

    print()
    print(
        "Selected time:",
        best_start,
    )

    print(
        "Selected Beijing time:",
        format_beijing_time(
            best_start
        ),
    )

    print(
        "1min Final Count:",
        final_count,
    )

    print(
        "1min Candidate Count:",
        candidate_count,
    )

    return (
        best_start,
        final_count,
        candidate_count,
    )


def refine_dense_center(
    index,
    catalog,
    spatial_shortlist,
    selected_start: int,
):
    print()
    print(
        "===== Spatial Center Refinement ====="
    )

    best = None

    candidates = (
        spatial_shortlist[
            :CENTER_REFINE_COUNT
        ]
    )

    for (
        center_lon,
        center_lat,
        entry_weight,
    ) in candidates:

        (
            final_count,
            candidate_count,
        ) = probe_real_count(
            index,
            catalog,
            center_lon,
            center_lat,
            selected_start,
            60,
        )

        score = (
            final_count,
            candidate_count,
            entry_weight,
        )

        if (
            best is None
            or
            score > best[0]
        ):
            best = (
                score,
                center_lon,
                center_lat,
                final_count,
                candidate_count,
                entry_weight,
            )

    if best is None:
        raise RuntimeError(
            "Spatial-center refinement failed"
        )

    (
        _,
        center_lon,
        center_lat,
        final_count,
        candidate_count,
        entry_weight,
    ) = best

    print(
        "Refined center:",
        f"({center_lon:.10f}, "
        f"{center_lat:.10f})",
    )

    print(
        "1min Final Count:",
        final_count,
    )

    print(
        "1min Candidate Count:",
        candidate_count,
    )

    print(
        "Spatial Entry Weight:",
        entry_weight,
    )

    return (
        center_lon,
        center_lat,
        final_count,
        candidate_count,
        entry_weight,
    )

#
def select_experiment_setup(
    index,
    catalog,
):
    (
        catalog_min_time,
        catalog_max_time,
    ) = get_catalog_time_range(
        catalog
    )

    spatial_shortlist = (
        generate_spatial_shortlist(
            index
        )
    )
    #  这个也要注释
    lon = 116.335# Replace with your center longitude
    lat = 39.985 # Replace with your center latitude
    weight = spatial_entry_weight(index, lon, lat)
    spatial_shortlist = [(lon, lat, weight)]

    (
        initial_lon,
        initial_lat,
        initial_weight,
    ) = spatial_shortlist[0]

    print()
    print(
        "Initial dense center:",
        f"({initial_lon:.10f}, "
        f"{initial_lat:.10f})",
    )

    print(
        "Initial entry weight:",
        initial_weight,
    )

    (
        selected_start,
        _,
        _,
    ) = choose_dense_time(
        index,
        catalog,
        initial_lon,
        initial_lat,
        catalog_min_time,
        catalog_max_time,
    )

    (
        center_lon,
        center_lat,
        _,
        _,
        entry_weight,
    ) = refine_dense_center(
        index,
        catalog,
        spatial_shortlist,
        selected_start,
    )

    # After the center changes, perform global high-density temporal selection again.
    (
        selected_start,
        final_count,
        candidate_count,
    ) = choose_dense_time(
        index,
        catalog,
        center_lon,
        center_lat,
        catalog_min_time,
        catalog_max_time,
    )

    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        max(
            SPATIAL_SIDE_KM
        ),
    )

    print()
    print(
        "========================================"
    )

    print(
        "===== Selected Experiment Setup ====="
    )

    print(
        f"center_lon = "
        f"{center_lon:.10f}"
    )

    print(
        f"center_lat = "
        f"{center_lat:.10f}"
    )

    print(
        "BASE_QUERY_START =",
        selected_start,
    )

    print(
        "BASE_QUERY_START Beijing =",
        format_beijing_time(
            selected_start
        ),
    )

    print(
        "Selected 1min Final Count =",
        final_count,
    )

    print(
        "Selected 1min Candidate Count =",
        candidate_count,
    )

    print(
        "Spatial Entry Weight =",
        entry_weight,
    )

    print()
    print(
        "Maximum 5km×5km query box:"
    )

    print(
        f"MIN_LON = "
        f"{min_lon:.10f}"
    )

    print(
        f"MIN_LAT = "
        f"{min_lat:.10f}"
    )

    print(
        f"MAX_LON = "
        f"{max_lon:.10f}"
    )

    print(
        f"MAX_LAT = "
        f"{max_lat:.10f}"
    )

    if min_lon < MIN_ALLOWED_LON:
        raise RuntimeError(
            "Automatically selected result violates the longitude lower bound"
        )

    if min_lat < MIN_ALLOWED_LAT:
        raise RuntimeError(
            "Automatically selected result violates the latitude lower bound"
        )

    return (
        center_lon,
        center_lat,
        selected_start,
        final_count,
        candidate_count,
        entry_weight,
        catalog_min_time,
        catalog_max_time,
    )
# def select_experiment_setup(
#     index,
#     catalog,
# ):
#     # ========================================================
#     # Fixed spatial center
#     # ========================================================
#
#     center_lon = 116.335
#     center_lat = 39.985
#
#     entry_weight = spatial_entry_weight(
#         index,
#         center_lon,
#         center_lat,
#     )
#
#     # ========================================================
#     # Full catalog time range
#     # ========================================================
#
#     (
#         catalog_min_time,
#         catalog_max_time,
#     ) = get_catalog_time_range(
#         catalog
#     )
#
#     selected_start = catalog_min_time
#
#     final_count = 0
#     candidate_count = 0
#
#     (
#         min_lon,
#         min_lat,
#         max_lon,
#         max_lat,
#     ) = make_square(
#         center_lon,
#         center_lat,
#         max(SPATIAL_SIDE_KM),
#     )
#
#     print()
#     print("========================================")
#     print("===== Selected Experiment Setup =====")
#
#     print("center_lon =", center_lon)
#     print("center_lat =", center_lat)
#
#     print(
#         "Catalog min =",
#         catalog_min_time,
#         format_beijing_time(catalog_min_time),
#     )
#
#     print(
#         "Catalog max =",
#         catalog_max_time,
#         format_beijing_time(catalog_max_time),
#     )
#
#     print(
#         "Catalog span days =",
#         (catalog_max_time - catalog_min_time)
#         / (24 * 60 * 60),
#     )
#
#     print(
#         "BASE_QUERY_START =",
#         selected_start,
#     )
#
#     print(
#         "Spatial Entry Weight =",
#         entry_weight,
#     )
#
#     print(
#         "Query box:",
#         min_lon,
#         min_lat,
#         max_lon,
#         max_lat,
#     )
#
#     return (
#         center_lon,
#         center_lat,
#         selected_start,
#         final_count,
#         candidate_count,
#         entry_weight,
#         catalog_min_time,
#         catalog_max_time,
#     )

# ============================================================
# Formal experiment
# ============================================================

def measure_vo_size_bytes(
    vo,
) -> int:
    return int(
        save_composite_vo(
            vo=vo,
            file_path=str(
                TEMP_VO_FILE
            ),
        )
    )


def run_combination(
    index,
    catalog,
    base_query_start: int,
    center_lon: float,
    center_lat: float,
    spatial_side_km: float,
    temporal_label: str,
    temporal_seconds: int,
) -> ResultRow:

    (
        query_min_lon,
        query_min_lat,
        query_max_lon,
        query_max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        spatial_side_km,
    )

    if (
        query_min_lon
        <
        MIN_ALLOWED_LON
    ):
        raise RuntimeError(
            f"{spatial_side_km}km query box"
            " crosses the longitude lower bound"
        )

    if (
        query_min_lat
        <
        MIN_ALLOWED_LAT
    ):
        raise RuntimeError(
            f"{spatial_side_km}km query box"
            " crosses the latitude lower bound"
        )

    query_start = (
        base_query_start
    )

    query_end = (
        base_query_start
        +
        temporal_seconds
    )

    # ========================================================
    # Warmup
    # ========================================================

    for _ in range(
        WARMUP_REPEATS
    ):
        warm_vo = build_vo(
            index,
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
            query_start,
            query_end,
        )

        warm_report = verify_vo(
            index,
            warm_vo,
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
            query_start,
            query_end,
        )

        if not warm_report.success:
            raise RuntimeError(
                "Warmup verification failed"
            )

        fine_filter_candidates(
            candidate_ids=
            warm_report.candidate_ids,

            catalog=catalog,
            road=index.road,

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
        )

    # ========================================================
    # Formal repeats
    # ========================================================

    coarse_query_times = []
    fine_filtering_times = []
    query_times = []
    verification_times = []

    last_vo = None
    last_final_ids = None

    for _ in range(
        REPEATS
    ):

        # ----------------------------------------------------
        # Coarse Query Time
        # L0/L1/MMR + VO/VS generation
        # ----------------------------------------------------

        t0 = perf_counter()

        vo = build_vo(
            index,
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
            query_start,
            query_end,
        )

        coarse_query_time = (
            perf_counter()
            -
            t0
        )

        # ----------------------------------------------------
        # Verification Time
        # Independent and not included in Query Time
        # ----------------------------------------------------

        t0 = perf_counter()

        report = verify_vo(
            index,
            vo,
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
            query_start,
            query_end,
        )

        verification_time = (
            perf_counter()
            -
            t0
        )

        if not report.success:
            raise RuntimeError(
                "Verification failed"
            )

        # ----------------------------------------------------
        # Fine Filtering Time
        # Catalog has already been loaded into memory.
        # ----------------------------------------------------

        t0 = perf_counter()

        final_ids = (
            fine_filter_candidates(
                candidate_ids=
                report.candidate_ids,

                catalog=catalog,
                road=index.road,

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
            )
        )

        fine_filtering_time = (
            perf_counter()
            -
            t0
        )

        # Previously specified definition:
        query_time = (
            coarse_query_time
            +
            fine_filtering_time
        )

        coarse_query_times.append(
            coarse_query_time
        )

        fine_filtering_times.append(
            fine_filtering_time
        )

        query_times.append(
            query_time
        )

        verification_times.append(
            verification_time
        )

        last_vo = vo
        last_final_ids = (
            final_ids
        )

    if last_vo is None:
        raise RuntimeError(
            "No VO was generated"
        )

    if last_final_ids is None:
        raise RuntimeError(
            "No Fine Filtering result was generated"
        )

    # VO serialization is not included in Query/Verification timing.
    vo_size_bytes = (
        measure_vo_size_bytes(
            last_vo
        )
    )

    vo_size_mb = (
        vo_size_bytes
        /
        1_000_000.0
    )

    return ResultRow(
        spatial_side_km=
        spatial_side_km,

        temporal_label=
        temporal_label,

        temporal_seconds=
        temporal_seconds,

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

        coarse_query_time_s=
        statistics.mean(
            coarse_query_times
        ),

        coarse_query_time_std_s=(
            statistics.stdev(
                coarse_query_times
            )
            if
            len(
                coarse_query_times
            )
            >
            1
            else
            0.0
        ),

        fine_filtering_time_s=
        statistics.mean(
            fine_filtering_times
        ),

        fine_filtering_time_std_s=(
            statistics.stdev(
                fine_filtering_times
            )
            if
            len(
                fine_filtering_times
            )
            >
            1
            else
            0.0
        ),

        query_time_s=
        statistics.mean(
            query_times
        ),

        query_time_std_s=(
            statistics.stdev(
                query_times
            )
            if
            len(
                query_times
            )
            >
            1
            else
            0.0
        ),

        verification_time_s=
        statistics.mean(
            verification_times
        ),

        verification_time_std_s=(
            statistics.stdev(
                verification_times
            )
            if
            len(
                verification_times
            )
            >
            1
            else
            0.0
        ),

        vo_size_bytes=
        vo_size_bytes,

        vo_size_mb=
        vo_size_mb,

        verification_set_count=
        len(
            last_vo.verification_set
        ),

        candidate_trajectory_count=
        len(
            last_vo.candidate_ids
        ),

        final_trajectory_count=
        len(
            last_final_ids
        ),
    )


# ============================================================
# Save results
# ============================================================

RESULT_FIELDS = [
    "spatial_side_km",
    "temporal_label",
    "temporal_seconds",

    "query_min_lon",
    "query_min_lat",
    "query_max_lon",
    "query_max_lat",

    "query_start",
    "query_end",

    "coarse_query_time_s",
    "coarse_query_time_std_s",

    "fine_filtering_time_s",
    "fine_filtering_time_std_s",

    "query_time_s",
    "query_time_std_s",

    "verification_time_s",
    "verification_time_std_s",

    "vo_size_bytes",
    "vo_size_mb",

    "verification_set_count",
    "candidate_trajectory_count",
    "final_trajectory_count",
]


def write_long_results(
    rows,
):
    with RESULT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
            RESULT_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "spatial_side_km":
                    f"{row.spatial_side_km:.1f}",

                    "temporal_label":
                    row.temporal_label,

                    "temporal_seconds":
                    row.temporal_seconds,

                    "query_min_lon":
                    f"{row.query_min_lon:.10f}",

                    "query_min_lat":
                    f"{row.query_min_lat:.10f}",

                    "query_max_lon":
                    f"{row.query_max_lon:.10f}",

                    "query_max_lat":
                    f"{row.query_max_lat:.10f}",

                    "query_start":
                    row.query_start,

                    "query_end":
                    row.query_end,

                    "coarse_query_time_s":
                    f"{row.coarse_query_time_s:.9f}",

                    "coarse_query_time_std_s":
                    f"{row.coarse_query_time_std_s:.9f}",

                    "fine_filtering_time_s":
                    f"{row.fine_filtering_time_s:.9f}",

                    "fine_filtering_time_std_s":
                    f"{row.fine_filtering_time_std_s:.9f}",

                    "query_time_s":
                    f"{row.query_time_s:.9f}",

                    "query_time_std_s":
                    f"{row.query_time_std_s:.9f}",

                    "verification_time_s":
                    f"{row.verification_time_s:.9f}",

                    "verification_time_std_s":
                    f"{row.verification_time_std_s:.9f}",

                    "vo_size_bytes":
                    row.vo_size_bytes,

                    "vo_size_mb":
                    f"{row.vo_size_mb:.9f}",

                    "verification_set_count":
                    row.verification_set_count,

                    "candidate_trajectory_count":
                    row.candidate_trajectory_count,

                    "final_trajectory_count":
                    row.final_trajectory_count,
                }
            )


def write_matrix(
    file_path: Path,
    rows,
    value_getter,
):
    lookup = {
        (
            row.spatial_side_km,
            row.temporal_label,
        ):
        value_getter(
            row
        )

        for row in rows
    }

    with file_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "spatial_side_km",
                *[
                    label
                    for (
                        label,
                        _
                    )
                    in
                    TEMPORAL_RANGES
                ],
            ]
        )

        for side_km in (
            SPATIAL_SIDE_KM
        ):
            writer.writerow(
                [
                    side_km,
                    *[
                        lookup[
                            (
                                side_km,
                                label,
                            )
                        ]
                        for (
                            label,
                            _
                        )
                        in
                        TEMPORAL_RANGES
                    ],
                ]
            )


def save_selected_setup(
    index,
    center_lon,
    center_lat,
    selected_start,
    selected_final_count,
    selected_candidate_count,
    entry_weight,
    catalog_min_time,
    catalog_max_time,
):
    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = make_square(
        center_lon,
        center_lat,
        max(
            SPATIAL_SIDE_KM
        ),
    )

    skeleton = index.skeleton
    root = skeleton.root_index

    with DENSE_REGION_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "constraint_min_lon",
                "constraint_min_lat",
                "road_mbr_min_lon",
                "road_mbr_min_lat",
                "road_mbr_max_lon",
                "road_mbr_max_lat",
                "center_lon",
                "center_lat",
                "base_query_start",
                "base_query_start_beijing",
                "catalog_min_time",
                "catalog_max_time",
                "selected_1min_final_count",
                "selected_1min_candidate_count",
                "spatial_entry_weight",
                "max_query_side_km",
                "max_query_min_lon",
                "max_query_min_lat",
                "max_query_max_lon",
                "max_query_max_lat",
                "formal_repeats",
                "warmup_repeats",
            ]
        )

        writer.writerow(
            [
                MIN_ALLOWED_LON,
                MIN_ALLOWED_LAT,

                skeleton.min_lon[
                    root
                ],

                skeleton.min_lat[
                    root
                ],

                skeleton.max_lon[
                    root
                ],

                skeleton.max_lat[
                    root
                ],

                f"{center_lon:.10f}",
                f"{center_lat:.10f}",

                selected_start,

                format_beijing_time(
                    selected_start
                ),

                catalog_min_time,
                catalog_max_time,

                selected_final_count,
                selected_candidate_count,
                entry_weight,

                max(
                    SPATIAL_SIDE_KM
                ),

                f"{min_lon:.10f}",
                f"{min_lat:.10f}",
                f"{max_lon:.10f}",
                f"{max_lat:.10f}",

                REPEATS,
                WARMUP_REPEATS,
            ]
        )


# ============================================================
# main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== Beijing 5×6 Cross Range Benchmark "
        f"[{VERSION}] ====="
    )

    print()
    print(
        "Spatial ranges:",
        SPATIAL_SIDE_KM,
    )

    print(
        "Temporal ranges:",
        [
            label
            for (
                label,
                _
            )
            in
            TEMPORAL_RANGES
        ],
    )

    print(
        "Total number of combinations:",
        len(
            SPATIAL_SIDE_KM
        )
        *
        len(
            TEMPORAL_RANGES
        ),
    )

    print(
        "Formal runs per group:",
        REPEATS,
    )

    print(
        "Warmup runs per group:",
        WARMUP_REPEATS,
    )

    print()
    print(
        "Beijing spatial lower bounds:"
    )

    print(
        "longitude >=",
        MIN_ALLOWED_LON,
    )

    print(
        "latitude >=",
        MIN_ALLOWED_LAT,
    )

    print()
    print(
        "Query Time definition:"
    )

    print(
        "Query Time = "
        "Coarse Query Time "
        "+ Fine Filtering Time"
    )

    print(
        "Verification Time is independent and "
        "not included in Query Time"
    )

    # ========================================================
    # Load index ONCE
    # ========================================================

    print()
    print(
        "===== Load Persisted Beijing Index ====="
    )

    if not Path(
        INDEX_FILE
    ).exists():
        raise FileNotFoundError(
            "Beijing index not found: "
            f"{INDEX_FILE}\n"
            "Please run build_beijing_index.py first"
        )

    t0 = perf_counter()

    index = load_index(
        INDEX_FILE
    )

    index_load_time = (
        perf_counter()
        -
        t0
    )

    print(
        "Index Load Time:",
        f"{index_load_time:.6f} s",
    )

    print(
        "(not included in Query Time)"
    )

    # ========================================================
    # Build catalog in memory ONCE
    # ========================================================

    print()
    print(
        "===== Build Beijing Trajectory Catalog In Memory ====="
    )

    t0 = perf_counter()

    catalog = (
        build_beijing_catalog(
            index
        )
    )

    catalog_build_time = (
        perf_counter()
        -
        t0
    )

    print(
        "Catalog Build Time:",
        f"{catalog_build_time:.6f} s",
    )

    print(
        "(not included in Query Time / Fine Filtering Time)"
    )

    # ========================================================
    # Automatically select a high-density spatial region + high-density time
    # ========================================================

    (
        center_lon,
        center_lat,
        base_query_start,
        selected_final_count,
        selected_candidate_count,
        entry_weight,
        catalog_min_time,
        catalog_max_time,
    ) = select_experiment_setup(
        index,
        catalog,
    )

    save_selected_setup(
        index,
        center_lon,
        center_lat,
        base_query_start,
        selected_final_count,
        selected_candidate_count,
        entry_weight,
        catalog_min_time,
        catalog_max_time,
    )

    # ========================================================
    # 30 formal experiment groups
    # ========================================================

    rows = []

    total = (
        len(
            SPATIAL_SIDE_KM
        )
        *
        len(
            TEMPORAL_RANGES
        )
    )

    current = 0

    print()
    print(
        "========================================"
    )

    print(
        "===== Cross Benchmark ====="
    )

    for side_km in (
        SPATIAL_SIDE_KM
    ):
        for (
            temporal_label,
            temporal_seconds,
        ) in TEMPORAL_RANGES:

            current += 1

            print()
            print(
                f"[{current}/{total}] "
                f"{side_km:.0f}km × "
                f"{side_km:.0f}km, "
                f"{temporal_label}"
            )

            row = run_combination(
                index=index,
                catalog=catalog,

                base_query_start=
                base_query_start,

                center_lon=
                center_lon,

                center_lat=
                center_lat,

                spatial_side_km=
                side_km,

                temporal_label=
                temporal_label,

                temporal_seconds=
                temporal_seconds,
            )

            rows.append(
                row
            )

            print(
                "Coarse Query Time:",
                f"{row.coarse_query_time_s:.9f} s",
            )

            print(
                "Fine Filtering Time:",
                f"{row.fine_filtering_time_s:.9f} s",
            )

            print(
                "Query Time "
                "(Coarse + Filter):",
                f"{row.query_time_s:.9f} s",
            )

            print(
                "Verification Time:",
                f"{row.verification_time_s:.9f} s",
            )

            print(
                "VO Size:",
                f"{row.vo_size_mb:.9f} MB",
            )

            print(
                "Candidate Count:",
                row.candidate_trajectory_count,
            )

            print(
                "Final Result Count:",
                row.final_trajectory_count,
            )

    # ========================================================
    # Save
    # ========================================================

    write_long_results(
        rows
    )

    write_matrix(
        QUERY_TIME_MATRIX_FILE,
        rows,
        lambda r:
        f"{r.query_time_s:.9f}",
    )

    write_matrix(
        COARSE_QUERY_TIME_MATRIX_FILE,
        rows,
        lambda r:
        f"{r.coarse_query_time_s:.9f}",
    )

    write_matrix(
        FINE_FILTERING_TIME_MATRIX_FILE,
        rows,
        lambda r:
        f"{r.fine_filtering_time_s:.9f}",
    )

    write_matrix(
        VERIFICATION_TIME_MATRIX_FILE,
        rows,
        lambda r:
        f"{r.verification_time_s:.9f}",
    )

    write_matrix(
        VO_SIZE_MATRIX_FILE,
        rows,
        lambda r:
        f"{r.vo_size_mb:.9f}",
    )

    write_matrix(
        CANDIDATE_MATRIX_FILE,
        rows,
        lambda r:
        r.candidate_trajectory_count,
    )

    write_matrix(
        FINAL_RESULT_MATRIX_FILE,
        rows,
        lambda r:
        r.final_trajectory_count,
    )

    try:
        if TEMP_VO_FILE.exists():
            TEMP_VO_FILE.unlink()
    except OSError:
        pass

    print()
    print(
        "========================================"
    )

    print(
        "All 30 Beijing experiment groups completed"
    )

    print()
    print(
        "Complete results:",
        RESULT_FILE,
    )

    print(
        "Query Time matrix:",
        QUERY_TIME_MATRIX_FILE,
    )

    print(
        "Coarse Query Time matrix:",
        COARSE_QUERY_TIME_MATRIX_FILE,
    )

    print(
        "Fine Filtering Time matrix:",
        FINE_FILTERING_TIME_MATRIX_FILE,
    )

    print(
        "Verification Time matrix:",
        VERIFICATION_TIME_MATRIX_FILE,
    )

    print(
        "VO Size matrix:",
        VO_SIZE_MATRIX_FILE,
    )

    print(
        "Candidate Count matrix:",
        CANDIDATE_MATRIX_FILE,
    )

    print(
        "Final Result Count matrix:",
        FINAL_RESULT_MATRIX_FILE,
    )

    print(
        "Automatically selected experiment region and time:",
        DENSE_REGION_FILE,
    )


if __name__ == "__main__":
    main()

