from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from composite_vo import CompositeVOBuilder
from index_io import load_index
from verifier import verify_composite_vo
from vo_io import save_composite_vo
from trajectory_catalog import (
    load_trajectory_catalog,
    fine_filter_candidates,
)


VERSION = "FILTER_V3"

INDEX_FILE = "index_output/chengdu_trajectory_index.dat"
TRAJECTORY_CSV = "catalog_output/chengdu_trajectory_catalog.csv"

OUTPUT_DIR = Path("benchmark_output_all_time")

RESULT_FILE = OUTPUT_DIR / "chengdu_cross_range_results.csv"
QUERY_TIME_MATRIX_FILE = OUTPUT_DIR / "chengdu_query_time_matrix_s.csv"
COARSE_QUERY_TIME_MATRIX_FILE = OUTPUT_DIR / "chengdu_coarse_query_time_matrix_s.csv"
FINE_FILTERING_TIME_MATRIX_FILE = OUTPUT_DIR / "chengdu_fine_filtering_time_matrix_s.csv"
VERIFICATION_TIME_MATRIX_FILE = OUTPUT_DIR / "chengdu_verification_time_matrix_s.csv"
VO_SIZE_MATRIX_FILE = OUTPUT_DIR / "chengdu_vo_size_matrix_mb.csv"
CANDIDATE_MATRIX_FILE = OUTPUT_DIR / "chengdu_candidate_count_matrix.csv"
FINAL_RESULT_MATRIX_FILE = OUTPUT_DIR / "chengdu_final_result_count_matrix.csv"
DENSE_REGION_FILE = OUTPUT_DIR / "chengdu_selected_dense_region.csv"
TEMP_VO_FILE = OUTPUT_DIR / "_temp_vo_size.dat"

MIN_QUERY_LAT = 30.65

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
    # None uses the loaded index's full time range, independent of BASE_QUERY_START.
    # ("all", None),
]

BASE_QUERY_START = 1541260553

REPEATS = 30
WARMUP_REPEATS = 1

DENSE_GRID_KM = 1.0
DENSE_TOP_K = 30
DENSE_PROBE_SECONDS = 24 * 60 * 60


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

    # User requirement:
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


def km_to_lat_degree(km: float) -> float:
    return km / 111.32


def km_to_lon_degree(km: float, latitude: float) -> float:
    denominator = 111.32 * math.cos(math.radians(latitude))
    if denominator <= 0:
        raise ValueError("Invalid latitude")
    return km / denominator


def make_square(
    center_lon: float,
    center_lat: float,
    side_km: float,
) -> tuple[float, float, float, float]:
    half_km = side_km / 2.0
    half_lat = km_to_lat_degree(half_km)
    half_lon = km_to_lon_degree(half_km, center_lat)

    return (
        center_lon - half_lon,
        center_lat - half_lat,
        center_lon + half_lon,
        center_lat + half_lat,
    )


def center_is_valid(
    center_lon: float,
    center_lat: float,
    skeleton,
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

    if min_lat < MIN_QUERY_LAT:
        return False

    root = skeleton.root_index

    return (
        min_lon >= skeleton.min_lon[root]
        and min_lat >= skeleton.min_lat[root]
        and max_lon <= skeleton.max_lon[root]
        and max_lat <= skeleton.max_lat[root]
    )


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
        verification_set=vo.verification_set,
        road=index.road,
        trusted_root=index.l0_index.root_hash,

        query_min_lon=query_min_lon,
        query_min_lat=query_min_lat,
        query_max_lon=query_max_lon,
        query_max_lat=query_max_lat,

        query_start=query_start,
        query_end=query_end,
    )


def find_dense_grid_candidates(index):
    road = index.road
    store = index.store
    skeleton = index.skeleton

    root = skeleton.root_index
    reference_lat = (
        skeleton.min_lat[root] + skeleton.max_lat[root]
    ) / 2.0

    cell_lat_degree = km_to_lat_degree(DENSE_GRID_KM)
    cell_lon_degree = km_to_lon_degree(
        DENSE_GRID_KM,
        reference_lat,
    )

    cells = {}

    for eid in range(road.max_eid + 1):
        if not road.edge_exists(eid):
            continue

        weight = int(store.entry_counts[eid])

        if weight <= 0:
            continue

        u, v = road.get_edge_nodes(eid)

        edge_center_lon = (
            road.node_lon[u] + road.node_lon[v]
        ) / 2.0

        edge_center_lat = (
            road.node_lat[u] + road.node_lat[v]
        ) / 2.0

        gx = math.floor(edge_center_lon / cell_lon_degree)
        gy = math.floor(edge_center_lat / cell_lat_degree)

        key = (gx, gy)

        state = cells.get(key)

        if state is None:
            cells[key] = [
                weight,
                edge_center_lon * weight,
                edge_center_lat * weight,
            ]
        else:
            state[0] += weight
            state[1] += edge_center_lon * weight
            state[2] += edge_center_lat * weight

    if not cells:
        raise RuntimeError("No usable non-empty edges found")

    ranked = sorted(
        cells.values(),
        key=lambda x: x[0],
        reverse=True,
    )

    candidates = []

    for weight, weighted_lon, weighted_lat in ranked:
        center_lon = weighted_lon / weight
        center_lat = weighted_lat / weight

        if not center_is_valid(
            center_lon,
            center_lat,
            skeleton,
        ):
            continue

        candidates.append(
            (
                center_lon,
                center_lat,
                int(weight),
            )
        )

        if len(candidates) >= DENSE_TOP_K:
            break

    if not candidates:
        raise RuntimeError(
            f"No valid high-density region found under the MIN_QUERY_LAT={MIN_QUERY_LAT} constraint"
        )

    return candidates


def choose_dense_center(index):
    candidates = find_dense_grid_candidates(index)

    probe_end = (
        BASE_QUERY_START
        + DENSE_PROBE_SECONDS
    )

    print()
    print("===== Dense Region Selection =====")
    print("Latitude lower bound:", MIN_QUERY_LAT)
    print(
        "Require MIN_LAT of the maximum 5km query box >=",
        MIN_QUERY_LAT,
    )

    best = None

    for i, (
        center_lon,
        center_lat,
        entry_weight,
    ) in enumerate(candidates, 1):

        (
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        ) = make_square(
            center_lon,
            center_lat,
            DENSE_GRID_KM,
        )

        # Probe is not included in the formal Query Time.
        vo = build_vo(
            index,

            min_lon,
            min_lat,
            max_lon,
            max_lat,

            BASE_QUERY_START,
            probe_end,
        )

        candidate_count = len(vo.candidate_ids)

        print(
            f"{i:02d}  "
            f"center=({center_lon:.6f}, {center_lat:.6f})  "
            f"entry_weight={entry_weight}  "
            f"24h_candidates={candidate_count}"
        )

        score = (
            candidate_count,
            entry_weight,
        )

        if best is None or score > best[0]:
            best = (
                score,
                center_lon,
                center_lat,
                candidate_count,
                entry_weight,
            )

    if best is None:
        raise RuntimeError("Failed to select a high-density region")

    (
        _,
        center_lon,
        center_lat,
        candidate_count,
        entry_weight,
    ) = best

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

    print()
    print("Selected center:")
    print(f"lon = {center_lon:.10f}")
    print(f"lat = {center_lat:.10f}")

    print()
    print("Maximum 5km×5km query range:")
    print(f"MIN_LON = {min_lon:.10f}")
    print(f"MIN_LAT = {min_lat:.10f}")
    print(f"MAX_LON = {max_lon:.10f}")
    print(f"MAX_LAT = {max_lat:.10f}")

    print()
    print(
        "1km/24h candidate count =",
        candidate_count,
    )

    return (
        center_lon,
        center_lat,
        candidate_count,
        entry_weight,
    )


def measure_vo_size_bytes(vo) -> int:
    # File serialization occurs outside all timed intervals.
    return int(
        save_composite_vo(
            vo=vo,
            file_path=str(TEMP_VO_FILE),
        )
    )


def resolve_query_time(index, temporal_seconds: int | None) -> tuple[int, int]:
    if temporal_seconds is None:
        root = int(index.l0_index.root_index)
        start = int(index.l0_index.node_min_start[root])
        end = int(index.l0_index.node_max_end[root])
        if start > end:
            raise ValueError("Cannot query the full time range of an empty index")
        return start, end
    return BASE_QUERY_START, BASE_QUERY_START + temporal_seconds


def run_combination(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    spatial_side_km: float,
    temporal_label: str,
    temporal_seconds: int | None,
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

    if query_min_lat < MIN_QUERY_LAT:
        raise RuntimeError(
            f"{spatial_side_km}km query box MIN_LAT="
            f"{query_min_lat} < {MIN_QUERY_LAT}"
        )

    query_start, query_end = resolve_query_time(index, temporal_seconds)
    temporal_seconds = query_end - query_start

    # ========================================================
    # Warmup
    # ========================================================

    for _ in range(WARMUP_REPEATS):
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
            candidate_ids=warm_report.candidate_ids,
            catalog=catalog,
            road=index.road,

            query_min_lon=query_min_lon,
            query_min_lat=query_min_lat,
            query_max_lon=query_max_lon,
            query_max_lat=query_max_lat,

            query_start=query_start,
            query_end=query_end,
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

    for _ in range(REPEATS):

        # ----------------------------------------------------
        # Coarse Query Time
        # L0/L1/MMR + VO/VS generation only
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
            perf_counter() - t0
        )

        # ----------------------------------------------------
        # Verification Time
        # Independent, NOT included in Query Time
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
            perf_counter() - t0
        )

        if not report.success:
            raise RuntimeError(
                "Verification failed"
            )

        # ----------------------------------------------------
        # Fine Filtering Time
        #
        # The catalog has already been loaded into memory.
        # Only the following operations are measured here:
        # trajectory_id dict lookup
        # + directed linear interpolation
        # + exact spatiotemporal filtering
        # ----------------------------------------------------

        t0 = perf_counter()

        final_ids = fine_filter_candidates(
            candidate_ids=report.candidate_ids,
            catalog=catalog,
            road=index.road,

            query_min_lon=query_min_lon,
            query_min_lat=query_min_lat,
            query_max_lon=query_max_lon,
            query_max_lat=query_max_lat,

            query_start=query_start,
            query_end=query_end,
        )

        fine_filtering_time = (
            perf_counter() - t0
        )

        # User-specified Query Time:
        query_time = (
            coarse_query_time
            + fine_filtering_time
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
        last_final_ids = final_ids

    if last_vo is None:
        raise RuntimeError("No VO was generated")

    if last_final_ids is None:
        raise RuntimeError(
            "No fine-filtering result was generated"
        )

    # VO size measurement outside timing.
    vo_size_bytes = measure_vo_size_bytes(
        last_vo
    )

    vo_size_mb = (
        vo_size_bytes
        /
        1_000_000.0
    )

    return ResultRow(
        spatial_side_km=spatial_side_km,
        temporal_label=temporal_label,
        temporal_seconds=temporal_seconds,

        query_min_lon=query_min_lon,
        query_min_lat=query_min_lat,
        query_max_lon=query_max_lon,
        query_max_lat=query_max_lat,

        query_start=query_start,
        query_end=query_end,

        coarse_query_time_s=statistics.mean(
            coarse_query_times
        ),

        coarse_query_time_std_s=(
            statistics.stdev(
                coarse_query_times
            )
            if len(coarse_query_times) > 1
            else 0.0
        ),

        fine_filtering_time_s=statistics.mean(
            fine_filtering_times
        ),

        fine_filtering_time_std_s=(
            statistics.stdev(
                fine_filtering_times
            )
            if len(fine_filtering_times) > 1
            else 0.0
        ),

        query_time_s=statistics.mean(
            query_times
        ),

        query_time_std_s=(
            statistics.stdev(
                query_times
            )
            if len(query_times) > 1
            else 0.0
        ),

        verification_time_s=statistics.mean(
            verification_times
        ),

        verification_time_std_s=(
            statistics.stdev(
                verification_times
            )
            if len(verification_times) > 1
            else 0.0
        ),

        vo_size_bytes=vo_size_bytes,
        vo_size_mb=vo_size_mb,

        verification_set_count=len(
            last_vo.verification_set
        ),

        candidate_trajectory_count=len(
            last_vo.candidate_ids
        ),

        final_trajectory_count=len(
            last_final_ids
        ),
    )


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


def write_long_results(rows):
    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_FIELDS,
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
        ): value_getter(row)
        for row in rows
    }

    with open(
        file_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "spatial_side_km",
                *[
                    label
                    for label, _
                    in TEMPORAL_RANGES
                ],
            ]
        )

        for side_km in SPATIAL_SIDE_KM:
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
                        for label, _
                        in TEMPORAL_RANGES
                    ],
                ]
            )


def save_selected_center(
    center_lon,
    center_lat,
    probe_candidate_count,
    entry_weight,
):
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

    with open(
        DENSE_REGION_FILE,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "center_lon",
                "center_lat",
                "min_allowed_query_lat",
                "max_query_side_km",
                "max_query_min_lon",
                "max_query_min_lat",
                "max_query_max_lon",
                "max_query_max_lat",
                "probe_candidate_count",
                "entry_density_weight",
            ]
        )

        writer.writerow(
            [
                f"{center_lon:.10f}",
                f"{center_lat:.10f}",
                MIN_QUERY_LAT,
                max(SPATIAL_SIDE_KM),
                f"{min_lon:.10f}",
                f"{min_lat:.10f}",
                f"{max_lon:.10f}",
                f"{max_lat:.10f}",
                probe_candidate_count,
                entry_weight,
            ]
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"===== {len(SPATIAL_SIDE_KM)}×{len(TEMPORAL_RANGES)} Cross Range Benchmark [{VERSION}] ====="
    )

    print()
    print("Spatial ranges:", SPATIAL_SIDE_KM)
    print(
        "Temporal ranges:",
        [
            label
            for label, _
            in TEMPORAL_RANGES
        ],
    )
    print(
        "Total combinations:",
        len(SPATIAL_SIDE_KM)
        *
        len(TEMPORAL_RANGES),
    )

    print()
    print(
        "Query Time definition:"
    )
    print(
        "Query Time = Coarse Query Time + Fine Filtering Time"
    )
    print(
        "Verification Time is independent and is not included in Query Time"
    )

    # ========================================================
    # Load index ONCE
    # ========================================================

    print()
    print(
        "===== Load Persisted Index ====="
    )

    t0 = perf_counter()

    index = load_index(
        INDEX_FILE
    )

    index_load_time = (
        perf_counter() - t0
    )

    print(
        "Index Load Time:",
        f"{index_load_time:.6f} s",
    )

    print(
        "(Not included in Query Time)"
    )

    # ========================================================
    # Load catalog ONCE
    # ========================================================

    print()
    print(
        "===== Load Trajectory Catalog ====="
    )

    t0 = perf_counter()

    catalog = load_trajectory_catalog(
        TRAJECTORY_CSV
    )

    catalog_load_time = (
        perf_counter() - t0
    )

    print(
        "Number of catalog trajectories:",
        len(catalog),
    )

    print(
        "Catalog Load Time:",
        f"{catalog_load_time:.6f} s",
    )

    print(
        "(Not included in Query Time / Fine Filtering Time)"
    )


    # ========================================================
    # Dense center
    # ========================================================

    # (
    #     center_lon,
    #     center_lat,
    #     probe_candidate_count,
    #     entry_weight,
    # ) = choose_dense_center(
    #     index
    # )
    #
    # save_selected_center(
    #     center_lon,
    #     center_lat,
    #     probe_candidate_count,
    #     entry_weight,
    # )

    # ========================================================
    # Fixed query center
    # ========================================================

    center_lon = 104.06
    center_lat = 30.665

    probe_candidate_count = 0
    entry_weight = 0

    save_selected_center(
        center_lon,
        center_lat,
        probe_candidate_count,
        entry_weight,
    )

    rows = []

    total = (
        len(SPATIAL_SIDE_KM)
        *
        len(TEMPORAL_RANGES)
    )

    current = 0

    print()
    print(
        "========================================"
    )
    print(
        "===== Cross Benchmark ====="
    )

    for side_km in SPATIAL_SIDE_KM:
        for (
            temporal_label,
            temporal_seconds,
        ) in TEMPORAL_RANGES:

            current += 1

            print()
            print(
                f"[{current}/{total}] "
                f"{side_km:.0f}km × {side_km:.0f}km, "
                f"{temporal_label}"
            )

            row = run_combination(
                index=index,
                catalog=catalog,

                center_lon=center_lon,
                center_lat=center_lat,

                spatial_side_km=side_km,

                temporal_label=temporal_label,
                temporal_seconds=temporal_seconds,
            )

            rows.append(row)

            print(
                "Coarse Query Time:",
                f"{row.coarse_query_time_s:.9f} s",
            )

            print(
                "Fine Filtering Time:",
                f"{row.fine_filtering_time_s:.9f} s",
            )

            print(
                "Query Time (Coarse + Filter):",
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
    # Save results
    # ========================================================

    write_long_results(rows)

    write_matrix(
        QUERY_TIME_MATRIX_FILE,
        rows,
        lambda r: f"{r.query_time_s:.9f}",
    )

    write_matrix(
        COARSE_QUERY_TIME_MATRIX_FILE,
        rows,
        lambda r: f"{r.coarse_query_time_s:.9f}",
    )

    write_matrix(
        FINE_FILTERING_TIME_MATRIX_FILE,
        rows,
        lambda r: f"{r.fine_filtering_time_s:.9f}",
    )

    write_matrix(
        VERIFICATION_TIME_MATRIX_FILE,
        rows,
        lambda r: f"{r.verification_time_s:.9f}",
    )

    write_matrix(
        VO_SIZE_MATRIX_FILE,
        rows,
        lambda r: f"{r.vo_size_mb:.9f}",
    )

    write_matrix(
        CANDIDATE_MATRIX_FILE,
        rows,
        lambda r: r.candidate_trajectory_count,
    )

    write_matrix(
        FINAL_RESULT_MATRIX_FILE,
        rows,
        lambda r: r.final_trajectory_count,
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
        "All 30 experiment groups completed"
    )

    print()
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
        "Final Result Count matrix:",
        FINAL_RESULT_MATRIX_FILE,
    )


if __name__ == "__main__":
    main()
