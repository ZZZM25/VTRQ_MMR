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
from query_utils import edge_intersects_query


VERSION = "XIAN_AUTO_DENSE_V2_FIXED"

INDEX_FILE = "index_output/xian_trajectory_index.dat"
TRAJECTORY_CSV = "catalog_output/xian_trajectory_catalog.csv"

OUTPUT_DIR = Path("benchmark_output")

RESULT_FILE = OUTPUT_DIR / "xian_cross_range_results.csv"
QUERY_TIME_MATRIX_FILE = OUTPUT_DIR / "xian_query_time_matrix_s.csv"
COARSE_QUERY_TIME_MATRIX_FILE = OUTPUT_DIR / "xian_coarse_query_time_matrix_s.csv"
FINE_FILTERING_TIME_MATRIX_FILE = OUTPUT_DIR / "xian_fine_filtering_time_matrix_s.csv"
VERIFICATION_TIME_MATRIX_FILE = OUTPUT_DIR / "xian_verification_time_matrix_s.csv"
VO_SIZE_MATRIX_FILE = OUTPUT_DIR / "xian_vo_size_matrix_mb.csv"
CANDIDATE_MATRIX_FILE = OUTPUT_DIR / "xian_candidate_count_matrix.csv"
FINAL_RESULT_MATRIX_FILE = OUTPUT_DIR / "xian_final_result_count_matrix.csv"
DENSE_REGION_FILE = OUTPUT_DIR / "xian_selected_experiment_setup.csv"
TEMP_VO_FILE = OUTPUT_DIR / "_temp_vo_size.dat"

# ============================================================
# Xi'an experiment region constraints
#
# All 1~5 km query boxes must be fully contained within:
#   Longitude 108.92 ~ 108.98
#   Latitude 34.23 ~ 34.28
# ============================================================

ROI_MIN_LON = 108.91
ROI_MAX_LON = 109.1
ROI_MIN_LAT = 34.1
ROI_MAX_LAT = 34.29

SPATIAL_SIDE_KM = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
]

TEMPORAL_RANGES = [
    ("1min", 60),
    ("30min", 30 * 60),
    ("1h", 60 * 60),
    ("4h", 4 * 60 * 60),
    ("12h", 12 * 60 * 60),
    ("24h", 24 * 60 * 60),

]

REPEATS = 30
WARMUP_REPEATS = 1

# Grid step for spatial center candidates.
# Since the 5 km query box nearly fills the given ROI, the valid center region is already small,
# so a 0.25 km step is sufficiently fine.
CENTER_GRID_STEP_KM = 0.25

# Automatic time selection:
# First scan a real 1-minute query every 30 minutes,
# then refine minute by minute around the best point.
TIME_COARSE_STEP_SECONDS = 30 * 60
TIME_FINE_RADIUS_SECONDS = 30 * 60
TIME_FINE_STEP_SECONDS = 60

# To ensure the formal 24-hour query is fully contained within the data time range,
# the automatically selected BASE_QUERY_START must satisfy:
#   BASE_QUERY_START + 24h <= catalog_max_time
MAX_EXPERIMENT_WINDOW_SECONDS = 24 * 60 * 60


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
    """
    The maximum 5 km × 5 km query box must be fully contained within both:
      1. The user-specified Xi'an ROI;
      2. The actual road-network L0 overall MBR.

    If the maximum box is valid, the 1~4 km boxes with the same center are also valid.
    """
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

    if min_lon < ROI_MIN_LON:
        return False
    if max_lon > ROI_MAX_LON:
        return False
    if min_lat < ROI_MIN_LAT:
        return False
    if max_lat > ROI_MAX_LAT:
        return False

    root = skeleton.root_index

    if min_lon < skeleton.min_lon[root]:
        return False
    if max_lon > skeleton.max_lon[root]:
        return False
    if min_lat < skeleton.min_lat[root]:
        return False
    if max_lat > skeleton.max_lat[root]:
        return False

    return True


def _inclusive_values(
    lower: float,
    upper: float,
    step: float,
) -> list[float]:
    if lower > upper:
        return []

    if step <= 0:
        raise ValueError("step must be > 0")

    values = []
    x = lower

    while x <= upper + 1e-12:
        values.append(x)
        x += step

    # Add the upper bound and midpoint to avoid missing the best location due to the step size.
    values.append(upper)
    values.append((lower + upper) / 2.0)

    return sorted(set(round(v, 12) for v in values))


def generate_center_candidates(index):
    """
    Generate a grid of valid center points within the user-specified ROI.

    Note: centers are constrained using the maximum 5 km query box, so none of the 30 spatial queries will exceed the boundary.
    """
    skeleton = index.skeleton

    max_side_km = max(SPATIAL_SIDE_KM)
    half_km = max_side_km / 2.0
    half_lat = km_to_lat_degree(half_km)

    lat_lower = ROI_MIN_LAT + half_lat
    lat_upper = ROI_MAX_LAT - half_lat

    # Also consider the road-network MBR.
    root = skeleton.root_index
    lat_lower = max(
        lat_lower,
        skeleton.min_lat[root] + half_lat,
    )
    lat_upper = min(
        lat_upper,
        skeleton.max_lat[root] - half_lat,
    )

    if lat_lower > lat_upper:
        raise RuntimeError(
            "The given latitude range cannot accommodate a 5 km × 5 km query box"
        )

    lat_step = km_to_lat_degree(CENTER_GRID_STEP_KM)
    lat_values = _inclusive_values(
        lat_lower,
        lat_upper,
        lat_step,
    )

    centers = []

    for center_lat in lat_values:
        half_lon = km_to_lon_degree(
            half_km,
            center_lat,
        )

        lon_lower = max(
            ROI_MIN_LON + half_lon,
            skeleton.min_lon[root] + half_lon,
        )
        lon_upper = min(
            ROI_MAX_LON - half_lon,
            skeleton.max_lon[root] - half_lon,
        )

        if lon_lower > lon_upper:
            continue

        lon_step = km_to_lon_degree(
            CENTER_GRID_STEP_KM,
            center_lat,
        )

        lon_values = _inclusive_values(
            lon_lower,
            lon_upper,
            lon_step,
        )

        for center_lon in lon_values:
            if center_is_valid(
                center_lon,
                center_lat,
                skeleton,
            ):
                centers.append(
                    (
                        center_lon,
                        center_lat,
                    )
                )

    centers = sorted(set(centers))

    if not centers:
        raise RuntimeError(
            "No valid center capable of accommodating a 5 km query box was found within the specified longitude/latitude range"
        )

    return centers


def spatial_entry_weight(
    index,
    center_lon: float,
    center_lat: float,
) -> int:
    """
    Use the total number of entries on roads that actually intersect the 1 km × 1 km area as the spatial-density pre-score.

    This step is used only to find the initial high-density center and is not included in any formal experiment timing.
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

    for eid in range(road.max_eid + 1):
        if not road.edge_exists(eid):
            continue

        count = int(store.entry_counts[eid])
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


def get_catalog_time_range(catalog) -> tuple[int, int]:
    """
    Obtain the true minimum and maximum times from the trajectory catalog already loaded in memory.
    This scan is not included in Query/Fine Filtering time.
    """
    min_time = None
    max_time = None

    for segments in catalog.values():
        for seg in segments:
            start = int(seg.start_time)
            end = int(seg.end_time)

            if min_time is None or start < min_time:
                min_time = start

            if max_time is None or end > max_time:
                max_time = end

    if min_time is None or max_time is None:
        raise RuntimeError("trajectory catalog is empty")

    return min_time, max_time



# ============================================================
# Composite VO Query
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
    """
    Execute the formal server-side coarse-grained spatiotemporal query,
    generating the Composite VO, Verification Set, and Candidate IDs.

    Fine Filtering is not performed here.
    """
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


# ============================================================
# Client Verification
# ============================================================

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
    """
    The client verifies the Composite VO + Verification Set,
    and returns the verified candidate trajectory IDs.

    Verification Time measures only this function.
    """
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

def probe_real_count(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    query_start: int,
    probe_seconds: int = 60,
) -> tuple[int, int]:
    """
    Returns:
      final_count, candidate_count

    Use the actual Composite VO query + Fine Filtering,
    but the entire Probe process is not included in the formal Benchmark timing.
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

    query_end = query_start + probe_seconds

    vo = build_vo(
        index,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        query_start,
        query_end,
    )

    # Verification time is not measured during the Probe stage, but candidates are used only after verification,
    # keeping the semantics consistent with the formal client workflow.
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
        raise RuntimeError("Automatic region-selection Probe verification failed")

    final_ids = fine_filter_candidates(
        candidate_ids=report.candidate_ids,
        catalog=catalog,
        road=index.road,

        query_min_lon=min_lon,
        query_min_lat=min_lat,
        query_max_lon=max_lon,
        query_max_lat=max_lat,

        query_start=query_start,
        query_end=query_end,
    )

    return (
        len(final_ids),
        len(report.candidate_ids),
    )


def choose_initial_dense_center(index, centers):
    print()
    print("===== Spatial Density Preselection =====")
    print(
        "ROI:",
        f"LON[{ROI_MIN_LON}, {ROI_MAX_LON}]",
        f"LAT[{ROI_MIN_LAT}, {ROI_MAX_LAT}]",
    )
    print("Number of valid center candidates:", len(centers))

    ranked = []

    for center_lon, center_lat in centers:
        weight = spatial_entry_weight(
            index,
            center_lon,
            center_lat,
        )

        ranked.append(
            (
                weight,
                center_lon,
                center_lat,
            )
        )

    ranked.sort(reverse=True)

    for i, (
        weight,
        center_lon,
        center_lat,
    ) in enumerate(ranked[:10], 1):
        print(
            f"{i:02d}  "
            f"center=({center_lon:.6f}, {center_lat:.6f})  "
            f"entry_weight={weight}"
        )

    best_weight, best_lon, best_lat = ranked[0]

    print()
    print(
        "Initial dense center:",
        f"({best_lon:.10f}, {best_lat:.10f})",
    )

    return best_lon, best_lat, best_weight


def choose_dense_time(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    catalog_min_time: int,
    catalog_max_time: int,
):
    """
    To ensure that even the shortest 1-minute experiment contains many real trajectories,
    Automatic time selection primarily scores the number of real trajectories after 1-minute Fine Filtering.

    First round: scan every 30 minutes;
    Second round: refine every 1 minute within ±30 minutes of the best point.

    The subsequent 24-hour query must also not exceed the maximum data time.
    """
    latest_start = (
        catalog_max_time
        - MAX_EXPERIMENT_WINDOW_SECONDS
    )

    if latest_start < catalog_min_time:
        raise RuntimeError(
            "The effective time span of the new dataset is less than 24 hours, so the current experiment cannot be executed"
        )

    print()
    print("===== Dense Time Selection =====")
    print("Catalog min time:", catalog_min_time)
    print("Catalog max time:", catalog_max_time)
    print("Latest valid experiment start:", latest_start)
    print(
        "Coarse time scan step:",
        TIME_COARSE_STEP_SECONDS,
        "seconds",
    )

    best = None
    current = catalog_min_time
    scanned = 0

    while current <= latest_start:
        final_count, candidate_count = probe_real_count(
            index,
            catalog,
            center_lon,
            center_lat,
            current,
            60,
        )

        scanned += 1

        score = (
            final_count,
            candidate_count,
        )

        if best is None or score > best[0]:
            best = (
                score,
                current,
                final_count,
                candidate_count,
            )

            print(
                "New best:",
                f"start={current}",
                f"final_1min={final_count}",
                f"candidate_1min={candidate_count}",
            )

        current += TIME_COARSE_STEP_SECONDS

    if best is None:
        raise RuntimeError("Automatic time selection failed")

    _, coarse_best_start, _, _ = best

    fine_start = max(
        catalog_min_time,
        coarse_best_start - TIME_FINE_RADIUS_SECONDS,
    )

    fine_end = min(
        latest_start,
        coarse_best_start + TIME_FINE_RADIUS_SECONDS,
    )

    current = fine_start

    while current <= fine_end:
        final_count, candidate_count = probe_real_count(
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

        current += TIME_FINE_STEP_SECONDS

    (
        _,
        best_start,
        final_count,
        candidate_count,
    ) = best

    print()
    print("Selected time:", best_start)
    print("1min Final Count:", final_count)
    print("1min Candidate Count:", candidate_count)

    return (
        best_start,
        final_count,
        candidate_count,
    )


def refine_dense_center(
    index,
    catalog,
    centers,
    selected_start: int,
):
    """
    At the identified dense 1-minute time point, reselect the spatial center using the actual Fine Filtering result.
    """
    print()
    print("===== Spatial Center Refinement =====")

    best = None

    for i, (
        center_lon,
        center_lat,
    ) in enumerate(centers, 1):
        final_count, candidate_count = probe_real_count(
            index,
            catalog,
            center_lon,
            center_lat,
            selected_start,
            60,
        )

        entry_weight = spatial_entry_weight(
            index,
            center_lon,
            center_lat,
        )

        score = (
            final_count,
            candidate_count,
            entry_weight,
        )

        if best is None or score > best[0]:
            best = (
                score,
                center_lon,
                center_lat,
                final_count,
                candidate_count,
                entry_weight,
            )

    if best is None:
        raise RuntimeError("Spatial center refinement failed")

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
        f"({center_lon:.10f}, {center_lat:.10f})",
    )
    print("1min Final Count:", final_count)
    print("1min Candidate Count:", candidate_count)

    return (
        center_lon,
        center_lat,
        final_count,
        candidate_count,
        entry_weight,
    )


def refine_time_locally(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    selected_start: int,
    catalog_min_time: int,
    catalog_max_time: int,
):
    """
    After the spatial center changes, fine-tune the time minute by minute within ±30 minutes around the final center.
    """
    latest_start = (
        catalog_max_time
        - MAX_EXPERIMENT_WINDOW_SECONDS
    )

    start = max(
        catalog_min_time,
        selected_start - TIME_FINE_RADIUS_SECONDS,
    )
    end = min(
        latest_start,
        selected_start + TIME_FINE_RADIUS_SECONDS,
    )

    best = None
    current = start

    while current <= end:
        final_count, candidate_count = probe_real_count(
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

        if best is None or score > best[0]:
            best = (
                score,
                current,
                final_count,
                candidate_count,
            )

        current += 60

    if best is None:
        raise RuntimeError("Final time refinement failed")

    _, best_start, final_count, candidate_count = best

    return (
        best_start,
        final_count,
        candidate_count,
    )

#
# def select_experiment_setup(index, catalog):
#     """
#     Automatically determine the unique setup for this Xi'an experiment:
#       - Spatial center;
#       - BASE_QUERY_START.
#
#     The selection process occurs entirely before the formal Benchmark and is not included in any experiment timing.
#     """
#     centers = generate_center_candidates(index)
#     (
#         initial_lon,
#         initial_lat,
#         _,
#     ) = choose_initial_dense_center(
#         index,
#         centers,
#     )
#
#     (
#         catalog_min_time,
#         catalog_max_time,
#     ) = get_catalog_time_range(catalog)
#
#     (
#         selected_start,
#         _,
#         _,
#     ) = choose_dense_time(
#         index,
#         catalog,
#         initial_lon,
#         initial_lat,
#         catalog_min_time,
#         catalog_max_time,
#     )
#
#     (
#         center_lon,
#         center_lat,
#         _,
#         _,
#         entry_weight,
#     ) = refine_dense_center(
#         index,
#         catalog,
#         centers,
#         selected_start,
#     )
#
#     (
#         selected_start,
#         final_count,
#         candidate_count,
#     ) = refine_time_locally(
#         index,
#         catalog,
#         center_lon,
#         center_lat,
#         selected_start,
#         catalog_min_time,
#         catalog_max_time,
#     )
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
#     print(f"center_lon = {center_lon:.10f}")
#     print(f"center_lat = {center_lat:.10f}")
#     print(f"BASE_QUERY_START = {selected_start}")
#     print("Selected 1min Final Count =", final_count)
#     print("Selected 1min Candidate Count =", candidate_count)
#     print("Spatial Entry Weight =", entry_weight)
#     print()
#     print("Maximum 5 km × 5 km query box:")
#     print(f"MIN_LON = {min_lon:.10f}")
#     print(f"MIN_LAT = {min_lat:.10f}")
#     print(f"MAX_LON = {max_lon:.10f}")
#     print(f"MAX_LAT = {max_lat:.10f}")
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
def select_experiment_setup(index, catalog):

    # 固定查询中心
    centers = [(108.95, 34.25)]

    (
        center_lon,
        center_lat,
        entry_weight,
    ) = choose_initial_dense_center(
        index,
        centers,
    )

    (
        catalog_min_time,
        catalog_max_time,
    ) = get_catalog_time_range(catalog)

    # 全时间查询，从数据最早时间开始
    selected_start = 1539619200

    print()
    print("===== Selected Experiment Setup =====")
    print("center_lon =", center_lon)
    print("center_lat =", center_lat)

    print("catalog_min_time =", catalog_min_time)
    print("catalog_max_time =", catalog_max_time)

    print(
        "catalog_span_days =",
        (catalog_max_time - catalog_min_time) / 86400
    )

    print("BASE_QUERY_START =", selected_start)

    return (
        center_lon,
        center_lat,
        selected_start,
        0,
        0,
        entry_weight,
        catalog_min_time,
        catalog_max_time,
    )

def measure_vo_size_bytes(vo) -> int:
    # File serialization occurs outside all timed intervals.
    return int(
        save_composite_vo(
            vo=vo,
            file_path=str(TEMP_VO_FILE),
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
        query_min_lon < ROI_MIN_LON
        or query_max_lon > ROI_MAX_LON
        or query_min_lat < ROI_MIN_LAT
        or query_max_lat > ROI_MAX_LAT
    ):
        raise RuntimeError(
            f"{spatial_side_km}km query box exceeds the specified Xi\'an ROI"
        )

    query_start = base_query_start
    query_end = (
        base_query_start
        + temporal_seconds
    )

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
                "roi_min_lon",
                "roi_min_lat",
                "roi_max_lon",
                "roi_max_lat",
                "center_lon",
                "center_lat",
                "base_query_start",
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
            ]
        )

        writer.writerow(
            [
                ROI_MIN_LON,
                ROI_MIN_LAT,
                ROI_MAX_LON,
                ROI_MAX_LAT,
                f"{center_lon:.10f}",
                f"{center_lat:.10f}",
                selected_start,
                catalog_min_time,
                catalog_max_time,
                selected_final_count,
                selected_candidate_count,
                entry_weight,
                max(SPATIAL_SIDE_KM),
                f"{min_lon:.10f}",
                f"{min_lat:.10f}",
                f"{max_lon:.10f}",
                f"{max_lat:.10f}",
            ]
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"===== Xi'an 5×6 Cross Range Benchmark [{VERSION}] ====="
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
        "Specified Xi\'an ROI:",
        f"LON[{ROI_MIN_LON}, {ROI_MAX_LON}]",
        f"LAT[{ROI_MIN_LAT}, {ROI_MAX_LAT}]",
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
    # Automatically select a high-density spatial center + high-density time start
    #
    # The selection process is not included in formal Query/Verify/Filter timing.
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

    save_selected_center(
        center_lon,
        center_lat,
        base_query_start,
        selected_final_count,
        selected_candidate_count,
        entry_weight,
        catalog_min_time,
        catalog_max_time,
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
                base_query_start=base_query_start,

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

    print(
        "Automatically selected experiment region and time:",
        DENSE_REGION_FILE,
    )


if __name__ == "__main__":
    main()
