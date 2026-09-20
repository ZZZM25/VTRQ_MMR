
from __future__ import annotations

import os
from time import perf_counter

from road_network import RoadNetwork

from vo_io import (
    load_composite_vo_tokens,
    load_verification_set,
    load_trusted_root,
)

from verifier import (
    verify_composite_vo,
)

from trajectory_catalog import (
    load_trajectory_catalog,
    fine_filter_candidates,
)


def main():
    NODE_FILE = (
        r"E:\MMR_Trajectory_range\xian_nodes.txt"
    )

    EDGE_FILE = (
        r"E:\MMR_Trajectory_range\xian_edges.txt"
    )

    VO_FILE = (
        "query_output/xian_composite_vo.dat"
    )

    VS_FILE = (
        "query_output/xian_verification_set.dat"
    )

    ROOT_FILE = (
        "query_output/xian_root_S.dat"
    )

    TRAJECTORY_CSV = (
        "catalog_output/xian_trajectory_catalog.csv"
    )

    # ========================================================
    # Query
    #
    # Must be exactly consistent with server_query.py.
    # ========================================================

    QUERY_MIN_LON = 104.0400
    QUERY_MIN_LAT = 30.6550

    QUERY_MAX_LON = 104.0900
    QUERY_MAX_LAT = 30.7050

    QUERY_START = 1538415951
    QUERY_END = 1538628400

    print(
        "===== Client ====="
    )

    for file_path in (
        VO_FILE,
        VS_FILE,
        ROOT_FILE,
        TRAJECTORY_CSV,
    ):

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

    # ========================================================
    # 1. Public Road Load
    #
    # Not included in Verification Time
    # Not included in Fine Filtering Time
    # ========================================================

    road = RoadNetwork()

    road.load(
        node_file=
        NODE_FILE,

        edge_file=
        EDGE_FILE,
    )

    # ========================================================
    # 2. Load VO / VS / root
    #
    # Not included in Verification Time
    # ========================================================

    tokens = (
        load_composite_vo_tokens(
            VO_FILE
        )
    )

    verification_set = (
        load_verification_set(
            VS_FILE
        )
    )

    trusted_root = (
        load_trusted_root(
            ROOT_FILE
        )
    )

    # ========================================================
    # 3. Verification
    #
    # Only verify_composite_vo is timed.
    # ========================================================

    verification_t0 = (
        perf_counter()
    )

    report = (
        verify_composite_vo(

            tokens=
            tokens,

            verification_set=
            verification_set,

            road=
            road,

            trusted_root=
            trusted_root,

            query_min_lon=
            QUERY_MIN_LON,

            query_min_lat=
            QUERY_MIN_LAT,

            query_max_lon=
            QUERY_MAX_LON,

            query_max_lat=
            QUERY_MAX_LAT,

            query_start=
            QUERY_START,

            query_end=
            QUERY_END,
        )
    )

    verification_time = (
        perf_counter()
        -
        verification_t0
    )

    print()
    print(
        "===== Verification ====="
    )

    print(
        "Verification successful:",
        report.success,
    )

    print(
        "Root matches exactly:",
        (
            report.reconstructed_root
            ==
            report.trusted_root
        ),
    )

    print(
        "Number of verified candidates:",
        len(
            report.candidate_ids
        ),
    )

    print(
        "Verification Time:",
        f"{verification_time:.9f} s",
    )

    # ========================================================
    # 4. Load Trajectory Catalog
    #
    # CSV loading is explicitly excluded from
    # Fine Filtering Time.
    # ========================================================

    print()
    print(
        "===== Load Trajectory Catalog ====="
    )

    catalog_load_t0 = (
        perf_counter()
    )

    catalog = (
        load_trajectory_catalog(
            TRAJECTORY_CSV
        )
    )

    catalog_load_time = (
        perf_counter()
        -
        catalog_load_t0
    )

    print(
        "Number of catalog trajectories:",
        len(
            catalog
        ),
    )

    print(
        "Catalog Load Time:",
        f"{catalog_load_time:.9f} s",
    )

    print(
        "(Not included in Fine Filtering Time)"
    )

    # ========================================================
    # 5. Fine Filtering
    #
    # Timing starts here.
    # ========================================================

    filtering_t0 = (
        perf_counter()
    )

    final_trajectory_ids = (
        fine_filter_candidates(

            candidate_ids=
            report.candidate_ids,

            catalog=
            catalog,

            road=
            road,

            query_min_lon=
            QUERY_MIN_LON,

            query_min_lat=
            QUERY_MIN_LAT,

            query_max_lon=
            QUERY_MAX_LON,

            query_max_lat=
            QUERY_MAX_LAT,

            query_start=
            QUERY_START,

            query_end=
            QUERY_END,
        )
    )

    fine_filtering_time = (
        perf_counter()
        -
        filtering_t0
    )

    print()
    print(
        "===== Fine Filtering ====="
    )

    print(
        "Number of candidates before fine filtering:",
        len(
            report.candidate_ids
        ),
    )

    print(
        "Number of results after fine filtering:",
        len(
            final_trajectory_ids
        ),
    )

    print(
        "Fine Filtering Time:",
        f"{fine_filtering_time:.9f} s",
    )

    print()
    print(
        "===== Final Trajectory IDs ====="
    )

    for i, trajectory_id in enumerate(
        final_trajectory_ids[:50]
    ):

        print(
            i,
            trajectory_id.hex(),
        )

    print()
    print(
        "client_verify.py completed successfully"
    )


if __name__ == "__main__":
    main()

