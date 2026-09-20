
from __future__ import annotations

import os
from time import perf_counter


from composite_vo import (
    CompositeVOBuilder,
)

from index_io import (
    load_index,
)

from vo_io import (
    save_composite_vo,
    save_verification_set,
    save_trusted_root,
)


def main():

    # ========================================================
    # Prebuilt Complete Server Index
    # ========================================================

    INDEX_FILE = (
        "index_output/xian_trajectory_index.dat"
    )

    # ========================================================
    # Query
    # ========================================================

    QUERY_MIN_LON = 104.0400
    QUERY_MIN_LAT = 30.6550

    QUERY_MAX_LON = 104.0900
    QUERY_MAX_LAT = 30.7050

    QUERY_START = 1538415951
    QUERY_END = 1538628400

    # ========================================================
    # Server Response
    # ========================================================

    OUTPUT_DIR = (
        "query_output"
    )

    VO_FILE = os.path.join(
        OUTPUT_DIR,
        "xian_composite_vo.dat",
    )

    VS_FILE = os.path.join(
        OUTPUT_DIR,
        "xian_verification_set.dat",
    )

    ROOT_FILE = os.path.join(
        OUTPUT_DIR,
        "xian_root_S.dat",
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load Complete Index
    #
    # Very important:
    #
    # Index file loading + deserialization
    # are explicitly excluded from Query Time.
    # ========================================================

    print(
        "===== Server: Load Persisted Index ====="
    )

    load_t0 = (
        perf_counter()
    )

    index = (
        load_index(
            INDEX_FILE
        )
    )

    index_load_time = (
        perf_counter()
        - load_t0
    )

    road = index.road
    store = index.store
    mmr_index = index.mmr_index
    skeleton = index.skeleton
    l1_index = index.l1_index
    l0_index = index.l0_index

    trusted_root = (
        l0_index.root_hash
    )

    print()

    print(
        "Index loading completed"
    )

    print(
        "Index Load Time:",
        f"{index_load_time:.9f} s",
    )

    print(
        "(Not included in Query Time)"
    )

    print()

    print(
        "root_S:"
    )

    print(
        trusted_root.hex()
    )

    # ========================================================
    # 2. Server Query
    #
    # Only this section is included in Query Time.
    #
    # Includes:
    #
    # L0/L1/MMR query
    # VO generation
    # Verification Set generation
    #
    # Excludes:
    #
    # index load
    # VO saving
    # VS saving
    # root saving
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "===== Server Query ====="
    )

    query_t0 = (
        perf_counter()
    )

    vo = (
        CompositeVOBuilder(

            road=
            road,

            store=
            store,

            mmr_index=
            mmr_index,

            skeleton=
            skeleton,

            l1_index=
            l1_index,

            l0_index=
            l0_index,

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
        ).build()
    )

    query_time = (
        perf_counter()
        - query_t0
    )

    print()

    print(
        "VO Token count:",
        len(
            vo.tokens
        ),
    )

    print(
        "Verification Set count:",
        len(
            vo.verification_set
        ),
    )

    print(
        "Candidate trajectory_id count:",
        len(
            vo.candidate_ids
        ),
    )

    print()

    print(
        "Query Time:",
        f"{query_time:.9f} s",
    )

    # ========================================================
    # 3. Save Server Response
    #
    # Explicitly excluded from Query Time.
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "===== Save Server Response ====="
    )

    save_t0 = (
        perf_counter()
    )

    vo_size = (
        save_composite_vo(
            vo=
            vo,

            file_path=
            VO_FILE,
        )
    )

    vs_size = (
        save_verification_set(
            verification_set=
            vo.verification_set,

            file_path=
            VS_FILE,
        )
    )

    # --------------------------------------------------------
    # Simulate the on-chain root during testing.
    #
    # In the formal system, root_S comes from the blockchain
    # and is not part of the Server Response.
    # --------------------------------------------------------

    save_trusted_root(
        root_hash=
        trusted_root,

        file_path=
        ROOT_FILE,
    )

    response_save_time = (
        perf_counter()
        - save_t0
    )

    response_size = (
        vo_size
        + vs_size
    )

    print()

    print(
        "VO size:",
        vo_size,
        "bytes",
    )

    print(
        "Verification Set size:",
        vs_size,
        "bytes",
    )

    print(
        "Server Response size:",
        response_size,
        "bytes",
    )

    print(
        "Server Response size:",
        f"{response_size / 1024.0:.3f} KiB",
    )

    print()

    print(
        "Response Save Time:",
        f"{response_save_time:.9f} s",
    )

    print(
        "(Not included in Query Time)"
    )

    print()

    print(
        "===== Candidate trajectory_id ====="
    )

    for i, trajectory_id in enumerate(
        vo.candidate_ids[:20]
    ):

        print(
            i,
            trajectory_id.hex(),
        )

    print()

    print(
        "server_query.py completed successfully"
    )


if __name__ == "__main__":
    main()
