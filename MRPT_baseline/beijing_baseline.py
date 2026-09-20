from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

from baseline.authenticated_index import (
    BaselineIndex,
    authenticate_spatial_tree,
    build_edge_commitments,
)
from baseline.benchmark_utils import (
    SPATIAL_SIDES,
    TEMPORAL_RANGES,
    load_setup,
    make_square,
)
from baseline.entry_store import PackedEdgeEntryStore
from baseline.models import TrajectorySegment
from baseline.persistence import (
    load_catalog,
    load_index,
    save_catalog,
    save_index,
)
from baseline.road_network import RoadNetwork
from baseline.server import BaselineServerQuery
from baseline.spatial_tree import BinarySpatialTreeBuilder
from baseline.trajectory_catalog import (
    PackedTrajectoryCatalog,
    fine_filter_candidates,
)
from baseline.trajectory_id import compute_trajectory_id
from baseline.verifier import verify_vo
from baseline.vo_io import serialize_vo


# ============================================================
# Beijing baseline: one-file entry point for VTRQ Baseline v3
#
# This file does NOT modify Chengdu/Xi'an code paths.
# It directly consumes the already-preprocessed Beijing files.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

BEIJING_DIR = Path(r"E:\MMR_Trajectory_range\beijing_preprocessed")
NODE_FILE = BEIJING_DIR / "beijing_nodes.txt"
EDGE_FILE = BEIJING_DIR / "beijing_edges.txt"
TRAJECTORY_FILE = BEIJING_DIR / "beijing_trajectories.csv"

OURS_SETUP_FILE = PROJECT_DIR / "configs" / "beijing_setup.csv"
OURS_CANDIDATE_MATRIX_FILE = Path(
    r"E:\MMR_Trajectory_range\benchmark_output_beijing\candidate_count_matrix.csv"
)
OURS_FINAL_MATRIX_FILE = Path(
    r"E:\MMR_Trajectory_range\benchmark_output_beijing\final_result_count_matrix.csv"
)
OURS_RESULT_FILE = OURS_CANDIDATE_MATRIX_FILE.with_name("cross_range_results.csv")

THETA = 64
REPEATS = 30
WARMUP_REPEATS = 1

INDEX_FILE = PROJECT_DIR / "index_output" / "beijing_baseline_index.dat"
CATALOG_FILE = PROJECT_DIR / "catalog_output" / "beijing_baseline_catalog.dat"
TRUSTED_ROOT_FILE = PROJECT_DIR / "trusted_root" / "beijing_root.txt"
OUTPUT_DIR = PROJECT_DIR / "benchmark_output" / "beijing_baseline"


# ============================================================
# Helpers
# ============================================================


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")


def _set_large_csv_field_limit() -> None:
    """points_data can be very large in the Beijing CSV."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_points(text: str, row_no: int) -> list:
    try:
        points = json.loads(text)
    except Exception as exc:
        raise ValueError(
            f"row {row_no}: points_data JSON parse failed"
        ) from exc
    if not isinstance(points, list):
        raise ValueError(f"row {row_no}: points_data must be a JSON list")
    return points


def _load_matrix(path: Path) -> dict[tuple[float, str], int]:
    out: dict[tuple[float, str], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "spatial_side_km" not in reader.fieldnames:
            raise ValueError(f"invalid matrix file: {path}")
        for row in reader:
            side = float(row["spatial_side_km"])
            for label in reader.fieldnames:
                if label == "spatial_side_km":
                    continue
                cell = (row.get(label) or "").strip()
                if cell:
                    out[(side, label)] = int(float(cell))
    return out


def _load_query_rows(path: Path) -> dict[tuple[float, str], dict]:
    """Use saved query parameters, even if the current setup has changed."""
    required = {
        "spatial_side_km", "temporal_label", "query_min_lon", "query_min_lat",
        "query_max_lon", "query_max_lat", "query_start", "query_end",
    }
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"query parameters missing from results: {path}")
        for row in reader:
            key = (float(row["spatial_side_km"]), row["temporal_label"])
            if key in out:
                raise ValueError(f"duplicate query cell {key} in results: {path}")
            out[key] = row
    return out


def _write_matrix_partial(path: Path, rows: list[dict], value_field: str) -> None:
    """Write completed cells only; safe while the 30-combination run is ongoing."""
    labels = [x[0] for x in TEMPORAL_RANGES]
    lookup = {
        (float(r["spatial_side_km"]), str(r["temporal_label"])): r[value_field]
        for r in rows
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spatial_side_km", *labels])
        for side in SPATIAL_SIDES:
            w.writerow(
                [side]
                + [lookup.get((float(side), label), "") for label in labels]
            )


def _write_outputs(rows: list[dict]) -> None:
    if not rows:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_file = OUTPUT_DIR / "cross_range_results.csv"
    with result_file.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    matrices = [
        ("query_time_matrix_s.csv", "query_time_s"),
        ("server_query_time_matrix_s.csv", "server_query_time_s"),
        ("fine_filtering_time_matrix_s.csv", "fine_filtering_time_s"),
        ("verification_time_matrix_s.csv", "verification_time_s"),
        ("vo_size_matrix_mb.csv", "vo_size_mb"),
        ("candidate_count_matrix.csv", "candidate_trajectory_count"),
        ("final_result_count_matrix.csv", "final_trajectory_count"),
        ("scanned_entries_matrix.csv", "scanned_entries"),
        ("checked_cross_edges_matrix.csv", "checked_cross_edges"),
    ]
    for name, field in matrices:
        _write_matrix_partial(OUTPUT_DIR / name, rows, field)


# ============================================================
# Beijing dataset loading
# ============================================================


def build_beijing_store_and_catalog(road: RoadNetwork):
    """
    Read Beijing's existing preprocessed trajectory CSV directly.

    Input format:
      source_traj_id(or traj_id), points_data

    points_data:
      [[node_id, lon, lat, timestamp], ...]

    For each adjacent pair u@t1 -> v@t2:
      - if u == v: stationary step, no road Entry
      - otherwise resolve the undirected road EID
      - create Entry(eid, trajectory_id, t1, t2)
      - keep the directed TrajectorySegment for Fine Filtering

    trajectory_id is computed exactly as in the current Beijing Ours path:
      SHA256(eid_path || first_movement_start || last_movement_end)

    The EID order is preserved and repeated EIDs are preserved.

    Timing policy is identical to Baseline v3:
      CSV reading / JSON parsing / trajectory parsing / catalog preparation
      are outside Index Construction Time; only insert_by_start() is timed
      here and added to the later authenticated-index construction time.
    """
    _require_file(TRAJECTORY_FILE)
    _set_large_csv_field_limit()

    store = PackedEdgeEntryStore(road.max_eid)
    catalog = PackedTrajectoryCatalog()
    seen: set[bytes] = set()

    total_raw = 0
    total_trajectories = 0
    total_entries = 0
    total_stationary = 0
    skipped_short = 0
    skipped_no_movement = 0
    ordered_insert_time_s = 0.0

    with TRAJECTORY_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])

        if "points_data" not in fields:
            raise ValueError(
                "beijing_trajectories.csv missing column: points_data"
            )
        if "source_traj_id" not in fields and "traj_id" not in fields:
            raise ValueError(
                "beijing_trajectories.csv missing source_traj_id/traj_id"
            )

        for row_no, row in enumerate(reader, 2):
            total_raw += 1
            points = _load_points(row["points_data"], row_no)

            if len(points) < 2:
                skipped_short += 1
                continue

            normalized: list[tuple[int, int]] = []
            previous_time = None

            for point_index, point in enumerate(points):
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) < 4
                ):
                    raise ValueError(
                        f"row {row_no} point {point_index}: "
                        "expected [node_id,lon,lat,timestamp]"
                    )

                node_id = int(point[0])
                timestamp = int(point[3])

                if previous_time is not None and timestamp < previous_time:
                    raise ValueError(
                        f"row {row_no}: timestamp goes backwards: "
                        f"{previous_time} -> {timestamp}"
                    )

                normalized.append((node_id, timestamp))
                previous_time = timestamp

            eids: list[int] = []
            segments: list[TrajectorySegment] = []

            for i in range(len(normalized) - 1):
                u, start = normalized[i]
                v, end = normalized[i + 1]

                if u == v:
                    total_stationary += 1
                    continue

                if end < start:
                    raise ValueError(
                        f"row {row_no} segment {i}: end < start"
                    )

                try:
                    eid = road.resolve_edge(u, v)
                except KeyError as exc:
                    raise KeyError(
                        f"row {row_no}: road has no edge {u}->{v}"
                    ) from exc

                eids.append(int(eid))
                segments.append(
                    TrajectorySegment(
                        start_node=int(u),
                        start_time=int(start),
                        end_node=int(v),
                        end_time=int(end),
                    )
                )

            # Match the existing Beijing Ours benchmark/catalog behavior:
            # trajectories with no movement edge do not enter the query catalog.
            if not eids:
                skipped_no_movement += 1
                continue

            trajectory_start = int(segments[0].start_time)
            trajectory_end = int(segments[-1].end_time)
            tid = compute_trajectory_id(
                eids, trajectory_start, trajectory_end
            )

            if tid in seen:
                raise ValueError(
                    f"row {row_no}: duplicate trajectory_id {tid.hex()}"
                )
            seen.add(tid)

            # Catalog preparation is outside the timed insertion.
            catalog.add(tid, segments)

            # Baseline v3 requirement: insert each arriving Entry immediately
            # at its correct start-time position. Do NOT append then sort.
            t_insert = perf_counter()
            for eid, seg in zip(eids, segments):
                store.insert_by_start(
                    eid, tid, seg.start_time, seg.end_time
                )
            ordered_insert_time_s += perf_counter() - t_insert

            total_trajectories += 1
            total_entries += len(eids)

            if total_raw % 10000 == 0:
                print(
                    f"read {total_raw} rows; kept {total_trajectories} "
                    f"trajectories; {total_entries} Entries"
                )

    store.validate_start_order()

    print(
        "Beijing raw CSV parsed: "
        f"raw={total_raw}, kept={total_trajectories}, "
        f"entries={total_entries}, stationary={total_stationary}, "
        f"short_skipped={skipped_short}, "
        f"no_movement_skipped={skipped_no_movement}"
    )

    return (
        store,
        catalog,
        total_trajectories,
        total_entries,
        ordered_insert_time_s,
    )


# ============================================================
# Build baseline index
# ============================================================


def build_index() -> None:
    _require_file(NODE_FILE)
    _require_file(EDGE_FILE)
    _require_file(TRAJECTORY_FILE)

    road = RoadNetwork()
    road.load(str(NODE_FILE), str(EDGE_FILE))
    print(
        f"road: {len(road.valid_node_ids())} nodes, "
        f"{len(road.valid_edge_ids())} edges"
    )

    print("===== Beijing Trajectory Preprocessing =====")
    print("Input:", TRAJECTORY_FILE)
    print("trajectory_id is recomputed from EID path and movement start/end exactly as Ours.")
    print("CSV/JSON parsing, EID mapping and catalog preparation are excluded from Index Construction Time.")
    print("Per-Entry ordered insertion is INCLUDED in Index Construction Time.")

    (
        store,
        catalog,
        ntr,
        nseg,
        ordered_insert_time,
    ) = build_beijing_store_and_catalog(road)

    print(f"trajectories={ntr}, entries/segments={nseg}")

    print("===== Beijing Baseline Index Construction =====")
    t0 = perf_counter()

    edge_roots, edge_hashes, suffix_states = build_edge_commitments(
        road, store
    )
    tree = BinarySpatialTreeBuilder(road, THETA).build()
    root = authenticate_spatial_tree(tree, edge_hashes)

    post_insert_index_time = perf_counter() - t0
    index_time = ordered_insert_time + post_insert_index_time

    index = BaselineIndex(
        road=road,
        store=store,
        tree=tree,
        edge_entry_roots=edge_roots,
        edge_hashes=edge_hashes,
        edge_suffix_states=suffix_states,
        root_hash=root,
        theta=tree.theta,
        trajectory_files=(str(TRAJECTORY_FILE),),
    )

    size = save_index(index, INDEX_FILE)
    cat_size = save_catalog(catalog, CATALOG_FILE)
    TRUSTED_ROOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRUSTED_ROOT_FILE.write_text(root.hex() + "\n", encoding="ascii")

    print(f"Ordered Entry Insertion Time: {ordered_insert_time:.9f} s")
    print(f"Post-insertion Index Time: {post_insert_index_time:.9f} s")
    print(f"Index Construction Time: {index_time:.9f} s")
    print(f"root_S: {root.hex()}")
    print(f"index file: {INDEX_FILE} ({size/1024/1024:.3f} MiB)")
    print(f"catalog file: {CATALOG_FILE} ({cat_size/1024/1024:.3f} MiB)")
    print(f"trusted root file: {TRUSTED_ROOT_FILE}")


# ============================================================
# Benchmark
# ============================================================


def _run_one_combination(
    index,
    catalog,
    center_lon: float,
    center_lat: float,
    base_start: int,
    side: float,
    label: str,
    seconds: int,
) -> dict:
    qminx, qminy, qmaxx, qmaxy = make_square(
        center_lon, center_lat, side
    )
    qs = int(base_start)
    qe = qs + int(seconds)

    # Warmup: identical server / verifier / fine-filter pipeline.
    for _ in range(WARMUP_REPEATS):
        r = BaselineServerQuery(
            index, qminx, qminy, qmaxx, qmaxy, qs, qe
        ).build()
        rep = verify_vo(
            r.vo,
            index.road,
            index.root_hash,
            qminx,
            qminy,
            qmaxx,
            qmaxy,
            qs,
            qe,
            r.candidate_ids,
        )
        if not rep.success:
            raise RuntimeError(
                f"warmup verification failed: {side}km x {label}"
            )
        fine_filter_candidates(
            rep.candidate_ids,
            catalog,
            index.road,
            qminx,
            qminy,
            qmaxx,
            qmaxy,
            qs,
            qe,
        )

    server_times: list[float] = []
    verify_times: list[float] = []
    fine_times: list[float] = []

    last = None
    last_rep = None
    last_final = None

    for _ in range(REPEATS):
        t0 = perf_counter()
        r = BaselineServerQuery(
            index, qminx, qminy, qmaxx, qmaxy, qs, qe
        ).build()
        server_times.append(perf_counter() - t0)

        t0 = perf_counter()
        rep = verify_vo(
            r.vo,
            index.road,
            index.root_hash,
            qminx,
            qminy,
            qmaxx,
            qmaxy,
            qs,
            qe,
            r.candidate_ids,
        )
        verify_times.append(perf_counter() - t0)
        if not rep.success:
            raise RuntimeError(
                f"verification failed: {side}km x {label}"
            )

        t0 = perf_counter()
        final = fine_filter_candidates(
            rep.candidate_ids,
            catalog,
            index.road,
            qminx,
            qminy,
            qmaxx,
            qmaxy,
            qs,
            qe,
        )
        fine_times.append(perf_counter() - t0)

        last = r
        last_rep = rep
        last_final = final

    if last is None or last_rep is None or last_final is None:
        raise RuntimeError("no benchmark result generated")

    vo_size = len(serialize_vo(last.vo))
    query_times = [x + y for x, y in zip(server_times, fine_times)]

    return {
        "spatial_side_km": side,
        "temporal_label": label,
        "temporal_seconds": seconds,
        "query_min_lon": qminx,
        "query_min_lat": qminy,
        "query_max_lon": qmaxx,
        "query_max_lat": qmaxy,
        "query_start": qs,
        "query_end": qe,
        "server_query_time_s": statistics.mean(server_times),
        "server_query_time_std_s": (
            statistics.stdev(server_times) if len(server_times) > 1 else 0.0
        ),
        "fine_filtering_time_s": statistics.mean(fine_times),
        "fine_filtering_time_std_s": (
            statistics.stdev(fine_times) if len(fine_times) > 1 else 0.0
        ),
        "query_time_s": statistics.mean(query_times),
        "verification_time_s": statistics.mean(verify_times),
        "verification_time_std_s": (
            statistics.stdev(verify_times) if len(verify_times) > 1 else 0.0
        ),
        "vo_size_bytes": vo_size,
        "vo_size_mb": vo_size / 1_000_000.0,
        "verification_set_count": len(last.vo.verification_set),
        "candidate_trajectory_count": len(last_rep.candidate_ids),
        "final_trajectory_count": len(last_final),
        "visited_spatial_nodes": last.stats.visited_spatial_nodes,
        "pruned_spatial_nodes": last.stats.pruned_spatial_nodes,
        "checked_cross_edges": last.stats.checked_cross_edges,
        "checked_leaf_edges": last.stats.checked_leaf_edges,
        "spatial_matched_edges": last.stats.spatial_matched_edges,
        "scanned_entries": last.stats.scanned_entries,
        "boundary_witness_entries": last.stats.boundary_witness_entries,
    }


def benchmark() -> None:
    _require_file(INDEX_FILE)
    _require_file(CATALOG_FILE)
    _require_file(OURS_SETUP_FILE)

    index = load_index(INDEX_FILE)
    catalog = load_catalog(CATALOG_FILE)

    center_lon, center_lat, base_start = load_setup(
        OURS_SETUP_FILE, None
    )
    if base_start is None:
        raise ValueError(
            f"base_query_start missing from Ours setup: {OURS_SETUP_FILE}"
        )

    print(
        f"===== Beijing Baseline {len(SPATIAL_SIDES)}x{len(TEMPORAL_RANGES)} Benchmark ====="
    )
    print(f"center_lon = {center_lon:.10f}")
    print(f"center_lat = {center_lat:.10f}")
    print(f"base_query_start = {base_start}")
    print("setup =", OURS_SETUP_FILE)
    print("repeats =", REPEATS)
    print("warmup =", WARMUP_REPEATS)
    print()

    rows: list[dict] = []

    for side in SPATIAL_SIDES:
        for label, seconds in TEMPORAL_RANGES:
            row = _run_one_combination(
                index=index,
                catalog=catalog,
                center_lon=center_lon,
                center_lat=center_lat,
                base_start=base_start,
                side=side,
                label=label,
                seconds=seconds,
            )
            rows.append(row)

            # Save immediately after each completed combination.
            _write_outputs(rows)

            print(
                f"{side:.0f}km x {label}: "
                f"Query={row['query_time_s']:.6f}s "
                f"Verify={row['verification_time_s']:.6f}s "
                f"VO={row['vo_size_mb']:.3f}MB "
                f"candidate={row['candidate_trajectory_count']} "
                f"final={row['final_trajectory_count']} "
                f"scanned={row['scanned_entries']}"
            )

    print("saved:", OUTPUT_DIR / "cross_range_results.csv")
    compare_with_ours_counts()


# ============================================================
# Fairness check against Ours counts
# ============================================================


def compare_with_ours_counts() -> None:
    baseline_candidate = OUTPUT_DIR / "candidate_count_matrix.csv"
    baseline_final = OUTPUT_DIR / "final_result_count_matrix.csv"
    baseline_results = OUTPUT_DIR / "cross_range_results.csv"

    print()
    print("===== Ours/Baseline Count Check =====")

    required = [
        baseline_candidate,
        baseline_final,
        baseline_results,
        OURS_CANDIDATE_MATRIX_FILE,
        OURS_FINAL_MATRIX_FILE,
        OURS_RESULT_FILE,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("skipped; missing files:")
        for p in missing:
            print(" -", p)
        return

    bc = _load_matrix(baseline_candidate)
    bf = _load_matrix(baseline_final)
    oc = _load_matrix(OURS_CANDIDATE_MATRIX_FILE)
    of = _load_matrix(OURS_FINAL_MATRIX_FILE)
    baseline_queries = _load_query_rows(baseline_results)
    ours_queries = _load_query_rows(OURS_RESULT_FILE)

    candidate_mismatches = []
    final_mismatches = []
    compared = 0

    for key, baseline_query in baseline_queries.items():
        side, label = key
        ours_query = ours_queries.get(key)
        if ours_query is None:
            print(f"skipped {side:g}km x {label}: no corresponding Ours query")
            continue
        # Ours writes coordinates to 10 decimal places. Allow rounding only.
        same_bounds = all(
            math.isclose(
                float(baseline_query[field]), float(ours_query[field]),
                rel_tol=0.0, abs_tol=1e-9,
            )
            for field in (
                "query_min_lon", "query_min_lat", "query_max_lon", "query_max_lat"
            )
        )
        same_time = all(
            int(baseline_query[field]) == int(ours_query[field])
            for field in ("query_start", "query_end")
        )
        if not same_bounds or not same_time:
            print(
                f"skipped {side:g}km x {label}: saved query bounds/time differ; "
                "run Ours with the same query to compare counts"
            )
            continue
        if any(key not in matrix for matrix in (bc, bf, oc, of)):
            print(f"skipped {side:g}km x {label}: count matrix cell missing")
            continue
        compared += 1
        if bc[key] != oc[key]:
            candidate_mismatches.append((key, bc[key], oc[key]))
        if bf[key] != of[key]:
            final_mismatches.append((key, bf[key], of[key]))

    if compared == 0:
        print("Count check skipped: no comparable saved queries with complete counts.")
        return

    print(f"Compared cells: {compared}")
    if not candidate_mismatches:
        print("Candidate counts: MATCH")
    else:
        print("Candidate counts: MISMATCH")
        for (side, label), b, o in candidate_mismatches:
            print(f"  {side:g}km x {label}: baseline={b}, ours={o}")

    if not final_mismatches:
        print("Final counts: MATCH")
    else:
        print("Final counts: MISMATCH")
        for (side, label), b, o in final_mismatches:
            print(f"  {side:g}km x {label}: baseline={b}, ours={o}")

    if candidate_mismatches or final_mismatches:
        raise RuntimeError(
            "Beijing fairness check failed: Baseline/Ours counts differ"
        )

    print(f"All {compared} comparable saved cells match Ours.")


# ============================================================
# Dataset-only sanity check
# ============================================================


def check() -> None:
    _require_file(NODE_FILE)
    _require_file(EDGE_FILE)
    _require_file(TRAJECTORY_FILE)
    _require_file(OURS_SETUP_FILE)

    road = RoadNetwork()
    road.load(str(NODE_FILE), str(EDGE_FILE))

    print("===== Beijing Input Check =====")
    print("nodes:", len(road.valid_node_ids()))
    print("edges:", len(road.valid_edge_ids()))
    print("trajectory file:", TRAJECTORY_FILE)

    center_lon, center_lat, base_start = load_setup(
        OURS_SETUP_FILE, None
    )
    print(f"center_lon = {center_lon:.10f}")
    print(f"center_lat = {center_lat:.10f}")
    print("base_query_start =", base_start)

    # Parse all trajectory rows, map adjacent nodes to EIDs, and validate ordering.
    # This also intentionally builds the in-memory structures, but does not save them.
    store, catalog, ntr, nseg, insertion = build_beijing_store_and_catalog(road)
    print("trajectories =", ntr)
    print("entries =", nseg)
    print("catalog trajectories =", len(catalog.offset_count))
    print("store entries =", store.total_entries())
    print(f"ordered insertion measured = {insertion:.9f} s")
    print("CHECK PASS")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Beijing one-file runner for VTRQ Baseline OrderedInsertion v3"
    )
    ap.add_argument(
        "action",
        choices=("check", "build", "benchmark", "all", "compare"),
        help=(
            "check=input validation; build=build Beijing baseline index; "
            "benchmark=run configured query ranges; all=build+benchmark; "
            "compare=compare candidate/final matrices with Ours"
        ),
    )
    args = ap.parse_args()

    if args.action == "check":
        check()
    elif args.action == "build":
        build_index()
    elif args.action == "benchmark":
        benchmark()
    elif args.action == "all":
        build_index()
        benchmark()
    elif args.action == "compare":
        compare_with_ours_counts()
    else:
        raise AssertionError(args.action)


if __name__ == "__main__":
    main()
