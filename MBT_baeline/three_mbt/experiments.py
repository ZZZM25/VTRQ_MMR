from __future__ import annotations

import csv
import json
import math
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .index import ThreeMBTIndex
from .models import GPSPoint, QueryWindow
from .plaintext_catalog import TrajectoryCatalog, DiskTrajectoryCatalog, run_plaintext_stage

SPATIAL_SIDE_KM = [1, 2, 3, 4, 5]
TEMPORAL_WINDOWS = [
    ("1min", 60),
    ("30min", 30 * 60),
    ("1h", 60 * 60),
    ("4h", 4 * 60 * 60),
    ("12h", 12 * 60 * 60),
    ("24h", 24 * 60 * 60),
]
SCALING_PERCENTAGES = [20, 40, 60, 80, 100]
UPDATE_TRAJECTORY_COUNTS = [1000, 2000, 4000, 8000, 10000]

CITY_QUERY_SETUP = {
    "chengdu": {
        "center_lon": 104.11,
        "center_lat": 30.75,
        "base_query_start": 1541260553,
    },
    "xian": {
        "center_lon": 108.9471716270,
        "center_lat": 34.25752622065,
        "base_query_start": 1538569860,
    },
    "beijing": {
        "center_lon": 116.32962783915,
        "center_lat": 39.98746508120,
        "base_query_start": 1239679020,
    },
}


def make_square(center_lon: float, center_lat: float, side_km: float):
    half_km = side_km / 2.0
    half_lat = half_km / 111.32
    denom = 111.32 * math.cos(math.radians(center_lat))
    if denom <= 0:
        raise ValueError("invalid latitude")
    half_lon = half_km / denom
    return (
        center_lon - half_lon,
        center_lat - half_lat,
        center_lon + half_lon,
        center_lat + half_lat,
    )


def _ordered_trajectory_ids(points: Iterable[GPSPoint]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(points, key=lambda x: x.point_id):
        if p.trajectory_id not in seen:
            seen.add(p.trajectory_id)
            out.append(p.trajectory_id)
    return out


def _write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs)


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _write_matrix(output_dir: Path, filename: str, rows: list[dict], value_key: str) -> None:
    lookup = {(r["temporal_label"], int(r["spatial_side_km"])): r[value_key] for r in rows}
    matrix_rows = []
    for temporal_label, _seconds in TEMPORAL_WINDOWS:
        row: dict[str, object] = {"temporal": temporal_label}
        for side in SPATIAL_SIDE_KM:
            row[f"{side}km"] = lookup[(temporal_label, side)]
        matrix_rows.append(row)
    _write_csv(output_dir / filename, matrix_rows)


def _check_catalog_covers_index(index: ThreeMBTIndex, catalog) -> None:
    # Sequential-scan plaintext catalogs intentionally do not pre-scan the entire
    # CSV before the experiment. Candidate lookup itself checks that every
    # requested trajectory_id exists.
    if getattr(catalog, "sequential_scan_only", False):
        return
    # Large indexed disk catalogs must not be materialized into a Python set.
    if hasattr(catalog, "missing_from"):
        missing = catalog.missing_from(index.trajectory_ids_seen, limit=5)
        if missing:
            raise ValueError(
                f"plaintext catalog misses indexed trajectory IDs; sample={missing}"
            )
        return
    missing = index.trajectory_ids_seen - catalog.trajectory_ids()
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(
            f"plaintext catalog misses {len(missing)} indexed trajectory IDs; sample={preview}"
        )


def run_paper_query_experiment(
    index: ThreeMBTIndex,
    catalog,
    city: str,
    output_dir: str | Path,
    repeats: int = 30,
    warmups: int = 1,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    """Formal 5x6 query experiment.

    Flow per repetition:
      1. server: Lon -> Lat -> Time MBT query, then TID intersection
      2. client: verify three VOs, rebuild three roots, aggregate, compare trusted root
      3. CSV/catalog lookup by verified candidate TIDs (explicitly excluded from Query Time)
      4. client: recompute each plaintext trajectory_id and compare
      5. exact Fine Filtering

    Formal Query Time = coarse MBT query + TID intersection + Fine Filtering.
    CSV lookup is excluded.

    Formal Verification Time = 3 MBT VO/root verification + global-root comparison
                               + client candidate recomputation + plaintext ID verification.
    """
    city = city.lower()
    if city not in CITY_QUERY_SETUP:
        raise ValueError(f"unsupported city: {city}")
    _check_catalog_covers_index(index, catalog)

    setup = CITY_QUERY_SETUP[city]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    trusted_global_root = bytes.fromhex(index.combined_root_hex)
    total_cells = len(SPATIAL_SIDE_KM) * len(TEMPORAL_WINDOWS)
    cell_no = 0
    catalog_total_rows_hint = len(index.trajectory_ids_seen)

    for side_km in SPATIAL_SIDE_KM:
        min_lon, min_lat, max_lon, max_lat = make_square(
            setup["center_lon"], setup["center_lat"], side_km
        )
        for temporal_label, temporal_seconds in TEMPORAL_WINDOWS:
            q = QueryWindow(
                min_lon=min_lon,
                max_lon=max_lon,
                min_lat=min_lat,
                max_lat=max_lat,
                start_time=float(setup["base_query_start"]),
                end_time=float(setup["base_query_start"] + temporal_seconds),
            )
            cell_no += 1
            cell_prefix = f"Cell {cell_no:02d}/{total_cells} | {side_km}km x {temporal_label}"
            if progress is not None:
                progress(f"{cell_prefix} | start", cell_no - 1, total_cells)

            def stage_progress(label: str, done: int, total: int) -> None:
                if progress is not None:
                    progress(f"{cell_prefix} | {label}", done, total)

            for warmup_no in range(1, warmups + 1):
                if progress is not None:
                    progress(f"{cell_prefix} | warmup {warmup_no}/{warmups}", warmup_no - 1, max(warmups, 1))
                warm_result = index.query(q, progress=stage_progress)
                warm_verify = index.verify(
                    warm_result, trusted_global_root, progress=stage_progress
                )
                warm_plain = run_plaintext_stage(
                    catalog,
                    warm_verify.verified_candidate_trajectory_ids,
                    q,
                    progress=stage_progress,
                    catalog_total_rows_hint=catalog_total_rows_hint,
                )
                _ = warm_plain.final_trajectory_ids
                if progress is not None:
                    progress(f"{cell_prefix} | warmup {warmup_no}/{warmups} complete", warmup_no, max(warmups, 1))

            samples: dict[str, list[float]] = defaultdict(list)
            last_result = None
            last_verification = None
            last_plaintext = None

            for repeat_no in range(1, repeats + 1):
                if progress is not None:
                    progress(
                        f"{cell_prefix} | formal repeat {repeat_no}/{repeats}",
                        repeat_no - 1,
                        max(repeats, 1),
                    )
                result = index.query(q, progress=stage_progress)
                verification = index.verify(
                    result, trusted_global_root, progress=stage_progress
                )
                plaintext_stage = run_plaintext_stage(
                    catalog,
                    verification.verified_candidate_trajectory_ids,
                    q,
                    progress=stage_progress,
                    catalog_total_rows_hint=catalog_total_rows_hint,
                )
                if progress is not None:
                    progress(
                        f"{cell_prefix} | formal repeat {repeat_no}/{repeats} complete",
                        repeat_no,
                        max(repeats, 1),
                    )

                last_result = result
                last_verification = verification
                last_plaintext = plaintext_stage

                # Formal timing definition:
                # Query = three MBT range queries + trajectory-ID intersection only.
                # Verification = authenticated VO/root verification + exact trajectory check.
                # Plaintext CSV lookup is excluded from both.
                query_time_s = result.coarse_query_time_s
                verification_time_s = (
                    verification.mbt_verification_time_s
                    + plaintext_stage.fine_filtering_time_s
                )
                filtering_time_s = (
                    result.trajectory_id_intersection_time_s
                    + plaintext_stage.fine_filtering_time_s
                )

                values = {
                    "query_time_s": query_time_s,
                    "coarse_query_time_s": result.coarse_query_time_s,
                    "lon_query_time_s": result.lon_query_time_s,
                    "lat_query_time_s": result.lat_query_time_s,
                    "time_query_time_s": result.time_query_time_s,
                    "trajectory_id_intersection_time_s": result.trajectory_id_intersection_time_s,
                    "filtering_time_s": filtering_time_s,
                    "fine_filtering_time_s": plaintext_stage.fine_filtering_time_s,
                    "exact_result_verification_time_s": plaintext_stage.fine_filtering_time_s,
                    "plaintext_lookup_time_s_excluded": plaintext_stage.lookup_time_s,
                    "verification_time_s": verification_time_s,
                    "mbt_verification_time_s": verification.mbt_verification_time_s,
                    "plaintext_verification_time_s": plaintext_stage.plaintext_verification_time_s,
                    "lon_verification_time_s": verification.lon_verification_time_s,
                    "lat_verification_time_s": verification.lat_verification_time_s,
                    "time_verification_time_s": verification.time_verification_time_s,
                    "root_aggregation_time_s": verification.root_aggregation_time_s,
                    "candidate_recompute_time_s": verification.candidate_recompute_time_s,
                }
                for key, value in values.items():
                    samples[key].append(float(value))

            assert last_result is not None
            assert last_verification is not None
            assert last_plaintext is not None

            vo_size_bytes = last_result.vo_size_bytes(index.combined_root_hex)
            row = {
                "city": city,
                "spatial_side_km": side_km,
                "temporal_label": temporal_label,
                "temporal_seconds": temporal_seconds,
                "query_min_lon": min_lon,
                "query_min_lat": min_lat,
                "query_max_lon": max_lon,
                "query_max_lat": max_lat,
                "query_start": int(q.start_time),
                "query_end": int(q.end_time),
                "query_time_s": _mean(samples["query_time_s"]),
                "query_time_std_s": _std(samples["query_time_s"]),
                "coarse_query_time_s": _mean(samples["coarse_query_time_s"]),
                "verification_time_s": _mean(samples["verification_time_s"]),
                "verification_time_std_s": _std(samples["verification_time_s"]),
                "mbt_verification_time_s": _mean(samples["mbt_verification_time_s"]),
                "plaintext_verification_time_s": _mean(samples["plaintext_verification_time_s"]),
                "vo_size_bytes": vo_size_bytes,
                "vo_size_mb": vo_size_bytes / 1_000_000.0,
                "lon_query_time_s": _mean(samples["lon_query_time_s"]),
                "lat_query_time_s": _mean(samples["lat_query_time_s"]),
                "time_query_time_s": _mean(samples["time_query_time_s"]),
                "trajectory_id_intersection_time_s": _mean(samples["trajectory_id_intersection_time_s"]),
                "filtering_time_s": _mean(samples["filtering_time_s"]),
                "fine_filtering_time_s": _mean(samples["fine_filtering_time_s"]),
                "exact_result_verification_time_s": _mean(samples["exact_result_verification_time_s"]),
                "plaintext_lookup_time_s_excluded": _mean(samples["plaintext_lookup_time_s_excluded"]),
                "lon_verification_time_s": _mean(samples["lon_verification_time_s"]),
                "lat_verification_time_s": _mean(samples["lat_verification_time_s"]),
                "time_verification_time_s": _mean(samples["time_verification_time_s"]),
                "root_aggregation_time_s": _mean(samples["root_aggregation_time_s"]),
                "candidate_recompute_time_s": _mean(samples["candidate_recompute_time_s"]),
                "lon_trajectory_count": len(last_result.lon_trajectory_ids),
                "lat_trajectory_count": len(last_result.lat_trajectory_ids),
                "time_trajectory_count": len(last_result.time_trajectory_ids),
                "candidate_trajectory_count": len(last_verification.verified_candidate_trajectory_ids),
                "final_result_count": len(last_plaintext.final_trajectory_ids),
            }
            rows.append(row)
            if progress is not None:
                progress(
                    f"{cell_prefix} | complete | candidates={row['candidate_trajectory_count']} final={row['final_result_count']}",
                    cell_no,
                    total_cells,
                )

    _write_csv(output_dir / "cross_range_results.csv", rows)
    matrix_fields = {
        "query_time_matrix_s.csv": "query_time_s",
        "verification_time_matrix_s.csv": "verification_time_s",
        "vo_size_matrix_mb.csv": "vo_size_mb",
        "lon_query_time_matrix_s.csv": "lon_query_time_s",
        "lat_query_time_matrix_s.csv": "lat_query_time_s",
        "time_query_time_matrix_s.csv": "time_query_time_s",
        "trajectory_id_intersection_time_matrix_s.csv": "trajectory_id_intersection_time_s",
        "filtering_time_matrix_s.csv": "filtering_time_s",
        "fine_filtering_time_matrix_s.csv": "fine_filtering_time_s",
        "exact_result_verification_time_matrix_s.csv": "exact_result_verification_time_s",
        "plaintext_lookup_time_excluded_matrix_s.csv": "plaintext_lookup_time_s_excluded",
        "mbt_verification_time_matrix_s.csv": "mbt_verification_time_s",
        "plaintext_verification_time_matrix_s.csv": "plaintext_verification_time_s",
        "lon_verification_time_matrix_s.csv": "lon_verification_time_s",
        "lat_verification_time_matrix_s.csv": "lat_verification_time_s",
        "time_verification_time_matrix_s.csv": "time_verification_time_s",
        "candidate_count_matrix.csv": "candidate_trajectory_count",
        "final_result_count_matrix.csv": "final_result_count",
    }
    for filename, value_key in matrix_fields.items():
        _write_matrix(output_dir, filename, rows, value_key)

    experiment_setup = {
        "city": city,
        **setup,
        "spatial_side_km": SPATIAL_SIDE_KM,
        "temporal_windows": [{"label": x, "seconds": y} for x, y in TEMPORAL_WINDOWS],
        "formal_repeats": repeats,
        "warmup_repeats": warmups,
        "query_order": ["MBT-Lon", "MBT-Lat", "MBT-Time"],
        "coarse_filter": "T_lon INTERSECT T_lat INTERSECT T_time (trajectory IDs)",
        "query_time_definition": (
            "Lon MBT + Lat MBT + Time MBT + trajectory-ID intersection + exact Fine Filtering; "
            "plaintext CSV/catalog lookup excluded"
        ),
        "verification_time_definition": (
            "three MBT VO/path/root reconstruction + combined-root comparison + client candidate "
            "recomputation + plaintext trajectory-ID verification"
        ),
        "plaintext_lookup_in_query_time": False,
        "trusted_combined_root": index.combined_root_hex,
    }
    (output_dir / "experiment_setup.json").write_text(
        json.dumps(experiment_setup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def build_index_with_formal_timing(
    points: list[GPSPoint],
    leaf_capacity: int = 128,
    fanout: int = 64,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[ThreeMBTIndex, float]:
    """Only the three MBT builds + combined root are timed.

    Progress printing is deliberately invoked outside each timed chunk, so enabling
    the build progress bar does not add terminal-I/O time to formal construction time.
    """
    idx = ThreeMBTIndex(leaf_capacity, fanout)
    formal_elapsed = 0.0

    stages = [
        ("MBT-Lon", idx.mbt_lon.bulk_build),
        ("MBT-Lat", idx.mbt_lat.bulk_build),
        ("MBT-Time", idx.mbt_time.bulk_build),
    ]
    for stage_no, (label, builder) in enumerate(stages, start=1):
        if progress_callback is not None:
            progress_callback(f"{label} building", stage_no - 1, 3)
        t0 = time.perf_counter()
        builder(points)
        formal_elapsed += time.perf_counter() - t0
        if progress_callback is not None:
            progress_callback(f"{label} complete", stage_no, 3)

    # Keep the same metadata/root work inside formal timing as the previous build
    # implementation, but exclude progress printing itself.
    t0 = time.perf_counter()
    idx.point_count = len(points)
    idx.trajectory_ids_seen = {p.trajectory_id for p in points}
    _ = idx.combined_root
    formal_elapsed += time.perf_counter() - t0

    return idx, formal_elapsed


def run_scalability_experiment(
    points: list[GPSPoint],
    output_dir: str | Path,
    city: str,
    percentages: list[int] | None = None,
    leaf_capacity: int = 128,
    fanout: int = 64,
) -> list[dict]:
    percentages = percentages or list(SCALING_PERCENTAGES)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tids = _ordered_trajectory_ids(points)
    rows: list[dict] = []

    for percent in percentages:
        n = max(1, math.floor(len(tids) * percent / 100.0))
        chosen = set(tids[:n])
        subset = [p for p in points if p.trajectory_id in chosen]
        idx, construction_time = build_index_with_formal_timing(
            subset, leaf_capacity, fanout
        )
        index_size_bytes = idx.serialized_size_bytes()
        rows.append(
            {
                "city": city,
                "percent": percent,
                "trajectory_count": n,
                "point_count": len(subset),
                "construction_time_s": construction_time,
                "index_size_bytes": index_size_bytes,
                "index_size_mib": index_size_bytes / (1024.0 * 1024.0),
                "combined_root": idx.combined_root_hex,
            }
        )

    _write_csv(output_dir / "index_scaling_results.csv", rows)
    _write_csv(
        output_dir / "construction_time_matrix_s.csv",
        [{"city": city, **{f"{r['percent']}%": r["construction_time_s"] for r in rows}}],
    )
    _write_csv(
        output_dir / "index_size_matrix_mib.csv",
        [{"city": city, **{f"{r['percent']}%": r["index_size_mib"] for r in rows}}],
    )
    return rows


def run_update_experiment(
    points: list[GPSPoint],
    output_dir: str | Path,
    city: str,
    base_fraction: float = 0.60,
    update_trajectory_counts: list[int] | None = None,
    leaf_capacity: int = 128,
    fanout: int = 64,
) -> list[dict]:
    update_trajectory_counts = update_trajectory_counts or list(UPDATE_TRAJECTORY_COUNTS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tids = _ordered_trajectory_ids(points)
    base_n = max(1, math.floor(len(tids) * base_fraction))
    base_tids = set(tids[:base_n])
    remaining = tids[base_n:]
    base_points = [p for p in points if p.trajectory_id in base_tids]

    base_idx, _ = build_index_with_formal_timing(base_points, leaf_capacity, fanout)
    base_blob = pickle.dumps(base_idx, protocol=pickle.HIGHEST_PROTOCOL)
    rows: list[dict] = []

    for requested in update_trajectory_counts:
        actual = min(requested, len(remaining))
        chosen = set(remaining[:actual])
        update_points = [p for p in points if p.trajectory_id in chosen]
        idx: ThreeMBTIndex = pickle.loads(base_blob)
        old_root = idx.combined_root_hex

        t0 = time.perf_counter()
        idx.incremental_insert_points(update_points)
        _ = idx.combined_root
        t1 = time.perf_counter()

        rows.append(
            {
                "city": city,
                "base_fraction": base_fraction,
                "base_trajectory_count": base_n,
                "requested_update_trajectories": requested,
                "actual_update_trajectories": actual,
                "update_point_count": len(update_points),
                "update_time_s": t1 - t0,
                "old_combined_root": old_root,
                "new_combined_root": idx.combined_root_hex,
            }
        )

    _write_csv(output_dir / "index_update_results.csv", rows)
    _write_csv(
        output_dir / "update_time_matrix_s.csv",
        [{"city": city, **{f"+{r['requested_update_trajectories']}": r["update_time_s"] for r in rows}}],
    )
    return rows
