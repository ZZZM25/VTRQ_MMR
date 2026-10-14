from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from pathlib import Path

from .build import build_index, index_size_bytes
from .catalog import run_exact_stage
from .models import QueryWindow
from .prepare import load_prepared_metadata, point_prefix
from .tree import AuthenticatedPointRTree
from .update import incremental_update
from .vo import verify_vo

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

# Exact centers / starts used by the Three-MBT experiment grid.
CITY_QUERY_SETUP = {
    "chengdu": {
        "center_lon": 104.0540760857,
        "center_lat": 30.67289263645,
        "base_query_start": 1541260553,
    },
    "xian": {
        "center_lon": 108.95,
        "center_lat": 34.25,
        "base_query_start": 1539619200,
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


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs)


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _matrix(rows: list[dict], value_key: str) -> list[dict]:
    lookup = {(int(r["spatial_side_km"]), r["temporal_label"]): r[value_key] for r in rows}
    out = []
    for side in SPATIAL_SIDE_KM:
        row = {"spatial": f"{side}km"}
        for label, _ in TEMPORAL_WINDOWS:
            row[label] = lookup[(side, label)]
        out.append(row)
    return out


def run_query_experiment(
    index_dir: str | Path,
    catalog_path: str | Path,
    city: str,
    output_dir: str | Path,
    repeats: int = 1,
    warmups: int = 0,
) -> list[dict]:
    city = city.lower()
    setup = CITY_QUERY_SETUP[city]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    tree = AuthenticatedPointRTree(index_dir)
    rows: list[dict] = []
    total_cells = len(SPATIAL_SIDE_KM) * len(TEMPORAL_WINDOWS)
    cell = 0

    for side in SPATIAL_SIDE_KM:
        min_lon, min_lat, max_lon, max_lat = make_square(
            setup["center_lon"], setup["center_lat"], side
        )
        for temporal_label, seconds in TEMPORAL_WINDOWS:
            cell += 1
            q = QueryWindow(
                min_lon=min_lon,
                min_lat=min_lat,
                max_lon=max_lon,
                max_lat=max_lat,
                start_time=float(setup["base_query_start"]),
                end_time=float(setup["base_query_start"] + seconds),
            )
            print(f"[Query] {city} cell {cell}/{total_cells} | {side}km x {temporal_label} | start")

            for _ in range(warmups):
                qr = tree.query(q, temp_dir)
                vv = verify_vo(qr.vo_path, q, tree.root_hash, qr.candidate_trajectory_ids)
                ex = run_exact_stage(catalog_path, vv.verified_candidate_trajectory_ids, q, temp_dir)
                Path(qr.vo_path).unlink(missing_ok=True)
                _ = ex.final_trajectory_ids

            q_times: list[float] = []
            coarse_query_times: list[float] = []
            v_times: list[float] = []
            vo_verify_times: list[float] = []
            exact_times: list[float] = []
            lookup_times: list[float] = []
            id_times: list[float] = []
            last_qr = None
            last_ex = None

            for rep in range(repeats):
                qr = tree.query(q, temp_dir)
                vv = verify_vo(qr.vo_path, q, tree.root_hash, qr.candidate_trajectory_ids)
                ex = run_exact_stage(catalog_path, vv.verified_candidate_trajectory_ids, q, temp_dir)

                # Unified paper timing semantics (same as Ours/MMR):
                # Query = R-tree coarse query + VO generation
                #       + exact complete-trajectory fine filtering.
                # Verification = authenticated VO/root/candidate verification only.
                # Plaintext sequential lookup and trajectory-ID diagnostic are excluded.
                coarse_query_times.append(qr.query_time_s)
                exact_times.append(ex.exact_result_verification_time_s)
                q_times.append(qr.query_time_s + ex.exact_result_verification_time_s)
                vo_verify_times.append(vv.vo_verification_time_s)
                v_times.append(vv.vo_verification_time_s)
                lookup_times.append(ex.lookup_time_s)
                id_times.append(ex.plaintext_id_check_time_s)
                last_qr = qr
                last_ex = ex
                print(
                    f"[Query] {city} {side}km x {temporal_label} | repeat {rep+1}/{repeats} complete | "
                    f"space-points={qr.spatial_match_point_count:,} | "
                    f"space+time-points={qr.spatiotemporal_match_point_count:,} | "
                    f"candidate={len(qr.candidate_trajectory_ids):,} | final={len(ex.final_trajectory_ids):,}"
                )
                Path(qr.vo_path).unlink(missing_ok=True)

            assert last_qr is not None and last_ex is not None
            row = {
                "city": city,
                "spatial_side_km": side,
                "temporal_label": temporal_label,
                "temporal_seconds": seconds,
                "query_min_lon": min_lon,
                "query_min_lat": min_lat,
                "query_max_lon": max_lon,
                "query_max_lat": max_lat,
                "query_start": int(q.start_time),
                "query_end": int(q.end_time),
                "query_time_s": _mean(q_times),
                "query_time_std_s": _std(q_times),
                "rtree_coarse_query_time_s": _mean(coarse_query_times),
                "fine_filtering_time_s": _mean(exact_times),
                "verification_time_s": _mean(v_times),
                "verification_time_std_s": _std(v_times),
                "rtree_vo_verification_time_s": _mean(vo_verify_times),
                "exact_result_verification_time_s": _mean(exact_times),
                "plaintext_lookup_time_s_excluded": _mean(lookup_times),
                "plaintext_id_check_time_s_excluded": _mean(id_times),
                "vo_size_bytes": last_qr.vo_size_bytes,
                "vo_size_mb": last_qr.vo_size_bytes / 1_000_000.0,
                "expanded_leaf_count": last_qr.expanded_leaf_count,
                "spatial_match_point_count": last_qr.spatial_match_point_count,
                "spatiotemporal_match_point_count": last_qr.spatiotemporal_match_point_count,
                "candidate_trajectory_count": len(last_qr.candidate_trajectory_ids),
                "final_result_count": len(last_ex.final_trajectory_ids),
            }
            rows.append(row)
            _write_csv(output_dir / "cross_range_results.csv", rows)
            print(f"[Query] {city} cell {cell}/{total_cells} | complete")

    _write_csv(output_dir / "query_time_matrix_s.csv", _matrix(rows, "query_time_s"))
    _write_csv(output_dir / "verification_time_matrix_s.csv", _matrix(rows, "verification_time_s"))
    _write_csv(output_dir / "vo_size_matrix_mb.csv", _matrix(rows, "vo_size_mb"))
    (output_dir / "experiment_setup.json").write_text(
        json.dumps(
            {
                "city": city,
                "spatial_side_km": SPATIAL_SIDE_KM,
                "temporal_windows": TEMPORAL_WINDOWS,
                "query_setup": setup,
                "formal_repeats": repeats,
                "warmup_repeats": warmups,
                "candidate_rule": "same leaf point must satisfy lon/lat spatial rectangle AND timestamp interval before trajectory_id is emitted",
                "query_time": "R-tree traversal + leaf spatial/time checks + candidate TID dedup + VO generation + exact complete-trajectory fine filtering",
                "verification_time": "VO/root/candidate verification only",
                "plaintext_lookup": "excluded from Query and Verification",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    return rows


def run_scaling_experiment(
    prepared_dir: str | Path,
    output_dir: str | Path,
    city: str,
    leaf_capacity: int,
    fanout: int,
    sort_chunk_rows: int = 500_000,
) -> list[dict]:
    prepared_dir = Path(prepared_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prep = load_prepared_metadata(prepared_dir)
    total = int(prep["trajectory_count"])
    rows: list[dict] = []

    for pct in SCALING_PERCENTAGES:
        traj_n = max(1, math.floor(total * pct / 100.0))
        tmp_index = output_dir / f"_tmp_{pct}pct"
        print(f"[Scaling] {city}: {pct}% = {traj_n:,} trajectories")
        meta, elapsed = build_index(
            prepared_dir,
            tmp_index,
            trajectory_count=traj_n,
            leaf_capacity=leaf_capacity,
            fanout=fanout,
            sort_chunk_rows=sort_chunk_rows,
        )
        size = index_size_bytes(tmp_index)
        row = {
            "city": city,
            "percent": pct,
            "trajectory_count": traj_n,
            "point_count": meta["point_count"],
            "construction_time_s": elapsed,
            "index_size_bytes": size,
            "index_size_mib": size / (1024.0 * 1024.0),
            "root_hash": meta["root_hash_hex"],
        }
        rows.append(row)
        _write_csv(output_dir / "index_scaling_results.csv", rows)
        shutil.rmtree(tmp_index, ignore_errors=True)

    _write_csv(
        output_dir / "construction_time_matrix_s.csv",
        [{"percent": r["percent"], "construction_time_s": r["construction_time_s"]} for r in rows],
    )
    _write_csv(
        output_dir / "index_size_matrix_mib.csv",
        [{"percent": r["percent"], "index_size_mib": r["index_size_mib"]} for r in rows],
    )
    return rows


def run_update_experiment(
    prepared_dir: str | Path,
    output_dir: str | Path,
    city: str,
    leaf_capacity: int,
    fanout: int,
    sort_chunk_rows: int = 500_000,
) -> list[dict]:
    prepared_dir = Path(prepared_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prep = load_prepared_metadata(prepared_dir)
    total = int(prep["trajectory_count"])
    base_n = max(1, math.floor(total * 0.60))
    base_points = point_prefix(prepared_dir, base_n)
    work = output_dir / "_work"
    base_index = work / "base_60pct"
    work.mkdir(parents=True, exist_ok=True)

    print(
        f"[Update] {city}: building fixed 60% base index: "
        f"{base_n:,} trajectories / {base_points:,} points (excluded from update timing)"
    )
    base_meta, _base_build_time = build_index(
        prepared_dir,
        base_index,
        trajectory_count=base_n,
        leaf_capacity=leaf_capacity,
        fanout=fanout,
        sort_chunk_rows=sort_chunk_rows,
    )

    rows: list[dict] = []
    try:
        for requested in UPDATE_TRAJECTORY_COUNTS:
            actual = min(requested, total - base_n)
            if actual <= 0:
                continue
            case_work = work / f"update_{requested}"
            print(f"[Update] {city}: +{requested:,} from the same fixed 60% base")
            updated, elapsed = incremental_update(
                prepared_dir,
                base_index,
                base_n,
                requested,
                case_work,
                leaf_capacity=leaf_capacity,
                fanout=fanout,
            )
            row = {
                "city": city,
                "base_fraction": 0.6,
                "base_trajectory_count": base_n,
                "base_point_count": base_meta["point_count"],
                "requested_update_trajectories": requested,
                "actual_update_trajectories": updated["actual_update_trajectories"],
                "update_point_count": updated["update_point_count"],
                "update_time_s": elapsed,
                "affected_base_leaf_count": updated["affected_base_leaf_count"],
                "result_leaf_count": updated["result_leaf_count"],
                "old_root_hash": updated["old_root_hash_hex"],
                "new_root_hash": updated["new_root_hash_hex"],
            }
            rows.append(row)
            _write_csv(output_dir / "index_update_results.csv", rows)
            shutil.rmtree(case_work, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    _write_csv(
        output_dir / "update_time_matrix_s.csv",
        [{"inserted_trajectories": r["requested_update_trajectories"], "update_time_s": r["update_time_s"]} for r in rows],
    )
    return rows
