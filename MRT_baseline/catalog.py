from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .geometry import trajectory_intersects_query
from .models import ExactVerificationResult, QueryWindow
from .trajectory_id import compute_trajectory_id


def _raise_csv_field_limit() -> None:
    """Allow plaintext trajectory payloads larger than Python csv's 128 KiB default.

    Full trajectories are intentionally stored in one CSV field. Large Chengdu/Xi'an
    trajectories can therefore exceed the default 131072-byte parser limit.
    This changes only the CSV parser limit; it does not change experiment timing or data.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _gps_points(points_format: str, points_data: list) -> list[tuple[float, float, float]]:
    if points_format == "time_lat_lon":
        return [(float(p[0]), float(p[1]), float(p[2])) for p in points_data]
    if points_format == "node_lon_lat_time":
        return [(float(p[3]), float(p[2]), float(p[1])) for p in points_data]
    raise ValueError(f"unsupported points_format: {points_format}")


def materialize_candidates(
    catalog_path: str | Path,
    candidate_ids: list[str],
    temp_dir: str | Path,
) -> tuple[Path, float]:
    """One sequential plaintext scan. Entire time is excluded from formal metrics."""
    catalog_path = Path(catalog_path)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    needed = set(candidate_ids)

    fd, temp_name = tempfile.mkstemp(prefix="candidate_payloads_", suffix=".csv", dir=temp_dir)
    os.close(fd)
    Path(temp_name).unlink(missing_ok=True)
    temp_path = Path(temp_name)

    _raise_csv_field_limit()
    t0 = time.perf_counter()
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as src, temp_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        found: set[str] = set()
        if needed:
            for row in reader:
                tid = str(row["trajectory_id"]).strip()
                if tid in needed:
                    writer.writerow(row)
                    found.add(tid)
                    if len(found) == len(needed):
                        break
        if found != needed:
            missing = sorted(needed - found)[:5]
            raise ValueError(f"plaintext catalog missing candidate IDs: {missing}")
    return temp_path, time.perf_counter() - t0


def exact_verify_materialized(
    materialized_path: str | Path,
    query: QueryWindow,
) -> tuple[list[str], float, float]:
    """Exact full-trajectory verification.

    JSON/CSV decoding is outside the measured computation timer. The trajectory-ID
    recomputation is diagnostic only and excluded from formal Verification Time.
    """
    final_ids: list[str] = []
    id_time = 0.0
    exact_time = 0.0

    _raise_csv_field_limit()
    with Path(materialized_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row["trajectory_id"]).strip()
            node_path = [int(x) for x in json.loads(row["road_node_path"])]
            fmt = str(row["points_format"]).strip()
            points_data = json.loads(row["points_data"])
            gps = _gps_points(fmt, points_data)

            t0 = time.perf_counter()
            computed = compute_trajectory_id(node_path, int(gps[0][0]), int(gps[-1][0]))
            id_time += time.perf_counter() - t0
            if computed != tid:
                raise ValueError(f"trajectory_id mismatch: {tid}")

            t1 = time.perf_counter()
            matched = trajectory_intersects_query(gps, query)
            exact_time += time.perf_counter() - t1
            if matched:
                final_ids.append(tid)

    final_ids.sort()
    return final_ids, id_time, exact_time


def run_exact_stage(
    catalog_path: str | Path,
    candidate_ids: list[str],
    query: QueryWindow,
    temp_dir: str | Path,
) -> ExactVerificationResult:
    temp_path, lookup_time = materialize_candidates(catalog_path, candidate_ids, temp_dir)
    try:
        final_ids, id_time, exact_time = exact_verify_materialized(temp_path, query)
    finally:
        temp_path.unlink(missing_ok=True)
    return ExactVerificationResult(
        final_trajectory_ids=final_ids,
        lookup_time_s=lookup_time,
        plaintext_id_check_time_s=id_time,
        exact_result_verification_time_s=exact_time,
    )
