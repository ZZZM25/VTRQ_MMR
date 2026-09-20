from __future__ import annotations

import csv
import json
import math
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Callable

from .datasets import iter_dataset_records
from .disk_index import DiskThreeMBTIndex
from .external_build import (
    _emit_internal_progress,
    _merged_runs,
    _open_point_store,
    _write_sorted_runs,
)

StatusCallback = Callable[[str], None]
LoadProgress = Callable[[int, int, Path], None]
BuildProgress = Callable[[str, int, int], None]
InternalProgress = Callable[[str, str, int, int], None]

SCALING_PERCENTAGES = [20, 40, 60, 80, 100]
UPDATE_TRAJECTORY_COUNTS = [1000, 2000, 4000, 8000, 10000]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_point_store(
    *,
    input_path: str | Path,
    dataset: str,
    temp_db: Path,
    beijing_subdivisions: int,
    legacy_sidecar: str | Path | None,
    load_progress: LoadProgress | None,
) -> tuple[sqlite3.Connection, list[str], list[int]]:
    """Parse raw data once into a disk point store; this is outside formal timing.

    Returns (connection, trajectory_ids, cumulative_point_counts), where
    cumulative_point_counts[n] is the point count in the first n trajectories.
    """
    temp_db.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_point_store(temp_db)
    trajectory_ids: list[str] = []
    prefix_points: list[int] = [0]
    batch: list[tuple[int, int, float, float, float]] = []
    batch_size = 50_000
    point_id = 0

    try:
        conn.execute("BEGIN")
        for rec in iter_dataset_records(
            input_path,
            dataset,
            beijing_subdivisions=beijing_subdivisions,
            legacy_sidecar=legacy_sidecar,
            file_progress=load_progress,
        ):
            tid_idx = len(trajectory_ids)
            if tid_idx >= 2**32:
                raise OverflowError("trajectory count exceeds uint32 disk format")
            trajectory_ids.append(rec.trajectory_id)
            for ts, lat, lon in rec.gps_points:
                batch.append((point_id, tid_idx, float(lon), float(lat), float(ts)))
                point_id += 1
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT INTO points(point_id,tid_idx,lon,lat,ts) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
            prefix_points.append(point_id)
        if batch:
            conn.executemany(
                "INSERT INTO points(point_id,tid_idx,lon,lat,ts) VALUES (?,?,?,?,?)",
                batch,
            )
            batch.clear()
        conn.commit()
        return conn, trajectory_ids, prefix_points
    except Exception:
        conn.close()
        raise


def _build_subset_from_store(
    *,
    conn: sqlite3.Connection,
    all_trajectory_ids: list[str],
    trajectory_limit: int,
    point_count: int,
    index_path: Path,
    leaf_capacity: int,
    fanout: int,
    build_progress: BuildProgress | None,
    internal_progress: InternalProgress | None,
) -> tuple[DiskThreeMBTIndex, float]:
    """Build a prefix subset from the prepared store with paper construction timing."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(str(index_path) + ".data")
    if index_path.exists():
        index_path.unlink()
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    trajectory_ids = list(all_trajectory_ids[:trajectory_limit])
    idx = DiskThreeMBTIndex(trajectory_ids, leaf_capacity, fanout)
    idx.bind_storage(storage_dir)
    idx.point_count = int(point_count)

    formal_elapsed = 0.0
    stages = [
        ("MBT-Lon", "lon", idx.mbt_lon, storage_dir / "lon.entries.bin"),
        ("MBT-Lat", "lat", idx.mbt_lat, storage_dir / "lat.entries.bin"),
        ("MBT-Time", "ts", idx.mbt_time, storage_dir / "time.entries.bin"),
    ]
    for stage_no, (label, column, tree, entries_path) in enumerate(stages, start=1):
        if build_progress is not None:
            build_progress(f"{label} building", stage_no - 1, 3)
        run_dir = storage_dir / f"_{column}_sort_runs"
        t0 = time.perf_counter()
        run_paths, ui_overhead = _write_sorted_runs(
            conn,
            column=column,
            run_dir=run_dir,
            label=label,
            point_count=point_count,
            internal_progress=internal_progress,
            where_sql="tid_idx < ?",
            where_params=(int(trajectory_limit),),
        )
        merge_ui_overhead = 0.0

        def merge_progress(done: int, total: int) -> None:
            nonlocal merge_ui_overhead
            merge_ui_overhead += _emit_internal_progress(
                internal_progress, label, "Merge", done, total
            )

        tree.build_from_sorted_rows(
            _merged_runs(run_paths),
            entries_path,
            total_rows=point_count,
            progress=merge_progress if internal_progress is not None else None,
        )
        elapsed = time.perf_counter() - t0
        formal_elapsed += max(0.0, elapsed - ui_overhead - merge_ui_overhead)
        shutil.rmtree(run_dir, ignore_errors=True)
        if internal_progress is not None:
            internal_progress(label, "Tree", 1, 1)
        if build_progress is not None:
            build_progress(f"{label} complete", stage_no, 3)

    t0 = time.perf_counter()
    _ = idx.combined_root
    formal_elapsed += time.perf_counter() - t0
    idx.save(index_path)
    return idx, formal_elapsed


def run_disk_scalability_experiment(
    *,
    input_path: str | Path,
    dataset: str,
    output_dir: str | Path,
    leaf_capacity: int = 128,
    fanout: int = 64,
    percentages: list[int] | None = None,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
    load_progress: LoadProgress | None = None,
    build_progress: BuildProgress | None = None,
    internal_progress: InternalProgress | None = None,
    status: StatusCallback | None = None,
) -> list[dict]:
    percentages = percentages or list(SCALING_PERCENTAGES)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_db = work_dir / "all_points.sqlite"

    if status:
        status("loading raw dataset once into disk point store (excluded from construction timing)")
    conn, trajectory_ids, prefix_points = _prepare_point_store(
        input_path=input_path,
        dataset=dataset,
        temp_db=temp_db,
        beijing_subdivisions=beijing_subdivisions,
        legacy_sidecar=legacy_sidecar,
        load_progress=load_progress,
    )
    rows: list[dict] = []
    total_trajectories = len(trajectory_ids)
    try:
        for percent in percentages:
            n = max(1, math.floor(total_trajectories * int(percent) / 100.0))
            point_count = prefix_points[n]
            if status:
                status(
                    f"{percent}%: building {n:,} trajectories / {point_count:,} indexed points"
                )
            index_path = work_dir / f"{dataset}_{percent}pct_three_mbt.pkl"
            idx, construction_time = _build_subset_from_store(
                conn=conn,
                all_trajectory_ids=trajectory_ids,
                trajectory_limit=n,
                point_count=point_count,
                index_path=index_path,
                leaf_capacity=leaf_capacity,
                fanout=fanout,
                build_progress=build_progress,
                internal_progress=internal_progress,
            )
            size_bytes = idx.disk_size_bytes(index_path)
            rows.append(
                {
                    "city": dataset,
                    "percent": int(percent),
                    "trajectory_count": n,
                    "point_count": point_count,
                    "construction_time_s": construction_time,
                    "index_size_bytes": size_bytes,
                    "index_size_mib": size_bytes / (1024.0 * 1024.0),
                    "combined_root": idx.combined_root_hex,
                }
            )
            # The experiment retains exact metrics/roots, not five huge temporary indexes.
            try:
                index_path.unlink()
            except FileNotFoundError:
                pass
            shutil.rmtree(Path(str(index_path) + ".data"), ignore_errors=True)

        _write_csv(output_dir / "index_scaling_results.csv", rows)
        _write_csv(
            output_dir / "construction_time_matrix_s.csv",
            [{"city": dataset, **{f"{r['percent']}%": r["construction_time_s"] for r in rows}}],
        )
        _write_csv(
            output_dir / "index_size_matrix_mib.csv",
            [{"city": dataset, **{f"{r['percent']}%": r["index_size_mib"] for r in rows}}],
        )
        (output_dir / "experiment_setup.json").write_text(
            json.dumps(
                {
                    "city": dataset,
                    "percentages": percentages,
                    "leaf_capacity": leaf_capacity,
                    "fanout": fanout,
                    "sampling_rule": "natural trajectory order; first floor(N*percent/100) trajectories",
                    "timed_scope": "per-dimension external sort of selected points + packed MBT leaf/hash build + combined root; raw I/O/parsing excluded",
                    "temporary_indexes_retained": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return rows
    finally:
        conn.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def _truncate_to_sizes(index: DiskThreeMBTIndex, sizes: dict[str, int]) -> None:
    for tree in (index.mbt_lon, index.mbt_lat, index.mbt_time):
        path = tree.entries_path
        with path.open("r+b") as f:
            f.truncate(sizes[tree.dimension])


def run_disk_update_experiment(
    *,
    input_path: str | Path,
    dataset: str,
    output_dir: str | Path,
    leaf_capacity: int = 128,
    fanout: int = 64,
    base_fraction: float = 0.60,
    update_trajectory_counts: list[int] | None = None,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
    load_progress: LoadProgress | None = None,
    status: StatusCallback | None = None,
) -> list[dict]:
    """Disk-backed 60%-base incremental update experiment.

    Raw parsing and the 60% base build are setup and excluded from Update Time.
    For each requested batch, Update Time includes sorting only the new points,
    inserting them into touched packed leaves (append/split), rebuilding Merkle
    internal metadata, and recomputing the combined root.  The base index itself is
    not rebuilt from raw points for any update batch.
    """
    update_trajectory_counts = update_trajectory_counts or list(UPDATE_TRAJECTORY_COUNTS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_db = work_dir / "all_points.sqlite"

    if status:
        status("loading raw dataset once into disk point store (excluded from update timing)")
    conn, trajectory_ids, prefix_points = _prepare_point_store(
        input_path=input_path,
        dataset=dataset,
        temp_db=temp_db,
        beijing_subdivisions=beijing_subdivisions,
        legacy_sidecar=legacy_sidecar,
        load_progress=load_progress,
    )
    total_trajectories = len(trajectory_ids)
    base_n = max(1, math.floor(total_trajectories * float(base_fraction)))
    base_point_count = prefix_points[base_n]
    base_path = work_dir / f"{dataset}_base60_three_mbt.pkl"

    try:
        if status:
            status(
                f"building 60% base index: {base_n:,} trajectories / {base_point_count:,} points (excluded from update timing)"
            )
        base_idx, _base_build_time = _build_subset_from_store(
            conn=conn,
            all_trajectory_ids=trajectory_ids,
            trajectory_limit=base_n,
            point_count=base_point_count,
            index_path=base_path,
            leaf_capacity=leaf_capacity,
            fanout=fanout,
            build_progress=None,
            internal_progress=None,
        )
        old_root = base_idx.combined_root_hex
        base_sizes = {
            tree.dimension: tree.entries_path.stat().st_size
            for tree in (base_idx.mbt_lon, base_idx.mbt_lat, base_idx.mbt_time)
        }
        del base_idx

        rows: list[dict] = []
        remaining = max(0, total_trajectories - base_n)
        for requested in update_trajectory_counts:
            actual = min(int(requested), remaining)
            end_tid = base_n + actual
            update_point_count = prefix_points[end_tid] - prefix_points[base_n]
            if status:
                status(
                    f"+{requested:,}: incremental insert {actual:,} trajectories / {update_point_count:,} points"
                )

            idx = DiskThreeMBTIndex.load(base_path)
            idx.trajectory_ids.extend(trajectory_ids[base_n:end_tid])
            idx.trajectory_ids_seen = set(idx.trajectory_ids)
            for tree in (idx.mbt_lon, idx.mbt_lat, idx.mbt_time):
                tree.trajectory_ids = idx.trajectory_ids
                if hasattr(tree, "_vo_tid_json_sizes"):
                    delattr(tree, "_vo_tid_json_sizes")

            total_elapsed = 0.0
            try:
                stages = [
                    ("MBT-Lon", "lon", idx.mbt_lon),
                    ("MBT-Lat", "lat", idx.mbt_lat),
                    ("MBT-Time", "ts", idx.mbt_time),
                ]
                for label, column, tree in stages:
                    run_dir = work_dir / f"update_{requested}_{column}_runs"
                    t0 = time.perf_counter()
                    run_paths, _ui = _write_sorted_runs(
                        conn,
                        column=column,
                        run_dir=run_dir,
                        label=label,
                        point_count=update_point_count,
                        internal_progress=None,
                        where_sql="tid_idx >= ? AND tid_idx < ?",
                        where_params=(base_n, end_tid),
                    )
                    inserted = tree.batch_insert_sorted_rows(_merged_runs(run_paths))
                    total_elapsed += time.perf_counter() - t0
                    shutil.rmtree(run_dir, ignore_errors=True)
                    if inserted != update_point_count:
                        raise RuntimeError(
                            f"{label}: inserted {inserted} points, expected {update_point_count}"
                        )

                t0 = time.perf_counter()
                idx.point_count = base_point_count + update_point_count
                new_root = idx.combined_root_hex
                total_elapsed += time.perf_counter() - t0

                rows.append(
                    {
                        "city": dataset,
                        "base_fraction": base_fraction,
                        "base_trajectory_count": base_n,
                        "base_point_count": base_point_count,
                        "requested_update_trajectories": int(requested),
                        "actual_update_trajectories": actual,
                        "update_point_count": update_point_count,
                        "update_time_s": total_elapsed,
                        "old_combined_root": old_root,
                        "new_combined_root": new_root,
                    }
                )
            finally:
                # Every requested batch starts from the exact same 60% base.  New
                # leaf versions are append-only, so truncation restores the base
                # files without copying/rebuilding gigabytes of existing data.
                _truncate_to_sizes(idx, base_sizes)

        _write_csv(output_dir / "index_update_results.csv", rows)
        _write_csv(
            output_dir / "update_time_matrix_s.csv",
            [{"city": dataset, **{f"+{r['requested_update_trajectories']}": r["update_time_s"] for r in rows}}],
        )
        (output_dir / "experiment_setup.json").write_text(
            json.dumps(
                {
                    "city": dataset,
                    "base_fraction": base_fraction,
                    "update_trajectory_counts": update_trajectory_counts,
                    "leaf_capacity": leaf_capacity,
                    "fanout": fanout,
                    "base_rule": "first floor(N*0.60) trajectories in natural order",
                    "update_rule": "next N trajectories in natural order; each requested N starts independently from the same 60% base",
                    "timed_scope": "sort only new points + incremental touched-leaf append/split + Merkle internal metadata/root update; raw parsing and 60% base build excluded",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return rows
    finally:
        conn.close()
        try:
            base_path.unlink()
        except FileNotFoundError:
            pass
        shutil.rmtree(Path(str(base_path) + ".data"), ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
