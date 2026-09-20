from __future__ import annotations

import heapq
import shutil
import sqlite3
import struct
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterator

from .datasets import iter_dataset_records
from .disk_index import DiskThreeMBTIndex


LoadProgress = Callable[[int, int, Path], None]
BuildProgress = Callable[[str, int, int], None]
InternalProgress = Callable[[str, str, int, int], None]

# key(float64), point_id(uint64), trajectory ordinal(uint32)
_SORT_RECORD = struct.Struct(">dQI")


def _open_point_store(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA cache_size=-131072")  # ~128 MiB SQLite page cache
    conn.execute(
        "CREATE TABLE points ("
        "point_id INTEGER NOT NULL, "
        "tid_idx INTEGER NOT NULL, "
        "lon REAL NOT NULL, "
        "lat REAL NOT NULL, "
        "ts REAL NOT NULL"
        ")"
    )
    return conn


def _emit_internal_progress(
    callback: InternalProgress | None,
    label: str,
    phase: str,
    done: int,
    total: int,
) -> float:
    """Call the UI callback and return its wall-clock overhead."""
    if callback is None:
        return 0.0
    t0 = time.perf_counter()
    callback(label, phase, done, total)
    return time.perf_counter() - t0


def _write_sorted_runs(
    conn: sqlite3.Connection,
    *,
    column: str,
    run_dir: Path,
    label: str,
    point_count: int,
    internal_progress: InternalProgress | None,
    chunk_rows: int = 500_000,
    where_sql: str = "",
    where_params: tuple = (),
) -> tuple[list[Path], float]:
    """Create independently sorted binary runs with exact record-count progress."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("run-*.bin"):
        old.unlink()

    where_clause = f" WHERE {where_sql}" if where_sql else ""
    cursor = conn.execute(
        f"SELECT {column}, point_id, tid_idx FROM points{where_clause}",
        where_params,
    )
    run_paths: list[Path] = []
    processed = 0
    ui_overhead = _emit_internal_progress(
        internal_progress, label, "Sort", 0, point_count
    )

    while True:
        rows = cursor.fetchmany(chunk_rows)
        if not rows:
            break
        rows = [(float(k), int(pid), int(tid)) for k, pid, tid in rows]
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
        run_path = run_dir / f"run-{len(run_paths):05d}.bin"
        with run_path.open("wb") as f:
            buf = bytearray(_SORT_RECORD.size * len(rows))
            pos = 0
            for key, point_id, tid_idx in rows:
                _SORT_RECORD.pack_into(buf, pos, key, point_id, tid_idx)
                pos += _SORT_RECORD.size
            f.write(buf)
        run_paths.append(run_path)
        processed += len(rows)
        ui_overhead += _emit_internal_progress(
            internal_progress, label, "Sort", processed, point_count
        )

    return run_paths, ui_overhead


def _iter_run(f) -> Iterator[tuple[float, int, int]]:
    size = _SORT_RECORD.size
    read_records = 8192
    while True:
        raw = f.read(size * read_records)
        if not raw:
            return
        if len(raw) % size:
            raise IOError("truncated external-sort run")
        for pos in range(0, len(raw), size):
            key, point_id, tid_idx = _SORT_RECORD.unpack_from(raw, pos)
            yield float(key), int(point_id), int(tid_idx)


def _merged_runs(run_paths: list[Path]):
    """Yield a global (key, point_id, tid_idx) ordering from sorted run files."""
    with ExitStack() as stack:
        iterators = [
            _iter_run(stack.enter_context(path.open("rb"))) for path in run_paths
        ]
        yield from heapq.merge(*iterators)


def build_disk_index_from_dataset(
    *,
    input_path: str | Path,
    dataset: str,
    index_path: str | Path,
    leaf_capacity: int = 128,
    fanout: int = 64,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
    load_progress: LoadProgress | None = None,
    build_progress: BuildProgress | None = None,
    internal_progress: InternalProgress | None = None,
) -> tuple[DiskThreeMBTIndex, float]:
    """Memory-safe full build with real per-tree internal progress.

    Raw JSON/CSV parsing and insertion into a temporary SQLite point store happen
    before formal timing.  For each dimension, formal construction timing includes
    external run generation/sorting, k-way merge, packed MBT leaf/hash creation,
    and final three-root aggregation.  Progress-print wall time is subtracted from
    formal construction time so enabling the UI does not contaminate paper timing.
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    storage_dir = Path(str(index_path) + ".data")
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    temp_db = storage_dir / "_build_points.sqlite"

    conn = _open_point_store(temp_db)
    trajectory_ids: list[str] = []
    point_count = 0
    batch: list[tuple[int, int, float, float, float]] = []
    batch_size = 50_000

    try:
        records = iter_dataset_records(
            input_path,
            dataset,
            beijing_subdivisions=beijing_subdivisions,
            legacy_sidecar=legacy_sidecar,
            file_progress=load_progress,
        )
        conn.execute("BEGIN")
        for rec in records:
            tid_idx = len(trajectory_ids)
            if tid_idx >= 2**32:
                raise OverflowError("trajectory count exceeds uint32 disk format")
            trajectory_ids.append(rec.trajectory_id)
            for ts, lat, lon in rec.gps_points:
                batch.append((point_count, tid_idx, float(lon), float(lat), float(ts)))
                point_count += 1
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT INTO points(point_id,tid_idx,lon,lat,ts) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
        if batch:
            conn.executemany(
                "INSERT INTO points(point_id,tid_idx,lon,lat,ts) VALUES (?,?,?,?,?)",
                batch,
            )
            batch.clear()
        conn.commit()

        idx = DiskThreeMBTIndex(trajectory_ids, leaf_capacity, fanout)
        idx.bind_storage(storage_dir)
        idx.point_count = point_count

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
        return idx, formal_elapsed
    finally:
        conn.close()
        try:
            temp_db.unlink()
        except FileNotFoundError:
            pass
