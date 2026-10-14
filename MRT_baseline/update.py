from __future__ import annotations

import heapq
import itertools
import shutil
import struct
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .build import _build_internal_levels, load_index_metadata
from .formats import POINT_DTYPE, POINT_STRUCT, leaf_hash, root_commitment
from .models import LeafMeta
from .morton import morton_keys
from .prepare import load_prepared_metadata, point_prefix

UPDATE_STRUCT = struct.Struct("<IQQ32sddd")
UPDATE_DTYPE = np.dtype([
    ("leaf", "<u4"),
    ("morton", "<u8"),
    ("point_id", "<u8"),
    ("tid", "V32"),
    ("lon", "<f8"),
    ("lat", "<f8"),
    ("ts", "<f8"),
], align=False)
assert UPDATE_DTYPE.itemsize == UPDATE_STRUCT.size == 76


def _write_update_runs(
    prepared_points: Path,
    start_point: int,
    end_point: int,
    bounds,
    leaf_max_morton: np.ndarray,
    run_dir: Path,
    chunk_rows: int = 250_000,
) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.glob("run-*.bin"):
        p.unlink()
    total_file_points = prepared_points.stat().st_size // POINT_STRUCT.size
    mm = np.memmap(prepared_points, dtype=POINT_DTYPE, mode="r", shape=(total_file_points,))
    paths: list[Path] = []
    total = end_point - start_point
    done = 0
    for start in range(start_point, end_point, chunk_rows):
        end = min(start + chunk_rows, end_point)
        src = mm[start:end]
        keys = morton_keys(src["lon"], src["lat"], bounds)
        targets = np.searchsorted(leaf_max_morton, keys, side="left")
        targets = np.minimum(targets, len(leaf_max_morton) - 1).astype(np.uint32)
        order = np.lexsort((src["point_id"], keys, targets))
        out = np.empty(end - start, dtype=UPDATE_DTYPE)
        out["leaf"] = targets[order]
        out["morton"] = keys[order]
        out["point_id"] = src["point_id"][order]
        out["tid"] = src["tid"][order]
        out["lon"] = src["lon"][order]
        out["lat"] = src["lat"][order]
        out["ts"] = src["ts"][order]
        path = run_dir / f"run-{len(paths):05d}.bin"
        out.tofile(path)
        paths.append(path)
        done += end - start
        print(f"[Update] route/sort new points: {done:,}/{total:,}")
        del out, order, targets, keys
    del mm
    return paths


def _iter_update_run(path: Path) -> Iterator[tuple[int, int, int, bytes, float, float, float]]:
    with path.open("rb", buffering=8 * 1024 * 1024) as f:
        while True:
            raw = f.read(UPDATE_STRUCT.size * 8192)
            if not raw:
                return
            if len(raw) % UPDATE_STRUCT.size:
                raise IOError("truncated update run")
            for pos in range(0, len(raw), UPDATE_STRUCT.size):
                leaf, morton, pid, tid, lon, lat, ts = UPDATE_STRUCT.unpack_from(raw, pos)
                yield int(leaf), int(morton), int(pid), tid, float(lon), float(lat), float(ts)


def _merged_update_runs(paths: list[Path]):
    yield from heapq.merge(*[_iter_update_run(p) for p in paths])


def incremental_update(
    prepared_dir: str | Path,
    base_index_dir: str | Path,
    base_trajectory_count: int,
    update_trajectory_count: int,
    work_dir: str | Path,
    leaf_capacity: int | None = None,
    fanout: int | None = None,
) -> tuple[dict, float]:
    """True incremental point insertion from an immutable fixed base index.

    It does not rebuild base points. New points are routed to existing Morton-packed
    R-tree leaves, touched leaves are merged/split into a delta point file, then
    internal Merkle metadata/root are rebuilt from leaf metadata only.
    """
    prepared_dir = Path(prepared_dir)
    base_index_dir = Path(base_index_dir)
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prep = load_prepared_metadata(prepared_dir)
    meta = load_index_metadata(base_index_dir)
    leaf_capacity = int(leaf_capacity or meta["leaf_capacity"])
    fanout = int(fanout or meta["fanout"])
    base_n = int(base_trajectory_count)
    total_traj = int(prep["trajectory_count"])
    actual = max(0, min(int(update_trajectory_count), total_traj - base_n))
    if actual <= 0:
        raise ValueError("no trajectories available for requested update")

    start_point = point_prefix(prepared_dir, base_n)
    end_point = point_prefix(prepared_dir, base_n + actual)
    new_point_count = end_point - start_point
    base_leaves: list[LeafMeta] = meta["levels"][0]
    leaf_max = np.asarray([x.max_morton for x in base_leaves], dtype=np.uint64)
    bounds = tuple(float(x) for x in meta["bounds"])
    prepared_points = prepared_dir / prep["points_file"]
    base_points = base_index_dir / meta["point_files"][0]
    run_dir = work_dir / "_update_runs"
    delta_points = work_dir / "points.delta.bin"

    t0 = time.perf_counter()
    run_paths = _write_update_runs(
        prepared_points, start_point, end_point, bounds, leaf_max, run_dir
    )

    replacements: dict[int, list[LeafMeta]] = {}
    delta_offset = 0
    affected_leaf_count = 0
    new_leaf_count = 0

    with base_points.open("rb", buffering=8 * 1024 * 1024) as base_f, \
         delta_points.open("wb", buffering=8 * 1024 * 1024) as delta_f:

        merged = _merged_update_runs(run_paths)
        for leaf_idx, group in itertools.groupby(merged, key=lambda x: x[0]):
            leaf_idx = int(leaf_idx)
            base_leaf = base_leaves[leaf_idx]
            base_f.seek(base_leaf.offset)
            base_raw = base_f.read(base_leaf.count * POINT_STRUCT.size)
            if len(base_raw) != base_leaf.count * POINT_STRUCT.size:
                raise IOError("short base leaf read during update")
            base_arr = np.frombuffer(base_raw, dtype=POINT_DTYPE, count=base_leaf.count)
            base_keys = morton_keys(base_arr["lon"], base_arr["lat"], bounds)
            base_records = [
                (
                    int(base_keys[i]), int(base_arr["point_id"][i]), bytes(base_arr["tid"][i]),
                    float(base_arr["lon"][i]), float(base_arr["lat"][i]), float(base_arr["ts"][i]),
                )
                for i in range(len(base_arr))
            ]

            def new_records():
                for _target, morton, pid, tid, lon, lat, ts in group:
                    yield morton, pid, tid, lon, lat, ts

            combined = heapq.merge(base_records, new_records())
            chunks: list[LeafMeta] = []
            bucket: list[tuple[int, int, bytes, float, float, float]] = []

            def flush() -> None:
                nonlocal delta_offset, new_leaf_count
                if not bucket:
                    return
                raw = bytearray(POINT_STRUCT.size * len(bucket))
                pos = 0
                min_lon = float("inf")
                min_lat = float("inf")
                max_lon = float("-inf")
                max_lat = float("-inf")
                for morton, pid, tid, lon, lat, ts in bucket:
                    POINT_STRUCT.pack_into(raw, pos, pid, tid, lon, lat, ts)
                    pos += POINT_STRUCT.size
                    min_lon = min(min_lon, lon)
                    min_lat = min(min_lat, lat)
                    max_lon = max(max_lon, lon)
                    max_lat = max(max_lat, lat)
                raw_b = bytes(raw)
                delta_f.write(raw_b)
                chunks.append(
                    LeafMeta(
                        min_lon, min_lat, max_lon, max_lat,
                        leaf_hash(raw_b, len(bucket)),
                        1,
                        delta_offset,
                        len(bucket),
                        int(bucket[0][0]),
                        int(bucket[-1][0]),
                    )
                )
                delta_offset += len(raw_b)
                new_leaf_count += 1
                bucket.clear()

            for rec in combined:
                bucket.append(rec)
                if len(bucket) >= leaf_capacity:
                    flush()
            flush()
            replacements[leaf_idx] = chunks
            affected_leaf_count += 1
            if affected_leaf_count % 1000 == 0:
                print(f"[Update] rewritten base leaves: {affected_leaf_count:,}")

    shutil.rmtree(run_dir, ignore_errors=True)
    updated_leaves: list[LeafMeta] = []
    for i, leaf in enumerate(base_leaves):
        updated_leaves.extend(replacements.get(i, [leaf]))
    levels = _build_internal_levels(updated_leaves, fanout)
    new_root = root_commitment(levels[-1][0])
    elapsed = time.perf_counter() - t0

    result = {
        "old_root_hash": meta["root_hash"],
        "old_root_hash_hex": meta["root_hash_hex"],
        "new_root_hash": new_root,
        "new_root_hash_hex": new_root.hex(),
        "actual_update_trajectories": actual,
        "update_point_count": new_point_count,
        "affected_base_leaf_count": affected_leaf_count,
        "result_leaf_count": len(updated_leaves),
        "delta_leaf_count": new_leaf_count,
        "levels": levels,
        "delta_points_path": str(delta_points),
    }
    return result, elapsed
