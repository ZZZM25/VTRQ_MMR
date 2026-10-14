from __future__ import annotations

import heapq
import json
import pickle
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator

import numpy as np

from .formats import (
    POINT_DTYPE,
    POINT_STRUCT,
    RUN_DTYPE,
    RUN_STRUCT,
    internal_hash,
    leaf_hash,
    root_commitment,
)
from .geometry import bbox_union
from .models import InternalMeta, LeafMeta
from .morton import morton_keys
from .prepare import load_prepared_metadata, point_prefix

INDEX_VERSION = 1


def _build_internal_levels(leaves: list[LeafMeta], fanout: int) -> list[list]:
    if not leaves:
        raise ValueError("R-tree has no leaves")
    levels: list[list] = [leaves]
    current = leaves
    while len(current) > 1:
        parents: list[InternalMeta] = []
        for start in range(0, len(current), fanout):
            children = current[start : start + fanout]
            box = bbox_union([c.bbox for c in children])
            parents.append(
                InternalMeta(
                    box[0], box[1], box[2], box[3],
                    internal_hash(children),
                    start,
                    len(children),
                )
            )
        levels.append(parents)
        current = parents
    return levels


def _write_sorted_runs(
    prepared_points_path: Path,
    point_count: int,
    bounds,
    run_dir: Path,
    chunk_rows: int = 500_000,
) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.glob("run-*.bin"):
        p.unlink()
    mm = np.memmap(prepared_points_path, dtype=POINT_DTYPE, mode="r", shape=(point_count,))
    runs: list[Path] = []
    for start in range(0, point_count, chunk_rows):
        end = min(start + chunk_rows, point_count)
        src = mm[start:end]
        keys = morton_keys(src["lon"], src["lat"], bounds)
        order = np.lexsort((src["point_id"], keys))
        out = np.empty(end - start, dtype=RUN_DTYPE)
        out["morton"] = keys[order]
        out["point_id"] = src["point_id"][order]
        out["tid"] = src["tid"][order]
        out["lon"] = src["lon"][order]
        out["lat"] = src["lat"][order]
        out["ts"] = src["ts"][order]
        path = run_dir / f"run-{len(runs):05d}.bin"
        out.tofile(path)
        runs.append(path)
        print(f"[Build] sort runs: {end:,}/{point_count:,} points")
        del out, order, keys
    del mm
    return runs


def _iter_run(path: Path) -> Iterator[tuple[int, int, bytes, float, float, float]]:
    with path.open("rb", buffering=8 * 1024 * 1024) as f:
        read_records = 8192
        while True:
            raw = f.read(RUN_STRUCT.size * read_records)
            if not raw:
                return
            if len(raw) % RUN_STRUCT.size:
                raise IOError(f"truncated sort run: {path}")
            for pos in range(0, len(raw), RUN_STRUCT.size):
                morton, point_id, tid, lon, lat, ts = RUN_STRUCT.unpack_from(raw, pos)
                yield int(morton), int(point_id), tid, float(lon), float(lat), float(ts)


def _merged_runs(run_paths: list[Path]):
    # Tuple natural order starts with (morton, point_id), giving deterministic packing.
    iterators = [_iter_run(p) for p in run_paths]
    yield from heapq.merge(*iterators)


def _pack_merged_points(
    merged,
    final_points_path: Path,
    point_count: int,
    leaf_capacity: int,
    fanout: int,
) -> list[list]:
    leaves: list[LeafMeta] = []
    processed = 0
    offset = 0
    leaf_records: list[tuple[int, int, bytes, float, float, float]] = []

    def flush_leaf(out_f) -> None:
        nonlocal offset
        if not leaf_records:
            return
        raw = bytearray(POINT_STRUCT.size * len(leaf_records))
        pos = 0
        min_lon = float("inf")
        min_lat = float("inf")
        max_lon = float("-inf")
        max_lat = float("-inf")
        min_morton = leaf_records[0][0]
        max_morton = leaf_records[-1][0]
        for morton, point_id, tid, lon, lat, ts in leaf_records:
            POINT_STRUCT.pack_into(raw, pos, point_id, tid, lon, lat, ts)
            pos += POINT_STRUCT.size
            min_lon = min(min_lon, lon)
            min_lat = min(min_lat, lat)
            max_lon = max(max_lon, lon)
            max_lat = max(max_lat, lat)
        raw_b = bytes(raw)
        out_f.write(raw_b)
        leaves.append(
            LeafMeta(
                min_lon, min_lat, max_lon, max_lat,
                leaf_hash(raw_b, len(leaf_records)),
                0,
                offset,
                len(leaf_records),
                int(min_morton),
                int(max_morton),
            )
        )
        offset += len(raw_b)
        leaf_records.clear()

    with final_points_path.open("wb", buffering=8 * 1024 * 1024) as out_f:
        for rec in merged:
            leaf_records.append(rec)
            processed += 1
            if len(leaf_records) >= leaf_capacity:
                flush_leaf(out_f)
            if processed % 1_000_000 == 0 or processed == point_count:
                print(f"[Build] merge/tree: {processed:,}/{point_count:,} points")
        flush_leaf(out_f)

    if processed != point_count:
        raise ValueError(f"merged point count mismatch: {processed} != {point_count}")
    return _build_internal_levels(leaves, fanout)


def save_index_metadata(index_dir: Path, metadata: dict) -> None:
    with (index_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f, protocol=5)


def load_index_metadata(index_dir: str | Path) -> dict:
    with (Path(index_dir) / "metadata.pkl").open("rb") as f:
        return pickle.load(f)


def index_size_bytes(index_dir: str | Path) -> int:
    root = Path(index_dir)
    total = 0
    meta = root / "metadata.pkl"
    if meta.exists():
        total += meta.stat().st_size
    for p in root.glob("points*.bin"):
        if p.is_file():
            total += p.stat().st_size
    return total


def build_index(
    prepared_dir: str | Path,
    index_dir: str | Path,
    trajectory_count: int | None = None,
    leaf_capacity: int = 128,
    fanout: int = 64,
    sort_chunk_rows: int = 500_000,
) -> tuple[dict, float]:
    """Disk-backed packed authenticated 2D point R-tree build.

    Raw JSON/CSV parsing is already completed by prepare and is excluded. Formal
    construction includes external Morton sorting, leaf writing/hashing, internal
    R-tree construction, root commitment, and metadata writing.
    """
    prepared_dir = Path(prepared_dir)
    index_dir = Path(index_dir)
    prep = load_prepared_metadata(prepared_dir)
    total_traj = int(prep["trajectory_count"])
    traj_n = total_traj if trajectory_count is None else min(int(trajectory_count), total_traj)
    point_count = point_prefix(prepared_dir, traj_n)
    if point_count <= 0:
        raise ValueError("selected subset contains no points")

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    run_dir = index_dir / "_sort_runs"
    final_points = index_dir / "points.bin"
    prepared_points = prepared_dir / prep["points_file"]
    bounds = tuple(float(x) for x in prep["bounds"])

    t0 = time.perf_counter()
    run_paths = _write_sorted_runs(
        prepared_points, point_count, bounds, run_dir, chunk_rows=sort_chunk_rows
    )
    levels = _pack_merged_points(
        _merged_runs(run_paths), final_points, point_count, leaf_capacity, fanout
    )
    shutil.rmtree(run_dir, ignore_errors=True)
    root_node = levels[-1][0]
    root = root_commitment(root_node)
    metadata = {
        "version": INDEX_VERSION,
        "city": prep["city"],
        "trajectory_count": traj_n,
        "point_count": point_count,
        "leaf_capacity": int(leaf_capacity),
        "fanout": int(fanout),
        "bounds": bounds,
        "levels": levels,
        "point_files": ["points.bin"],
        "root_hash": root,
        "root_hash_hex": root.hex(),
        "semantics": "Authenticated packed 2D R-tree over GPS points. Internal nodes contain only spatial MBR+hash. Leaf entries contain lon,lat,timestamp,trajectory_id,point_id; timestamp is tested only after spatial leaf hit.",
    }
    save_index_metadata(index_dir, metadata)
    elapsed = time.perf_counter() - t0

    report = {
        "city": prep["city"],
        "trajectory_count": traj_n,
        "point_count": point_count,
        "leaf_count": len(levels[0]),
        "construction_time_s": elapsed,
        "index_size_bytes": index_size_bytes(index_dir),
        "index_size_mib": index_size_bytes(index_dir) / (1024.0 * 1024.0),
        "root_hash": root.hex(),
    }
    (index_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[Build] done | trajectories={traj_n:,} | points={point_count:,} | "
        f"leaves={len(levels[0]):,} | time={elapsed:.3f}s | "
        f"size={report['index_size_mib']:.3f} MiB"
    )
    return metadata, elapsed
