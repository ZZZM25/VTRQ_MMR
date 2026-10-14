from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

from .datasets import iter_dataset
from .formats import POINT_STRUCT

CATALOG_FIELDS = [
    "trajectory_id",
    "source_traj_id",
    "road_node_path",
    "points_format",
    "points_data",
]
PREFIX_STRUCT = struct.Struct("<Q")


def prepare_dataset(
    input_path: str | Path,
    city: str,
    prepared_dir: str | Path,
    catalog_path: str | Path,
    beijing_subdivisions: int = 10,
) -> dict:
    """Parse raw data once into a natural-trajectory-order point store.

    This phase is preprocessing and is excluded from all formal paper timings.
    """
    input_path = Path(input_path)
    prepared_dir = Path(prepared_dir)
    catalog_path = Path(catalog_path)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    points_path = prepared_dir / "points.natural.bin"
    prefix_path = prepared_dir / "trajectory_prefix_points.bin"
    meta_path = prepared_dir / "metadata.json"

    trajectory_count = 0
    point_count = 0
    min_lon = math.inf
    min_lat = math.inf
    max_lon = -math.inf
    max_lat = -math.inf

    with points_path.open("wb", buffering=8 * 1024 * 1024) as point_f, \
         prefix_path.open("wb", buffering=8 * 1024 * 1024) as prefix_f, \
         catalog_path.open("w", encoding="utf-8-sig", newline="") as cat_f:
        writer = csv.DictWriter(cat_f, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        prefix_f.write(PREFIX_STRUCT.pack(0))

        for rec in iter_dataset(input_path, city, beijing_subdivisions=beijing_subdivisions):
            tid_raw = bytes.fromhex(rec.trajectory_id)
            raw = bytearray(POINT_STRUCT.size * len(rec.gps_points))
            pos = 0
            for ts, lat, lon in rec.gps_points:
                POINT_STRUCT.pack_into(
                    raw, pos, int(point_count), tid_raw, float(lon), float(lat), float(ts)
                )
                pos += POINT_STRUCT.size
                point_count += 1
                min_lon = min(min_lon, float(lon))
                min_lat = min(min_lat, float(lat))
                max_lon = max(max_lon, float(lon))
                max_lat = max(max_lat, float(lat))
            point_f.write(raw)

            writer.writerow({
                "trajectory_id": rec.trajectory_id,
                "source_traj_id": rec.source_traj_id,
                "road_node_path": json.dumps(rec.road_node_path, separators=(",", ":")),
                "points_format": rec.points_format,
                "points_data": json.dumps(rec.plaintext_points, separators=(",", ":"), ensure_ascii=False),
            })
            trajectory_count += 1
            prefix_f.write(PREFIX_STRUCT.pack(point_count))
            if trajectory_count % 1000 == 0:
                print(
                    f"[Prepare] {city}: trajectories={trajectory_count:,}, "
                    f"indexed-points={point_count:,}"
                )

    if trajectory_count == 0 or point_count == 0:
        raise ValueError(f"empty prepared dataset: {city}")

    metadata = {
        "version": 1,
        "city": city,
        "trajectory_count": trajectory_count,
        "point_count": point_count,
        "bounds": [min_lon, min_lat, max_lon, max_lat],
        "points_file": points_path.name,
        "prefix_file": prefix_path.name,
        "point_record_size": POINT_STRUCT.size,
        "beijing_subdivisions": int(beijing_subdivisions),
        "catalog": str(catalog_path),
        "semantics": {
            "chengdu_xian": "raw original trajectory[2] GPS points only; road edges are not indexed or queried",
            "beijing": "each adjacent [node_id,lon,lat,timestamp] source pair linearly split into 10 segments by default; N -> 10*(N-1)+1 indexed points",
            "rtree": "2D longitude/latitude R-tree; timestamp stored only in leaf point entry and checked only after spatial leaf hit",
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Prepare] {city}: done | trajectories={trajectory_count:,} | "
        f"indexed-points={point_count:,} | {prepared_dir}"
    )
    return metadata


def load_prepared_metadata(prepared_dir: str | Path) -> dict:
    return json.loads((Path(prepared_dir) / "metadata.json").read_text(encoding="utf-8"))


def point_prefix(prepared_dir: str | Path, trajectory_count: int) -> int:
    prepared_dir = Path(prepared_dir)
    meta = load_prepared_metadata(prepared_dir)
    n = max(0, min(int(trajectory_count), int(meta["trajectory_count"])))
    prefix_path = prepared_dir / meta["prefix_file"]
    with prefix_path.open("rb") as f:
        f.seek(n * PREFIX_STRUCT.size)
        raw = f.read(PREFIX_STRUCT.size)
    if len(raw) != PREFIX_STRUCT.size:
        raise IOError("truncated trajectory prefix file")
    return int(PREFIX_STRUCT.unpack(raw)[0])
