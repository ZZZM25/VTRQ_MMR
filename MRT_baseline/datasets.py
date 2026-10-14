from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator

from .models import TrajectoryRecord
from .trajectory_id import compute_trajectory_id


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_literal(value: str):
    value = value.strip()
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _natural_key(path: Path):
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name))


def list_json_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}]
    files.sort(key=_natural_key)
    if not files:
        raise FileNotFoundError(f"no JSON/JSONL files under {path}")
    return files


def _load_json_items(path: Path):
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        items = json.loads(raw)
    if isinstance(items, dict):
        items = items.get("trajectories", items.get("data", [items]))
    if not isinstance(items, list):
        raise ValueError(f"top-level JSON must be list/dict: {path}")
    return items


def _extract_chengdu_xian(obj) -> tuple[list[int], list[tuple[float, float, float]]]:
    """Extract the original GPS samples only.

    Archived formal shape:
      trajectory[0] = road-node path (ID material only; NOT indexed)
      trajectory[2][segment][point] = [timestamp, latitude, longitude]

    trajectory[1] / road edges are intentionally ignored by this baseline.
    """
    if isinstance(obj, dict):
        node_path = obj.get("node_path", obj.get("road_node_path", obj.get("nodes")))
        points = obj.get("points", obj.get("gps_points", obj.get("points_data")))
        if node_path is not None and points is not None:
            gps: list[tuple[float, float, float]] = []
            for p in points:
                if len(p) >= 4:
                    _node, lon, lat, ts = p[:4]
                    gps.append((float(ts), float(lat), float(lon)))
                elif len(p) >= 3:
                    ts, lat, lon = p[:3]
                    gps.append((float(ts), float(lat), float(lon)))
            return [int(x) for x in node_path], gps
        obj = obj.get("trajectory", obj.get("data", obj))

    if not isinstance(obj, (list, tuple)) or len(obj) <= 2:
        raise ValueError("unrecognized Chengdu/Xi'an trajectory JSON shape")
    node_path_raw = obj[0]
    groups = obj[2]
    if not isinstance(node_path_raw, (list, tuple)):
        raise ValueError("trajectory[0] must be road-node path")
    if not isinstance(groups, (list, tuple)):
        raise ValueError("trajectory[2] must be GPS segment groups")

    node_path = [int(x) for x in node_path_raw]
    gps: list[tuple[float, float, float]] = []
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for p in group:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                ts, lat, lon = p[:3]
                gps.append((float(ts), float(lat), float(lon)))
    if not node_path or not gps:
        raise ValueError("empty road-node path or GPS points")
    return node_path, gps


def _validate_time(gps: list[tuple[float, float, float]], label: str) -> None:
    previous = None
    for ts, _lat, _lon in gps:
        if previous is not None and ts < previous:
            raise ValueError(f"timestamp decreases: {label}")
        previous = ts


def iter_chengdu_xian(path: str | Path) -> Iterator[TrajectoryRecord]:
    ordinal = 0
    for json_path in list_json_files(path):
        items = _load_json_items(json_path)
        for local_ordinal, obj in enumerate(items):
            node_path, gps = _extract_chengdu_xian(obj)
            _validate_time(gps, f"{json_path}:{local_ordinal}")
            tid = compute_trajectory_id(node_path, int(gps[0][0]), int(gps[-1][0]))
            plaintext_points = [
                [int(ts) if float(ts).is_integer() else float(ts), float(lat), float(lon)]
                for ts, lat, lon in gps
            ]
            yield TrajectoryRecord(
                ordinal=ordinal,
                trajectory_id=tid,
                source_traj_id=f"{json_path.name}:{local_ordinal}",
                road_node_path=node_path,
                gps_points=gps,
                plaintext_points=plaintext_points,
                points_format="time_lat_lon",
            )
            ordinal += 1


def iter_beijing(path: str | Path, subdivisions: int = 10) -> Iterator[TrajectoryRecord]:
    """Read Beijing and linearly split every adjacent source pair into 10 segments.

    Source CSV point format: [node_id, longitude, latitude, timestamp].
    Indexed point count for N source points is 10*(N-1)+1.
    The R-tree uses only interpolated longitude/latitude/timestamp, never node_id.
    """
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    _raise_csv_field_limit()
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"source_traj_id", "points_data"}.issubset(fields):
            raise ValueError("Beijing CSV must contain source_traj_id,points_data")

        seen: set[str] = set()
        for ordinal, row in enumerate(reader):
            source_tid = str(row["source_traj_id"]).strip()
            if not source_tid:
                raise ValueError(f"empty source_traj_id at row {ordinal+2}")
            if source_tid in seen:
                raise ValueError(f"duplicate source_traj_id: {source_tid}")
            seen.add(source_tid)
            raw_points = _parse_literal(row["points_data"])
            if not isinstance(raw_points, list) or not raw_points:
                raise ValueError(f"row={ordinal+2}: points_data must be non-empty list")

            nodes: list[int] = []
            source_gps: list[tuple[float, float, float]] = []
            plaintext: list[list[float | int]] = []
            for p in raw_points:
                if not isinstance(p, (list, tuple)) or len(p) < 4:
                    raise ValueError("Beijing point must be [node_id,lon,lat,timestamp]")
                node, lon, lat, ts = p[:4]
                nodes.append(int(node))
                source_gps.append((float(ts), float(lat), float(lon)))
                plaintext.append([int(node), float(lon), float(lat), int(ts)])
            _validate_time(source_gps, source_tid)
            tid = compute_trajectory_id(nodes, int(source_gps[0][0]), int(source_gps[-1][0]))

            if len(source_gps) <= 1:
                indexed = list(source_gps)
            else:
                indexed: list[tuple[float, float, float]] = []
                for i in range(len(source_gps) - 1):
                    t0, lat0, lon0 = source_gps[i]
                    t1, lat1, lon1 = source_gps[i + 1]
                    for step in range(subdivisions):
                        r = step / subdivisions
                        indexed.append((
                            t0 + r * (t1 - t0),
                            lat0 + r * (lat1 - lat0),
                            lon0 + r * (lon1 - lon0),
                        ))
                indexed.append(source_gps[-1])

            yield TrajectoryRecord(
                ordinal=ordinal,
                trajectory_id=tid,
                source_traj_id=source_tid,
                road_node_path=nodes,
                gps_points=indexed,
                plaintext_points=plaintext,
                points_format="node_lon_lat_time",
            )


def iter_dataset(path: str | Path, city: str, beijing_subdivisions: int = 10) -> Iterator[TrajectoryRecord]:
    city = city.lower()
    if city in {"chengdu", "xian", "xi'an", "xi_an"}:
        yield from iter_chengdu_xian(path)
    elif city in {"beijing", "bj"}:
        yield from iter_beijing(path, subdivisions=beijing_subdivisions)
    else:
        raise ValueError(f"unsupported city: {city}")
