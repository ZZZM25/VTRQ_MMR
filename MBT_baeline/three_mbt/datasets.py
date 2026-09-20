from __future__ import annotations

import ast
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .models import GPSPoint
from .plaintext_catalog import TrajectoryCatalog, TrajectoryPlaintext
from .trajectory_id import compute_paper_trajectory_id


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10



@dataclass(slots=True)
class TrajectoryRecord:
    ordinal: int
    trajectory_id: str
    source_traj_id: str
    road_node_path: list[int]
    gps_points: list[tuple[float, float, float]]  # timestamp, lat, lon
    plaintext_points: list[list[float | int]]
    points_format: str

    def plaintext(self) -> TrajectoryPlaintext:
        return TrajectoryPlaintext(
            trajectory_id=self.trajectory_id,
            source_traj_id=self.source_traj_id,
            road_node_path=list(self.road_node_path),
            points_data=[list(p) for p in self.plaintext_points],
            points_format=self.points_format,
        )


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


def list_chengdu_xian_json_files(path: str | Path) -> list[Path]:
    """One JSON/JSONL file or a directory of daily files in natural numeric order."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"expected .json/.jsonl file, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
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
    """Extract paper node path and raw GPS points from the archived city JSON shape.

    Formal archived shape used by the supplied sample:
      trajectory[0] = road-node path
      trajectory[2][segment][point] = [timestamp, latitude, longitude]
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

    if isinstance(obj, (list, tuple)) and len(obj) > 2:
        node_path_raw = obj[0]
        groups = obj[2]
        if not isinstance(node_path_raw, (list, tuple)):
            raise ValueError("Chengdu/Xi'an trajectory[0] must be road-node path")
        node_path = [int(x) for x in node_path_raw]
        gps: list[tuple[float, float, float]] = []
        if not isinstance(groups, (list, tuple)):
            raise ValueError("Chengdu/Xi'an trajectory[2] must be GPS groups")
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

    raise ValueError("unrecognized Chengdu/Xi'an trajectory JSON shape")


def load_ordinal_sidecar(path: str | Path | None) -> dict[int, str]:
    if path in (None, ""):
        return {}
    mapping: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"ordinal", "trajectory_id"}.issubset(fields):
            raise ValueError("Chengdu/Xi'an sidecar must contain ordinal,trajectory_id")
        for row in reader:
            ordinal = int(row["ordinal"])
            tid = str(row["trajectory_id"]).strip().lower()
            if ordinal in mapping:
                raise ValueError(f"duplicate ordinal in sidecar: {ordinal}")
            mapping[ordinal] = tid
    return mapping


def load_beijing_sidecar(path: str | Path | None) -> dict[str, str]:
    if path in (None, ""):
        return {}
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"source_traj_id", "trajectory_id"}.issubset(fields):
            raise ValueError("Beijing sidecar must contain source_traj_id,trajectory_id")
        for row in reader:
            source_tid = str(row["source_traj_id"]).strip()
            tid = str(row["trajectory_id"]).strip().lower()
            if source_tid in mapping and mapping[source_tid] != tid:
                raise ValueError(f"conflicting Beijing sidecar mapping: {source_tid}")
            mapping[source_tid] = tid
    return mapping


def _paper_id(node_path: list[int], gps: list[tuple[float, float, float]]) -> str:
    if not gps:
        raise ValueError("trajectory has no GPS points")
    previous = None
    for ts, _lat, _lon in gps:
        if previous is not None and ts < previous:
            raise ValueError("trajectory timestamp decreases")
        previous = ts
    start = int(gps[0][0])
    end = int(gps[-1][0])
    return compute_paper_trajectory_id(node_path, start, end)


def iter_chengdu_xian_records(
    path: str | Path,
    legacy_sidecar: str | Path | None = None,
    file_progress: Callable[[int, int, Path], None] | None = None,
) -> Iterator[TrajectoryRecord]:
    """All daily JSONs: natural filename order, then original trajectory order."""
    audit = load_ordinal_sidecar(legacy_sidecar)
    files = list_chengdu_xian_json_files(path)
    global_ordinal = 0
    for file_index, json_path in enumerate(files, start=1):
        items = _load_json_items(json_path)
        for local_ordinal, obj in enumerate(items):
            node_path, gps = _extract_chengdu_xian(obj)
            tid = _paper_id(node_path, gps)
            if audit:
                expected = audit.get(global_ordinal)
                if expected is None:
                    raise ValueError(f"sidecar missing ordinal={global_ordinal}")
                if expected.lower() != tid:
                    raise ValueError(
                        f"paper trajectory_id mismatch at ordinal={global_ordinal}: "
                        f"computed={tid}, sidecar={expected}"
                    )
            # The original files do not expose a separate source_traj_id.  Use a
            # deterministic file/local ordinal label for traceability only.
            source_tid = f"{json_path.name}:{local_ordinal}"
            plaintext_points = [
                [int(ts) if float(ts).is_integer() else float(ts), float(lat), float(lon)]
                for ts, lat, lon in gps
            ]
            yield TrajectoryRecord(
                ordinal=global_ordinal,
                trajectory_id=tid,
                source_traj_id=source_tid,
                road_node_path=node_path,
                gps_points=gps,
                plaintext_points=plaintext_points,
                points_format="time_lat_lon",
            )
            global_ordinal += 1
        if file_progress is not None:
            file_progress(file_index, len(files), json_path)
    if audit and len(audit) != global_ordinal:
        raise ValueError(
            f"sidecar rows={len(audit)} but dataset trajectories={global_ordinal}"
        )


def iter_beijing_records(
    path: str | Path,
    subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
) -> Iterator[TrajectoryRecord]:
    """Read the one Beijing CSV; each adjacent source point is split into 10 subsegments.

    Plaintext remains the original source_traj_id,points_data record.  Interpolated
    GPS points are used only for Three-MBT indexing / exact Fine Filtering.
    """
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    _raise_csv_field_limit()
    audit = load_beijing_sidecar(legacy_sidecar)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    seen_source: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"source_traj_id", "points_data"}.issubset(fields):
            raise ValueError("Beijing CSV must contain source_traj_id,points_data")

        for ordinal, row in enumerate(reader):
            source_tid = str(row["source_traj_id"]).strip()
            if not source_tid:
                raise ValueError(f"empty source_traj_id at row={ordinal + 2}")
            if source_tid in seen_source:
                raise ValueError(f"duplicate Beijing source_traj_id: {source_tid}")
            seen_source.add(source_tid)

            raw_points = _parse_literal(row["points_data"])
            if not isinstance(raw_points, list) or not raw_points:
                raise ValueError(f"row={ordinal + 2}: points_data must be non-empty list")

            node_path: list[int] = []
            source_gps: list[tuple[float, float, float]] = []
            normalized_plaintext: list[list[float | int]] = []
            previous_ts = None
            for p in raw_points:
                if not isinstance(p, (list, tuple)) or len(p) < 4:
                    raise ValueError("Beijing point must be [node_id,lon,lat,timestamp]")
                node_id, lon, lat, ts = p[:4]
                tsf = float(ts)
                if previous_ts is not None and tsf < previous_ts:
                    raise ValueError(f"timestamp decreases for source_traj_id={source_tid}")
                previous_ts = tsf
                node_path.append(int(node_id))
                source_gps.append((tsf, float(lat), float(lon)))
                normalized_plaintext.append([int(node_id), float(lon), float(lat), int(ts)])

            tid = _paper_id(node_path, source_gps)
            if audit:
                expected = audit.get(source_tid)
                if expected is None:
                    raise ValueError(f"Beijing sidecar missing source_traj_id={source_tid}")
                if expected.lower() != tid:
                    raise ValueError(
                        f"paper trajectory_id mismatch for source_traj_id={source_tid}: "
                        f"computed={tid}, sidecar={expected}"
                    )

            if len(source_gps) <= 1:
                indexed_gps = source_gps
            else:
                indexed_gps: list[tuple[float, float, float]] = []
                for i in range(len(source_gps) - 1):
                    t0, lat0, lon0 = source_gps[i]
                    t1, lat1, lon1 = source_gps[i + 1]
                    for step in range(subdivisions):
                        ratio = step / subdivisions
                        indexed_gps.append(
                            (
                                t0 + ratio * (t1 - t0),
                                lat0 + ratio * (lat1 - lat0),
                                lon0 + ratio * (lon1 - lon0),
                            )
                        )
                indexed_gps.append(source_gps[-1])

            yield TrajectoryRecord(
                ordinal=ordinal,
                trajectory_id=tid,
                source_traj_id=source_tid,
                road_node_path=node_path,
                gps_points=indexed_gps,
                plaintext_points=normalized_plaintext,
                points_format="node_lon_lat_time",
            )

    if audit:
        extra = set(audit) - seen_source
        if extra:
            raise ValueError(f"Beijing sidecar has {len(extra)} IDs not present in CSV")


def iter_dataset_records(
    path: str | Path,
    dataset: str,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
    file_progress: Callable[[int, int, Path], None] | None = None,
) -> Iterator[TrajectoryRecord]:
    ds = dataset.lower()
    if ds in {"chengdu", "xian", "xi'an", "xi_an"}:
        yield from iter_chengdu_xian_records(
            path,
            legacy_sidecar=legacy_sidecar,
            file_progress=file_progress,
        )
        return
    if ds in {"beijing", "bj"}:
        yield from iter_beijing_records(
            path,
            subdivisions=beijing_subdivisions,
            legacy_sidecar=legacy_sidecar,
        )
        return
    raise ValueError(f"unsupported dataset: {dataset}")


def records_to_points(
    records: Iterable[TrajectoryRecord],
    start_point_id: int = 0,
) -> list[GPSPoint]:
    out: list[GPSPoint] = []
    point_id = int(start_point_id)
    for rec in records:
        for ts, lat, lon in rec.gps_points:
            out.append(
                GPSPoint(
                    point_id=point_id,
                    trajectory_id=rec.trajectory_id,
                    timestamp=float(ts),
                    latitude=float(lat),
                    longitude=float(lon),
                )
            )
            point_id += 1
    return out


def load_dataset_records(
    path: str | Path,
    dataset: str,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
) -> list[TrajectoryRecord]:
    return list(
        iter_dataset_records(
            path,
            dataset,
            beijing_subdivisions=beijing_subdivisions,
            legacy_sidecar=legacy_sidecar,
        )
    )


def load_dataset(
    path: str | Path,
    dataset: str,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
    file_progress: Callable[[int, int, Path], None] | None = None,
) -> list[GPSPoint]:
    records = iter_dataset_records(
        path,
        dataset,
        beijing_subdivisions=beijing_subdivisions,
        legacy_sidecar=legacy_sidecar,
        file_progress=file_progress,
    )
    return records_to_points(records)


def write_plaintext_catalog(
    input_path: str | Path,
    dataset: str,
    output_csv: str | Path,
    beijing_subdivisions: int = 10,
    legacy_sidecar: str | Path | None = None,
) -> int:
    records = iter_dataset_records(
        input_path,
        dataset,
        beijing_subdivisions=beijing_subdivisions,
        legacy_sidecar=legacy_sidecar,
    )
    return TrajectoryCatalog.write_csv(output_csv, (rec.plaintext() for rec in records))
