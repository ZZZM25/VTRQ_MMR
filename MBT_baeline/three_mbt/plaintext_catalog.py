from __future__ import annotations

import csv
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .geometry import trajectory_intersects_query
from .models import GPSPoint, QueryWindow, stable_trajectory_sort_key
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
class TrajectoryPlaintext:
    trajectory_id: str
    source_traj_id: str
    road_node_path: list[int]
    points_data: list[list[float | int]]
    points_format: str

    def gps_points(self) -> list[GPSPoint]:
        out: list[GPSPoint] = []
        if self.points_format == "node_lon_lat_time":
            for i, p in enumerate(self.points_data):
                if len(p) < 4:
                    raise ValueError("node_lon_lat_time point requires 4 fields")
                _node_id, lon, lat, ts = p[:4]
                out.append(
                    GPSPoint(i, self.trajectory_id, float(ts), float(lat), float(lon))
                )
            return out
        if self.points_format == "time_lat_lon":
            for i, p in enumerate(self.points_data):
                if len(p) < 3:
                    raise ValueError("time_lat_lon point requires 3 fields")
                ts, lat, lon = p[:3]
                out.append(
                    GPSPoint(i, self.trajectory_id, float(ts), float(lat), float(lon))
                )
            return out
        raise ValueError(f"unsupported points_format: {self.points_format}")

    def time_bounds(self) -> tuple[int, int]:
        if not self.points_data:
            raise ValueError("points_data is empty")
        if self.points_format == "node_lon_lat_time":
            return int(self.points_data[0][3]), int(self.points_data[-1][3])
        if self.points_format == "time_lat_lon":
            return int(self.points_data[0][0]), int(self.points_data[-1][0])
        raise ValueError(f"unsupported points_format: {self.points_format}")

    def effective_node_path(self) -> list[int]:
        if self.road_node_path:
            return [int(x) for x in self.road_node_path]
        if self.points_format == "node_lon_lat_time":
            return [int(p[0]) for p in self.points_data]
        raise ValueError("road_node_path is required for this plaintext format")

    def recompute_trajectory_id(self) -> str:
        start, end = self.time_bounds()
        return compute_paper_trajectory_id(self.effective_node_path(), start, end)

    def verify_trajectory_id(self) -> bool:
        return self.recompute_trajectory_id() == self.trajectory_id


class TrajectoryCatalog:
    """In-memory map loaded before formal query timing.

    The backing CSV is the requested trajectory_id -> plaintext file. Loading and
    per-candidate lookup are deliberately outside formal Query Time.
    """

    CSV_FIELDS = [
        "trajectory_id",
        "source_traj_id",
        "road_node_path",
        "points_format",
        "points_data",
    ]

    def __init__(self, records: Iterable[TrajectoryPlaintext] = ()) -> None:
        self._records: dict[str, TrajectoryPlaintext] = {}
        for rec in records:
            self.add(rec)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, rec: TrajectoryPlaintext) -> None:
        tid = str(rec.trajectory_id)
        if tid in self._records:
            raise ValueError(f"duplicate trajectory_id in plaintext catalog: {tid}")
        self._records[tid] = rec

    def get(self, trajectory_id: str) -> TrajectoryPlaintext:
        tid = str(trajectory_id)
        try:
            return self._records[tid]
        except KeyError as exc:
            raise KeyError(f"plaintext catalog missing trajectory_id: {tid}") from exc

    def get_many(self, trajectory_ids: Iterable[str]) -> list[TrajectoryPlaintext]:
        return [self.get(tid) for tid in sorted(set(trajectory_ids), key=stable_trajectory_sort_key)]

    def trajectory_ids(self) -> set[str]:
        return set(self._records)

    @classmethod
    def load_csv(cls, path: str | Path) -> "TrajectoryCatalog":
        _raise_csv_field_limit()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        out = cls()
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            required = {"trajectory_id", "points_data"}
            if not required.issubset(fields):
                raise ValueError(
                    f"plaintext catalog must contain {sorted(required)}; got {reader.fieldnames}"
                )
            for row_no, row in enumerate(reader, 2):
                tid = str(row["trajectory_id"]).strip()
                source_tid = str(row.get("source_traj_id", "")).strip()
                node_text = str(row.get("road_node_path", "")).strip()
                node_path = json.loads(node_text) if node_text else []
                fmt = str(row.get("points_format", "")).strip() or "node_lon_lat_time"
                points = json.loads(row["points_data"])
                if not isinstance(points, list):
                    raise ValueError(f"row {row_no}: points_data must be a JSON list")
                rec = TrajectoryPlaintext(
                    trajectory_id=tid,
                    source_traj_id=source_tid,
                    road_node_path=[int(x) for x in node_path],
                    points_data=points,
                    points_format=fmt,
                )
                out.add(rec)
        return out

    @classmethod
    def write_csv(
        cls,
        path: str | Path,
        records: Iterable[TrajectoryPlaintext],
    ) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cls.CSV_FIELDS)
            writer.writeheader()
            for rec in records:
                writer.writerow(
                    {
                        "trajectory_id": rec.trajectory_id,
                        "source_traj_id": rec.source_traj_id,
                        "road_node_path": json.dumps(
                            rec.road_node_path, separators=(",", ":")
                        ),
                        "points_format": rec.points_format,
                        "points_data": json.dumps(
                            rec.points_data, separators=(",", ":"), ensure_ascii=False
                        ),
                    }
                )
                count += 1
        return count


class DiskTrajectoryCatalog:
    """Sequential-scan access to the trajectory plaintext CSV.

    No auxiliary offset/index file is created.  For a set of candidate
    trajectory IDs, get_many() scans the CSV once, collects matching rows, and
    stops as soon as all candidates are found.  The entire lookup is measured
    separately by run_plaintext_stage() and excluded from formal Query Time and
    Verification Time.
    """

    def __init__(self, csv_path: str | Path) -> None:
        _raise_csv_field_limit()
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(self.csv_path)
        self._fieldnames = self._read_header()

    sequential_scan_only = True

    def __len__(self) -> int:
        count = 0
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for _row in reader:
                count += 1
        return count

    def close(self) -> None:
        return None

    def __enter__(self) -> "DiskTrajectoryCatalog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_header(self) -> list[str]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                fields = [str(x) for x in next(reader)]
            except StopIteration as exc:
                raise ValueError("plaintext catalog is empty") from exc
        required = {"trajectory_id", "points_data"}
        if not required.issubset(set(fields)):
            raise ValueError(
                f"plaintext catalog must contain {sorted(required)}; got {fields}"
            )
        return fields

    def _dict_to_plaintext(self, row: dict[str, str], row_no: int) -> TrajectoryPlaintext:
        tid = str(row["trajectory_id"]).strip()
        source_tid = str(row.get("source_traj_id", "")).strip()
        node_text = str(row.get("road_node_path", "")).strip()
        node_path = json.loads(node_text) if node_text else []
        fmt = str(row.get("points_format", "")).strip() or "node_lon_lat_time"
        points = json.loads(row["points_data"])
        if not isinstance(points, list):
            raise ValueError(f"row {row_no}: points_data must be a JSON list")
        return TrajectoryPlaintext(
            trajectory_id=tid,
            source_traj_id=source_tid,
            road_node_path=[int(x) for x in node_path],
            points_data=points,
            points_format=fmt,
        )

    def get(self, trajectory_id: str) -> TrajectoryPlaintext:
        records = self.get_many([trajectory_id])
        return records[0]

    def get_many(
        self,
        trajectory_ids: Iterable[str],
        *,
        progress: Callable[[str, int, int], None] | None = None,
        total_rows_hint: int | None = None,
    ) -> list[TrajectoryPlaintext]:
        tids = sorted(
            set(str(x) for x in trajectory_ids),
            key=stable_trajectory_sort_key,
        )
        if not tids:
            if progress is not None:
                progress("CSV scan (0 candidates)", 1, 1)
            return []

        wanted = set(tids)
        found: dict[str, TrajectoryPlaintext] = {}
        total_hint = max(1, int(total_rows_hint or 1))
        report_step = max(1, total_hint // 100) if total_rows_hint else 5000
        last_report = 0
        rows_scanned = 0
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            required = {"trajectory_id", "points_data"}
            if not required.issubset(fields):
                raise ValueError(
                    f"plaintext catalog must contain {sorted(required)}; got {reader.fieldnames}"
                )
            for row_no, row in enumerate(reader, 2):
                rows_scanned += 1
                if progress is not None and rows_scanned - last_report >= report_step:
                    if total_rows_hint:
                        progress(
                            f"CSV scan: found {len(found)}/{len(wanted)} candidates",
                            min(rows_scanned, total_hint),
                            total_hint,
                        )
                    else:
                        progress(
                            f"CSV scan: {rows_scanned:,} rows, found {len(found)}/{len(wanted)}",
                            0,
                            1,
                        )
                    last_report = rows_scanned

                tid = str(row["trajectory_id"]).strip()
                if tid not in wanted:
                    continue
                if tid in found:
                    raise ValueError(f"duplicate trajectory_id in plaintext catalog: {tid}")
                found[tid] = self._dict_to_plaintext(row, row_no)
                if len(found) == len(wanted):
                    break

        missing = [tid for tid in tids if tid not in found]
        if missing:
            raise KeyError(
                f"plaintext catalog missing {len(missing)} trajectory IDs; sample={missing[:5]}"
            )
        if progress is not None:
            progress(
                f"CSV scan complete: {len(found)}/{len(wanted)} found after {rows_scanned:,} rows",
                1,
                1,
            )
        return [found[tid] for tid in tids]

    def missing_from(self, trajectory_ids: Iterable[str], *, limit: int | None = None) -> list[str]:
        """Compatibility helper using one sequential CSV scan.

        Formal query execution does not need a side index.  This method exists
        only for callers that explicitly request catalog-coverage checking.
        """
        tids = list(dict.fromkeys(str(x) for x in trajectory_ids))
        if not tids:
            return []
        remaining = set(tids)
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                remaining.discard(str(row["trajectory_id"]).strip())
                if not remaining:
                    break
        missing = [tid for tid in tids if tid in remaining]
        if limit is not None:
            missing = missing[:limit]
        return missing

    def trajectory_ids(self) -> set[str]:
        out: set[str] = set()
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.add(str(row["trajectory_id"]).strip())
        return out


@dataclass(slots=True)
class PlaintextStageResult:
    payloads: list[TrajectoryPlaintext]
    final_trajectory_ids: list[str]
    lookup_time_s: float
    plaintext_verification_time_s: float
    fine_filtering_time_s: float


def run_plaintext_stage(
    catalog: TrajectoryCatalog,
    candidate_trajectory_ids: Iterable[str],
    query: QueryWindow,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    catalog_total_rows_hint: int | None = None,
) -> PlaintextStageResult:
    """Lookup -> verify plaintext ID -> exact Fine Filtering.

    lookup_time_s is measured only for diagnostics and MUST NOT be added to
    formal Query Time.
    """
    candidate_ids = sorted(set(candidate_trajectory_ids), key=stable_trajectory_sort_key)

    t0 = time.perf_counter()
    if isinstance(catalog, DiskTrajectoryCatalog):
        payloads = catalog.get_many(
            candidate_ids,
            progress=progress,
            total_rows_hint=catalog_total_rows_hint,
        )
    else:
        payloads = catalog.get_many(candidate_ids)
        if progress is not None:
            progress(f"Plaintext lookup: {len(payloads)} trajectories", 1, 1)
    t1 = time.perf_counter()

    payload_ids = [p.trajectory_id for p in payloads]
    if payload_ids != candidate_ids:
        raise ValueError("plaintext payload set differs from verified candidate set")

    for payload in payloads:
        if not payload.verify_trajectory_id():
            raise ValueError(
                f"plaintext trajectory_id verification failed: {payload.trajectory_id}"
            )
    t2 = time.perf_counter()

    final_ids: list[str] = []
    total_payloads = len(payloads)
    report_step = max(1, total_payloads // 100) if total_payloads else 1
    for i, payload in enumerate(payloads, 1):
        if trajectory_intersects_query(payload.gps_points(), query):
            final_ids.append(payload.trajectory_id)
        if progress is not None and (i == total_payloads or i % report_step == 0):
            progress(
                f"Exact result verification: matched {len(final_ids)}/{i}",
                i,
                max(total_payloads, 1),
            )
    if progress is not None and total_payloads == 0:
        progress("Exact result verification: 0 candidates", 1, 1)
    t3 = time.perf_counter()

    return PlaintextStageResult(
        payloads=payloads,
        final_trajectory_ids=final_ids,
        lookup_time_s=t1 - t0,
        plaintext_verification_time_s=t2 - t1,
        fine_filtering_time_s=t3 - t2,
    )
