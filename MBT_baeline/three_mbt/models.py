from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class GPSPoint:
    point_id: int
    trajectory_id: str
    timestamp: float
    latitude: float
    longitude: float

    def key(self, dimension: str) -> float:
        if dimension == "lon":
            return float(self.longitude)
        if dimension == "lat":
            return float(self.latitude)
        if dimension == "time":
            return float(self.timestamp)
        raise ValueError(f"unsupported dimension: {dimension}")


@dataclass(frozen=True, slots=True)
class QueryWindow:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if self.min_lon > self.max_lon:
            raise ValueError("min_lon > max_lon")
        if self.min_lat > self.max_lat:
            raise ValueError("min_lat > max_lat")
        if self.start_time > self.end_time:
            raise ValueError("start_time > end_time")


@dataclass(frozen=True, slots=True)
class FineFilterResult:
    trajectory_id: str
    matched: bool


def stable_trajectory_sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def normalize_points(points: Iterable[GPSPoint]) -> list[GPSPoint]:
    return sorted(
        points,
        key=lambda p: (stable_trajectory_sort_key(p.trajectory_id), p.timestamp, p.point_id),
    )
