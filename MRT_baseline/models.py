from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryWindow:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    start_time: float
    end_time: float


@dataclass(slots=True)
class TrajectoryRecord:
    ordinal: int
    trajectory_id: str
    source_traj_id: str
    road_node_path: list[int]
    gps_points: list[tuple[float, float, float]]  # timestamp, latitude, longitude
    plaintext_points: list[list[float | int]]
    points_format: str


@dataclass(frozen=True, slots=True)
class LeafMeta:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    digest: bytes
    point_file_id: int
    offset: int
    count: int
    min_morton: int
    max_morton: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.min_lon, self.min_lat, self.max_lon, self.max_lat


@dataclass(frozen=True, slots=True)
class InternalMeta:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    digest: bytes
    first_child: int
    child_count: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.min_lon, self.min_lat, self.max_lon, self.max_lat


@dataclass(slots=True)
class QueryResult:
    candidate_trajectory_ids: list[str]
    query_time_s: float
    vo_path: str
    vo_size_bytes: int
    spatial_match_point_count: int
    spatiotemporal_match_point_count: int
    expanded_leaf_count: int


@dataclass(slots=True)
class VOVerificationResult:
    verified_candidate_trajectory_ids: list[str]
    vo_verification_time_s: float
    reconstructed_root_hex: str


@dataclass(slots=True)
class ExactVerificationResult:
    final_trajectory_ids: list[str]
    lookup_time_s: float
    plaintext_id_check_time_s: float
    exact_result_verification_time_s: float
