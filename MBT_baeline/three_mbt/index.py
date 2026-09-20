from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .hashing import aggregate_three_roots
from .mbt import MerkleBTree
from .models import GPSPoint, QueryWindow, stable_trajectory_sort_key
from .vo_size import composite_three_mbt_vo_size


@dataclass(slots=True)
class QueryResult:
    query: QueryWindow
    lon_trajectory_ids: set[str]
    lat_trajectory_ids: set[str]
    time_trajectory_ids: set[str]
    candidate_trajectory_ids: list[str]
    vo_lon: dict
    vo_lat: dict
    vo_time: dict
    lon_query_time_s: float
    lat_query_time_s: float
    time_query_time_s: float
    trajectory_id_intersection_time_s: float
    coarse_query_time_s: float
    precomputed_vo_size_bytes: int | None = None
    precomputed_trusted_global_root_hex: str | None = None

    def composite_vo_bundle(self, trusted_global_root_hex: str) -> dict:
        return {
            "version": 2,
            "query": {
                "min_lon": self.query.min_lon,
                "max_lon": self.query.max_lon,
                "min_lat": self.query.min_lat,
                "max_lat": self.query.max_lat,
                "start_time": self.query.start_time,
                "end_time": self.query.end_time,
            },
            "trusted_global_root": trusted_global_root_hex,
            "vo_lon": self.vo_lon,
            "vo_lat": self.vo_lat,
            "vo_time": self.vo_time,
        }

    def vo_size_bytes(self, trusted_global_root_hex: str) -> int:
        """Return exact compact-JSON VO size.

        Normal query paths compute this incrementally while the VO is being built,
        so this method is constant-time even for very large proofs.  The streaming
        fallback is retained only for manually constructed/legacy QueryResult
        objects that do not carry the precomputed size.
        """
        if (
            self.precomputed_vo_size_bytes is not None
            and self.precomputed_trusted_global_root_hex == trusted_global_root_hex
        ):
            return int(self.precomputed_vo_size_bytes)

        encoder = json.JSONEncoder(separators=(",", ":"), sort_keys=True)
        total = 0
        for chunk in encoder.iterencode(
            self.composite_vo_bundle(trusted_global_root_hex)
        ):
            total += len(chunk.encode("utf-8"))
        return total


@dataclass(slots=True)
class VerificationResult:
    verified: bool
    lon_verification_time_s: float
    lat_verification_time_s: float
    time_verification_time_s: float
    root_aggregation_time_s: float
    candidate_recompute_time_s: float
    mbt_verification_time_s: float
    reconstructed_root_lon: str
    reconstructed_root_lat: str
    reconstructed_root_time: str
    reconstructed_global_root: str
    verified_candidate_trajectory_ids: list[str]


class ThreeMBTIndex:
    """Three authenticated MBTs over raw/interpolated GPS points.

    Every leaf entry is:
        (dimension_key, point_id, paper_trajectory_id)

    The three range queries return trajectory-ID sets directly.  Coarse filtering
    is the intersection of those three trajectory-ID sets.
    """

    def __init__(self, leaf_capacity: int = 128, fanout: int = 64) -> None:
        self.mbt_lon = MerkleBTree("lon", leaf_capacity, fanout)
        self.mbt_lat = MerkleBTree("lat", leaf_capacity, fanout)
        self.mbt_time = MerkleBTree("time", leaf_capacity, fanout)
        self.point_count = 0
        self.trajectory_ids_seen: set[str] = set()

    @property
    def combined_root(self) -> bytes:
        return aggregate_three_roots(
            self.mbt_lon.root_hash,
            self.mbt_lat.root_hash,
            self.mbt_time.root_hash,
        )

    @property
    def combined_root_hex(self) -> str:
        return self.combined_root.hex()

    def bulk_build(self, points: Iterable[GPSPoint]) -> None:
        pts = list(points)
        self.mbt_lon.bulk_build(pts)
        self.mbt_lat.bulk_build(pts)
        self.mbt_time.bulk_build(pts)
        self.point_count = len(pts)
        self.trajectory_ids_seen = {p.trajectory_id for p in pts}
        _ = self.combined_root

    def incremental_insert_points(self, points: Iterable[GPSPoint]) -> None:
        pts = list(points)
        for p in pts:
            self.mbt_lon.insert_point(p)
        for p in pts:
            self.mbt_lat.insert_point(p)
        for p in pts:
            self.mbt_time.insert_point(p)
        self.point_count += len(pts)
        self.trajectory_ids_seen.update(p.trajectory_id for p in pts)
        _ = self.combined_root

    def query(
        self,
        q: QueryWindow,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> QueryResult:
        """Server-side coarse query only.

        Order is fixed and serial: Lon -> Lat -> Time -> trajectory-ID intersection.
        No plaintext-file lookup occurs in this method.
        """
        t0 = time.perf_counter()
        lon_tids, vo_lon, vo_lon_size = self.mbt_lon.range_query_with_size(q.min_lon, q.max_lon)
        lon_time = time.perf_counter() - t0
        if progress is not None:
            progress("MBT-Lon query", 1, 4)

        t0 = time.perf_counter()
        lat_tids, vo_lat, vo_lat_size = self.mbt_lat.range_query_with_size(q.min_lat, q.max_lat)
        lat_time = time.perf_counter() - t0
        if progress is not None:
            progress("MBT-Lat query", 2, 4)

        t0 = time.perf_counter()
        time_tids, vo_time, vo_time_size = self.mbt_time.range_query_with_size(q.start_time, q.end_time)
        time_time = time.perf_counter() - t0
        if progress is not None:
            progress("MBT-Time query", 3, 4)

        t0 = time.perf_counter()
        candidate_ids = sorted(
            lon_tids & lat_tids & time_tids,
            key=stable_trajectory_sort_key,
        )
        intersection_time = time.perf_counter() - t0
        if progress is not None:
            progress("TID intersection", 4, 4)

        trusted_root_hex = self.combined_root_hex
        total_vo_size = composite_three_mbt_vo_size(
            min_lon=q.min_lon,
            max_lon=q.max_lon,
            min_lat=q.min_lat,
            max_lat=q.max_lat,
            start_time=q.start_time,
            end_time=q.end_time,
            trusted_global_root_hex=trusted_root_hex,
            vo_lon_size=vo_lon_size,
            vo_lat_size=vo_lat_size,
            vo_time_size=vo_time_size,
        )

        return QueryResult(
            query=q,
            lon_trajectory_ids=lon_tids,
            lat_trajectory_ids=lat_tids,
            time_trajectory_ids=time_tids,
            candidate_trajectory_ids=candidate_ids,
            vo_lon=vo_lon,
            vo_lat=vo_lat,
            vo_time=vo_time,
            lon_query_time_s=lon_time,
            lat_query_time_s=lat_time,
            time_query_time_s=time_time,
            trajectory_id_intersection_time_s=intersection_time,
            coarse_query_time_s=lon_time + lat_time + time_time + intersection_time,
            precomputed_vo_size_bytes=total_vo_size,
            precomputed_trusted_global_root_hex=trusted_root_hex,
        )

    def verify(
        self,
        result: QueryResult,
        trusted_global_root: bytes | str | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> VerificationResult:
        """Client-side verification of all three VOs and the combined trusted root.

        The client reconstructs each MBT root from the revealed leaves + sibling
        stubs. It then computes:

            H(Root_lon || Root_lat || Root_time)

        and compares that value with the trusted/on-chain global root.  Only after
        this succeeds does the client accept the authenticated TID sets and compute
        their intersection independently.
        """
        q = result.query
        expected_meta = [
            (result.vo_lon, "lon", q.min_lon, q.max_lon),
            (result.vo_lat, "lat", q.min_lat, q.max_lat),
            (result.vo_time, "time", q.start_time, q.end_time),
        ]
        for proof, dimension, low, high in expected_meta:
            if proof.get("dimension") != dimension:
                raise ValueError(f"VO dimension mismatch: expected {dimension}")
            if float(proof.get("low")) != float(low) or float(proof.get("high")) != float(high):
                raise ValueError(f"VO query bounds mismatch for {dimension}")

        if trusted_global_root is None:
            trusted = self.combined_root
        elif isinstance(trusted_global_root, str):
            trusted = bytes.fromhex(trusted_global_root)
        else:
            trusted = bytes(trusted_global_root)

        t0 = time.perf_counter()
        lon_tids, root_lon = MerkleBTree.reconstruct_range_proof(result.vo_lon)
        lon_verify_time = time.perf_counter() - t0
        if progress is not None:
            progress("VO-Lon verify", 1, 5)

        t0 = time.perf_counter()
        lat_tids, root_lat = MerkleBTree.reconstruct_range_proof(result.vo_lat)
        lat_verify_time = time.perf_counter() - t0
        if progress is not None:
            progress("VO-Lat verify", 2, 5)

        t0 = time.perf_counter()
        time_tids, root_time = MerkleBTree.reconstruct_range_proof(result.vo_time)
        time_verify_time = time.perf_counter() - t0
        if progress is not None:
            progress("VO-Time verify", 3, 5)

        t0 = time.perf_counter()
        reconstructed_global = aggregate_three_roots(root_lon, root_lat, root_time)
        if reconstructed_global != trusted:
            raise ValueError("combined trusted root mismatch")
        root_aggregation_time = time.perf_counter() - t0
        if progress is not None:
            progress("Combined-root verify", 4, 5)

        t0 = time.perf_counter()
        verified_candidates = sorted(
            lon_tids & lat_tids & time_tids,
            key=stable_trajectory_sort_key,
        )
        if verified_candidates != result.candidate_trajectory_ids:
            raise ValueError("server candidate trajectory-ID intersection mismatch")
        # The server also returns per-tree TID sets; do not trust them silently.
        if lon_tids != result.lon_trajectory_ids:
            raise ValueError("server Lon trajectory-ID set mismatch")
        if lat_tids != result.lat_trajectory_ids:
            raise ValueError("server Lat trajectory-ID set mismatch")
        if time_tids != result.time_trajectory_ids:
            raise ValueError("server Time trajectory-ID set mismatch")
        candidate_recompute_time = time.perf_counter() - t0
        if progress is not None:
            progress("Candidate recompute", 5, 5)

        return VerificationResult(
            verified=True,
            lon_verification_time_s=lon_verify_time,
            lat_verification_time_s=lat_verify_time,
            time_verification_time_s=time_verify_time,
            root_aggregation_time_s=root_aggregation_time,
            candidate_recompute_time_s=candidate_recompute_time,
            mbt_verification_time_s=(
                lon_verify_time + lat_verify_time + time_verify_time
                + root_aggregation_time + candidate_recompute_time
            ),
            reconstructed_root_lon=root_lon.hex(),
            reconstructed_root_lat=root_lat.hex(),
            reconstructed_root_time=root_time.hex(),
            reconstructed_global_root=reconstructed_global.hex(),
            verified_candidate_trajectory_ids=verified_candidates,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "ThreeMBTIndex":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("file does not contain ThreeMBTIndex")
        return obj

    def serialized_size_bytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def stats(self) -> dict:
        return {
            "points": self.point_count,
            "trajectories": len(self.trajectory_ids_seen),
            "combined_root": self.combined_root_hex,
            "lon": self.mbt_lon.stats(),
            "lat": self.mbt_lat.stats(),
            "time": self.mbt_time.stats(),
        }
