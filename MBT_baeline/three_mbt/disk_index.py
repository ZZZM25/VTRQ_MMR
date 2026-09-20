from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Callable

from .disk_mbt import DiskMerkleBTree
from .hashing import aggregate_three_roots
from .index import QueryResult, VerificationResult
from .models import QueryWindow, stable_trajectory_sort_key
from .vo_size import composite_three_mbt_vo_size


class DiskThreeMBTIndex:
    """Three-MBT index with compact leaf records on disk and Merkle metadata in RAM."""

    FORMAT = "THREE-MBT-DISK-v1"

    def __init__(
        self,
        trajectory_ids: list[str],
        leaf_capacity: int = 128,
        fanout: int = 64,
    ) -> None:
        self.trajectory_ids = list(trajectory_ids)
        self.mbt_lon = DiskMerkleBTree(
            "lon", leaf_capacity, fanout, trajectory_ids=self.trajectory_ids,
            entries_filename="lon.entries.bin",
        )
        self.mbt_lat = DiskMerkleBTree(
            "lat", leaf_capacity, fanout, trajectory_ids=self.trajectory_ids,
            entries_filename="lat.entries.bin",
        )
        self.mbt_time = DiskMerkleBTree(
            "time", leaf_capacity, fanout, trajectory_ids=self.trajectory_ids,
            entries_filename="time.entries.bin",
        )
        self.point_count = 0
        self.trajectory_ids_seen = set(self.trajectory_ids)
        self._storage_dir: str | None = None

    def bind_storage(self, storage_dir: str | Path) -> None:
        self._storage_dir = str(Path(storage_dir))
        self.mbt_lon.bind_storage(storage_dir)
        self.mbt_lat.bind_storage(storage_dir)
        self.mbt_time.bind_storage(storage_dir)

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

    def query(
        self,
        q: QueryWindow,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> QueryResult:
        # Progress callbacks are deliberately invoked outside the timed sections.
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
        lon_tids, root_lon = DiskMerkleBTree.reconstruct_range_proof(result.vo_lon)
        lon_verify_time = time.perf_counter() - t0
        if progress is not None:
            progress("VO-Lon verify", 1, 5)

        t0 = time.perf_counter()
        lat_tids, root_lat = DiskMerkleBTree.reconstruct_range_proof(result.vo_lat)
        lat_verify_time = time.perf_counter() - t0
        if progress is not None:
            progress("VO-Lat verify", 2, 5)

        t0 = time.perf_counter()
        time_tids, root_time = DiskMerkleBTree.reconstruct_range_proof(result.vo_time)
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
                lon_verify_time
                + lat_verify_time
                + time_verify_time
                + root_aggregation_time
                + candidate_recompute_time
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
        if self._storage_dir is None:
            self.bind_storage(Path(str(path) + ".data"))
        with path.open("wb") as f:
            pickle.dump({"format": self.FORMAT, "index": self}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "DiskThreeMBTIndex":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("format") != cls.FORMAT:
            raise TypeError("file does not contain DiskThreeMBTIndex")
        obj = payload["index"]
        if not isinstance(obj, cls):
            raise TypeError("invalid DiskThreeMBTIndex payload")
        obj.bind_storage(Path(str(path) + ".data"))
        return obj

    def disk_size_bytes(self, manifest_path: str | Path | None = None) -> int:
        total = 0
        if manifest_path is not None:
            p = Path(manifest_path)
            if p.exists():
                total += p.stat().st_size
        total += self.mbt_lon.disk_bytes()
        total += self.mbt_lat.disk_bytes()
        total += self.mbt_time.disk_bytes()
        return total

    def serialized_size_bytes(self) -> int:
        # Metadata-only serialization; disk leaves are counted separately by disk_size_bytes().
        return len(pickle.dumps({"format": self.FORMAT, "index": self}, protocol=pickle.HIGHEST_PROTOCOL))

    def stats(self) -> dict:
        return {
            "points": self.point_count,
            "trajectories": len(self.trajectory_ids_seen),
            "combined_root": self.combined_root_hex,
            "lon": self.mbt_lon.stats(),
            "lat": self.mbt_lat.stats(),
            "time": self.mbt_time.stats(),
        }
