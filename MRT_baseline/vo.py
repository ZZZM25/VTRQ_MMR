from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .formats import BBOX, POINT_DTYPE, POINT_STRUCT, U16, U32, internal_hash, leaf_hash, root_commitment
from .geometry import bbox_intersects, bbox_union
from .models import InternalMeta, LeafMeta, QueryWindow, VOVerificationResult
from .tree import VO_INTERNAL, VO_LEAF, VO_PRUNED


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated VO")
    return data


def verify_vo(
    vo_path: str | Path,
    query: QueryWindow,
    trusted_root: bytes,
    server_candidate_ids: list[str],
) -> VOVerificationResult:
    """Verify spatial pruning, leaf records, same-point time test, candidates, root."""
    candidates: set[bytes] = set()

    with Path(vo_path).open("rb", buffering=8 * 1024 * 1024) as f:
        def parse_node():
            tag = _read_exact(f, 1)[0]

            if tag == VO_PRUNED:
                box = BBOX.unpack(_read_exact(f, BBOX.size))
                digest = _read_exact(f, 32)
                if bbox_intersects(box, query):
                    raise ValueError("invalid VO pruning: pruned MBR intersects spatial query")
                return InternalMeta(*box, digest, 0, 0)

            if tag == VO_INTERNAL:
                child_count = U16.unpack(_read_exact(f, U16.size))[0]
                if child_count <= 0:
                    raise ValueError("empty internal node in VO")
                children = [parse_node() for _ in range(child_count)]
                box = bbox_union([c.bbox for c in children])
                return InternalMeta(*box, internal_hash(children), 0, child_count)

            if tag == VO_LEAF:
                count = U32.unpack(_read_exact(f, U32.size))[0]
                if count <= 0:
                    raise ValueError("empty leaf in VO")
                raw = _read_exact(f, count * POINT_STRUCT.size)
                arr = np.frombuffer(raw, dtype=POINT_DTYPE, count=count)
                box = (
                    float(arr["lon"].min()), float(arr["lat"].min()),
                    float(arr["lon"].max()), float(arr["lat"].max()),
                )
                if not bbox_intersects(box, query):
                    raise ValueError("expanded leaf does not intersect spatial query")

                space_mask = (
                    (arr["lon"] >= query.min_lon)
                    & (arr["lon"] <= query.max_lon)
                    & (arr["lat"] >= query.min_lat)
                    & (arr["lat"] <= query.max_lat)
                )
                both = space_mask & (arr["ts"] >= query.start_time) & (arr["ts"] <= query.end_time)
                for tid in arr[both]["tid"]:
                    candidates.add(bytes(tid))
                return LeafMeta(
                    *box, leaf_hash(raw, count), 0, 0, count, 0, 0
                )

            raise ValueError(f"unknown VO tag: {tag}")

        t0 = time.perf_counter()
        root = parse_node()
        if f.read(1):
            raise ValueError("extra bytes after VO root")
        reconstructed = root_commitment(root)
        if reconstructed != trusted_root:
            raise ValueError(
                f"R-tree root mismatch: reconstructed={reconstructed.hex()} trusted={trusted_root.hex()}"
            )
        verified = sorted(x.hex() for x in candidates)
        if verified != sorted(server_candidate_ids):
            raise ValueError("candidate trajectory set differs from VO recomputation")
        elapsed = time.perf_counter() - t0

    return VOVerificationResult(
        verified_candidate_trajectory_ids=verified,
        vo_verification_time_s=elapsed,
        reconstructed_root_hex=reconstructed.hex(),
    )
