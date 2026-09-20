from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from road_network import RoadNetwork
from crypto import hash_mmr_leaf, hash_mmr_internal, time_overlap
from l1_merkle import hash_l1_leaf, hash_l1_internal
from l0_authenticated import hash_l0_leaf, hash_l0_internal
from query_utils import UINT32_MAX, normalize_rectangle, rectangle_overlap, edge_intersects_query, is_empty_time_range
from composite_vo import (
    TOKEN_L0_PRUNED_LEAF, TOKEN_L0_PRUNED_INTERNAL, TOKEN_L0_LEAF, TOKEN_L0_INTERNAL,
    TOKEN_L1_EDGE_OPAQUE, TOKEN_L1_EDGE_EXPANDED, TOKEN_L1_INTERNAL,
    TOKEN_MMR_PRUNED_LEAF, TOKEN_MMR_PRUNED_INTERNAL, TOKEN_MMR_RESULT_SLOT,
    TOKEN_MMR_INTERNAL, TOKEN_MMR_ROOT,
)

HASH_SIZE = 32
_U32 = struct.Struct(">I")
_U32_2 = struct.Struct(">II")
_sha256 = hashlib.sha256

@dataclass(slots=True)
class VerificationReport:
    success: bool
    reconstructed_root: bytes
    trusted_root: bytes
    token_count: int
    verification_entry_count: int
    consumed_verification_entries: int
    candidate_ids: list[bytes]

STACK_MMR_NODE = 1
STACK_MMR_ROOT = 2
STACK_L1_NODE = 3
STACK_L0_NODE = 4

@dataclass(slots=True)
class StackItem:
    kind: int
    node_hash: bytes
    min_start: int = UINT32_MAX
    max_end: int = 0
    k: int = 0
    edge_eid: int = -1
    min_lon: float = 0.0
    min_lat: float = 0.0
    max_lon: float = 0.0
    max_lat: float = 0.0

class VerificationError(RuntimeError):
    pass

def calculate_empty_mmr_root() -> bytes:
    return _sha256(_U32.pack(0)).digest()

EMPTY_MMR_ROOT = calculate_empty_mmr_root()

def validate_hash(value: bytes, name: str) -> None:
    if not isinstance(value, bytes) or len(value) != HASH_SIZE:
        raise VerificationError(f"{name} is not a valid 32-byte hash")

def validate_eid(eid: int, road: RoadNetwork) -> None:
    if eid < 0 or eid > road.max_eid:
        raise VerificationError(f"Invalid eid={eid}")
    if not road.edge_exists(eid):
        raise VerificationError(f"eid={eid} is not a valid road edge")

def merge_edge_eid(left_eid: int, right_eid: int) -> int:
    if left_eid >= 0 and right_eid >= 0 and left_eid != right_eid:
        raise VerificationError("Different eid values appeared in Verification Entries within the same MMR")
    return left_eid if left_eid >= 0 else right_eid

def l0_can_prune(
    min_lon, min_lat, max_lon, max_lat, min_start, max_end,
    query_min_lon, query_min_lat, query_max_lon, query_max_lat,
    query_start, query_end,
) -> bool:
    if not rectangle_overlap(
        min_lon, min_lat, max_lon, max_lat,
        query_min_lon, query_min_lat, query_max_lon, query_max_lat,
    ):
        return True
    if is_empty_time_range(min_start, max_end):
        return True
    if not time_overlap(min_start, max_end, query_start, query_end):
        return True
    return False

def calculate_mmr_root_from_peaks(k: int, peaks: list[StackItem]) -> bytes:
    h = _sha256()
    h.update(_U32.pack(k))
    for peak in peaks:
        h.update(peak.node_hash)
        h.update(_U32_2.pack(peak.min_start, peak.max_end))
    return h.digest()

def verify_composite_vo(
    tokens: list[tuple],
    verification_set,
    road: RoadNetwork,
    trusted_root: bytes,
    query_min_lon: float,
    query_min_lat: float,
    query_max_lon: float,
    query_max_lat: float,
    query_start: int,
    query_end: int,
) -> VerificationReport:
    validate_hash(trusted_root, "trusted_root")
    if query_start > query_end:
        raise VerificationError("query_start > query_end")

    query_min_lon, query_min_lat, query_max_lon, query_max_lat = normalize_rectangle(
        query_min_lon, query_min_lat, query_max_lon, query_max_lat
    )

    stack: list[StackItem] = []
    verification_index = 0

    for token_position, token in enumerate(tokens):
        token_type = token[0]

        if token_type == TOKEN_MMR_PRUNED_LEAF:
            _, trajectory_id, start, end = token
            if len(trajectory_id) != 32:
                raise VerificationError("MMR_PRUNED_LEAF trajectory_id has invalid length")
            if start > end:
                raise VerificationError("MMR_PRUNED_LEAF start > end")
            if time_overlap(start, end, query_start, query_end):
                raise VerificationError("MMR_PRUNED_LEAF actually overlaps the query time range")
            stack.append(StackItem(
                kind=STACK_MMR_NODE,
                node_hash=hash_mmr_leaf(trajectory_id, start, end),
                min_start=start,
                max_end=end,
            ))
            continue

        if token_type == TOKEN_MMR_PRUNED_INTERNAL:
            _, min_start, max_end, left_hash, right_hash = token
            validate_hash(left_hash, "MMR pruned left hash")
            validate_hash(right_hash, "MMR pruned right hash")
            if min_start > max_end:
                raise VerificationError("MMR_PRUNED_INTERNAL has an invalid time range")
            if time_overlap(min_start, max_end, query_start, query_end):
                raise VerificationError("MMR_PRUNED_INTERNAL overlaps the query time range and cannot be pruned")
            stack.append(StackItem(
                kind=STACK_MMR_NODE,
                node_hash=hash_mmr_internal(min_start, max_end, left_hash, right_hash),
                min_start=min_start,
                max_end=max_end,
            ))
            continue

        if token_type == TOKEN_MMR_RESULT_SLOT:
            if verification_index >= len(verification_set):
                raise VerificationError("Insufficient Verification Set entries")
            entry = verification_set[verification_index]
            verification_index += 1
            validate_eid(entry.eid, road)
            if len(entry.trajectory_id) != 32:
                raise VerificationError("Verification Entry trajectory_id has invalid length")
            if entry.start > entry.end:
                raise VerificationError("Verification Entry start > end")
            if not time_overlap(entry.start, entry.end, query_start, query_end):
                raise VerificationError("Verification Entry does not overlap the query time range")
            if not edge_intersects_query(
                eid=entry.eid, road=road,
                query_min_lon=query_min_lon, query_min_lat=query_min_lat,
                query_max_lon=query_max_lon, query_max_lat=query_max_lat,
            ):
                raise VerificationError(f"Verification Entry eid={entry.eid} does not intersect the query region")
            stack.append(StackItem(
                kind=STACK_MMR_NODE,
                node_hash=hash_mmr_leaf(entry.trajectory_id, entry.start, entry.end),
                min_start=entry.start,
                max_end=entry.end,
                edge_eid=entry.eid,
            ))
            continue

        if token_type == TOKEN_MMR_INTERNAL:
            _, min_start, max_end = token
            if len(stack) < 2:
                raise VerificationError("Insufficient stack elements for MMR_INTERNAL")
            right = stack.pop()
            left = stack.pop()
            if left.kind != STACK_MMR_NODE or right.kind != STACK_MMR_NODE:
                raise VerificationError("Invalid child type for MMR_INTERNAL")
            if min_start != min(left.min_start, right.min_start) or max_end != max(left.max_end, right.max_end):
                raise VerificationError("Incorrect min_start/max_end aggregation for MMR_INTERNAL")
            edge_eid = merge_edge_eid(left.edge_eid, right.edge_eid)
            stack.append(StackItem(
                kind=STACK_MMR_NODE,
                node_hash=hash_mmr_internal(min_start, max_end, left.node_hash, right.node_hash),
                min_start=min_start,
                max_end=max_end,
                edge_eid=edge_eid,
            ))
            continue

        if token_type == TOKEN_MMR_ROOT:
            _, k_e = token
            if k_e <= 0:
                raise VerificationError("k_e must be > 0 for an expanded MMR")
            peak_count = k_e.bit_count()
            if len(stack) < peak_count:
                raise VerificationError("Insufficient number of Peaks for MMR_ROOT")
            peaks = []
            for _ in range(peak_count):
                peak = stack.pop()
                if peak.kind != STACK_MMR_NODE:
                    raise VerificationError("A non-MMR node appeared before MMR_ROOT")
                peaks.append(peak)
            peaks.reverse()
            edge_eid = -1
            for peak in peaks:
                edge_eid = merge_edge_eid(edge_eid, peak.edge_eid)
            stack.append(StackItem(
                kind=STACK_MMR_ROOT,
                node_hash=calculate_mmr_root_from_peaks(k_e, peaks),
                min_start=min(p.min_start for p in peaks),
                max_end=max(p.max_end for p in peaks),
                k=k_e,
                edge_eid=edge_eid,
            ))
            continue

        if token_type == TOKEN_L1_EDGE_OPAQUE:
            _, eid, k_e, min_start, max_end, root_e = token
            validate_eid(eid, road)
            validate_hash(root_e, "root_e")
            if k_e == 0:
                if min_start != UINT32_MAX or max_end != 0:
                    raise VerificationError(f"eid={eid} has invalid empty-MMR time sentinel values")
                if root_e != EMPTY_MMR_ROOT:
                    raise VerificationError(f"eid={eid} has an invalid empty-MMR root")
            else:
                if min_start == UINT32_MAX or min_start > max_end:
                    raise VerificationError(f"eid={eid} has an invalid non-empty MMR time range")
            spatial_match = edge_intersects_query(
                eid=eid, road=road,
                query_min_lon=query_min_lon, query_min_lat=query_min_lat,
                query_max_lon=query_max_lon, query_max_lat=query_max_lat,
            )
            opaque_legal = (
                (not spatial_match)
                or k_e == 0
                or not time_overlap(min_start, max_end, query_start, query_end)
            )
            if not opaque_legal:
                raise VerificationError(f"eid={eid} should expand its MMR but OPAQUE was used")
            stack.append(StackItem(
                kind=STACK_L1_NODE,
                node_hash=hash_l1_leaf(
                    eid=eid, k_e=k_e, min_start=min_start, max_end=max_end, root_e=root_e
                ),
            ))
            continue

        if token_type == TOKEN_L1_EDGE_EXPANDED:
            _, eid, k_e, min_start, max_end = token
            validate_eid(eid, road)
            if not stack:
                raise VerificationError("No MMR_ROOT before L1_EDGE_EXPANDED")
            mmr_root = stack.pop()
            if mmr_root.kind != STACK_MMR_ROOT:
                raise VerificationError("The item before L1_EDGE_EXPANDED is not MMR_ROOT")
            if mmr_root.k != k_e:
                raise VerificationError(f"eid={eid} k_e does not match MMR_ROOT")
            if mmr_root.edge_eid >= 0 and mmr_root.edge_eid != eid:
                raise VerificationError("eid in the Verification Set does not match the L1 Edge eid")
            if mmr_root.min_start != min_start or mmr_root.max_end != max_end:
                raise VerificationError(f"eid={eid} MMR time range does not match the L1 Edge")
            if k_e <= 0:
                raise VerificationError("An empty MMR should not be EXPANDED")
            if not edge_intersects_query(
                eid=eid, road=road,
                query_min_lon=query_min_lon, query_min_lat=query_min_lat,
                query_max_lon=query_max_lon, query_max_lat=query_max_lat,
            ):
                raise VerificationError(f"eid={eid} is EXPANDED but does not intersect the query region")
            if not time_overlap(min_start, max_end, query_start, query_end):
                raise VerificationError(f"eid={eid} is EXPANDED but does not overlap the query time range")
            stack.append(StackItem(
                kind=STACK_L1_NODE,
                node_hash=hash_l1_leaf(
                    eid=eid, k_e=k_e, min_start=min_start, max_end=max_end, root_e=mmr_root.node_hash
                ),
            ))
            continue

        if token_type == TOKEN_L1_INTERNAL:
            if len(stack) < 2:
                raise VerificationError("Insufficient stack elements for L1_INTERNAL")
            right = stack.pop()
            left = stack.pop()
            if left.kind != STACK_L1_NODE or right.kind != STACK_L1_NODE:
                raise VerificationError("Invalid child type for L1_INTERNAL")
            stack.append(StackItem(
                kind=STACK_L1_NODE,
                node_hash=hash_l1_internal(left.node_hash, right.node_hash),
            ))
            continue

        if token_type == TOKEN_L0_PRUNED_LEAF:
            _, min_lon, min_lat, max_lon, max_lat, min_start, max_end, l1_root = token
            validate_hash(l1_root, "L0 pruned L1root")
            if not l0_can_prune(
                min_lon, min_lat, max_lon, max_lat, min_start, max_end,
                query_min_lon, query_min_lat, query_max_lon, query_max_lat,
                query_start, query_end,
            ):
                raise VerificationError("L0_PRUNED_LEAF does not satisfy the pruning condition")
            stack.append(StackItem(
                kind=STACK_L0_NODE,
                node_hash=hash_l0_leaf(
                    min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
                    min_start=min_start, max_end=max_end, l1_root=l1_root,
                ),
                min_start=min_start, max_end=max_end,
                min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
            ))
            continue

        if token_type == TOKEN_L0_PRUNED_INTERNAL:
            (
                _, min_lon, min_lat, max_lon, max_lat, min_start, max_end,
                left_hash, mid_hash, right_hash,
            ) = token
            validate_hash(left_hash, "L0 left hash")
            validate_hash(mid_hash, "L0 mid hash")
            validate_hash(right_hash, "L0 right hash")
            if not l0_can_prune(
                min_lon, min_lat, max_lon, max_lat, min_start, max_end,
                query_min_lon, query_min_lat, query_max_lon, query_max_lat,
                query_start, query_end,
            ):
                raise VerificationError("L0_PRUNED_INTERNAL does not satisfy the pruning condition")
            stack.append(StackItem(
                kind=STACK_L0_NODE,
                node_hash=hash_l0_internal(
                    min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
                    min_start=min_start, max_end=max_end,
                    left_hash=left_hash, mid_hash=mid_hash, right_hash=right_hash,
                ),
                min_start=min_start, max_end=max_end,
                min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
            ))
            continue

        if token_type == TOKEN_L0_LEAF:
            _, min_lon, min_lat, max_lon, max_lat, min_start, max_end = token
            if not stack:
                raise VerificationError("No L1 root before L0_LEAF")
            l1_root = stack.pop()
            if l1_root.kind != STACK_L1_NODE:
                raise VerificationError("The item before L0_LEAF is not an L1 root")
            if l0_can_prune(
                min_lon, min_lat, max_lon, max_lat, min_start, max_end,
                query_min_lon, query_min_lat, query_max_lon, query_max_lat,
                query_start, query_end,
            ):
                raise VerificationError("L0_LEAF could have been pruned directly but was expanded")
            stack.append(StackItem(
                kind=STACK_L0_NODE,
                node_hash=hash_l0_leaf(
                    min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
                    min_start=min_start, max_end=max_end, l1_root=l1_root.node_hash,
                ),
                min_start=min_start, max_end=max_end,
                min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
            ))
            continue

        if token_type == TOKEN_L0_INTERNAL:
            _, min_lon, min_lat, max_lon, max_lat, min_start, max_end = token
            if len(stack) < 3:
                raise VerificationError("Insufficient stack elements for L0_INTERNAL")
            right = stack.pop()
            mid = stack.pop()
            left = stack.pop()
            if left.kind != STACK_L0_NODE or mid.kind != STACK_L0_NODE or right.kind != STACK_L0_NODE:
                raise VerificationError("Invalid child type for L0_INTERNAL")
            expected_min_start = min(left.min_start, mid.min_start, right.min_start)
            expected_max_end = max(left.max_end, mid.max_end, right.max_end)
            if min_start != expected_min_start or max_end != expected_max_end:
                raise VerificationError("Incorrect time-range aggregation for L0_INTERNAL")
            stack.append(StackItem(
                kind=STACK_L0_NODE,
                node_hash=hash_l0_internal(
                    min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
                    min_start=min_start, max_end=max_end,
                    left_hash=left.node_hash, mid_hash=mid.node_hash, right_hash=right.node_hash,
                ),
                min_start=min_start, max_end=max_end,
                min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
            ))
            continue

        raise VerificationError(f"Unknown Token type: {token_type}, position={token_position}")

    if verification_index != len(verification_set):
        raise VerificationError(
            f"Verification Set was not fully consumed: {verification_index}/{len(verification_set)}"
        )

    if len(stack) != 1:
        raise VerificationError(f"Stack size after verification is {len(stack)}, expected 1")

    root_item = stack[0]
    if root_item.kind != STACK_L0_NODE:
        raise VerificationError("The final stack item is not the L0 Root")

    reconstructed_root = root_item.node_hash

    if reconstructed_root != trusted_root:
        raise VerificationError(
            "\nroot_S verification failed\n"
            f"reconstructed = {reconstructed_root.hex()}\n"
            f"trusted       = {trusted_root.hex()}"
        )

    candidate_ids = sorted({entry.trajectory_id for entry in verification_set})

    return VerificationReport(
        success=True,
        reconstructed_root=reconstructed_root,
        trusted_root=trusted_root,
        token_count=len(tokens),
        verification_entry_count=len(verification_set),
        consumed_verification_entries=verification_index,
        candidate_ids=candidate_ids,
    )
