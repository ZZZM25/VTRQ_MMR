from __future__ import annotations

"""
Paper-aligned edge block layer.

The old implementation built a binary L1 Merkle tree under every SEM-tree leaf.
The paper definition used in this revision instead authenticates the complete,
eid-sorted edge-record list E_h as:

    rec_e  = (e, g_e, k_e, tau_e, root_e)
    H_e(E_h) = H(H(rec_1) || ... || H(rec_n))

The module name and a few compatibility method names are intentionally retained
so the rest of the project can migrate without changing every import at once.
"""

import hashlib
import struct
from array import array

from edge_mmr import EdgeMMRIndex
from road_network import RoadNetwork
from spatial_skeleton import SpatialSkeleton, NODE_LEAF
from query_utils import UINT32_MAX

HASH_SIZE = 32
_EDGE_RECORD_STRUCT = struct.Struct(">IddddIII32s")
EMPTY_EDGE_BLOCK_HASH = hashlib.sha256(b"").digest()


def edge_geometry(eid: int, road: RoadNetwork) -> tuple[float, float, float, float]:
    """Return g_e in the fixed endpoint order stored by the road file: u then v."""
    u = road.edge_u[eid]
    v = road.edge_v[eid]
    if u < 0 or v < 0:
        raise ValueError(f"Invalid eid={eid}")
    return (
        road.node_lon[u], road.node_lat[u],
        road.node_lon[v], road.node_lat[v],
    )


def hash_edge_record(
    eid: int,
    u_lon: float,
    u_lat: float,
    v_lon: float,
    v_lat: float,
    k_e: int,
    min_start: int,
    max_end: int,
    root_e: bytes,
) -> bytes:
    """H(rec_e), where rec_e=(e,g_e,k_e,tau_e,root_e)."""
    if len(root_e) != HASH_SIZE:
        raise ValueError("root_e must be 32 bytes")
    return hashlib.sha256(
        _EDGE_RECORD_STRUCT.pack(
            int(eid),
            float(u_lon), float(u_lat), float(v_lon), float(v_lat),
            int(k_e), int(min_start), int(max_end), root_e,
        )
    ).digest()


def hash_edge_block_records(record_hashes) -> bytes:
    h = hashlib.sha256()
    for value in record_hashes:
        if len(value) != HASH_SIZE:
            raise ValueError("edge record hash must be 32 bytes")
        h.update(value)
    return h.digest()


def edge_record_tuple(eid: int, mmr_index: EdgeMMRIndex, road: RoadNetwork) -> tuple:
    k_e, min_start, max_end, root_e = mmr_index.get_edge_state(eid)
    u_lon, u_lat, v_lon, v_lat = edge_geometry(eid, road)
    return (
        int(eid),
        float(u_lon), float(u_lat), float(v_lon), float(v_lat),
        int(k_e), int(min_start), int(max_end), root_e,
    )


def hash_edge_block(edge_ids, mmr_index: EdgeMMRIndex, road: RoadNetwork) -> bytes:
    """Compute H_e(E_h) in strictly increasing eid order."""
    eids = [int(x) for x in edge_ids]
    if eids != sorted(eids):
        eids.sort()
    h = hashlib.sha256()
    for eid in eids:
        rec = edge_record_tuple(eid, mmr_index, road)
        h.update(hash_edge_record(*rec))
    return h.digest()


class EdgeBlockIndex:
    __slots__ = (
        "l0_l1_root_hashes",   # compatibility name: stores H_e(E_h)
        "l0_min_start",
        "l0_max_end",
        "l0_l1_root_node",     # compatibility sentinel only; no L1 tree exists
        "eid_to_l0_leaf",
        "total_leaves",
        "tree_count",
        "record_count",
    )

    def __init__(self, l0_node_count: int, max_eid: int):
        self.l0_l1_root_hashes = bytearray(EMPTY_EDGE_BLOCK_HASH * l0_node_count)
        self.l0_min_start = array("I", [UINT32_MAX] * l0_node_count)
        self.l0_max_end = array("I", [0] * l0_node_count)
        self.l0_l1_root_node = array("i", [-1] * l0_node_count)
        self.eid_to_l0_leaf = array("i", [-1] * (max_eid + 1))
        self.total_leaves = 0
        self.tree_count = 0
        self.record_count = 0

    def get_l1_root(self, l0_node_index: int) -> bytes:
        """Compatibility API: returns H_e(E_h), not a Merkle-tree root."""
        offset = l0_node_index * HASH_SIZE
        return bytes(self.l0_l1_root_hashes[offset:offset + HASH_SIZE])

    def get_edge_block_hash(self, l0_node_index: int) -> bytes:
        return self.get_l1_root(l0_node_index)

    def set_l1_root(self, l0_node_index: int, root_hash: bytes) -> None:
        if len(root_hash) != HASH_SIZE:
            raise ValueError("edge block hash must be 32 bytes")
        offset = l0_node_index * HASH_SIZE
        self.l0_l1_root_hashes[offset:offset + HASH_SIZE] = root_hash


# Compatibility alias retained for existing imports.
L1MerkleIndex = EdgeBlockIndex


def build_all_l1_trees(
    skeleton: SpatialSkeleton,
    mmr_index: EdgeMMRIndex,
    road: RoadNetwork,
) -> EdgeBlockIndex:
    """Build all eid-sorted E_h lists and their H_e(E_h) digests."""
    index = EdgeBlockIndex(len(skeleton.node_type), road.max_eid)

    for l0_leaf in range(len(skeleton.node_type)):
        if skeleton.node_type[l0_leaf] != NODE_LEAF:
            continue

        offset = int(skeleton.leaf_edge_offset[l0_leaf])
        count = int(skeleton.leaf_edge_count[l0_leaf])
        eids = [int(skeleton.leaf_edge_ids[offset + i]) for i in range(count)]
        if eids != sorted(eids):
            eids.sort()

        block_hash = hash_edge_block(eids, mmr_index, road)
        index.set_l1_root(l0_leaf, block_hash)

        min_start = UINT32_MAX
        max_end = 0
        for eid in eids:
            if index.eid_to_l0_leaf[eid] != -1:
                raise RuntimeError(f"eid={eid} appears in multiple L0 leaves")
            index.eid_to_l0_leaf[eid] = l0_leaf
            k_e, edge_min, edge_max, _ = mmr_index.get_edge_state(eid)
            if k_e > 0:
                min_start = min(min_start, int(edge_min))
                max_end = max(max_end, int(edge_max))

        index.l0_min_start[l0_leaf] = min_start
        index.l0_max_end[l0_leaf] = max_end
        index.l0_l1_root_node[l0_leaf] = l0_leaf if count > 0 else -1
        index.total_leaves += count
        index.record_count += count
        index.tree_count += 1

    return index


# New explicit name for callers that no longer want the legacy terminology.
build_all_edge_blocks = build_all_l1_trees
