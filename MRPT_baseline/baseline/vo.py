from __future__ import annotations

from dataclasses import dataclass

from .models import VerificationEntry

TOKEN_NODE_PRUNED_LEAF = 1
TOKEN_NODE_PRUNED_INTERNAL = 2
TOKEN_NODE_LEAF = 3
TOKEN_NODE_INTERNAL = 4
TOKEN_EDGE_OPAQUE = 10
TOKEN_EDGE_PREFIX = 11

TOKEN_NAMES = {
    TOKEN_NODE_PRUNED_LEAF: "NODE_PRUNED_LEAF",
    TOKEN_NODE_PRUNED_INTERNAL: "NODE_PRUNED_INTERNAL",
    TOKEN_NODE_LEAF: "NODE_LEAF",
    TOKEN_NODE_INTERNAL: "NODE_INTERNAL",
    TOKEN_EDGE_OPAQUE: "EDGE_OPAQUE",
    TOKEN_EDGE_PREFIX: "EDGE_PREFIX",
}


@dataclass(slots=True)
class BaselineVO:
    tokens: list[tuple]
    verification_set: list[VerificationEntry]


@dataclass(slots=True)
class QueryStats:
    visited_spatial_nodes: int = 0
    pruned_spatial_nodes: int = 0
    checked_cross_edges: int = 0
    checked_leaf_edges: int = 0
    spatial_matched_edges: int = 0
    scanned_entries: int = 0
    boundary_witness_entries: int = 0


@dataclass(slots=True)
class ServerQueryResponse:
    candidate_ids: list[bytes]
    vo: BaselineVO
    stats: QueryStats
