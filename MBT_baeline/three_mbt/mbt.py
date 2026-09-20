from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .hashing import hash_internal, hash_leaf_entries
from .models import GPSPoint
from .vo_size import (
    empty_proof_size,
    json_array_size,
    json_number_size,
    json_object_size,
    json_string_size,
    proof_bundle_size,
)


@dataclass(slots=True)
class MBTNode:
    min_key: float
    max_key: float
    hash_value: bytes


@dataclass(slots=True)
class LeafNode(MBTNode):
    entries: list[tuple[float, int, str]] = field(default_factory=list)


@dataclass(slots=True)
class InternalNode(MBTNode):
    children: list[MBTNode] = field(default_factory=list)


def _make_leaf(entries: list[tuple[float, int, str]]) -> LeafNode:
    if not entries:
        raise ValueError("leaf must not be empty")
    entries = sorted(entries, key=lambda x: (x[0], x[1], x[2]))
    return LeafNode(
        min_key=entries[0][0],
        max_key=entries[-1][0],
        hash_value=hash_leaf_entries(entries),
        entries=entries,
    )


def _make_internal(children: list[MBTNode]) -> InternalNode:
    if not children:
        raise ValueError("internal node must not be empty")
    children = sorted(children, key=lambda c: (c.min_key, c.max_key))
    payload = [(c.min_key, c.max_key, c.hash_value) for c in children]
    return InternalNode(
        min_key=children[0].min_key,
        max_key=max(c.max_key for c in children),
        hash_value=hash_internal(payload),
        children=children,
    )


class MerkleBTree:
    """Packed Merkle B-tree with true incremental insertion and split propagation.

    Leaves store (dimension_key, point_id, trajectory_id).
    Query VO expands overlapping nodes and replaces disjoint subtrees with
    (min_key, max_key, hash) stubs.
    """

    def __init__(self, dimension: str, leaf_capacity: int = 128, fanout: int = 64) -> None:
        if dimension not in {"lon", "lat", "time"}:
            raise ValueError("dimension must be lon/lat/time")
        if leaf_capacity < 2 or fanout < 2:
            raise ValueError("leaf_capacity and fanout must be >= 2")
        self.dimension = dimension
        self.leaf_capacity = int(leaf_capacity)
        self.fanout = int(fanout)
        self.root: MBTNode | None = None
        self.entry_count = 0

    @property
    def root_hash(self) -> bytes:
        return self.root.hash_value if self.root else b"\x00" * 32

    @property
    def root_hash_hex(self) -> str:
        return self.root_hash.hex()

    def bulk_build(self, points: Iterable[GPSPoint]) -> None:
        entries = [(p.key(self.dimension), p.point_id, p.trajectory_id) for p in points]
        entries.sort(key=lambda x: (x[0], x[1], x[2]))
        self.entry_count = len(entries)
        if not entries:
            self.root = None
            return
        level: list[MBTNode] = [
            _make_leaf(entries[i : i + self.leaf_capacity])
            for i in range(0, len(entries), self.leaf_capacity)
        ]
        while len(level) > 1:
            level = [
                _make_internal(level[i : i + self.fanout])
                for i in range(0, len(level), self.fanout)
            ]
        self.root = level[0]

    def insert_point(self, point: GPSPoint) -> None:
        entry = (point.key(self.dimension), point.point_id, point.trajectory_id)
        if self.root is None:
            self.root = _make_leaf([entry])
            self.entry_count = 1
            return
        new_root, sibling = self._insert_recursive(self.root, entry)
        if sibling is not None:
            self.root = _make_internal([new_root, sibling])
        else:
            self.root = new_root
        self.entry_count += 1

    def _insert_recursive(
        self,
        node: MBTNode,
        entry: tuple[float, int, str],
    ) -> tuple[MBTNode, MBTNode | None]:
        if isinstance(node, LeafNode):
            entries = list(node.entries)
            bisect.insort(entries, entry)
            if len(entries) <= self.leaf_capacity:
                return _make_leaf(entries), None
            mid = len(entries) // 2
            return _make_leaf(entries[:mid]), _make_leaf(entries[mid:])

        assert isinstance(node, InternalNode)
        children = list(node.children)
        key = entry[0]
        idx = self._child_index(children, key)
        child, sibling = self._insert_recursive(children[idx], entry)
        children[idx] = child
        if sibling is not None:
            children.insert(idx + 1, sibling)
        if len(children) <= self.fanout:
            return _make_internal(children), None
        mid = len(children) // 2
        return _make_internal(children[:mid]), _make_internal(children[mid:])

    @staticmethod
    def _child_index(children: list[MBTNode], key: float) -> int:
        # Choose first child whose max_key >= key; otherwise the rightmost child.
        maxes = [c.max_key for c in children]
        idx = bisect.bisect_left(maxes, key)
        return min(idx, len(children) - 1)

    def range_query(self, low: float, high: float) -> tuple[set[str], dict[str, Any]]:
        result, proof, _proof_size = self.range_query_with_size(low, high)
        return result, proof

    def range_query_with_size(
        self,
        low: float,
        high: float,
    ) -> tuple[set[str], dict[str, Any], int]:
        """Range query plus exact compact-JSON VO byte count.

        The byte count is accumulated as proof nodes are created, avoiding a
        second traversal/serialization of the complete VO after the query.
        """
        if low > high:
            raise ValueError("low > high")
        if self.root is None:
            proof = {"type": "empty", "dimension": self.dimension}
            return set(), proof, empty_proof_size(self.dimension)

        result: set[str] = set()
        root_proof, root_size = self._query_node_with_size(
            self.root, float(low), float(high), result
        )
        proof = {
            "version": 1,
            "dimension": self.dimension,
            "low": float(low),
            "high": float(high),
            "root": root_proof,
        }
        proof_size = proof_bundle_size(
            dimension=self.dimension,
            low=float(low),
            high=float(high),
            root_size=root_size,
        )
        return result, proof, proof_size

    def _query_node(
        self,
        node: MBTNode,
        low: float,
        high: float,
        result: set[str],
    ) -> dict[str, Any]:
        proof, _proof_size = self._query_node_with_size(node, low, high, result)
        return proof

    def _query_node_with_size(
        self,
        node: MBTNode,
        low: float,
        high: float,
        result: set[str],
    ) -> tuple[dict[str, Any], int]:
        if node.max_key < low or node.min_key > high:
            proof = {
                "type": "stub",
                "min_token": node.min_key,
                "max_token": node.max_key,
                "hash": node.hash_value.hex(),
            }
            size = json_object_size(
                [
                    ("type", json_string_size("stub")),
                    ("min_token", json_number_size(node.min_key)),
                    ("max_token", json_number_size(node.max_key)),
                    ("hash", json_string_size(proof["hash"])),
                ]
            )
            return proof, size

        if isinstance(node, LeafNode):
            entry_lists: list[list[float | int | str]] = []
            entry_sizes: list[int] = []
            for key, point_id, trajectory_id in node.entries:
                if low <= key <= high:
                    result.add(trajectory_id)
                entry_lists.append([key, point_id, trajectory_id])
                entry_sizes.append(
                    json_array_size(
                        [
                            json_number_size(key),
                            json_number_size(point_id),
                            json_string_size(trajectory_id),
                        ]
                    )
                )
            entries_size = json_array_size(entry_sizes)
            proof = {
                "type": "leaf",
                "min_token": node.min_key,
                "max_token": node.max_key,
                "entries": entry_lists,
            }
            size = json_object_size(
                [
                    ("type", json_string_size("leaf")),
                    ("min_token", json_number_size(node.min_key)),
                    ("max_token", json_number_size(node.max_key)),
                    ("entries", entries_size),
                ]
            )
            return proof, size

        assert isinstance(node, InternalNode)
        children: list[dict[str, Any]] = []
        child_sizes: list[int] = []
        for child in node.children:
            child_proof, child_size = self._query_node_with_size(
                child, low, high, result
            )
            children.append(child_proof)
            child_sizes.append(child_size)
        children_size = json_array_size(child_sizes)
        proof = {
            "type": "internal",
            "min_token": node.min_key,
            "max_token": node.max_key,
            "children": children,
        }
        size = json_object_size(
            [
                ("type", json_string_size("internal")),
                ("min_token", json_number_size(node.min_key)),
                ("max_token", json_number_size(node.max_key)),
                ("children", children_size),
            ]
        )
        return proof, size

    @staticmethod
    def reconstruct_range_proof(
        proof_bundle: dict[str, Any],
    ) -> tuple[set[str], bytes]:
        """Verify the VO structure, recover matching trajectory IDs, and rebuild root.

        The verifier trusts neither server-returned TID sets nor per-tree roots.
        Matching TIDs are recovered from revealed authenticated leaf entries.
        The returned root is later combined with the other two reconstructed roots
        and compared with the trusted global root.
        """
        if proof_bundle.get("type") == "empty":
            return set(), b"\x00" * 32
        low = float(proof_bundle["low"])
        high = float(proof_bundle["high"])
        matches: set[str] = set()

        def verify(node: dict[str, Any]) -> tuple[float, float, bytes]:
            typ = node["type"]
            node_min = float(node["min_token"])
            node_max = float(node["max_token"])
            if node_min > node_max:
                raise ValueError("invalid node range")
            if typ == "stub":
                if not (node_max < low or node_min > high):
                    raise ValueError("stub overlaps query range")
                return node_min, node_max, bytes.fromhex(node["hash"])
            if typ == "leaf":
                entries = [(float(k), int(pid), str(tid)) for k, pid, tid in node["entries"]]
                if not entries:
                    raise ValueError("empty revealed leaf")
                if entries != sorted(entries, key=lambda x: (x[0], x[1], x[2])):
                    raise ValueError("leaf entries not sorted")
                actual_min, actual_max = entries[0][0], entries[-1][0]
                if actual_min != node_min or actual_max != node_max:
                    raise ValueError("leaf bounds mismatch")
                for key, _point_id, trajectory_id in entries:
                    if low <= key <= high:
                        matches.add(trajectory_id)
                return node_min, node_max, hash_leaf_entries(entries)
            if typ == "internal":
                children_raw = node.get("children", [])
                if not children_raw:
                    raise ValueError("empty revealed internal node")
                verified = [verify(c) for c in children_raw]
                if verified != sorted(verified, key=lambda x: (x[0], x[1])):
                    raise ValueError("children not sorted")
                actual_min = verified[0][0]
                actual_max = max(v[1] for v in verified)
                if actual_min != node_min or actual_max != node_max:
                    raise ValueError("internal bounds mismatch")
                return node_min, node_max, hash_internal(verified)
            raise ValueError(f"unknown proof node type: {typ}")

        _min, _max, computed = verify(proof_bundle["root"])
        return matches, computed

    @staticmethod
    def verify_range_proof_trajectory_ids(
        proof_bundle: dict[str, Any],
        expected_root_hash: bytes | None = None,
    ) -> set[str]:
        matches, computed = MerkleBTree.reconstruct_range_proof(proof_bundle)
        if expected_root_hash is not None and computed != expected_root_hash:
            raise ValueError("root hash mismatch")
        return matches

    @staticmethod
    def verify_range_proof(
        proof_bundle: dict[str, Any],
        expected_root_hash: bytes,
    ) -> set[str]:
        return MerkleBTree.verify_range_proof_trajectory_ids(
            proof_bundle, expected_root_hash
        )

    def proof_size_bytes(self, proof: dict[str, Any]) -> int:
        return len(json.dumps(proof, separators=(",", ":"), sort_keys=True).encode("utf-8"))

    def stats(self) -> dict[str, int]:
        if self.root is None:
            return {"height": 0, "leaf_nodes": 0, "internal_nodes": 0, "entries": 0}
        leaf_nodes = 0
        internal_nodes = 0
        max_depth = 0
        stack: list[tuple[MBTNode, int]] = [(self.root, 1)]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if isinstance(node, LeafNode):
                leaf_nodes += 1
            else:
                internal_nodes += 1
                stack.extend((c, depth + 1) for c in node.children)
        return {
            "height": max_depth,
            "leaf_nodes": leaf_nodes,
            "internal_nodes": internal_nodes,
            "entries": self.entry_count,
        }
