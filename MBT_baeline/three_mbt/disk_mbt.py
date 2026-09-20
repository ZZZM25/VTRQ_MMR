from __future__ import annotations

import heapq
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable

from .hashing import hash_internal, hash_leaf_entries
from .mbt import MBTNode, InternalNode, MerkleBTree
from .vo_size import (
    empty_proof_size,
    json_array_size,
    json_number_size,
    json_object_size,
    json_string_size,
    proof_bundle_size,
)

_RECORD = struct.Struct(">dQI")  # key(float64), point_id(uint64), trajectory ordinal(uint32)


@dataclass(slots=True)
class DiskLeafNode(MBTNode):
    record_offset: int = 0
    record_count: int = 0


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


class DiskMerkleBTree:
    """Packed MBT whose leaf records are kept in a compact binary file.

    Only leaf metadata and internal hashes remain as Python objects.  This keeps
    full-city builds/queryable indexes feasible when the dataset contains tens of
    millions of GPS points.
    """

    def __init__(
        self,
        dimension: str,
        leaf_capacity: int = 128,
        fanout: int = 64,
        *,
        trajectory_ids: list[str] | None = None,
        entries_filename: str | None = None,
    ) -> None:
        if dimension not in {"lon", "lat", "time"}:
            raise ValueError("dimension must be lon/lat/time")
        self.dimension = dimension
        self.leaf_capacity = int(leaf_capacity)
        self.fanout = int(fanout)
        self.root: MBTNode | None = None
        self.entry_count = 0
        self.trajectory_ids = trajectory_ids if trajectory_ids is not None else []
        self.entries_filename = entries_filename or f"{dimension}.entries.bin"
        self._storage_dir: str | None = None

    def bind_storage(self, storage_dir: str | Path) -> None:
        self._storage_dir = str(Path(storage_dir))

    @property
    def entries_path(self) -> Path:
        if self._storage_dir is None:
            raise RuntimeError("disk MBT storage is not bound")
        return Path(self._storage_dir) / self.entries_filename

    @property
    def root_hash(self) -> bytes:
        return self.root.hash_value if self.root else b"\x00" * 32

    @property
    def root_hash_hex(self) -> str:
        return self.root_hash.hex()

    def build_from_sorted_rows(
        self,
        rows: Iterable[tuple[float, int, int]],
        entries_path: str | Path,
        *,
        total_rows: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build packed leaves from rows already sorted by key, point_id, tid_idx.

        ``progress`` receives the actual number of sorted records consumed.  It is
        intentionally optional so tests/library callers do not pay any display cost.
        """
        entries_path = Path(entries_path)
        entries_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries_filename = entries_path.name
        self.bind_storage(entries_path.parent)

        leaves: list[MBTNode] = []
        leaf_rows: list[tuple[float, int, int]] = []
        record_offset = 0
        total = 0

        def flush_leaf(f: BinaryIO) -> None:
            nonlocal record_offset, total, leaf_rows
            if not leaf_rows:
                return
            full_entries: list[tuple[float, int, str]] = []
            for key, point_id, tid_idx in leaf_rows:
                tid = self.trajectory_ids[tid_idx]
                full_entries.append((float(key), int(point_id), tid))
                f.write(_RECORD.pack(float(key), int(point_id), int(tid_idx)))
            leaf_hash = hash_leaf_entries(full_entries)
            leaves.append(
                DiskLeafNode(
                    min_key=float(full_entries[0][0]),
                    max_key=float(full_entries[-1][0]),
                    hash_value=leaf_hash,
                    record_offset=record_offset,
                    record_count=len(full_entries),
                )
            )
            record_offset += len(full_entries)
            total += len(full_entries)
            leaf_rows = []

        progress_total = int(total_rows or 0)
        progress_step = max(1, progress_total // 200) if progress_total else 1
        next_progress = progress_step
        consumed = 0

        with entries_path.open("wb") as f:
            for key, point_id, tid_idx in rows:
                leaf_rows.append((float(key), int(point_id), int(tid_idx)))
                consumed += 1
                if len(leaf_rows) >= self.leaf_capacity:
                    flush_leaf(f)
                if progress is not None and progress_total and consumed >= next_progress:
                    progress(consumed, progress_total)
                    next_progress = consumed + progress_step
            flush_leaf(f)

        if progress is not None and progress_total and consumed < progress_total:
            progress(consumed, progress_total)

        self.entry_count = total
        if not leaves:
            self.root = None
            return

        level: list[MBTNode] = leaves
        while len(level) > 1:
            level = [
                _make_internal(level[i : i + self.fanout])
                for i in range(0, len(level), self.fanout)
            ]
        self.root = level[0]


    def _iter_leaves_in_order(self) -> list[DiskLeafNode]:
        if self.root is None:
            return []
        out: list[DiskLeafNode] = []
        stack: list[MBTNode] = [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, DiskLeafNode):
                out.append(node)
            elif isinstance(node, InternalNode):
                stack.extend(reversed(node.children))
            else:
                raise TypeError(f"unexpected node type: {type(node)!r}")
        return out

    def _read_leaf_rows_raw(
        self,
        f: BinaryIO,
        leaf: DiskLeafNode,
    ) -> list[tuple[float, int, int]]:
        f.seek(leaf.record_offset * _RECORD.size)
        raw = f.read(leaf.record_count * _RECORD.size)
        if len(raw) != leaf.record_count * _RECORD.size:
            raise IOError("truncated disk MBT leaf file")
        out: list[tuple[float, int, int]] = []
        for pos in range(0, len(raw), _RECORD.size):
            key, point_id, tid_idx = _RECORD.unpack_from(raw, pos)
            out.append((float(key), int(point_id), int(tid_idx)))
        return out

    def _append_leaf_rows(
        self,
        f: BinaryIO,
        rows: list[tuple[float, int, int]],
    ) -> DiskLeafNode:
        if not rows:
            raise ValueError("cannot append empty leaf")
        f.seek(0, 2)
        byte_offset = f.tell()
        if byte_offset % _RECORD.size:
            raise IOError("unaligned packed MBT leaf file")
        record_offset = byte_offset // _RECORD.size
        full_entries: list[tuple[float, int, str]] = []
        buf = bytearray(_RECORD.size * len(rows))
        pos = 0
        for key, point_id, tid_idx in rows:
            if tid_idx < 0 or tid_idx >= len(self.trajectory_ids):
                raise IndexError(f"trajectory ordinal out of range: {tid_idx}")
            _RECORD.pack_into(buf, pos, float(key), int(point_id), int(tid_idx))
            pos += _RECORD.size
            full_entries.append(
                (float(key), int(point_id), self.trajectory_ids[int(tid_idx)])
            )
        f.write(buf)
        return DiskLeafNode(
            min_key=full_entries[0][0],
            max_key=full_entries[-1][0],
            hash_value=hash_leaf_entries(full_entries),
            record_offset=int(record_offset),
            record_count=len(rows),
        )

    def batch_insert_sorted_rows(
        self,
        rows: Iterable[tuple[float, int, int]],
    ) -> int:
        """Incrementally insert a sorted batch into the packed on-disk MBT.

        Existing leaf records are never rewritten in place.  Only leaves touched by
        the new keys are copied/split and appended to the packed file; untouched
        leaves keep their original offsets.  Internal Merkle metadata is rebuilt
        from the resulting leaf references in RAM.  This is a true incremental
        update of the existing disk index, not a rebuild from raw/base points.
        """
        iterator = iter(rows)
        try:
            current = next(iterator)
        except StopIteration:
            return 0

        if self.root is None:
            # Empty-tree update: the incoming rows are already sorted.
            def all_rows():
                yield current
                yield from iterator
            self.build_from_sorted_rows(all_rows(), self.entries_path)
            return self.entry_count

        leaves = self._iter_leaves_in_order()
        new_leaves: list[MBTNode] = []
        inserted = 0

        with self.entries_path.open("r+b") as f:
            for i, leaf in enumerate(leaves):
                is_last = i == len(leaves) - 1
                next_same_max = (
                    not is_last and leaves[i + 1].max_key == leaf.max_key
                )

                def belongs(key: float) -> bool:
                    if is_last:
                        return True
                    if next_same_max:
                        return key < leaf.max_key
                    return key <= leaf.max_key

                if current is None or not belongs(float(current[0])):
                    new_leaves.append(leaf)
                    continue

                old_rows = self._read_leaf_rows_raw(f, leaf)

                def update_segment():
                    nonlocal current, inserted
                    while current is not None and belongs(float(current[0])):
                        row = (float(current[0]), int(current[1]), int(current[2]))
                        inserted += 1
                        yield row
                        try:
                            current = next(iterator)
                        except StopIteration:
                            current = None

                merged = heapq.merge(old_rows, update_segment())
                chunk: list[tuple[float, int, int]] = []
                for row in merged:
                    chunk.append(row)
                    if len(chunk) >= self.leaf_capacity:
                        new_leaves.append(self._append_leaf_rows(f, chunk))
                        chunk = []
                if chunk:
                    new_leaves.append(self._append_leaf_rows(f, chunk))

        if current is not None:
            raise RuntimeError("incremental update rows were not fully consumed")

        self.entry_count += inserted
        level: list[MBTNode] = new_leaves
        while len(level) > 1:
            level = [
                _make_internal(level[i : i + self.fanout])
                for i in range(0, len(level), self.fanout)
            ]
        self.root = level[0] if level else None
        return inserted

    def _read_leaf_entries(
        self,
        f: BinaryIO,
        leaf: DiskLeafNode,
    ) -> list[tuple[float, int, str]]:
        f.seek(leaf.record_offset * _RECORD.size)
        raw = f.read(leaf.record_count * _RECORD.size)
        if len(raw) != leaf.record_count * _RECORD.size:
            raise IOError("truncated disk MBT leaf file")
        out: list[tuple[float, int, str]] = []
        pos = 0
        for _ in range(leaf.record_count):
            key, point_id, tid_idx = _RECORD.unpack_from(raw, pos)
            pos += _RECORD.size
            out.append((float(key), int(point_id), self.trajectory_ids[int(tid_idx)]))
        return out

    def _trajectory_id_json_sizes(self) -> list[int]:
        # Existing on-disk indexes were pickled before this cache existed, so keep
        # it lazy and non-required for load compatibility.
        sizes = getattr(self, "_vo_tid_json_sizes", None)
        if sizes is None:
            sizes = [json_string_size(tid) for tid in self.trajectory_ids]
            self._vo_tid_json_sizes = sizes
        return sizes

    def _read_leaf_entries_with_size(
        self,
        f: BinaryIO,
        leaf: DiskLeafNode,
    ) -> tuple[list[tuple[float, int, str]], int]:
        f.seek(leaf.record_offset * _RECORD.size)
        raw = f.read(leaf.record_count * _RECORD.size)
        if len(raw) != leaf.record_count * _RECORD.size:
            raise IOError("truncated disk MBT leaf file")

        tid_json_sizes = self._trajectory_id_json_sizes()
        out: list[tuple[float, int, str]] = []
        entries_payload_size = 0
        pos = 0
        for _ in range(leaf.record_count):
            key, point_id, tid_idx = _RECORD.unpack_from(raw, pos)
            pos += _RECORD.size
            tid_idx = int(tid_idx)
            trajectory_id = self.trajectory_ids[tid_idx]
            key = float(key)
            point_id = int(point_id)
            out.append((key, point_id, trajectory_id))

            # Compact JSON for one [key,point_id,trajectory_id] entry: brackets
            # + two commas + three scalar encodings.
            entries_payload_size += (
                4
                + json_number_size(key)
                + json_number_size(point_id)
                + tid_json_sizes[tid_idx]
            )

        # Outer entries list: [] plus commas between entry arrays.
        entries_size = 2 + entries_payload_size
        if leaf.record_count > 1:
            entries_size += leaf.record_count - 1
        return out, entries_size

    def range_query(self, low: float, high: float) -> tuple[set[str], dict[str, Any]]:
        result, proof, _proof_size = self.range_query_with_size(low, high)
        return result, proof

    def range_query_with_size(
        self,
        low: float,
        high: float,
    ) -> tuple[set[str], dict[str, Any], int]:
        if low > high:
            raise ValueError("low > high")
        if self.root is None:
            proof = {"type": "empty", "dimension": self.dimension}
            return set(), proof, empty_proof_size(self.dimension)
        result: set[str] = set()
        with self.entries_path.open("rb") as f:
            root_proof, root_size = self._query_node_with_size(
                self.root, float(low), float(high), result, f
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
        f: BinaryIO,
    ) -> dict[str, Any]:
        proof, _proof_size = self._query_node_with_size(node, low, high, result, f)
        return proof

    def _query_node_with_size(
        self,
        node: MBTNode,
        low: float,
        high: float,
        result: set[str],
        f: BinaryIO,
    ) -> tuple[dict[str, Any], int]:
        if node.max_key < low or node.min_key > high:
            hash_hex = node.hash_value.hex()
            proof = {
                "type": "stub",
                "min_token": node.min_key,
                "max_token": node.max_key,
                "hash": hash_hex,
            }
            size = json_object_size(
                [
                    ("type", json_string_size("stub")),
                    ("min_token", json_number_size(node.min_key)),
                    ("max_token", json_number_size(node.max_key)),
                    ("hash", json_string_size(hash_hex)),
                ]
            )
            return proof, size

        if isinstance(node, DiskLeafNode):
            entries, entries_size = self._read_leaf_entries_with_size(f, node)
            for key, _point_id, trajectory_id in entries:
                if low <= key <= high:
                    result.add(trajectory_id)
            proof = {
                "type": "leaf",
                "min_token": node.min_key,
                "max_token": node.max_key,
                "entries": [[k, pid, tid] for k, pid, tid in entries],
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

        if not isinstance(node, InternalNode):
            raise TypeError(f"unexpected node type: {type(node)!r}")
        children: list[dict[str, Any]] = []
        child_sizes: list[int] = []
        for child in node.children:
            child_proof, child_size = self._query_node_with_size(
                child, low, high, result, f
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
    def reconstruct_range_proof(proof_bundle: dict[str, Any]) -> tuple[set[str], bytes]:
        # VO format and hashes are intentionally identical to the in-memory MBT.
        return MerkleBTree.reconstruct_range_proof(proof_bundle)

    def stats(self) -> dict[str, int | str]:
        leaves = 0
        internals = 0
        if self.root is not None:
            stack = [self.root]
            while stack:
                node = stack.pop()
                if isinstance(node, DiskLeafNode):
                    leaves += 1
                elif isinstance(node, InternalNode):
                    internals += 1
                    stack.extend(node.children)
        return {
            "dimension": self.dimension,
            "entries": self.entry_count,
            "leaves": leaves,
            "internal_nodes": internals,
            "root": self.root_hash_hex,
        }

    def disk_bytes(self) -> int:
        try:
            return self.entries_path.stat().st_size
        except FileNotFoundError:
            return 0
