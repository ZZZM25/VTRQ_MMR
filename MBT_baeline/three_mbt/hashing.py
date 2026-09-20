from __future__ import annotations

import hashlib
import struct

DOMAIN_LEAF = b"THREE-MBT-LEAF-v1"
DOMAIN_INTERNAL = b"THREE-MBT-INTERNAL-v1"
DOMAIN_THREE_ROOT = b"THREE-MBT-ROOT-v1"


def _u64(v: int) -> bytes:
    return struct.pack(">Q", int(v))


def _f64(v: float) -> bytes:
    return struct.pack(">d", float(v))


def _blob(v: bytes) -> bytes:
    return _u64(len(v)) + v


def _trajectory_id_bytes(trajectory_id: str) -> bytes:
    text = str(trajectory_id)
    if len(text) == 64:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    return text.encode("utf-8")


def hash_leaf_entries(entries: list[tuple[float, int, str]]) -> bytes:
    h = hashlib.sha256()
    h.update(DOMAIN_LEAF)
    h.update(_u64(len(entries)))
    for key, point_id, trajectory_id in entries:
        h.update(_f64(key))
        h.update(_u64(point_id))
        h.update(_blob(_trajectory_id_bytes(trajectory_id)))
    return h.digest()


def hash_internal(children: list[tuple[float, float, bytes]]) -> bytes:
    h = hashlib.sha256()
    h.update(DOMAIN_INTERNAL)
    h.update(_u64(len(children)))
    for min_key, max_key, child_hash in children:
        h.update(_f64(min_key))
        h.update(_f64(max_key))
        h.update(_blob(child_hash))
    return h.digest()


def aggregate_three_roots(root_lon: bytes, root_lat: bytes, root_time: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(DOMAIN_THREE_ROOT)
    h.update(root_lon)
    h.update(root_lat)
    h.update(root_time)
    return h.digest()
