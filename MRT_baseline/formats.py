from __future__ import annotations

import hashlib
import struct

import numpy as np

# Formal indexed point leaf entry:
# point_id(uint64), trajectory_id(raw SHA256 32 bytes), lon, lat, timestamp.
POINT_STRUCT = struct.Struct("<Q32sddd")
POINT_DTYPE = np.dtype([
    ("point_id", "<u8"),
    ("tid", "V32"),
    ("lon", "<f8"),
    ("lat", "<f8"),
    ("ts", "<f8"),
], align=False)
assert POINT_DTYPE.itemsize == POINT_STRUCT.size == 64

# External-sort record = Morton key + exact point record.
RUN_STRUCT = struct.Struct("<QQ32sddd")
RUN_DTYPE = np.dtype([
    ("morton", "<u8"),
    ("point_id", "<u8"),
    ("tid", "V32"),
    ("lon", "<f8"),
    ("lat", "<f8"),
    ("ts", "<f8"),
], align=False)
assert RUN_DTYPE.itemsize == RUN_STRUCT.size == 72

U16 = struct.Struct("<H")
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
BBOX = struct.Struct("<dddd")


def leaf_hash(point_record_bytes: bytes, count: int) -> bytes:
    h = hashlib.sha256()
    h.update(b"L")
    h.update(U32.pack(int(count)))
    h.update(point_record_bytes)
    return h.digest()


def internal_hash(children) -> bytes:
    h = hashlib.sha256()
    h.update(b"I")
    h.update(U16.pack(len(children)))
    for child in children:
        h.update(BBOX.pack(*child.bbox))
        h.update(child.digest)
    return h.digest()


def root_commitment(node) -> bytes:
    h = hashlib.sha256()
    h.update(b"R")
    h.update(BBOX.pack(*node.bbox))
    h.update(node.digest)
    return h.digest()
