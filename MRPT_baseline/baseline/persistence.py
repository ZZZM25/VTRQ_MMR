from __future__ import annotations

import hashlib
import pickle
import struct
from pathlib import Path

MAGIC_INDEX = b"BLIDX002"
MAGIC_CATALOG = b"BLCAT002"
VERSION = 2
_HEADER = struct.Struct(">8sIQ32s")


def _save(obj, path, magic):
    payload = pickle.dumps(obj, protocol=5)
    digest = hashlib.sha256(payload).digest()
    raw = _HEADER.pack(magic, VERSION, len(payload), digest) + payload
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return len(raw)


def _load(path, magic):
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise ValueError("truncated file")
    got_magic, version, n, digest = _HEADER.unpack(raw[:_HEADER.size])
    if got_magic != magic or version != VERSION:
        raise ValueError("bad file format/version; rebuild baseline index with v2")
    payload = raw[_HEADER.size:]
    if len(payload) != n:
        raise ValueError("payload length mismatch")
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("payload checksum mismatch")
    return pickle.loads(payload)


def save_index(index, path):
    return _save(index, path, MAGIC_INDEX)


def load_index(path):
    return _load(path, MAGIC_INDEX)


def save_catalog(catalog, path):
    return _save(catalog, path, MAGIC_CATALOG)


def load_catalog(path):
    return _load(path, MAGIC_CATALOG)
