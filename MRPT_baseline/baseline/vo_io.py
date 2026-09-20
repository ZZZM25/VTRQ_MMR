from __future__ import annotations

import struct
from pathlib import Path

from .models import VerificationEntry
from .vo import BaselineVO
from .vo import (
    TOKEN_NODE_PRUNED_LEAF, TOKEN_NODE_PRUNED_INTERNAL, TOKEN_NODE_LEAF, TOKEN_NODE_INTERNAL,
    TOKEN_EDGE_OPAQUE, TOKEN_EDGE_PREFIX,
)

MAGIC = b"BLVO0002"
VERSION = 2
_HDR = struct.Struct(">8sIQQ")
_U8 = struct.Struct(">B")
_U32 = struct.Struct(">I")
_MBR = struct.Struct(">dddd")
_ENTRY = struct.Struct(">I32sII")


def serialize_vo(vo: BaselineVO) -> bytes:
    parts = []
    for t in vo.tokens:
        code = t[0]
        parts.append(_U8.pack(code))
        if code == TOKEN_NODE_PRUNED_LEAF:
            _, a, b, c, d, n, r = t
            parts += [_MBR.pack(a, b, c, d), _U32.pack(n), r]
        elif code == TOKEN_NODE_PRUNED_INTERNAL:
            _, a, b, c, d, lh, rh, n, gr = t
            parts += [_MBR.pack(a, b, c, d), lh, rh, _U32.pack(n), gr]
        elif code == TOKEN_NODE_LEAF:
            _, a, b, c, d, n = t
            parts += [_MBR.pack(a, b, c, d), _U32.pack(n)]
        elif code == TOKEN_NODE_INTERNAL:
            _, a, b, c, d, n = t
            parts += [_MBR.pack(a, b, c, d), _U32.pack(n)]
        elif code == TOKEN_EDGE_OPAQUE:
            _, eid, n, r = t
            parts += [_U32.pack(eid), _U32.pack(n), r]
        elif code == TOKEN_EDGE_PREFIX:
            _, eid, total_count, prefix_count, has_boundary, suffix_state = t
            parts += [
                _U32.pack(eid), _U32.pack(total_count), _U32.pack(prefix_count),
                _U8.pack(has_boundary), suffix_state,
            ]
        else:
            raise ValueError(f"unknown token {code}")

    token_blob = b"".join(parts)
    entry_blob = b"".join(
        _ENTRY.pack(e.eid, e.trajectory_id, e.start, e.end)
        for e in vo.verification_set
    )
    return _HDR.pack(MAGIC, VERSION, len(vo.tokens), len(vo.verification_set)) + token_blob + entry_blob


def save_vo(vo, path) -> int:
    raw = serialize_vo(vo)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return len(raw)


def load_vo(path) -> BaselineVO:
    raw = memoryview(Path(path).read_bytes())
    pos = 0
    if len(raw) < _HDR.size:
        raise ValueError("truncated VO")
    magic, version, nt, nv = _HDR.unpack(raw[:_HDR.size])
    pos += _HDR.size
    if magic != MAGIC or version != VERSION:
        raise ValueError("bad VO format; rebuild/re-query with baseline v2")

    tokens = []

    def take(n):
        nonlocal pos
        if pos + n > len(raw):
            raise ValueError("truncated VO")
        v = raw[pos:pos + n]
        pos += n
        return v

    for _ in range(nt):
        code = _U8.unpack(take(1))[0]
        if code == TOKEN_NODE_PRUNED_LEAF:
            m = _MBR.unpack(take(_MBR.size))
            n = _U32.unpack(take(4))[0]
            r = bytes(take(32))
            tokens.append((code, *m, n, r))
        elif code == TOKEN_NODE_PRUNED_INTERNAL:
            m = _MBR.unpack(take(_MBR.size))
            lh = bytes(take(32))
            rh = bytes(take(32))
            n = _U32.unpack(take(4))[0]
            gr = bytes(take(32))
            tokens.append((code, *m, lh, rh, n, gr))
        elif code in (TOKEN_NODE_LEAF, TOKEN_NODE_INTERNAL):
            m = _MBR.unpack(take(_MBR.size))
            n = _U32.unpack(take(4))[0]
            tokens.append((code, *m, n))
        elif code == TOKEN_EDGE_OPAQUE:
            eid = _U32.unpack(take(4))[0]
            n = _U32.unpack(take(4))[0]
            r = bytes(take(32))
            tokens.append((code, eid, n, r))
        elif code == TOKEN_EDGE_PREFIX:
            eid = _U32.unpack(take(4))[0]
            total_count = _U32.unpack(take(4))[0]
            prefix_count = _U32.unpack(take(4))[0]
            has_boundary = _U8.unpack(take(1))[0]
            suffix_state = bytes(take(32))
            tokens.append((code, eid, total_count, prefix_count, has_boundary, suffix_state))
        else:
            raise ValueError(f"unknown token {code}")

    vs = []
    for _ in range(nv):
        eid, tid, start, end = _ENTRY.unpack(take(_ENTRY.size))
        vs.append(VerificationEntry(eid, bytes(tid), start, end))
    if pos != len(raw):
        raise ValueError("trailing VO bytes")
    return BaselineVO(tokens, vs)
