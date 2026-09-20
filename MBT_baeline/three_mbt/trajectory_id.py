from __future__ import annotations

import hashlib
import struct
from typing import Iterable

_U32 = struct.Struct(">I")
UINT32_MAX = (1 << 32) - 1


def _pack_u32(value: int, name: str) -> bytes:
    value = int(value)
    if not 0 <= value <= UINT32_MAX:
        raise OverflowError(f"{name}={value} is outside uint32")
    return _U32.pack(value)


def compute_paper_trajectory_id(
    road_node_path: Iterable[int],
    trajectory_start: int,
    trajectory_end: int,
) -> str:
    """Compute the paper trajectory ID used by this project.

    Definition fixed by the experiment design:

        SHA256(
            uint32_be(node_0) || ... || uint32_be(node_n)
            || uint32_be(trajectory_start)
            || uint32_be(trajectory_end)
        )

    Node order is preserved and repeated nodes are preserved.
    The returned value is a 64-hex-character lowercase string.
    """
    nodes = [int(x) for x in road_node_path]
    if not nodes:
        raise ValueError("road_node_path must not be empty")
    start = int(trajectory_start)
    end = int(trajectory_end)
    if end < start:
        raise ValueError("trajectory_end < trajectory_start")

    h = hashlib.sha256()
    for node_id in nodes:
        h.update(_pack_u32(node_id, "node_id"))
    h.update(_pack_u32(start, "trajectory_start"))
    h.update(_pack_u32(end, "trajectory_end"))
    return h.hexdigest()


def is_sha256_hex(value: str) -> bool:
    text = str(value).strip()
    if len(text) != 64:
        return False
    try:
        bytes.fromhex(text)
    except ValueError:
        return False
    return True
