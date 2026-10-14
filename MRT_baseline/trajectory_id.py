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


def compute_trajectory_id(
    road_node_path: Iterable[int], trajectory_start: int, trajectory_end: int
) -> str:
    """Use the same stable trajectory-ID definition as the previous MBT baseline.

    The R-tree does NOT index or query the road-node path.  The path is used only
    to keep trajectory IDs identical across baselines.
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
