from __future__ import annotations

import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .build import load_index_metadata
from .formats import BBOX, POINT_DTYPE, POINT_STRUCT, U16, U32
from .geometry import bbox_intersects
from .models import QueryResult, QueryWindow

VO_PRUNED = 0
VO_INTERNAL = 1
VO_LEAF = 2


class AuthenticatedPointRTree:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.meta = load_index_metadata(self.index_dir)
        self.levels = self.meta["levels"]
        self.root_hash: bytes = self.meta["root_hash"]
        self.root_hash_hex: str = self.meta["root_hash_hex"]
        self.point_files = [self.index_dir / x for x in self.meta["point_files"]]

    def query(self, query: QueryWindow, vo_dir: str | Path) -> QueryResult:
        """Spatial R-tree traversal, then same-point timestamp check at expanded leaves."""
        vo_dir = Path(vo_dir)
        vo_dir.mkdir(parents=True, exist_ok=True)
        fd, vo_name = tempfile.mkstemp(prefix="gps_rtree_vo_", suffix=".bin", dir=vo_dir)
        os.close(fd)
        vo_path = Path(vo_name)

        candidates: set[bytes] = set()
        spatial_match_point_count = 0
        spatiotemporal_match_point_count = 0
        expanded_leaf_count = 0
        root_level = len(self.levels) - 1

        with ExitStack() as stack:
            point_handles = [
                stack.enter_context(p.open("rb", buffering=8 * 1024 * 1024))
                for p in self.point_files
            ]
            vo = stack.enter_context(vo_path.open("wb", buffering=8 * 1024 * 1024))

            def visit(level: int, node_index: int) -> None:
                nonlocal spatial_match_point_count, spatiotemporal_match_point_count, expanded_leaf_count
                node = self.levels[level][node_index]
                if not bbox_intersects(node.bbox, query):
                    vo.write(bytes((VO_PRUNED,)))
                    vo.write(BBOX.pack(*node.bbox))
                    vo.write(node.digest)
                    return

                if level > 0:
                    vo.write(bytes((VO_INTERNAL,)))
                    vo.write(U16.pack(node.child_count))
                    for child_index in range(node.first_child, node.first_child + node.child_count):
                        visit(level - 1, child_index)
                    return

                expanded_leaf_count += 1
                vo.write(bytes((VO_LEAF,)))
                vo.write(U32.pack(node.count))
                fh = point_handles[node.point_file_id]
                fh.seek(node.offset)
                raw = fh.read(node.count * POINT_STRUCT.size)
                if len(raw) != node.count * POINT_STRUCT.size:
                    raise IOError("short point leaf read")
                vo.write(raw)
                arr = np.frombuffer(raw, dtype=POINT_DTYPE, count=node.count)
                space_mask = (
                    (arr["lon"] >= query.min_lon)
                    & (arr["lon"] <= query.max_lon)
                    & (arr["lat"] >= query.min_lat)
                    & (arr["lat"] <= query.max_lat)
                )
                spatial_match_point_count += int(np.count_nonzero(space_mask))
                both = space_mask & (arr["ts"] >= query.start_time) & (arr["ts"] <= query.end_time)
                matched = arr[both]
                spatiotemporal_match_point_count += int(len(matched))
                for tid in matched["tid"]:
                    candidates.add(bytes(tid))

            t0 = time.perf_counter()
            visit(root_level, 0)
            vo.flush()
            t1 = time.perf_counter()

        return QueryResult(
            candidate_trajectory_ids=sorted(x.hex() for x in candidates),
            query_time_s=t1 - t0,
            vo_path=str(vo_path),
            vo_size_bytes=vo_path.stat().st_size,
            spatial_match_point_count=spatial_match_point_count,
            spatiotemporal_match_point_count=spatiotemporal_match_point_count,
            expanded_leaf_count=expanded_leaf_count,
        )
