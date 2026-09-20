from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from road_network import RoadNetwork
from edge_entries import EdgeEntryStore
from edge_mmr import (
    EdgeMMRIndex,
    get_peak_heights,
    get_peak_node_indices,
)
from spatial_skeleton import (
    SpatialSkeleton,
    NODE_LEAF,
)
from l1_merkle import L1MerkleIndex
from l0_authenticated import L0AuthenticatedIndex
from crypto import time_overlap
from query_utils import (
    normalize_rectangle,
    rectangle_overlap,
    edge_intersects_query,
    is_empty_time_range,
)


TOKEN_L0_PRUNED_LEAF = 1
TOKEN_L0_PRUNED_INTERNAL = 2
TOKEN_L0_LEAF = 3
TOKEN_L0_INTERNAL = 4

TOKEN_L1_EDGE_OPAQUE = 10
TOKEN_L1_EDGE_EXPANDED = 11
TOKEN_L1_INTERNAL = 12

TOKEN_MMR_PRUNED_LEAF = 20
TOKEN_MMR_PRUNED_INTERNAL = 21
TOKEN_MMR_RESULT_SLOT = 22
TOKEN_MMR_INTERNAL = 23
TOKEN_MMR_ROOT = 24


TOKEN_NAMES = {
    TOKEN_L0_PRUNED_LEAF: "L0_PRUNED_LEAF",
    TOKEN_L0_PRUNED_INTERNAL: "L0_PRUNED_INTERNAL",
    TOKEN_L0_LEAF: "L0_LEAF",
    TOKEN_L0_INTERNAL: "L0_INTERNAL",
    TOKEN_L1_EDGE_OPAQUE: "L1_EDGE_OPAQUE",
    TOKEN_L1_EDGE_EXPANDED: "L1_EDGE_EXPANDED",
    TOKEN_L1_INTERNAL: "L1_INTERNAL",
    TOKEN_MMR_PRUNED_LEAF: "MMR_PRUNED_LEAF",
    TOKEN_MMR_PRUNED_INTERNAL: "MMR_PRUNED_INTERNAL",
    TOKEN_MMR_RESULT_SLOT: "MMR_RESULT_SLOT",
    TOKEN_MMR_INTERNAL: "MMR_INTERNAL",
    TOKEN_MMR_ROOT: "MMR_ROOT",
}


@dataclass(slots=True)
class VerificationEntry:
    eid: int
    trajectory_id: bytes
    start: int
    end: int


@dataclass(slots=True)
class CompositeVO:
    tokens: list[tuple]
    verification_set: list[VerificationEntry]
    candidate_ids: list[bytes]


class CompositeVOBuilder:

    __slots__ = (
        "road",
        "store",
        "mmr_index",
        "skeleton",
        "l1_index",
        "l0_index",
        "query_min_lon",
        "query_min_lat",
        "query_max_lon",
        "query_max_lat",
        "query_start",
        "query_end",
        "tokens",
        "verification_set",
    )

    def __init__(
        self,
        road: RoadNetwork,
        store: EdgeEntryStore,
        mmr_index: EdgeMMRIndex,
        skeleton: SpatialSkeleton,
        l1_index: L1MerkleIndex,
        l0_index: L0AuthenticatedIndex,
        query_min_lon: float,
        query_min_lat: float,
        query_max_lon: float,
        query_max_lat: float,
        query_start: int,
        query_end: int,
    ):
        if query_start > query_end:
            raise ValueError("query_start > query_end")

        (
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
        ) = normalize_rectangle(
            query_min_lon,
            query_min_lat,
            query_max_lon,
            query_max_lat,
        )

        self.road = road
        self.store = store
        self.mmr_index = mmr_index
        self.skeleton = skeleton
        self.l1_index = l1_index
        self.l0_index = l0_index

        self.query_min_lon = query_min_lon
        self.query_min_lat = query_min_lat
        self.query_max_lon = query_max_lon
        self.query_max_lat = query_max_lat

        self.query_start = query_start
        self.query_end = query_end

        self.tokens: list[tuple] = []
        self.verification_set: list[VerificationEntry] = []

    def _l0_should_prune(
        self,
        node_index: int,
    ) -> bool:
        if not rectangle_overlap(
            self.skeleton.min_lon[node_index],
            self.skeleton.min_lat[node_index],
            self.skeleton.max_lon[node_index],
            self.skeleton.max_lat[node_index],
            self.query_min_lon,
            self.query_min_lat,
            self.query_max_lon,
            self.query_max_lat,
        ):
            return True

        min_start = self.l0_index.node_min_start[node_index]
        max_end = self.l0_index.node_max_end[node_index]

        if is_empty_time_range(
            min_start,
            max_end,
        ):
            return True

        if not time_overlap(
            min_start,
            max_end,
            self.query_start,
            self.query_end,
        ):
            return True

        return False

    def _emit_pruned_l0(
        self,
        node_index: int,
    ) -> None:
        min_lon = self.skeleton.min_lon[node_index]
        min_lat = self.skeleton.min_lat[node_index]
        max_lon = self.skeleton.max_lon[node_index]
        max_lat = self.skeleton.max_lat[node_index]

        min_start = self.l0_index.node_min_start[node_index]
        max_end = self.l0_index.node_max_end[node_index]

        if self.skeleton.node_type[node_index] == NODE_LEAF:
            l1_root = self.l1_index.get_l1_root(
                node_index
            )

            self.tokens.append(
                (
                    TOKEN_L0_PRUNED_LEAF,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    min_start,
                    max_end,
                    l1_root,
                )
            )
            return

        left = self.skeleton.left_child[node_index]
        mid = self.skeleton.mid_child[node_index]
        right = self.skeleton.right_child[node_index]

        self.tokens.append(
            (
                TOKEN_L0_PRUNED_INTERNAL,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                min_start,
                max_end,
                self.l0_index.get_node_hash(left),
                self.l0_index.get_node_hash(mid),
                self.l0_index.get_node_hash(right),
            )
        )

    def _emit_mmr_subtree(
        self,
        eid: int,
        node_index: int,
        height: int,
        leaf_start: int,
    ) -> None:
        min_start = self.mmr_index.node_min_start[node_index]
        max_end = self.mmr_index.node_max_end[node_index]

        if not time_overlap(
            min_start,
            max_end,
            self.query_start,
            self.query_end,
        ):
            if height == 0:
                trajectory_id, start, end = self.store.get_entry(
                    eid=eid,
                    index=leaf_start,
                )

                self.tokens.append(
                    (
                        TOKEN_MMR_PRUNED_LEAF,
                        trajectory_id,
                        start,
                        end,
                    )
                )
                return

            left_node = node_index - (1 << height)
            right_node = node_index - 1

            self.tokens.append(
                (
                    TOKEN_MMR_PRUNED_INTERNAL,
                    min_start,
                    max_end,
                    self.mmr_index.get_node_hash(left_node),
                    self.mmr_index.get_node_hash(right_node),
                )
            )
            return

        if height == 0:
            trajectory_id, start, end = self.store.get_entry(
                eid=eid,
                index=leaf_start,
            )

            self.verification_set.append(
                VerificationEntry(
                    eid=eid,
                    trajectory_id=trajectory_id,
                    start=start,
                    end=end,
                )
            )

            self.tokens.append(
                (
                    TOKEN_MMR_RESULT_SLOT,
                )
            )
            return

        left_node = node_index - (1 << height)
        right_node = node_index - 1

        child_height = height - 1
        left_leaf_count = 1 << child_height

        self._emit_mmr_subtree(
            eid=eid,
            node_index=left_node,
            height=child_height,
            leaf_start=leaf_start,
        )

        self._emit_mmr_subtree(
            eid=eid,
            node_index=right_node,
            height=child_height,
            leaf_start=leaf_start + left_leaf_count,
        )

        self.tokens.append(
            (
                TOKEN_MMR_INTERNAL,
                min_start,
                max_end,
            )
        )

    def _emit_edge_mmr(
        self,
        eid: int,
    ) -> None:
        k_e = self.mmr_index.edge_k[eid]

        if k_e <= 0:
            raise RuntimeError(
                f"eid={eid} has an empty MMR but entered MMR expansion"
            )

        node_offset = self.mmr_index.edge_node_offset[eid]

        peak_heights = get_peak_heights(
            k_e
        )

        peak_nodes = get_peak_node_indices(
            node_offset=node_offset,
            k=k_e,
        )

        leaf_start = 0

        for height, peak_node in zip(
            peak_heights,
            peak_nodes,
        ):
            self._emit_mmr_subtree(
                eid=eid,
                node_index=peak_node,
                height=height,
                leaf_start=leaf_start,
            )

            leaf_start += 1 << height

        self.tokens.append(
            (
                TOKEN_MMR_ROOT,
                k_e,
            )
        )

    def _emit_l1_edge(
        self,
        eid: int,
    ) -> None:
        (
            k_e,
            min_start,
            max_end,
            root_e,
        ) = self.mmr_index.get_edge_state(
            eid
        )

        spatial_match = edge_intersects_query(
            eid=eid,
            road=self.road,
            query_min_lon=self.query_min_lon,
            query_min_lat=self.query_min_lat,
            query_max_lon=self.query_max_lon,
            query_max_lat=self.query_max_lat,
        )

        if not spatial_match:
            self.tokens.append(
                (
                    TOKEN_L1_EDGE_OPAQUE,
                    eid,
                    k_e,
                    min_start,
                    max_end,
                    root_e,
                )
            )
            return

        if k_e == 0:
            self.tokens.append(
                (
                    TOKEN_L1_EDGE_OPAQUE,
                    eid,
                    k_e,
                    min_start,
                    max_end,
                    root_e,
                )
            )
            return

        if not time_overlap(
            min_start,
            max_end,
            self.query_start,
            self.query_end,
        ):
            self.tokens.append(
                (
                    TOKEN_L1_EDGE_OPAQUE,
                    eid,
                    k_e,
                    min_start,
                    max_end,
                    root_e,
                )
            )
            return

        self._emit_edge_mmr(
            eid
        )

        self.tokens.append(
            (
                TOKEN_L1_EDGE_EXPANDED,
                eid,
                k_e,
                min_start,
                max_end,
            )
        )

    def _emit_l1_subtree(
        self,
        node_index: int,
    ) -> None:
        eid = self.l1_index.node_eid[node_index]

        if eid >= 0:
            self._emit_l1_edge(
                eid
            )
            return

        left = self.l1_index.node_left[node_index]
        right = self.l1_index.node_right[node_index]

        self._emit_l1_subtree(
            left
        )

        self._emit_l1_subtree(
            right
        )

        self.tokens.append(
            (
                TOKEN_L1_INTERNAL,
            )
        )

    def _emit_l0_leaf(
        self,
        node_index: int,
    ) -> None:
        l1_root_node = self.l1_index.l0_l1_root_node[
            node_index
        ]

        if l1_root_node < 0:
            raise RuntimeError(
                f"L0 Leaf={node_index} has an empty L1 but entered expansion"
            )

        self._emit_l1_subtree(
            l1_root_node
        )

        self.tokens.append(
            (
                TOKEN_L0_LEAF,
                self.skeleton.min_lon[node_index],
                self.skeleton.min_lat[node_index],
                self.skeleton.max_lon[node_index],
                self.skeleton.max_lat[node_index],
                self.l0_index.node_min_start[node_index],
                self.l0_index.node_max_end[node_index],
            )
        )

    def _emit_l0_node(
        self,
        node_index: int,
    ) -> None:
        if self._l0_should_prune(
            node_index
        ):
            self._emit_pruned_l0(
                node_index
            )
            return

        if self.skeleton.node_type[node_index] == NODE_LEAF:
            self._emit_l0_leaf(
                node_index
            )
            return

        left = self.skeleton.left_child[node_index]
        mid = self.skeleton.mid_child[node_index]
        right = self.skeleton.right_child[node_index]

        self._emit_l0_node(
            left
        )

        self._emit_l0_node(
            mid
        )

        self._emit_l0_node(
            right
        )

        self.tokens.append(
            (
                TOKEN_L0_INTERNAL,
                self.skeleton.min_lon[node_index],
                self.skeleton.min_lat[node_index],
                self.skeleton.max_lon[node_index],
                self.skeleton.max_lat[node_index],
                self.l0_index.node_min_start[node_index],
                self.l0_index.node_max_end[node_index],
            )
        )

    def build(
        self,
    ) -> CompositeVO:
        self.tokens.clear()
        self.verification_set.clear()

        self._emit_l0_node(
            self.skeleton.root_index
        )

        candidate_ids = sorted(
            {
                entry.trajectory_id
                for entry in self.verification_set
            }
        )

        return CompositeVO(
            tokens=list(self.tokens),
            verification_set=list(self.verification_set),
            candidate_ids=candidate_ids,
        )


def print_vo_statistics(
    vo: CompositeVO,
) -> None:
    counter = Counter(
        token[0]
        for token in vo.tokens
    )

    print(
        "Total number of VO tokens:",
        len(vo.tokens),
    )

    print(
        "Verification Set count:",
        len(vo.verification_set),
    )

    print(
        "Candidate trajectory_id count:",
        len(vo.candidate_ids),
    )

    print()
    print(
        "===== Token Categories ====="
    )

    for token_type in sorted(
        counter
    ):
        print(
            TOKEN_NAMES[token_type],
            ":",
            counter[token_type],
        )


def print_verification_set(
    vo: CompositeVO,
    show_count: int = 10,
) -> None:
    print(
        "===== Verification Set ====="
    )

    for i, entry in enumerate(
        vo.verification_set[:show_count]
    ):
        print()
        print(
            f"VS[{i}]"
        )
        print(
            "eid:",
            entry.eid,
        )
        print(
            "trajectory_id:",
            entry.trajectory_id.hex(),
        )
        print(
            "start:",
            entry.start,
        )
        print(
            "end:",
            entry.end,
        )
