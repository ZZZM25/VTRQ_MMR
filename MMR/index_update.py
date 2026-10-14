"""Incremental arrival-order updates for the authenticated v2 index.

Only affected edge blocks are redirected. Historical MMR nodes are copied into
new per-edge blocks, new leaves and carry merges are hashed, and the affected
E_h blocks plus SEM-tree ancestors are written back in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import edge_mmr
from edge_entries import ENTRY_SIZE, UINT32_MAX, EdgeEntryStore, _ENTRY_STRUCT
from index_io import PersistedIndex
from l0_authenticated import hash_l0_leaf, hash_l0_internal, _child_phi
from l1_merkle import hash_edge_block
from spatial_skeleton import NODE_INTERNAL, NODE_LEAF

UPDATE_ALGORITHM = "arrival_order_append_mmr_v1"


@dataclass(frozen=True)
class UpdatePlan:
    affected_eids: tuple[int, ...]
    parents: tuple[int, ...]
    base_root: bytes
    base_entry_count: int


def prepare_index_update(index: PersistedIndex, delta: EdgeEntryStore) -> UpdatePlan:
    """Validate the prepared delta and locate static parents outside timing."""
    if delta is index.store:
        raise ValueError("delta must be separate from the base EntryStore")
    if len(delta.entry_counts) != len(index.store.entry_counts):
        raise ValueError("delta and base must use the same road network")
    eids = []
    for eid, count in enumerate(delta.entry_counts):
        if not count:
            continue
        if not index.road.edge_exists(eid):
            raise ValueError(f"delta contains unknown eid={eid}")
        leaf = int(index.l1_index.eid_to_l0_leaf[eid])
        if leaf < 0 or index.skeleton.node_type[leaf] != NODE_LEAF:
            raise ValueError(f"eid={eid} has no SEM-tree leaf")
        if len(delta.edge_buffers[eid]) != count * ENTRY_SIZE:
            raise ValueError(f"invalid delta Entry buffer for eid={eid}")
        if index.store.entry_counts[eid] != index.mmr_index.edge_k[eid]:
            raise ValueError(f"base Entry/MMR count mismatch for eid={eid}")
        eids.append(eid)
    if sum(delta.entry_counts) != delta.total_entries:
        raise ValueError("delta total_entries does not match its edge counts")

    sk = index.skeleton
    parents = [-1] * sk.node_count
    for node in range(sk.node_count):
        if sk.node_type[node] != NODE_INTERNAL:
            continue
        for child in (sk.left_child[node], sk.mid_child[node], sk.right_child[node]):
            if child < 0 or child >= node or parents[child] != -1:
                raise ValueError("invalid SEM-tree postorder/parent mapping")
            parents[child] = node
    return UpdatePlan(tuple(eids), tuple(parents), index.l0_index.root_hash,
                      index.store.total_entries)


def _update_blocks(index: PersistedIndex, eids: tuple[int, ...]) -> set[int]:
    sk, blocks, mmr = index.skeleton, index.l1_index, index.mmr_index
    leaves = {int(blocks.eid_to_l0_leaf[eid]) for eid in eids}
    for leaf in sorted(leaves):
        off, count = int(sk.leaf_edge_offset[leaf]), int(sk.leaf_edge_count[leaf])
        block_eids = sk.leaf_edge_ids[off:off + count]
        blocks.set_l1_root(leaf, hash_edge_block(block_eids, mmr, index.road))
        mn, mx = UINT32_MAX, 0
        for eid in block_eids:
            if mmr.edge_k[eid]:
                mn = min(mn, int(mmr.edge_min_start[eid]))
                mx = max(mx, int(mmr.edge_max_end[eid]))
        blocks.l0_min_start[leaf] = mn
        blocks.l0_max_end[leaf] = mx
    return leaves


def _update_l0(index: PersistedIndex, leaves: set[int], parents: tuple[int, ...]) -> int:
    sk, blocks, l0 = index.skeleton, index.l1_index, index.l0_index
    dirty = set(leaves)
    for leaf in leaves:
        node = parents[leaf]
        while node >= 0 and node not in dirty:
            dirty.add(node)
            node = parents[node]
    # Skeleton indices are postorder: children are committed before parents.
    for node in sorted(dirty):
        bounds = (sk.min_lon[node], sk.min_lat[node], sk.max_lon[node], sk.max_lat[node])
        if sk.node_type[node] == NODE_LEAF:
            mn, mx = int(blocks.l0_min_start[node]), int(blocks.l0_max_end[node])
            digest = hash_l0_leaf(*bounds, mn, mx, edge_block_hash=blocks.get_l1_root(node))
        else:
            children = (sk.left_child[node], sk.mid_child[node], sk.right_child[node])
            mn = min(l0.node_min_start[c] for c in children)
            mx = max(l0.node_max_end[c] for c in children)
            phis = [_child_phi(sk, l0, c) for c in children]
            digest = hash_l0_internal(*bounds, mn, mx,
                                     left_phi=phis[0], mid_phi=phis[1], right_phi=phis[2])
        l0.node_min_start[node] = mn
        l0.node_max_end[node] = mx
        l0.set_node_hash(node, digest)
    l0.root_hash = l0.get_node_hash(l0.root_index)
    return len(dirty)


def apply_index_update(index: PersistedIndex, delta: EdgeEntryStore,
                       plan: UpdatePlan, source_files=()) -> dict:
    """Commit a prepared batch and return measured stages.

    Existing MMR blocks are contiguous and have no spare capacity. Replacement
    blocks are appended to the node arena and their edge offsets are redirected.
    Old blocks remain unreachable until a future compaction. Their storage is
    reported explicitly; no global copying/compaction is hidden outside timing.
    The resulting PersistedIndex works with the existing queries and save_index.
    """
    if (index.l0_index.root_hash != plan.base_root
            or index.store.total_entries != plan.base_entry_count):
        raise ValueError("stale update plan; prepare again for the current index")
    eids = plan.affected_eids
    total_t0 = perf_counter()

    t0 = perf_counter()
    delta_records: dict[int, list[tuple[bytes, int, int]]] = {}
    for eid in eids:
        records = []
        for tid, start, end in _ENTRY_STRUCT.iter_unpack(delta.edge_buffers[eid]):
            index.store.append(eid, tid, start, end)
            records.append((tid, int(start), int(end)))
        delta_records[eid] = records
    index.store.total_trajectories += delta.total_trajectories
    index.store.processed_files += delta.processed_files
    index.trajectory_files = tuple(dict.fromkeys((*index.trajectory_files, *source_files)))
    entry_time = perf_counter() - t0

    t0 = perf_counter()
    previous_nodes = len(index.mmr_index.node_min_start)
    copied_nodes = 0
    hashed_nodes = 0
    for eid in eids:
        was_empty = index.mmr_index.edge_k[eid] == 0
        stats = edge_mmr.append_single_edge_mmr(
            eid, index.store, index.mmr_index, delta_records[eid]
        )
        copied_nodes += stats["copied_historical_mmr_nodes"]
        hashed_nodes += stats["hashed_new_mmr_nodes"]
        if was_empty:
            index.mmr_index.non_empty_edges += 1
    index.mmr_index.total_nodes = len(index.mmr_index.node_min_start)
    rebuilt_nodes = index.mmr_index.total_nodes - previous_nodes
    mmr_time = perf_counter() - t0

    t0 = perf_counter()
    leaves = _update_blocks(index, eids)
    block_time = perf_counter() - t0

    t0 = perf_counter()
    l0_count = _update_l0(index, leaves, plan.parents)
    l0_time = perf_counter() - t0
    total_time = perf_counter() - total_t0

    active_nodes = sum(index.mmr_index.edge_node_count)
    return {
        "update_algorithm": UPDATE_ALGORITHM,
        "inserted_entry_count": delta.total_entries,
        "affected_edge_count": len(eids),
        "affected_edge_block_count": len(leaves),
        "updated_l0_node_count": l0_count,
        "copied_historical_mmr_nodes": copied_nodes,
        "hashed_new_mmr_nodes": hashed_nodes,
        "rebuilt_mmr_node_count": rebuilt_nodes,
        "active_mmr_node_count": active_nodes,
        "allocated_mmr_node_count": index.mmr_index.total_nodes,
        "retained_obsolete_mmr_node_count": index.mmr_index.total_nodes - active_nodes,
        "update_time_s": total_time,
        "entry_apply_time_s": entry_time,
        "mmr_update_time_s": mmr_time,
        "l1_update_time_s": block_time,
        "l0_update_time_s": l0_time,
    }
