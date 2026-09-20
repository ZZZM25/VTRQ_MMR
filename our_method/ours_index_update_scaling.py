
# ============================================================
# DIRECT incremental index update scalability experiment
#
# Ours update path:
#   Entry append in arrival order
#   -> append-only tau-MMR new leaf/merge hashes
#   -> affected L1 authentication paths
#   -> affected L0 authentication paths
#
# No probe. No dynamic API guessing. No full rebuild inside Update Time.
#
# Experiment:
#   base = exactly the existing 60% authenticated index
#   delta sizes = 1000 / 2000 / 4000 / 8000 / 10000 trajectories
#
# Every point starts from the SAME 60% base index independently.
# Delta is the next N raw trajectories immediately after the 60% prefix.
#
# Raw I/O / selection / parsing / EID mapping / delta preparation are EXCLUDED.
# Serialization is EXCLUDED.
#
# IMPORTANT:
#   This harness intentionally has NO full-rebuild fallback.
#   If the project does not expose a real incremental update primitive,
#   the run fails instead of silently timing a rebuild.
# ============================================================

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = Path(r"E:\VTRQ_baseline")

CITIES = ("chengdu", "xian", "beijing")
INSERT_SIZES = (1000, 2000, 4000, 8000, 10000)

TOTAL_TRAJECTORIES = {
    "chengdu": 284_608,
    "xian": 45_851,
    "beijing": 31_390,
}
BASE_PERCENT = 60

BASELINE_CONFIGS = {
    "chengdu": BASELINE_ROOT / "configs" / "chengdu.json",
    "xian": BASELINE_ROOT / "configs" / "xian.json",
}

BEIJING_DIR = PROJECT_ROOT / "beijing_preprocessed"
BEIJING_NODE = BEIJING_DIR / "beijing_nodes.txt"
BEIJING_EDGE = BEIJING_DIR / "beijing_edges.txt"
BEIJING_TRAJECTORY = BEIJING_DIR / "beijing_trajectories.csv"
BEIJING_THETA = 64

BASE_INDEX_DIR = PROJECT_ROOT / "index_scaling_output" / "ours" / "indexes"
OUTPUT_ROOT = PROJECT_ROOT / "index_update_output" / "ours"
WORKER_DIR = OUTPUT_ROOT / "_worker_results"
RESULT_FILE = OUTPUT_ROOT / "index_update_results.csv"
MATRIX_FILE = OUTPUT_ROOT / "update_time_matrix_s.csv"


def base_count(city: str) -> int:
    return TOTAL_TRAJECTORIES[city] * BASE_PERCENT // 100


def final_count(city: str, insert_count: int) -> int:
    return base_count(city) + int(insert_count)


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolve_config_path(value: str, config_file: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    for q in (config_file.parent / p, BASELINE_ROOT / p, p):
        if q.exists():
            return q.resolve()
    return (config_file.parent / p).resolve()


def expand_glob(pattern: str, config_file: Path) -> list[Path]:
    p = Path(pattern)
    patterns = [str(p)] if p.is_absolute() else [
        str(config_file.parent / p),
        str(BASELINE_ROOT / p),
        str(p),
    ]
    found: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for value in glob.glob(pat):
            q = Path(value).resolve()
            key = str(q).lower()
            if key not in seen:
                seen.add(key)
                found.append(q)
    return found


def load_city_spec(city: str) -> tuple[Path, Path, int, list[Path]]:
    if city == "beijing":
        node = BEIJING_NODE.resolve()
        edge = BEIJING_EDGE.resolve()
        theta = BEIJING_THETA
        files = [BEIJING_TRAJECTORY.resolve()]
    else:
        cfg_file = BASELINE_CONFIGS[city]
        if not cfg_file.exists():
            raise FileNotFoundError(cfg_file)
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
        node = resolve_config_path(str(cfg["node_file"]), cfg_file)
        edge = resolve_config_path(str(cfg["edge_file"]), cfg_file)
        theta = int(cfg.get("theta", 64))
        values: list[Path] = []
        for pat in cfg.get("trajectory_globs", []):
            values.extend(expand_glob(str(pat), cfg_file))
        unique: list[Path] = []
        seen: set[str] = set()
        for p in values:
            k = str(p).lower()
            if k not in seen:
                seen.add(k)
                unique.append(p)
        files = sorted(unique, key=natural_key)

    for p in (node, edge, *files):
        if not p.exists():
            raise FileNotFoundError(p)
    if not files:
        raise FileNotFoundError(f"{city}: no trajectory files")
    return node, edge, theta, files


class SliceJSONLoader:
    """
    Wrap the project's original JSON loader and return only the raw trajectory
    slice [start, start+count), preserving the exact natural file order.
    """
    def __init__(self, base_loader, start: int, count: int):
        self.base_loader = base_loader
        self.start = int(start)
        self.stop = int(start + count)
        self.raw_seen = 0
        self.selected = 0
        self.used_files: list[str] = []

    def __call__(self, file_path):
        if self.raw_seen >= self.stop:
            return []

        data = self.base_loader(file_path)
        if not isinstance(data, list):
            raise ValueError(f"{file_path}: top-level trajectory data must be list")

        file_start = self.raw_seen
        file_stop = file_start + len(data)
        self.raw_seen = file_stop

        left = max(self.start, file_start)
        right = min(self.stop, file_stop)
        if right <= left:
            return []

        lo = left - file_start
        hi = right - file_start
        selected = data[lo:hi]
        self.selected += len(selected)
        if selected:
            self.used_files.append(str(Path(file_path).resolve()))
        return selected


class SliceBeijingLoader:
    def __init__(self, start: int, count: int):
        self.start = int(start)
        self.stop = int(start + count)
        self.selected = 0
        self.used_files: list[str] = []

    @staticmethod
    def _raise_csv_field_limit():
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                return
            except OverflowError:
                limit //= 10

    def __call__(self, file_path):
        self._raise_csv_field_limit()
        path = Path(file_path)
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if "points_data" not in fields:
                raise ValueError(f"{path}: missing points_data")
            for raw_index, row in enumerate(reader):
                if raw_index < self.start:
                    continue
                if raw_index >= self.stop:
                    break
                self.selected += 1
                source_traj_id = (
                    row.get("source_traj_id")
                    or row.get("traj_id")
                    or ""
                ).strip()
                points = json.loads(row["points_data"])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                rows.append({
                    "source_traj_id": source_traj_id,
                    "points": points,
                    "row_no": raw_index + 2,
                })
        if self.selected:
            self.used_files.append(str(path.resolve()))
        return rows


def base_index_file(city: str) -> Path:
    return (BASE_INDEX_DIR / f"{city}_060pct_ours.dat").resolve()


def affected_edges_from_store(store, road) -> list[int]:
    if not hasattr(store, "entry_counts"):
        raise RuntimeError(
            "Delta store has no entry_counts; cannot identify affected edges safely."
        )
    return [
        eid
        for eid in range(int(road.max_eid) + 1)
        if road.edge_exists(eid) and int(store.entry_counts[eid]) > 0
    ]



def block_full_rebuild_ours(edge_mmr, l1_merkle, l0_authenticated):
    """
    Replace only the public FULL-build functions during the timed update.
    A correct incremental implementation should not call them.
    """
    originals = []

    def blocked(name):
        def _blocked(*args, **kwargs):
            raise RuntimeError(
                f"FULL REBUILD FORBIDDEN inside Update Time: {name}"
            )
        return _blocked

    for module, name in (
        (edge_mmr, "build_all_edge_mmrs"),
        (l1_merkle, "build_all_l1_trees"),
        (l0_authenticated, "build_authenticated_l0"),
    ):
        if hasattr(module, name):
            originals.append((module, name, getattr(module, name)))
            setattr(module, name, blocked(f"{module.__name__}.{name}"))
    return originals


def restore_functions(originals):
    for module, name, fn in originals:
        setattr(module, name, fn)


def root_hex(index) -> str:
    value = index.l0_index.root_hash
    return value.hex() if isinstance(value, (bytes, bytearray)) else str(value)


def prepare_delta_store(city: str, insert_count: int, trajectory_files: list[Path], road):
    # Beijing adapter must be installed before importing edge_entries.
    if city == "beijing":
        from beijing_csv_adapter import install_beijing_adapter
        install_beijing_adapter()

    import edge_entries

    if not hasattr(edge_entries, "load_trajectory_json"):
        raise RuntimeError("edge_entries.py does not expose load_trajectory_json")
    if not hasattr(edge_entries, "parse_trajectory"):
        raise RuntimeError("edge_entries.py does not expose parse_trajectory")

    start = base_count(city)

    if city == "beijing":
        from beijing_csv_adapter import parse_beijing_trajectory
        loader = SliceBeijingLoader(start, insert_count)
        edge_entries.load_trajectory_json = loader
        edge_entries.parse_trajectory = parse_beijing_trajectory
    else:
        original_loader = edge_entries.load_trajectory_json
        loader = SliceJSONLoader(original_loader, start, insert_count)
        edge_entries.load_trajectory_json = loader

    t0 = perf_counter()
    delta_store = edge_entries.build_edge_entries_from_files(
        trajectory_files=[str(p) for p in trajectory_files],
        road=road,
    )
    preparation_time = perf_counter() - t0

    if loader.selected != insert_count:
        raise RuntimeError(
            f"{city}: requested delta {insert_count}, selected {loader.selected}. "
            "Check totals/input files."
        )
    return delta_store, loader, preparation_time


def verify_reference_root(index) -> tuple[bool, str]:
    """
    Full rebuild validation OUTSIDE update timing.
    This is optional and never contributes to update_time_s.
    """
    from edge_mmr import build_all_edge_mmrs
    from l1_merkle import build_all_l1_trees
    from l0_authenticated import build_authenticated_l0

    mmr = build_all_edge_mmrs(store=index.store, road=index.road)
    l1 = build_all_l1_trees(
        skeleton=index.skeleton,
        mmr_index=mmr,
        road=index.road,
    )
    l0 = build_authenticated_l0(
        skeleton=index.skeleton,
        l1_index=l1,
    )
    ref = l0.root_hash
    got = index.l0_index.root_hash
    return ref == got, ref.hex()




def _hash_at(obj, node_index: int) -> bytes:
    """Read a stored node hash without rebuilding an unchanged subtree."""
    fn = getattr(obj, "get_node_hash", None)
    if callable(fn):
        value = fn(node_index)
        if isinstance(value, (bytes, bytearray)) and len(value) == 32:
            return bytes(value)

    for name in ("node_hash", "node_hashes", "hashes", "node_hash_bytes"):
        if not hasattr(obj, name):
            continue
        raw = getattr(obj, name)
        try:
            value = raw[node_index]
            if isinstance(value, (bytes, bytearray)) and len(value) == 32:
                return bytes(value)
        except Exception:
            pass
        if isinstance(raw, (bytes, bytearray, memoryview)):
            p = node_index * 32
            value = bytes(raw[p:p+32])
            if len(value) == 32:
                return value
    raise RuntimeError(
        f"{type(obj).__name__} does not expose stored node hashes; "
        "expected get_node_hash() or node_hash/node_hashes storage"
    )


def _append_base_store(base_store, delta_store, affected_eids):
    """Append new Entries in arrival order. Ours never start-time sorts them."""
    append_fn = getattr(base_store, "append", None)
    add_fn = getattr(base_store, "add_entry", None)

    for eid in affected_eids:
        for i in range(int(delta_store.entry_counts[eid])):
            tid, start, end = delta_store.get_entry(eid, i)
            if callable(append_fn):
                append_fn(eid, tid, int(start), int(end))
                continue
            if callable(add_fn):
                add_fn(eid, tid, int(start), int(end))
                continue

            # Packed store fallback used by our compact Entry representation.
            if all(hasattr(base_store, x) for x in (
                "trajectory_id_bytes", "starts", "ends", "entry_counts"
            )):
                base_store.trajectory_id_bytes[eid].extend(tid)
                base_store.starts[eid].append(int(start))
                base_store.ends[eid].append(int(end))
                base_store.entry_counts[eid] += 1
                continue

            # Generic list-of-entries fallback.
            if hasattr(base_store, "entries"):
                base_store.entries[eid].append((tid, int(start), int(end)))
                if hasattr(base_store, "entry_counts"):
                    base_store.entry_counts[eid] += 1
                continue

            raise RuntimeError(
                f"Unsupported Ours EntryStore layout: {type(base_store).__name__}. "
                "Expected append()/add_entry() or packed Entry arrays."
            )


def _calculate_mmr_root(k: int, peaks) -> bytes:
    import hashlib, struct
    h = hashlib.sha256()
    h.update(struct.pack(">I", int(k)))
    for height, node_hash, min_start, max_end in peaks:
        h.update(node_hash)
        h.update(struct.pack(">II", int(min_start), int(max_end)))
    return h.digest()


def _incremental_edge_states(mmr_index, delta_store, affected_eids):
    """
    Real append-only tau-MMR update.
    Historical MMR nodes are read and reused; only new leaves and merge nodes hash.
    Returns eid -> (new_k, new_min_start, new_max_end, new_root).
    """
    from crypto import hash_mmr_leaf, hash_mmr_internal
    from edge_mmr import get_peak_heights, get_peak_node_indices
    UINT32_MAX = 0xFFFFFFFF

    updated = {}
    for eid in affected_eids:
        old_k, old_min, old_max, old_root = mmr_index.get_edge_state(eid)
        old_k = int(old_k)

        peaks = []
        if old_k > 0:
            heights = get_peak_heights(old_k)
            nodes = get_peak_node_indices(
                node_offset=int(mmr_index.edge_node_offset[eid]),
                k=old_k,
            )
            for height, node in zip(heights, nodes):
                peaks.append((
                    int(height),
                    bytes(mmr_index.get_node_hash(node)),
                    int(mmr_index.node_min_start[node]),
                    int(mmr_index.node_max_end[node]),
                ))

        new_k = old_k
        new_min = int(old_min)
        new_max = int(old_max)

        for i in range(int(delta_store.entry_counts[eid])):
            tid, start, end = delta_store.get_entry(eid, i)
            start = int(start); end = int(end)
            current = (0, hash_mmr_leaf(tid, start, end), start, end)
            new_k += 1
            if new_min == UINT32_MAX:
                new_min = start
            else:
                new_min = min(new_min, start)
            new_max = max(new_max, end)

            while peaks and peaks[-1][0] == current[0]:
                left = peaks.pop()
                right = current
                merged_min = min(left[2], right[2])
                merged_max = max(left[3], right[3])
                merged_hash = hash_mmr_internal(
                    merged_min, merged_max, left[1], right[1]
                )
                current = (left[0] + 1, merged_hash, merged_min, merged_max)
            peaks.append(current)

        root = _calculate_mmr_root(new_k, peaks)
        updated[eid] = (new_k, new_min, new_max, root)
    return updated


def _build_l1_metadata(index):
    l1 = index.l1_index
    n = len(l1.node_eid)
    parent = [-1] * n
    depth = [0] * n
    for i in range(n):
        if int(l1.node_eid[i]) >= 0:
            continue
        left = int(l1.node_left[i]); right = int(l1.node_right[i])
        if left >= 0: parent[left] = i
        if right >= 0: parent[right] = i

    roots = [int(x) for x in l1.l0_l1_root_node if int(x) >= 0]
    stack = [(r, 0) for r in roots]
    seen = set()
    while stack:
        node, d = stack.pop()
        if node in seen: continue
        seen.add(node); depth[node] = d
        if int(l1.node_eid[node]) < 0:
            stack.append((int(l1.node_left[node]), d+1))
            stack.append((int(l1.node_right[node]), d+1))

    edge_leaf = {}
    edge_l0_leaf = {}
    for l0_leaf, root in enumerate(l1.l0_l1_root_node):
        root = int(root)
        if root < 0: continue
        stack = [root]
        while stack:
            node = stack.pop()
            eid = int(l1.node_eid[node])
            if eid >= 0:
                edge_leaf[eid] = node
                edge_l0_leaf[eid] = l0_leaf
            else:
                stack.append(int(l1.node_left[node]))
                stack.append(int(l1.node_right[node]))
    return parent, depth, edge_leaf, edge_l0_leaf


def _incremental_l1_roots(index, updated_edge_states, metadata):
    from l1_merkle import hash_l1_leaf, hash_l1_internal
    l1 = index.l1_index
    parent, depth, edge_leaf, edge_l0_leaf = metadata

    changed = {}
    dirty = set()
    affected_l0 = set()

    for eid, (k_e, min_start, max_end, root_e) in updated_edge_states.items():
        node = edge_leaf[eid]
        changed[node] = hash_l1_leaf(
            eid=eid, k_e=k_e, min_start=min_start, max_end=max_end, root_e=root_e
        )
        affected_l0.add(edge_l0_leaf[eid])
        p = parent[node]
        while p >= 0:
            dirty.add(p)
            p = parent[p]

    for node in sorted(dirty, key=lambda x: depth[x], reverse=True):
        left = int(l1.node_left[node]); right = int(l1.node_right[node])
        left_hash = changed.get(left, _hash_at(l1, left))
        right_hash = changed.get(right, _hash_at(l1, right))
        changed[node] = hash_l1_internal(left_hash, right_hash)

    new_roots = {}
    for l0_leaf in affected_l0:
        root_node = int(l1.l0_l1_root_node[l0_leaf])
        new_roots[l0_leaf] = changed.get(root_node, _hash_at(l1, root_node))
    return new_roots, edge_l0_leaf


def _build_l0_metadata(skeleton):
    n = len(skeleton.node_type)
    parent = [-1] * n
    depth = [0] * n
    stack = [(int(skeleton.root_index), 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        # Leaves commonly carry -1 children; harmless to skip.
        for name in ("left_child", "mid_child", "right_child"):
            arr = getattr(skeleton, name)
            child = int(arr[node])
            if child >= 0:
                parent[child] = node
                stack.append((child, d+1))
    return parent, depth


def _incremental_l0_root(index, new_l1_roots, edge_l0_leaf, delta_store, affected_eids, metadata):
    from l0_authenticated import hash_l0_leaf, hash_l0_internal
    skeleton = index.skeleton
    l0 = index.l0_index
    parent, depth = metadata

    # Addition-only update: temporal envelopes can only expand.
    leaf_minmax = {}
    for eid in affected_eids:
        leaf = edge_l0_leaf[eid]
        mn, mx = leaf_minmax.get(
            leaf,
            (int(l0.node_min_start[leaf]), int(l0.node_max_end[leaf]))
        )
        for i in range(int(delta_store.entry_counts[eid])):
            _, s, e = delta_store.get_entry(eid, i)
            s = int(s); e = int(e)
            mn = s if mn == 0xFFFFFFFF else min(mn, s)
            mx = max(mx, e)
        leaf_minmax[leaf] = (mn, mx)

    changed_hash = {}
    changed_minmax = {}
    dirty = set()

    for leaf, l1_root in new_l1_roots.items():
        mn, mx = leaf_minmax[leaf]
        changed_minmax[leaf] = (mn, mx)
        changed_hash[leaf] = hash_l0_leaf(
            min_lon=skeleton.min_lon[leaf], min_lat=skeleton.min_lat[leaf],
            max_lon=skeleton.max_lon[leaf], max_lat=skeleton.max_lat[leaf],
            min_start=mn, max_end=mx, l1_root=l1_root,
        )
        p = parent[leaf]
        while p >= 0:
            dirty.add(p); p = parent[p]

    for node in sorted(dirty, key=lambda x: depth[x], reverse=True):
        left = int(skeleton.left_child[node])
        mid = int(skeleton.mid_child[node])
        right = int(skeleton.right_child[node])

        children = (left, mid, right)
        child_hashes = [changed_hash.get(c, _hash_at(l0, c)) for c in children]
        child_mm = [changed_minmax.get(
            c, (int(l0.node_min_start[c]), int(l0.node_max_end[c]))
        ) for c in children]
        mn = min(x[0] for x in child_mm)
        mx = max(x[1] for x in child_mm)
        changed_minmax[node] = (mn, mx)
        changed_hash[node] = hash_l0_internal(
            min_lon=skeleton.min_lon[node], min_lat=skeleton.min_lat[node],
            max_lon=skeleton.max_lon[node], max_lat=skeleton.max_lat[node],
            min_start=mn, max_end=mx,
            left_hash=child_hashes[0], mid_hash=child_hashes[1], right_hash=child_hashes[2],
        )

    root = int(skeleton.root_index)
    return changed_hash.get(root, _hash_at(l0, root))


def _verify_reference_root(index, calculated_root: bytes):
    from edge_mmr import build_all_edge_mmrs
    from l1_merkle import build_all_l1_trees
    from l0_authenticated import build_authenticated_l0
    mmr = build_all_edge_mmrs(store=index.store, road=index.road)
    l1 = build_all_l1_trees(skeleton=index.skeleton, mmr_index=mmr, road=index.road)
    l0 = build_authenticated_l0(skeleton=index.skeleton, l1_index=l1)
    return l0.root_hash == calculated_root, l0.root_hash.hex()


def run_one(city: str, insert_count: int, result_path: Path, verify_root: bool):
    from index_io import load_index

    index_file = base_index_file(city)
    if not index_file.exists():
        raise FileNotFoundError(
            f"Missing 60% base index: {index_file}\\nRun the 60% Ours scaling point first."
        )

    _, _, _, trajectory_files = load_city_spec(city)

    # EXCLUDED: base index loading.
    index = load_index(str(index_file))

    # EXCLUDED: raw I/O, selection, parsing, trajectory->EID mapping.
    delta_store, loader, prep_time = prepare_delta_store(
        city, insert_count, trajectory_files, index.road
    )
    affected_eids = affected_edges_from_store(delta_store, index.road)
    inserted_entries = sum(int(delta_store.entry_counts[eid]) for eid in affected_eids)

    # EXCLUDED harness preparation: locate stored authentication paths once.
    map_t0 = perf_counter()
    l1_meta = _build_l1_metadata(index)
    l0_meta = _build_l0_metadata(index.skeleton)
    path_map_time = perf_counter() - map_t0

    total_t0 = perf_counter()

    # 1) Apply new Entries in arrival order. No sorting.
    t0 = perf_counter()
    _append_base_store(index.store, delta_store, affected_eids)
    entry_apply_time = perf_counter() - t0

    # 2) tau-MMR ONLY-APPEND: reuse historical peaks/nodes.
    t0 = perf_counter()
    updated_edge_states = _incremental_edge_states(
        index.mmr_index, delta_store, affected_eids
    )
    mmr_update_time = perf_counter() - t0

    # 3) Rehash only affected L1 branches; unchanged sibling hashes are reused.
    t0 = perf_counter()
    new_l1_roots, edge_l0_leaf = _incremental_l1_roots(
        index, updated_edge_states, l1_meta
    )
    l1_update_time = perf_counter() - t0

    # 4) Rehash only affected L0 branches; unchanged sibling hashes are reused.
    t0 = perf_counter()
    calculated_root = _incremental_l0_root(
        index, new_l1_roots, edge_l0_leaf, delta_store, affected_eids, l0_meta
    )
    l0_update_time = perf_counter() - t0

    update_time = perf_counter() - total_t0

    reference_ok = ""
    reference_root = ""
    if verify_root:
        reference_ok, reference_root = _verify_reference_root(index, calculated_root)
        if not reference_ok:
            raise RuntimeError(
                f"Ours incremental root mismatch: got={calculated_root.hex()}, reference={reference_root}"
            )

    result = {
        "method": "ours",
        "city": city,
        "base_percent": BASE_PERCENT,
        "base_trajectory_count": base_count(city),
        "insert_trajectory_count": insert_count,
        "final_trajectory_count": final_count(city, insert_count),
        "inserted_entry_count": inserted_entries,
        "affected_edge_count": len(affected_eids),
        "update_time_s": update_time,
        "entry_apply_time_s": entry_apply_time,
        "mmr_update_time_s": mmr_update_time,
        "l1_update_time_s": l1_update_time,
        "l0_update_time_s": l0_update_time,
        "delta_preparation_time_excluded_s": prep_time,
        "path_mapping_time_excluded_s": path_map_time,
        "base_index_file": str(index_file),
        "updated_root": calculated_root.hex(),
        "reference_root_checked": bool(verify_root),
        "reference_root_match": reference_ok,
        "reference_root": reference_root,
        "selection_policy": (
            "next N raw trajectories immediately after fixed 60% prefix; "
            "each point independently reloads the same 60% base index"
        ),
        "timing_scope": (
            "Entry append + append-only tau-MMR new leaf/merge hashes + affected L1 paths + "
            "affected L0 paths; unchanged historical MMR nodes and sibling hashes reused; "
            "raw I/O/parsing/EID mapping/base load/path-location/serialization excluded"
        ),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\\n===== OURS UPDATE RESULT =====")
    print(f"city: {city}")
    print(f"base: {base_count(city)} trajectories (60%)")
    print(f"insert: {insert_count} trajectories")
    print(f"inserted entries: {inserted_entries}")
    print(f"affected edges: {len(affected_eids)}")
    print(f"Entry Apply Time: {entry_apply_time:.9f} s")
    print(f"tau-MMR Update Time: {mmr_update_time:.9f} s")
    print(f"L1 Path Update Time: {l1_update_time:.9f} s")
    print(f"L0 Path Update Time: {l0_update_time:.9f} s")
    print(f"Index Update Time: {update_time:.9f} s")
    print(f"updated root: {calculated_root.hex()}")
    if verify_root:
        print("reference root match:", reference_ok)

def collect_results():
    rows = []
    if WORKER_DIR.exists():
        for p in sorted(WORKER_DIR.glob("*.json")):
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def write_outputs(rows):
    if not rows:
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (CITIES.index(r["city"]), int(r["insert_trajectory_count"])))

    fields = [
        "method","city","base_percent","base_trajectory_count",
        "insert_trajectory_count","final_trajectory_count",
        "inserted_entry_count","affected_edge_count","update_time_s",
        "entry_apply_time_s","mmr_update_time_s","l1_update_time_s","l0_update_time_s",
        "delta_preparation_time_excluded_s","path_mapping_time_excluded_s","base_index_file","updated_root",
        "reference_root_checked","reference_root_match","reference_root",
        "selection_policy","timing_scope",
    ]
    with RESULT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    by = {(r["city"], int(r["insert_trajectory_count"])): r for r in rows}
    with MATRIX_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", *INSERT_SIZES])
        for city in CITIES:
            w.writerow([
                city,
                *[
                    by.get((city, n), {}).get("update_time_s", "")
                    for n in INSERT_SIZES
                ],
            ])


def show():
    print("Ours incremental update experiment")
    print("Every point independently starts from the SAME 60% index.\n")
    for city in CITIES:
        print(
            f"{city}: total={TOTAL_TRAJECTORIES[city]}, "
            f"base60={base_count(city)}, base_index={base_index_file(city)}"
        )
        for n in INSERT_SIZES:
            print(f"  +{n:5d} -> {final_count(city, n)}")
    print("\nOutput:")
    print(" ", RESULT_FILE)
    print(" ", MATRIX_FILE)



def run_parent(cities, inserts, overwrite: bool, verify_root: bool):
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    for city in cities:
        for n in inserts:
            result_path = WORKER_DIR / f"{city}_{n}.json"
            if result_path.exists() and not overwrite:
                print(f"SKIP existing: {city} +{n}")
                continue

            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "_worker", "--city", city, "--insert", str(n),
                "--result", str(result_path),
            ]
            if verify_root:
                cmd.append("--verify-root")
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

            write_outputs(collect_results())
    write_outputs(collect_results())
    print("\nOURS INDEX UPDATE SCALING PASS")
    print("results:", RESULT_FILE)
    print("matrix:", MATRIX_FILE)


def parse_city_arg(value: str):
    return list(CITIES) if value == "all" else [value]


def parse_insert_arg(value: str):
    if value == "all":
        return list(INSERT_SIZES)
    n = int(value)
    if n not in INSERT_SIZES:
        raise ValueError(f"insert must be one of {INSERT_SIZES}")
    return [n]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show")

    p_run = sub.add_parser("run")
    p_run.add_argument("--city", choices=(*CITIES, "all"), default="all")
    p_run.add_argument("--insert", default="all")
    p_run.add_argument("--overwrite", action="store_true")
    p_run.add_argument("--verify-root", action="store_true")

    p_worker = sub.add_parser("_worker")
    p_worker.add_argument("--city", choices=CITIES, required=True)
    p_worker.add_argument("--insert", type=int, choices=INSERT_SIZES, required=True)
    p_worker.add_argument("--result", type=Path, required=True)
    p_worker.add_argument("--verify-root", action="store_true")

    args = parser.parse_args()

    if args.cmd == "show":
        show()
    elif args.cmd == "run":
        run_parent(
            parse_city_arg(args.city),
            parse_insert_arg(args.insert),
            args.overwrite,
            args.verify_root,
        )
    else:
        run_one(args.city, args.insert, args.result, args.verify_root)


if __name__ == "__main__":
    main()
