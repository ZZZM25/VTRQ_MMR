
# ============================================================
# DIRECT incremental index update scalability experiment
#
# Baseline update path:
#   store.insert_by_start(...)
#   -> complete reaggregation for EACH affected edge
#   -> affected Binary Spatial Tree authentication paths only
#
# No probe. No dynamic API guessing. No full-network rebuild inside Update Time.
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
OURS_ROOT = Path(r"E:\MMR_Trajectory_range")

CITIES = ("chengdu", "xian", "beijing")
INSERT_SIZES = (1000, 2000, 4000, 8000, 10000)
BASE_PERCENT = 60

TOTAL_TRAJECTORIES = {
    "chengdu": 284_608,
    "xian": 45_851,
    "beijing": 31_390,
}

CONFIG_FILES = {
    "chengdu": PROJECT_ROOT / "configs" / "chengdu.json",
    "xian": PROJECT_ROOT / "configs" / "xian.json",
}

BEIJING_DIR = OURS_ROOT / "beijing_preprocessed"
BEIJING_NODE = BEIJING_DIR / "beijing_nodes.txt"
BEIJING_EDGE = BEIJING_DIR / "beijing_edges.txt"
BEIJING_TRAJECTORY = BEIJING_DIR / "beijing_trajectories.csv"
BEIJING_THETA = 64

BASE_INDEX_DIR = PROJECT_ROOT / "index_scaling_output" / "baseline" / "indexes"
OUTPUT_ROOT = PROJECT_ROOT / "index_update_output" / "baseline"
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
    for q in (config_file.parent / p, PROJECT_ROOT / p, p):
        if q.exists():
            return q.resolve()
    return (config_file.parent / p).resolve()


def expand_glob(pattern: str, config_file: Path) -> list[Path]:
    p = Path(pattern)
    patterns = [str(p)] if p.is_absolute() else [
        str(config_file.parent / p),
        str(PROJECT_ROOT / p),
        str(p),
    ]
    found: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for value in glob.glob(pat):
            q = Path(value).resolve()
            k = str(q).lower()
            if k not in seen:
                seen.add(k)
                found.append(q)
    return found


def load_city_spec(city: str) -> tuple[Path, Path, int, list[Path]]:
    if city == "beijing":
        node = BEIJING_NODE.resolve()
        edge = BEIJING_EDGE.resolve()
        theta = BEIJING_THETA
        files = [BEIJING_TRAJECTORY.resolve()]
    else:
        cfg_file = CONFIG_FILES[city]
        if not cfg_file.exists():
            raise FileNotFoundError(cfg_file)
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
        node = resolve_config_path(str(cfg["node_file"]), cfg_file)
        edge = resolve_config_path(str(cfg["edge_file"]), cfg_file)
        theta = int(cfg.get("theta", 64))
        values: list[Path] = []
        for pat in cfg.get("trajectory_globs", []):
            values.extend(expand_glob(str(pat), cfg_file))
        unique, seen = [], set()
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


def load_relaxed_json(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    obj, end = json.JSONDecoder().raw_decode(stripped)
    trailing = stripped[end:].strip()
    if trailing and set(trailing) != {"]"}:
        raise ValueError(f"unexpected trailing data in {path}: {trailing[:100]!r}")
    return obj


def raw_slice_json(files: list[Path], start: int, count: int):
    stop = start + count
    pos = 0
    selected = []
    used = []
    for path in files:
        if pos >= stop:
            break
        data = load_relaxed_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level must be list")
        file_start, file_stop = pos, pos + len(data)
        pos = file_stop
        left, right = max(start, file_start), min(stop, file_stop)
        if right <= left:
            continue
        selected.extend(data[left-file_start:right-file_start])
        used.append(str(path.resolve()))
    if len(selected) != count:
        raise RuntimeError(f"requested delta {count}, selected {len(selected)}")
    return selected, used


def raw_slice_beijing(path: Path, start: int, count: int):
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    stop = start + count
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "points_data" not in set(reader.fieldnames or []):
            raise ValueError(f"{path}: missing points_data")
        for raw_index, row in enumerate(reader):
            if raw_index < start:
                continue
            if raw_index >= stop:
                break
            rows.append({
                "row_no": raw_index + 2,
                "source_traj_id": (
                    row.get("source_traj_id")
                    or row.get("traj_id")
                    or ""
                ).strip(),
                "points": json.loads(row["points_data"]),
            })
    if len(rows) != count:
        raise RuntimeError(f"requested Beijing delta {count}, selected {len(rows)}")
    return rows, [str(path.resolve())]


def prepare_standard_delta(road, trajectories):
    """
    Parse to (trajectory_id, [(eid,start,end), ...]) OUTSIDE Update Time.
    No insertion into the base store happens here.
    """
    from baseline.trajectory_id import compute_trajectory_id

    prepared = []
    entry_count = 0
    affected = set()

    for raw_index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, list) or len(trajectory) < 3:
            raise ValueError(f"trajectory {raw_index}: must be list with >=3 items")
        node_pairs = trajectory[1]
        geometry_segments = trajectory[2]
        if len(node_pairs) != len(geometry_segments):
            raise ValueError(f"trajectory {raw_index}: node/geometry length mismatch")
        if not geometry_segments:
            continue

        eids = []
        intervals = []
        for i, (node_pair, points) in enumerate(zip(node_pairs, geometry_segments)):
            if not points:
                raise ValueError(f"trajectory {raw_index}, segment {i}: empty points")
            u, v = int(node_pair[0]), int(node_pair[1])
            eid = int(road.resolve_edge(u, v))
            start = int(points[0][0])
            if i + 1 < len(geometry_segments):
                nxt = geometry_segments[i + 1]
                if not nxt:
                    raise ValueError(f"trajectory {raw_index}, segment {i+1}: empty")
                end = int(nxt[0][0])
            else:
                end = int(points[-1][0])
            if end <= start:
                end = start + 1
            eids.append(eid)
            intervals.append((eid, start, end))
            affected.add(eid)

        if not eids:
            continue
        tid = compute_trajectory_id(eids, intervals[0][1], intervals[-1][2])
        prepared.append((tid, intervals))
        entry_count += len(intervals)

    return prepared, entry_count, sorted(affected)


def prepare_beijing_delta(road, rows):
    from baseline.trajectory_id import compute_trajectory_id

    prepared = []
    entry_count = 0
    affected = set()
    stationary = 0
    short = 0
    no_movement = 0

    for item in rows:
        row_no = int(item["row_no"])
        points = item["points"]
        if not isinstance(points, list) or len(points) < 2:
            short += 1
            continue

        normalized = []
        prev = None
        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) < 4:
                raise ValueError(f"row {row_no}, point {i}: invalid")
            node = int(point[0])
            ts = int(point[3])
            if prev is not None and ts < prev:
                raise ValueError(f"row {row_no}: timestamp goes backwards")
            normalized.append((node, ts))
            prev = ts

        eids, intervals = [], []
        for i in range(len(normalized)-1):
            u, start = normalized[i]
            v, end = normalized[i+1]
            if u == v:
                stationary += 1
                continue
            eid = int(road.resolve_edge(u, v))
            eids.append(eid)
            intervals.append((eid, start, end))
            affected.add(eid)

        if not eids:
            no_movement += 1
            continue

        tid = compute_trajectory_id(eids, intervals[0][1], intervals[-1][2])
        prepared.append((tid, intervals))
        entry_count += len(intervals)

    return prepared, entry_count, sorted(affected), {
        "stationary_segments": stationary,
        "short_trajectories": short,
        "no_movement_trajectories": no_movement,
    }


def base_index_file(city: str) -> Path:
    return (BASE_INDEX_DIR / f"{city}_060pct_baseline.dat").resolve()



def block_full_rebuild(authenticated_index_module):
    originals = []

    def blocked(name):
        def _blocked(*args, **kwargs):
            raise RuntimeError(f"FULL REBUILD FORBIDDEN inside Update Time: {name}")
        return _blocked

    for name in ("build_edge_commitments", "authenticate_spatial_tree"):
        if hasattr(authenticated_index_module, name):
            originals.append(
                (authenticated_index_module, name, getattr(authenticated_index_module, name))
            )
            setattr(
                authenticated_index_module,
                name,
                blocked(f"{authenticated_index_module.__name__}.{name}"),
            )
    return originals


def restore(originals):
    for module, name, fn in originals:
        setattr(module, name, fn)


def root_hex(index):
    value = index.root_hash
    return value.hex() if isinstance(value, (bytes, bytearray)) else str(value)


def verify_reference_root(index):
    from baseline.authenticated_index import build_edge_commitments, authenticate_spatial_tree
    edge_roots, edge_hashes, suffix_states = build_edge_commitments(index.road, index.store)
    ref = authenticate_spatial_tree(index.tree, edge_hashes)
    return ref == index.root_hash, ref.hex()




def _rebuild_one_edge(index, eid: int):
    """Reaggregate the COMPLETE updated ordered Entry list of one affected edge."""
    from baseline.crypto import ENTRY_LIST_TAIL, hash_entry, entry_list_prepend, hash_edge
    count = int(index.store.entry_counts[eid])
    packed = bytearray((count + 1) * 32)
    packed[count*32:(count+1)*32] = ENTRY_LIST_TAIL
    state = ENTRY_LIST_TAIL
    for i in range(count - 1, -1, -1):
        tid, start, end = index.store.get_entry(eid, i)
        state = entry_list_prepend(hash_entry(tid, start, end), state)
        p = i * 32
        packed[p:p+32] = state
    root = state if count else ENTRY_LIST_TAIL
    index.edge_entry_roots[eid] = root
    index.edge_hashes[eid] = hash_edge(eid, count, root)
    index.edge_suffix_states[eid] = bytes(packed)


def _build_spatial_metadata(tree, max_eid: int):
    n = len(tree.node_type)
    parent = [-1] * n
    depth = [0] * n
    owner = [-1] * (max_eid + 1)
    stack = [(int(tree.root_index), 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        for eid in tree.node_edges[node]:
            owner[int(eid)] = node
        left = int(tree.left_child[node]); right = int(tree.right_child[node])
        if left >= 0:
            parent[left] = node; stack.append((left, d+1))
        if right >= 0:
            parent[right] = node; stack.append((right, d+1))
    return parent, depth, owner


def _update_spatial_paths(index, affected_eids, metadata):
    """Update only owner nodes + their ancestors. Unchanged sibling hashes are reused."""
    from baseline.crypto import (
        EDGE_GROUP_INIT, edge_group_append, hash_leaf, hash_internal,
    )
    from baseline.spatial_tree import NODE_LEAF

    tree = index.tree
    parent, depth, owner = metadata
    local_dirty = {owner[eid] for eid in affected_eids}
    if -1 in local_dirty:
        raise RuntimeError("Affected edge not found in Binary Spatial Tree")

    dirty = set(local_dirty)
    for node in list(local_dirty):
        p = parent[node]
        while p >= 0:
            dirty.add(p); p = parent[p]

    changed_hash = {}
    for node in sorted(dirty, key=lambda x: depth[x], reverse=True):
        if node in local_dirty:
            group = EDGE_GROUP_INIT
            for eid in tree.node_edges[node]:
                group = edge_group_append(group, index.edge_hashes[eid])
            tree.edge_group_root[node] = group
        else:
            group = tree.edge_group_root[node]

        m = (
            tree.min_lon[node], tree.min_lat[node],
            tree.max_lon[node], tree.max_lat[node],
        )
        edges = tree.node_edges[node]
        if tree.node_type[node] == NODE_LEAF:
            h = hash_leaf(*m, len(edges), group)
        else:
            left = int(tree.left_child[node]); right = int(tree.right_child[node])
            lh = changed_hash.get(left, tree.node_hash[left])
            rh = changed_hash.get(right, tree.node_hash[right])
            h = hash_internal(*m, lh, rh, len(edges), group)
        tree.node_hash[node] = h
        changed_hash[node] = h

    root = int(tree.root_index)
    index.root_hash = changed_hash.get(root, tree.node_hash[root])
    return index.root_hash


def run_one(city: str, insert_count: int, result_path: Path, verify_root: bool):
    from baseline.persistence import load_index

    index_file = base_index_file(city)
    if not index_file.exists():
        raise FileNotFoundError(
            f"Missing 60% base index: {index_file}\\nRun the 60% Baseline scaling point first."
        )

    _, _, _, trajectory_files = load_city_spec(city)

    # EXCLUDED: base index load.
    index = load_index(index_file)

    # EXCLUDED: raw I/O + selection + parsing + trajectory->EID mapping.
    prep_t0 = perf_counter()
    start = base_count(city)
    if city == "beijing":
        rows, used = raw_slice_beijing(trajectory_files[0], start, insert_count)
        prepared, inserted_entries, affected_eids, extra = prepare_beijing_delta(index.road, rows)
    else:
        raw, used = raw_slice_json(trajectory_files, start, insert_count)
        prepared, inserted_entries, affected_eids = prepare_standard_delta(index.road, raw)
        extra = {}
    prep_time = perf_counter() - prep_t0

    # EXCLUDED harness preparation: static edge->tree-node/path map.
    map_t0 = perf_counter()
    spatial_meta = _build_spatial_metadata(index.tree, int(index.road.max_eid))
    path_map_time = perf_counter() - map_t0

    total_t0 = perf_counter()

    # 1) Start-time ordered insertion. Search + shifting are INCLUDED.
    t0 = perf_counter()
    for tid, intervals in prepared:
        for eid, start_time, end_time in intervals:
            index.store.insert_by_start(eid, tid, start_time, end_time)
    ordered_insertion_time = perf_counter() - t0

    # 2) For every affected edge, reaggregate its COMPLETE updated Entry list.
    t0 = perf_counter()
    for eid in affected_eids:
        _rebuild_one_edge(index, eid)
    edge_reaggregate_time = perf_counter() - t0

    # 3) Update only affected Binary Spatial Tree branches to root.
    t0 = perf_counter()
    new_root = _update_spatial_paths(index, affected_eids, spatial_meta)
    spatial_auth_update_time = perf_counter() - t0

    update_time = perf_counter() - total_t0

    reference_ok = ""
    reference_root = ""
    if verify_root:
        reference_ok, reference_root = verify_reference_root(index)
        if not reference_ok:
            raise RuntimeError(
                f"Baseline incremental root mismatch: got={new_root.hex()}, reference={reference_root}"
            )

    result = {
        "method": "baseline",
        "city": city,
        "base_percent": BASE_PERCENT,
        "base_trajectory_count": base_count(city),
        "insert_trajectory_count": insert_count,
        "final_trajectory_count": final_count(city, insert_count),
        "effective_insert_trajectory_count": len(prepared),
        "inserted_entry_count": inserted_entries,
        "affected_edge_count": len(affected_eids),
        "update_time_s": update_time,
        "ordered_insertion_time_s": ordered_insertion_time,
        "affected_edge_reaggregate_time_s": edge_reaggregate_time,
        "spatial_auth_update_time_s": spatial_auth_update_time,
        "delta_preparation_time_excluded_s": prep_time,
        "path_mapping_time_excluded_s": path_map_time,
        "base_index_file": str(index_file),
        "updated_root": new_root.hex(),
        "reference_root_checked": bool(verify_root),
        "reference_root_match": reference_ok,
        "reference_root": reference_root,
        "source_files_used": "|".join(used),
        "selection_policy": (
            "next N raw trajectories immediately after fixed 60% prefix; "
            "each point independently reloads the same 60% base index"
        ),
        "timing_scope": (
            "insert_by_start + complete reaggregation of each affected edge Entry commitment + "
            "affected Binary Spatial Tree paths only; unchanged edges/sibling hashes reused; "
            "raw I/O/parsing/EID mapping/base load/path-location/serialization excluded"
        ),
        **extra,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\\n===== BASELINE UPDATE RESULT =====")
    print(f"city: {city}")
    print(f"base: {base_count(city)} trajectories (60%)")
    print(f"insert: {insert_count} raw trajectories")
    print(f"effective inserted trajectories: {len(prepared)}")
    print(f"inserted entries: {inserted_entries}")
    print(f"affected edges: {len(affected_eids)}")
    print(f"Ordered Entry Insertion Time: {ordered_insertion_time:.9f} s")
    print(f"Affected-edge Reaggregation Time: {edge_reaggregate_time:.9f} s")
    print(f"Spatial Path Update Time: {spatial_auth_update_time:.9f} s")
    print(f"Index Update Time: {update_time:.9f} s")
    print(f"updated root: {new_root.hex()}")
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
        "effective_insert_trajectory_count","inserted_entry_count",
        "affected_edge_count","update_time_s","ordered_insertion_time_s",
        "affected_edge_reaggregate_time_s","spatial_auth_update_time_s",
        "delta_preparation_time_excluded_s","path_mapping_time_excluded_s","base_index_file","updated_root",
        "reference_root_checked","reference_root_match","reference_root",
        "source_files_used","selection_policy","timing_scope",
        "stationary_segments","short_trajectories","no_movement_trajectories",
    ]

    with RESULT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
    print("Baseline incremental update experiment")
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



def parse_city(value):
    return list(CITIES) if value == "all" else [value]


def parse_insert(value):
    if value == "all":
        return list(INSERT_SIZES)
    n = int(value)
    if n not in INSERT_SIZES:
        raise ValueError(f"insert must be one of {INSERT_SIZES}")
    return [n]


def run_parent(cities, inserts, overwrite, verify_root):
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
    print("\nBASELINE INDEX UPDATE SCALING PASS")
    print("results:", RESULT_FILE)
    print("matrix:", MATRIX_FILE)


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
            parse_city(args.city),
            parse_insert(args.insert),
            args.overwrite,
            args.verify_root,
        )
    else:
        run_one(args.city, args.insert, args.result, args.verify_root)


if __name__ == "__main__":
    main()
