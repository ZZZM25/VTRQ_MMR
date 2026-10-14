"""Arrival-order append-only MMR update experiment.

The 60% base and every delta use the original file/trajectory arrival order.
Each update appends new per-edge leaves, copies historical MMR nodes, hashes
only new leaves and carry merges, then writes E_h and SEM ancestor state.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from ours_index_scaling import CITIES, TOTAL_TRAJECTORIES, cutoff_for, load_city_spec
from ours_index_update_scaling import SliceJSONLoader, SliceBeijingLoader
from edge_mmr import build_all_edge_mmrs
from spatial_skeleton import SpatialSkeletonBuilder
from l1_merkle import build_all_l1_trees
from l0_authenticated import build_authenticated_l0
from index_io import PersistedIndex, load_index, save_index
from index_update import prepare_index_update, apply_index_update, UPDATE_ALGORITHM

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PERCENT = 60
INSERT_SIZES = (1000, 2000, 4000, 8000, 10000)
OUTPUT_ROOT = PROJECT_ROOT / "index_update_output" / "ours_append"
BASE_DIR = OUTPUT_ROOT / "indexes"
WORKER_DIR = OUTPUT_ROOT / "_worker_results"
CACHE_DIR = OUTPUT_ROOT / "_delta_cache"
RESULT_FILE = OUTPUT_ROOT / "index_update_results.csv"
MATRIX_FILE = OUTPUT_ROOT / "update_time_matrix_s.csv"


def base_count(city):
    return cutoff_for(city, BASE_PERCENT)


def final_count(city, n):
    return base_count(city) + n


def base_path(city):
    return (BASE_DIR / f"{city}_060pct_ours_append.dat").resolve()


def _make_loader(city, start, count):
    import edge_entries
    if city == "beijing":
        from beijing_csv_adapter import install_beijing_adapter, parse_beijing_trajectory
        install_beijing_adapter()
        loader = SliceBeijingLoader(start, count)
        edge_entries.load_trajectory_json = loader
        edge_entries.parse_trajectory = parse_beijing_trajectory
    else:
        from trajectory_parser import load_trajectory_json
        loader = SliceJSONLoader(load_trajectory_json, start, count)
        edge_entries.load_trajectory_json = loader
    return edge_entries, loader


def build_prefix(city):
    node, edge, theta, files = load_city_spec(city)
    from road_network import RoadNetwork
    road = RoadNetwork(); road.load(str(node), str(edge))
    edge_entries, loader = _make_loader(city, 0, base_count(city))
    t0 = perf_counter()
    store = edge_entries.build_edge_entries_from_files([str(p) for p in files], road)
    prep = perf_counter() - t0
    if loader.selected != base_count(city):
        raise RuntimeError(f"{city}: base selected {loader.selected} != {base_count(city)}")
    t0 = perf_counter()
    mmr = build_all_edge_mmrs(store, road, sort_records=False)
    skeleton = SpatialSkeletonBuilder(road, theta).build()
    blocks = build_all_l1_trees(skeleton, mmr, road)
    l0 = build_authenticated_l0(skeleton, blocks)
    construction = perf_counter() - t0
    idx = PersistedIndex(road, store, mmr, skeleton, blocks, l0, theta, tuple(loader.used_files))
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    size = save_index(idx, str(base_path(city)))
    print(f"BASE {city}: trajectories={store.total_trajectories}, entries={store.total_entries}, "
          f"build={construction:.3f}s, prep_excluded={prep:.3f}s, size={size}")


def delta_store(city, n, files, road, base_store=None, time_safe=False):
    cache_file = CACHE_DIR / f"{city}_next_{max(INSERT_SIZES)}.pkl"
    if cache_file.exists():
        payload = pickle.loads(cache_file.read_bytes())
        if int(payload["base_count"]) != base_count(city):
            raise RuntimeError(f"{cache_file}: cache base count mismatch")
        from edge_entries import EdgeEntryStore, _ENTRY_STRUCT
        store = EdgeEntryStore(road.max_eid)
        if not time_safe:
            for trajectory_id, entries in payload["trajectories"][:n]:
                for eid, start, end in entries:
                    store.append(int(eid), trajectory_id, int(start), int(end))
            store.total_trajectories = n
            store.processed_files = len(payload.get("used_files", ()))
            return store, _CachedLoader(n, payload.get("used_files", ())), 0.0
        if base_store is None:
            raise RuntimeError("arrival-order cache selection requires the 60% base store")
        # Scan only the final 40% candidate pool.  Keep an entry when its
        # start time is no earlier than the current tail of that edge; this
        # preserves direct append order for the selected edge records.
        tails = [-1] * len(base_store.edge_buffers)
        for eid, buffer in enumerate(base_store.edge_buffers):
            if buffer:
                for _, start, _ in _ENTRY_STRUCT.iter_unpack(buffer):
                    tails[eid] = int(start)
        inserted = 0
        raw_scanned = 0
        for trajectory_id, entries in payload["trajectories"]:
            raw_scanned += 1
            for eid, start, end in entries:
                eid, start, end = int(eid), int(start), int(end)
                if start < tails[eid]:
                    continue
                store.append(eid, trajectory_id, start, end)
                tails[eid] = start
                inserted += 1
                if inserted >= n:
                    break
            if inserted >= n:
                break
        if inserted != n:
            raise RuntimeError(f"{city}: only found {inserted} time-safe edge entries in cached final-40% candidates; need {n}")
        store.total_trajectories = raw_scanned
        store.processed_files = len(payload.get("used_files", ()))
        return store, _CachedLoader(n, payload.get("used_files", ()), raw_scanned), 0.0

    import edge_entries
    if city == "beijing":
        from beijing_csv_adapter import parse_beijing_trajectory
        loader = SliceBeijingLoader(base_count(city), n)
        edge_entries.load_trajectory_json = loader
        edge_entries.parse_trajectory = parse_beijing_trajectory
    else:
        from trajectory_parser import load_trajectory_json
        loader = SliceJSONLoader(load_trajectory_json, base_count(city), n)
        edge_entries.load_trajectory_json = loader
    t0 = perf_counter()
    store = edge_entries.build_edge_entries_from_files([str(p) for p in files], road)
    prep = perf_counter() - t0
    if loader.selected != n:
        raise RuntimeError(f"{city}: delta selected {loader.selected} != {n}")
    return store, loader, prep


class _CachedLoader:
    def __init__(self, selected, used_files, raw_scanned=None):
        self.selected = int(selected)
        self.raw_scanned = int(selected if raw_scanned is None else raw_scanned)
        self.used_files = list(used_files)


def build_delta_cache(city):
    """Parse the post-60% prefix once so the five timed workers reuse it."""
    cache_file = CACHE_DIR / f"{city}_next_{max(INSERT_SIZES)}.pkl"
    if cache_file.exists():
        return
    node, edge, theta, files = load_city_spec(city)
    from road_network import RoadNetwork
    from trajectory_id import calculate_trajectory_id
    import edge_entries
    road = RoadNetwork(); road.load(str(node), str(edge))
    start, count = base_count(city), max(INSERT_SIZES)
    edge_entries, loader = _make_loader(city, start, count)
    original_parse = edge_entries.parse_trajectory
    captured = []

    def capture_parse(trajectory, trajectory_index, road):
        eids, starts, ends = original_parse(trajectory, trajectory_index, road)
        if eids:
            tid = calculate_trajectory_id(eids, starts[0], ends[-1])
            captured.append((tid, tuple((int(e), int(s), int(en))
                                        for e, s, en in zip(eids, starts, ends))))
        else:
            captured.append((b"\x00" * 32, tuple()))
        return eids, starts, ends

    edge_entries.parse_trajectory = capture_parse
    edge_entries.build_edge_entries_from_files([str(p) for p in files], road)
    if loader.selected != count or len(captured) != count:
        raise RuntimeError(f"{city}: cache selected={loader.selected}, captured={len(captured)}, expected={count}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps({
        "base_count": base_count(city),
        "trajectories": captured,
        "used_files": tuple(loader.used_files),
    }, protocol=5))
    print(f"CACHE {city}: trajectories={len(captured)}, file={cache_file}")


@contextmanager
def forbid_rebuild():
    import edge_mmr, l1_merkle, l0_authenticated
    with ExitStack() as stack:
        for module, name in ((edge_mmr, "build_all_edge_mmrs"),
                             (l1_merkle, "build_all_l1_trees"),
                             (l0_authenticated, "build_authenticated_l0"),
                             (SpatialSkeletonBuilder, "build")):
            stack.enter_context(patch.object(module, name, side_effect=RuntimeError(
                f"full rebuild called inside update timing: {name}")))
        yield


def reference_root(index):
    # Arrival order is the reference order for this experiment.
    reference_store = copy.copy(index.store)
    reference_store.edge_buffers = index.store.edge_buffers.copy()
    mmr = build_all_edge_mmrs(reference_store, index.road, sort_records=False)
    blocks = build_all_l1_trees(index.skeleton, mmr, index.road)
    l0 = build_authenticated_l0(index.skeleton, blocks)
    return l0.root_hash, mmr, blocks


def run_one(city, n, result_path, verify_root=False, save_updated=False, time_safe=False):
    node, edge, theta, files = load_city_spec(city)
    idx_file = base_path(city)
    if not idx_file.exists():
        raise FileNotFoundError(f"missing append-order base; run build-bases first: {idx_file}")
    index = load_index(str(idx_file))
    if index.store.total_trajectories != base_count(city):
        raise ValueError("base index is not the requested 60% prefix")
    delta, loader, prep = delta_store(city, n, files, index.road, index.store, time_safe=time_safe)
    map_t0 = perf_counter(); plan = prepare_index_update(index, delta); map_time = perf_counter() - map_t0
    base_root = index.l0_index.root_hash.hex()
    # Full rebuild guards are active during the timed update.
    with forbid_rebuild():
        metrics = apply_index_update(index, delta, plan, source_files=loader.used_files)
    ref_root = ""
    if verify_root:
        got, ref_mmr, ref_blocks = reference_root(index)
        if got != index.l0_index.root_hash:
            raise RuntimeError(f"root mismatch: got={index.l0_index.root_hash.hex()} ref={got.hex()}")
        if ref_mmr.edge_root_hashes != index.mmr_index.edge_root_hashes:
            raise RuntimeError("MMR roots differ from arrival-order full rebuild")
        if ref_blocks.l0_l1_root_hashes != index.l1_index.l0_l1_root_hashes:
            raise RuntimeError("E_h blocks differ from arrival-order full rebuild")
        ref_root = got.hex()
    updated_path = ""
    save_time = 0.0
    if save_updated:
        updated_path = str(OUTPUT_ROOT / "updated_indexes" / f"{city}_060pct_plus{n}_ours_append.dat")
        t0 = perf_counter(); save_index(index, updated_path); save_time = perf_counter() - t0
    result = {
        "method": "ours_append", "city": city, "base_percent": BASE_PERCENT,
        "base_trajectory_count": base_count(city), "insert_trajectory_count": n,
        "final_trajectory_count": base_count(city) + delta.total_trajectories,
        "selected_raw_trajectory_count": getattr(loader, "raw_scanned", loader.selected),
        "selected_edge_entry_count": delta.total_entries,
        "delta_loaded_trajectory_count": delta.total_trajectories,
        **metrics, "delta_preparation_time_excluded_s": prep,
        "path_mapping_time_excluded_s": map_time, "serialization_time_excluded_s": save_time,
        "base_index_file": str(idx_file), "base_index_size_bytes": idx_file.stat().st_size,
        "base_root": base_root,
        "updated_root": index.l0_index.root_hash.hex(), "updated_index_file": updated_path,
        "reference_root_checked": verify_root, "reference_root_match": True if verify_root else "",
        "reference_root": ref_root,
        "selection_policy": ("scan the final 40% candidate pool in file/trajectory order; append the first N edge entries whose start time is no earlier than the current edge tail"
                             if time_safe else
                             "next N raw trajectories after fixed 60% prefix; per-edge MMR leaves retain file/trajectory arrival order"),
        "timing_scope": "append new Entries + copy historical MMR nodes + hash new leaves/carry merges + affected E_h + affected SEM paths; input, parsing, base load, validation and serialization excluded",
        "storage_policy": "updated per-edge MMR blocks are written into the node arena; old blocks remain unreachable until compaction",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{city} +{n}: update={metrics['update_time_s']:.6f}s, copied={metrics['copied_historical_mmr_nodes']}, hashed={metrics['hashed_new_mmr_nodes']}, root={result['updated_root']}")


def collect():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(WORKER_DIR.glob("*.json"))]


def write_outputs():
    rows = collect()
    if not rows: return
    rows.sort(key=lambda r: (CITIES.index(r['city']), int(r['insert_trajectory_count'])))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    import csv
    fields = list(rows[0])
    with RESULT_FILE.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    by = {(r['city'], int(r['insert_trajectory_count'])): r for r in rows}
    with MATRIX_FILE.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['city', *INSERT_SIZES])
        for city in CITIES: w.writerow([city, *[by.get((city, n), {}).get('update_time_s', '') for n in INSERT_SIZES]])


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build-bases')
    run = sub.add_parser('run'); run.add_argument('--city', choices=(*CITIES, 'all'), default='all'); run.add_argument('--insert', default='all'); run.add_argument('--overwrite', action='store_true'); run.add_argument('--verify-root', action='store_true'); run.add_argument('--save-updated-index', action='store_true'); run.add_argument('--time-safe', action='store_true')
    worker = sub.add_parser('_worker'); worker.add_argument('--city', choices=CITIES, required=True); worker.add_argument('--insert', type=int, choices=INSERT_SIZES, required=True); worker.add_argument('--result', type=Path, required=True); worker.add_argument('--verify-root', action='store_true'); worker.add_argument('--save-updated-index', action='store_true'); worker.add_argument('--time-safe', action='store_true')
    args = parser.parse_args()
    if args.cmd == 'build-bases':
        for city in CITIES: build_prefix(city)
    elif args.cmd == '_worker':
        run_one(args.city, args.insert, args.result, args.verify_root, args.save_updated_index, args.time_safe)
    else:
        if args.overwrite or not all(base_path(c).exists() for c in CITIES):
            for city in CITIES:
                if not base_path(city).exists(): build_prefix(city)
        cities = list(CITIES) if args.city == 'all' else [args.city]
        nums = list(INSERT_SIZES) if args.insert == 'all' else [int(args.insert)]
        for city in cities:
            build_delta_cache(city)
        WORKER_DIR.mkdir(parents=True, exist_ok=True)
        for city in cities:
            for n in nums:
                out = WORKER_DIR / f'{city}_{n}.json'
                if out.exists() and not args.overwrite:
                    print(f'SKIP {city} +{n}'); continue
                cmd = [sys.executable, str(Path(__file__).resolve()), '_worker', '--city', city, '--insert', str(n), '--result', str(out)]
                if args.verify_root: cmd.append('--verify-root')
                if args.save_updated_index: cmd.append('--save-updated-index')
                if args.time_safe: cmd.append('--time-safe')
                subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
                write_outputs()
        write_outputs()
        print('OURS ARRIVAL-ORDER UPDATE PASS')
        print(RESULT_FILE); print(MATRIX_FILE)


if __name__ == '__main__': main()
