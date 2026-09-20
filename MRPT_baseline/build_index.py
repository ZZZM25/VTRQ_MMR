from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from baseline.config import load_config, resolve_path, ensure_parent
from baseline.road_network import RoadNetwork
from baseline.dataset_builder import build_store_and_catalog
from baseline.spatial_tree import BinarySpatialTreeBuilder
from baseline.authenticated_index import BaselineIndex, build_edge_commitments, authenticate_spatial_tree
from baseline.persistence import save_index, save_catalog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cdir = Path(cfg["_config_dir"])
    node = resolve_path(cfg, cfg["node_file"])
    edge = resolve_path(cfg, cfg["edge_file"])
    index_file = resolve_path(cfg, cfg["index_file"])
    catalog_file = resolve_path(cfg, cfg["catalog_file"])
    root_file = resolve_path(cfg, cfg["trusted_root_file"])

    road = RoadNetwork()
    road.load(str(node), str(edge))
    print(f"road: {len(road.valid_node_ids())} nodes, {len(road.valid_edge_ids())} edges")

    print("===== Trajectory Preprocessing =====")
    print("JSON loading / trajectory parsing / catalog preparation are excluded from Index Construction Time.")
    print("Per-Entry ordered insertion is measured separately and INCLUDED in Index Construction Time.")
    store, catalog, tr_files, ntr, nseg, ordered_insert_time = build_store_and_catalog(
        road, cfg["trajectory_globs"], cdir
    )
    print(f"trajectories={ntr}, entries/segments={nseg}")

    print("===== Baseline Index Construction =====")
    t0 = perf_counter()

    # Entry lists are already maintained in start-time order during insertion.
    # Every Entry, result or non-result for future queries, participates in the
    # authenticated reverse hash-chain commitment.
    edge_roots, edge_hashes, suffix_states = build_edge_commitments(road, store)

    tree = BinarySpatialTreeBuilder(road, int(cfg.get("theta", 64))).build()
    root = authenticate_spatial_tree(tree, edge_hashes)
    post_insert_index_time = perf_counter() - t0
    index_time = ordered_insert_time + post_insert_index_time

    index = BaselineIndex(
        road, store, tree, edge_roots, edge_hashes, suffix_states,
        root, tree.theta, tuple(str(x) for x in tr_files),
    )
    size = save_index(index, index_file)
    cat_size = save_catalog(catalog, catalog_file)
    ensure_parent(root_file)
    root_file.write_text(root.hex() + "\n", encoding="ascii")

    print(f"Ordered Entry Insertion Time: {ordered_insert_time:.9f} s")
    print(f"Post-insertion Index Time: {post_insert_index_time:.9f} s")
    print(f"Index Construction Time: {index_time:.9f} s")
    print("  includes: per-Entry start-time position search/insertion + full Entry-list hashing + spatial index authentication")
    print(f"root_S: {root.hex()}")
    print(f"index file: {index_file} ({size/1024/1024:.3f} MiB)")
    print(f"catalog file: {catalog_file} ({cat_size/1024/1024:.3f} MiB)")
    print(f"trusted root file: {root_file}")


if __name__ == "__main__":
    main()
