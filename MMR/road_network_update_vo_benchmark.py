from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import math
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from road_network import RoadNetwork
from spatial_skeleton import SpatialSkeletonBuilder, NODE_LEAF

PROJECT_ROOT = Path(__file__).resolve().parent

CITY_CONFIG = {
    "chengdu": {
        "node_file": PROJECT_ROOT / "chengdu_nodes.txt",
        "edge_file": PROJECT_ROOT / "chengdu_edges.txt",
        "theta": 64,
    },
    "xian": {
        "node_file": PROJECT_ROOT / "xian_nodes.txt",
        "edge_file": PROJECT_ROOT / "xian_edges.txt",
        "theta": 64,
    },
    "beijing": {
        "node_file": PROJECT_ROOT / "beijing_preprocessed" / "beijing_nodes.txt",
        "edge_file": PROJECT_ROOT / "beijing_preprocessed" / "beijing_edges.txt",
        "theta": 64,
    },
}

DEFAULT_NEW_EDGE_COUNT = 10
DEFAULT_REPEATS = 10

OUTPUT_DIR = PROJECT_ROOT / "road_network_update_vo_output"
RESULT_FILE = OUTPUT_DIR / "road_network_update_vo_results.csv"
NEW_EDGE_FILE = OUTPUT_DIR / "selected_new_road_edges.csv"

# 5 columns x 2 rows: 10 geographically separated target regions.
TARGET_FRACTIONS = (
    (0.12, 0.28), (0.30, 0.28), (0.50, 0.28), (0.70, 0.28), (0.88, 0.28),
    (0.12, 0.72), (0.30, 0.72), (0.50, 0.72), (0.70, 0.72), (0.88, 0.72),
)

PREFERRED_MIN_EDGE_M = 80.0
PREFERRED_MAX_EDGE_M = 2000.0
RELAXED_MAX_EDGE_M = 5000.0
NEAREST_NODE_POOL = 64

DOMAIN_EDGE = b"ROAD-EDGE-v1"
DOMAIN_LEAF = b"ROAD-LEAF-v1"
DOMAIN_INTERNAL = b"ROAD-INTERNAL-v1"

_U32 = struct.Struct(">I")
_F64 = struct.Struct(">d")


@dataclass(slots=True, frozen=True)
class NewEdge:
    u: int
    v: int
    target_lon: float
    target_lat: float
    length_m: float


@dataclass(slots=True)
class SpatialAuthState:
    node_hashes: list[bytes]
    root_hash: bytes


@dataclass(slots=True)
class FullRoadVO:
    tokens: list[tuple]
    root_index: int


def hash_edge_record(eid, u, v, u_lon, u_lat, v_lon, v_lat) -> bytes:
    h = hashlib.sha256()
    h.update(DOMAIN_EDGE)
    h.update(int(eid).to_bytes(8, "big", signed=True))
    h.update(int(u).to_bytes(8, "big", signed=True))
    h.update(int(v).to_bytes(8, "big", signed=True))
    h.update(_F64.pack(float(u_lon)))
    h.update(_F64.pack(float(u_lat)))
    h.update(_F64.pack(float(v_lon)))
    h.update(_F64.pack(float(v_lat)))
    return h.digest()


def hash_leaf(min_lon, min_lat, max_lon, max_lat, origin, edge_hashes) -> bytes:
    h = hashlib.sha256()
    h.update(DOMAIN_LEAF)
    h.update(_F64.pack(float(min_lon)))
    h.update(_F64.pack(float(min_lat)))
    h.update(_F64.pack(float(max_lon)))
    h.update(_F64.pack(float(max_lat)))
    h.update(int(origin).to_bytes(4, "big", signed=True))
    h.update(_U32.pack(len(edge_hashes)))
    for value in edge_hashes:
        h.update(value)
    return h.digest()


def hash_internal(min_lon, min_lat, max_lon, max_lat, left_hash, mid_hash, right_hash) -> bytes:
    h = hashlib.sha256()
    h.update(DOMAIN_INTERNAL)
    h.update(_F64.pack(float(min_lon)))
    h.update(_F64.pack(float(min_lat)))
    h.update(_F64.pack(float(max_lon)))
    h.update(_F64.pack(float(max_lat)))
    h.update(left_hash)
    h.update(mid_hash)
    h.update(right_hash)
    return h.digest()


def clone_road(road: RoadNetwork) -> RoadNetwork:
    copied = RoadNetwork()
    copied.node_lon = list(road.node_lon)
    copied.node_lat = list(road.node_lat)
    copied.edge_u = list(road.edge_u)
    copied.edge_v = list(road.edge_v)
    copied.edge_min_lon = list(road.edge_min_lon)
    copied.edge_min_lat = list(road.edge_min_lat)
    copied.edge_max_lon = list(road.edge_max_lon)
    copied.edge_max_lat = list(road.edge_max_lat)
    copied.edge_lookup = dict(road.edge_lookup)
    copied._edge_count = int(road.edge_count)
    return copied


def insert_new_edge(road: RoadNetwork, u: int, v: int) -> int:
    if u == v:
        raise ValueError("self-loop is not allowed")

    key = road.edge_key(u, v)
    if key in road.edge_lookup:
        raise ValueError(f"edge ({u},{v}) already exists")

    eid = len(road.edge_u)
    road.edge_u.append(u)
    road.edge_v.append(v)

    u_lon = float(road.node_lon[u])
    u_lat = float(road.node_lat[u])
    v_lon = float(road.node_lon[v])
    v_lat = float(road.node_lat[v])

    road.edge_min_lon.append(min(u_lon, v_lon))
    road.edge_min_lat.append(min(u_lat, v_lat))
    road.edge_max_lon.append(max(u_lon, v_lon))
    road.edge_max_lat.append(max(u_lat, v_lat))

    road.edge_lookup[key] = eid
    road._edge_count += 1
    return eid


def road_mbr(road: RoadNetwork):
    return min(road.node_lon), min(road.node_lat), max(road.node_lon), max(road.node_lat)


def distance_m(lon1, lat1, lon2, lat2) -> float:
    mean_lat = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110_540.0
    return math.hypot(dx, dy)


def nearest_nodes(road, target_lon, target_lat, k):
    k = min(k, road.node_count)
    cos_lat = max(math.cos(math.radians(target_lat)), 1e-9)
    return heapq.nsmallest(
        k,
        range(road.node_count),
        key=lambda nid: (((road.node_lon[nid] - target_lon) * cos_lat) ** 2 +
                         (road.node_lat[nid] - target_lat) ** 2),
    )


def choose_pair_from_pool(road, pool, target_lon, target_lat, used_pairs,
                          min_length_m, max_length_m):
    best = None
    for i in range(len(pool)):
        u = pool[i]
        for j in range(i + 1, len(pool)):
            v = pool[j]
            key = road.edge_key(u, v)
            if key in road.edge_lookup or key in used_pairs:
                continue

            u_lon, u_lat = float(road.node_lon[u]), float(road.node_lat[u])
            v_lon, v_lat = float(road.node_lon[v]), float(road.node_lat[v])
            length = distance_m(u_lon, u_lat, v_lon, v_lat)
            if length < min_length_m or length > max_length_m:
                continue

            mid_lon = (u_lon + v_lon) * 0.5
            mid_lat = (u_lat + v_lat) * 0.5
            target_distance = distance_m(mid_lon, mid_lat, target_lon, target_lat)
            score = (target_distance, length, min(u, v), max(u, v))
            item = NewEdge(u, v, target_lon, target_lat, length)
            if best is None or score < best[0]:
                best = (score, item)
    return None if best is None else best[1]


def select_distributed_new_edges(road: RoadNetwork, count: int) -> list[NewEdge]:
    min_lon, min_lat, max_lon, max_lat = road_mbr(road)
    fractions = list(TARGET_FRACTIONS)

    if count > len(fractions):
        for i in range(count - len(fractions)):
            fx = 0.15 + 0.70 * ((i * 0.61803398875) % 1.0)
            fy = 0.15 + 0.70 * ((i * 0.41421356237) % 1.0)
            fractions.append((fx, fy))

    selected = []
    used_pairs = set()

    for fx, fy in fractions[:count]:
        target_lon = min_lon + (max_lon - min_lon) * fx
        target_lat = min_lat + (max_lat - min_lat) * fy
        pool = nearest_nodes(road, target_lon, target_lat, NEAREST_NODE_POOL)

        candidate = choose_pair_from_pool(
            road, pool, target_lon, target_lat, used_pairs,
            PREFERRED_MIN_EDGE_M, PREFERRED_MAX_EDGE_M,
        )
        if candidate is None:
            candidate = choose_pair_from_pool(
                road, pool, target_lon, target_lat, used_pairs,
                0.0, RELAXED_MAX_EDGE_M,
            )
        if candidate is None:
            pool = nearest_nodes(road, target_lon, target_lat, min(256, road.node_count))
            candidate = choose_pair_from_pool(
                road, pool, target_lon, target_lat, used_pairs,
                0.0, float("inf"),
            )
        if candidate is None:
            raise RuntimeError(f"cannot generate new road near {target_lon},{target_lat}")

        used_pairs.add(road.edge_key(candidate.u, candidate.v))
        selected.append(candidate)

    return selected


def build_spatial_auth(road: RoadNetwork, skeleton) -> SpatialAuthState:
    node_hashes = [b""] * skeleton.node_count

    for node_index in range(skeleton.node_count):
        if skeleton.node_type[node_index] == NODE_LEAF:
            offset = int(skeleton.leaf_edge_offset[node_index])
            count = int(skeleton.leaf_edge_count[node_index])
            edge_hashes = []

            for pos in range(offset, offset + count):
                eid = int(skeleton.leaf_edge_ids[pos])
                u = int(road.edge_u[eid])
                v = int(road.edge_v[eid])
                edge_hashes.append(hash_edge_record(
                    eid, u, v,
                    road.node_lon[u], road.node_lat[u],
                    road.node_lon[v], road.node_lat[v],
                ))

            node_hashes[node_index] = hash_leaf(
                skeleton.min_lon[node_index], skeleton.min_lat[node_index],
                skeleton.max_lon[node_index], skeleton.max_lat[node_index],
                int(skeleton.leaf_origin[node_index]), edge_hashes,
            )
        else:
            left = int(skeleton.left_child[node_index])
            mid = int(skeleton.mid_child[node_index])
            right = int(skeleton.right_child[node_index])
            node_hashes[node_index] = hash_internal(
                skeleton.min_lon[node_index], skeleton.min_lat[node_index],
                skeleton.max_lon[node_index], skeleton.max_lat[node_index],
                node_hashes[left], node_hashes[mid], node_hashes[right],
            )

    root_index = int(skeleton.root_index)
    return SpatialAuthState(node_hashes, node_hashes[root_index])


def build_full_road_vo(road: RoadNetwork, skeleton) -> FullRoadVO:
    tokens = []

    for node_index in range(skeleton.node_count):
        if skeleton.node_type[node_index] == NODE_LEAF:
            offset = int(skeleton.leaf_edge_offset[node_index])
            count = int(skeleton.leaf_edge_count[node_index])
            records = []

            for pos in range(offset, offset + count):
                eid = int(skeleton.leaf_edge_ids[pos])
                u = int(road.edge_u[eid])
                v = int(road.edge_v[eid])
                records.append((
                    eid, u, v,
                    float(road.node_lon[u]), float(road.node_lat[u]),
                    float(road.node_lon[v]), float(road.node_lat[v]),
                ))

            tokens.append((
                "L",
                float(skeleton.min_lon[node_index]),
                float(skeleton.min_lat[node_index]),
                float(skeleton.max_lon[node_index]),
                float(skeleton.max_lat[node_index]),
                int(skeleton.leaf_origin[node_index]),
                tuple(records),
            ))
        else:
            tokens.append((
                "I",
                float(skeleton.min_lon[node_index]),
                float(skeleton.min_lat[node_index]),
                float(skeleton.max_lon[node_index]),
                float(skeleton.max_lat[node_index]),
                int(skeleton.left_child[node_index]),
                int(skeleton.mid_child[node_index]),
                int(skeleton.right_child[node_index]),
            ))

    return FullRoadVO(tokens=tokens, root_index=int(skeleton.root_index))


def verify_full_road_vo(vo: FullRoadVO, trusted_root: bytes, expected_edge_count: int):
    node_hashes = []
    seen_eids = set()

    for node_index, token in enumerate(vo.tokens):
        if token[0] == "L":
            _, min_lon, min_lat, max_lon, max_lat, origin, records = token
            edge_hashes = []

            for record in records:
                eid, u, v, u_lon, u_lat, v_lon, v_lat = record
                if eid in seen_eids:
                    raise RuntimeError(f"duplicate eid in VO: {eid}")
                seen_eids.add(eid)
                edge_hashes.append(hash_edge_record(
                    eid, u, v, u_lon, u_lat, v_lon, v_lat
                ))

            node_hashes.append(hash_leaf(
                min_lon, min_lat, max_lon, max_lat, origin, edge_hashes
            ))

        elif token[0] == "I":
            _, min_lon, min_lat, max_lon, max_lat, left, mid, right = token
            if min(left, mid, right) < 0 or max(left, mid, right) >= node_index:
                raise RuntimeError(f"invalid child reference at node {node_index}")
            node_hashes.append(hash_internal(
                min_lon, min_lat, max_lon, max_lat,
                node_hashes[left], node_hashes[mid], node_hashes[right],
            ))
        else:
            raise RuntimeError(f"unknown token type: {token[0]}")

    reconstructed_root = node_hashes[vo.root_index]
    ok = reconstructed_root == trusted_root and len(seen_eids) == expected_edge_count
    return ok, len(seen_eids)


def estimate_vo_payload_bytes(vo: FullRoadVO) -> int:
    total = 0
    for token in vo.tokens:
        if token[0] == "L":
            records = token[6]
            total += 1 + 4 * 8 + 4 + 4
            total += len(records) * (3 * 4 + 4 * 8)
        else:
            total += 1 + 4 * 8 + 3 * 4
    return total


def run_city(city: str, new_edge_count: int, repeats: int):
    cfg = CITY_CONFIG[city]
    node_file = Path(cfg["node_file"])
    edge_file = Path(cfg["edge_file"])
    theta = int(cfg["theta"])

    if not node_file.exists():
        raise FileNotFoundError(node_file)
    if not edge_file.exists():
        raise FileNotFoundError(edge_file)

    print("\n" + "=" * 78)
    print("City:", city)
    print("=" * 78)

    # File loading is excluded from timing.
    base_road = RoadNetwork()
    base_road.load(node_file=str(node_file), edge_file=str(edge_file))
    print("Base node count:", base_road.node_count)
    print("Base edge count:", base_road.edge_count)

    # Synthetic road selection is excluded from timing.
    new_edges = select_distributed_new_edges(base_road, new_edge_count)
    print("\nSelected synthetic new roads:")
    for i, item in enumerate(new_edges, 1):
        print(
            f"{i:02d} u={item.u} v={item.v} length={item.length_m:.2f}m "
            f"u=({base_road.node_lon[item.u]:.6f},{base_road.node_lat[item.u]:.6f}) "
            f"v=({base_road.node_lon[item.v]:.6f},{base_road.node_lat[item.v]:.6f})"
        )

    # Base state outside update timing.
    base_skeleton = SpatialSkeletonBuilder(road=base_road, theta=theta).build()
    base_auth = build_spatial_auth(base_road, base_skeleton)

    insert_times = []
    skeleton_times = []
    auth_times = []
    update_times = []

    updated_road = None
    updated_skeleton = None
    updated_auth = None

    for repeat_id in range(repeats):
        road = clone_road(base_road)  # excluded

        total_t0 = perf_counter()

        t0 = perf_counter()
        for item in new_edges:
            insert_new_edge(road, item.u, item.v)
        insert_time = perf_counter() - t0

        t0 = perf_counter()
        skeleton = SpatialSkeletonBuilder(road=road, theta=theta).build()
        skeleton_time = perf_counter() - t0

        t0 = perf_counter()
        auth = build_spatial_auth(road, skeleton)
        auth_time = perf_counter() - t0

        update_time = perf_counter() - total_t0

        insert_times.append(insert_time)
        skeleton_times.append(skeleton_time)
        auth_times.append(auth_time)
        update_times.append(update_time)

        if repeat_id == 0:
            updated_road = road
            updated_skeleton = skeleton
            updated_auth = auth

    expected_updated_edges = base_road.edge_count + new_edge_count
    if updated_road.edge_count != expected_updated_edges:
        raise RuntimeError("updated edge count mismatch")
    if updated_auth.root_hash == base_auth.root_hash:
        raise RuntimeError("root did not change after road update")

    # Full-road VO generation and verification on UPDATED road network.
    vo_gen_times = []
    vo_verify_times = []
    full_vo = None
    verified_edge_count = 0

    for _ in range(repeats):
        t0 = perf_counter()
        vo = build_full_road_vo(updated_road, updated_skeleton)
        vo_gen_times.append(perf_counter() - t0)

        t0 = perf_counter()
        ok, verified_count = verify_full_road_vo(
            vo,
            updated_auth.root_hash,
            updated_road.edge_count,
        )
        vo_verify_times.append(perf_counter() - t0)

        if not ok:
            raise RuntimeError(f"{city}: full-road VO verification failed")

        full_vo = vo
        verified_edge_count = verified_count

    vo_payload_bytes = estimate_vo_payload_bytes(full_vo)

    def mean(x):
        return statistics.mean(x)

    def std(x):
        return statistics.stdev(x) if len(x) > 1 else 0.0

    print("\n===== Result =====")
    print(f"Road Update Time: {mean(update_times):.9f} s")
    print(f"  edge insert only: {mean(insert_times):.9f} s")
    print(f"  spatial skeleton rebuild: {mean(skeleton_times):.9f} s")
    print(f"  spatial auth rebuild: {mean(auth_times):.9f} s")
    print(f"Full Road VO Generation Time: {mean(vo_gen_times):.9f} s")
    print(f"Full Road VO Verification Time: {mean(vo_verify_times):.9f} s")
    print("Verified road edges:", verified_edge_count)
    print(f"Full Road VO logical payload: {vo_payload_bytes / (1024*1024):.6f} MiB")

    row = {
        "city": city,
        "theta": theta,
        "base_node_count": base_road.node_count,
        "base_edge_count": base_road.edge_count,
        "inserted_road_edge_count": new_edge_count,
        "updated_edge_count": updated_road.edge_count,
        "road_update_time_s": mean(update_times),
        "road_update_time_std_s": std(update_times),
        "edge_insert_only_time_s": mean(insert_times),
        "edge_insert_only_time_std_s": std(insert_times),
        "spatial_skeleton_rebuild_time_s": mean(skeleton_times),
        "spatial_skeleton_rebuild_time_std_s": std(skeleton_times),
        "spatial_auth_rebuild_time_s": mean(auth_times),
        "spatial_auth_rebuild_time_std_s": std(auth_times),
        "full_road_vo_generation_time_s": mean(vo_gen_times),
        "full_road_vo_generation_time_std_s": std(vo_gen_times),
        "full_road_vo_verification_time_s": mean(vo_verify_times),
        "full_road_vo_verification_time_std_s": std(vo_verify_times),
        "full_road_vo_verified_edge_count": verified_edge_count,
        "full_road_vo_payload_bytes": vo_payload_bytes,
        "base_spatial_root": base_auth.root_hash.hex(),
        "updated_spatial_root": updated_auth.root_hash.hex(),
    }

    return row, new_edges, base_road


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("all", "chengdu", "xian", "beijing"), default="all")
    parser.add_argument("--new-edges", type=int, default=DEFAULT_NEW_EDGE_COUNT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    if args.new_edges <= 0 or args.repeats <= 0:
        raise ValueError("--new-edges and --repeats must be > 0")

    cities = ["chengdu", "xian", "beijing"] if args.city == "all" else [args.city]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    selected_rows = []

    for city in cities:
        row, edges, road = run_city(city, args.new_edges, args.repeats)
        rows.append(row)

        for i, item in enumerate(edges, 1):
            selected_rows.append({
                "city": city,
                "new_edge_no": i,
                "u": item.u,
                "v": item.v,
                "u_lon": f"{road.node_lon[item.u]:.10f}",
                "u_lat": f"{road.node_lat[item.u]:.10f}",
                "v_lon": f"{road.node_lon[item.v]:.10f}",
                "v_lat": f"{road.node_lat[item.v]:.10f}",
                "length_m": f"{item.length_m:.3f}",
                "target_lon": f"{item.target_lon:.10f}",
                "target_lat": f"{item.target_lat:.10f}",
            })

    with RESULT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with NEW_EDGE_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "city", "new_edge_no", "u", "v",
            "u_lon", "u_lat", "v_lon", "v_lat",
            "length_m", "target_lon", "target_lat",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    print("\n" + "=" * 78)
    print("ALL DONE")
    print("Result CSV:", RESULT_FILE)
    print("Selected roads CSV:", NEW_EDGE_FILE)


if __name__ == "__main__":
    main()
