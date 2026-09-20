from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from itertools import islice
from pathlib import Path
from time import perf_counter

# ============================================================
# Baseline index scaling experiment
#
# Put this file in:
#   E:\VTRQ_baseline\baseline_index_scaling_v2.py
#
# No subset files are generated.
# No total-count pass is performed.
#
# Data amount:
#   first N trajectories in deterministic original order
#   20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100%
#
# Baseline Index Construction Time INCLUDES:
#   1) immediate start-time ordered Entry insertion
#   2) edge authenticated commitments
#   3) Binary Spatial Tree build/authentication
#
# EXCLUDES:
#   road loading
#   trajectory file reading
#   cutoff selection/counting
#   raw JSON/CSV parsing
#   EID mapping / trajectory preparation
#   catalog construction (not needed for this experiment)
#   index serialization
#
# Index Size:
#   ONLY the authenticated Baseline index file.
#   No trajectory catalog is counted.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OURS_ROOT = Path(r"E:\MMR_Trajectory_range")

OUTPUT_ROOT = PROJECT_ROOT / "index_scaling_output" / "baseline"
INDEX_DIR = OUTPUT_ROOT / "indexes"
WORKER_DIR = OUTPUT_ROOT / "_worker_results"

RESULT_FILE = OUTPUT_ROOT / "index_scaling_results.csv"
TIME_MATRIX_FILE = OUTPUT_ROOT / "construction_time_matrix_s.csv"
SIZE_MATRIX_FILE = OUTPUT_ROOT / "index_size_matrix_mib.csv"

CITIES = ("chengdu", "xian", "beijing")
PERCENTS = (20, 40, 60, 80, 100)

# Fixed once. The scaling run never scans the whole dataset to recount.
TOTAL_TRAJECTORIES = {
    "chengdu": 284608,
    "xian": 45851,
    "beijing": 31390,
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


# ============================================================
# Dataset/order helpers
# ============================================================

def cutoff_for(city: str, percent: int) -> int:
    total = TOTAL_TRAJECTORIES[city]
    return total if percent == 100 else total * percent // 100


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolve_config_path(
    value: str,
    config_file: Path,
) -> Path:
    p = Path(value)

    if p.is_absolute():
        return p.resolve()

    candidates = (
        config_file.parent / p,
        PROJECT_ROOT / p,
        p,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (config_file.parent / p).resolve()


def expand_glob(
    pattern: str,
    config_file: Path,
) -> list[Path]:
    p = Path(pattern)

    if p.is_absolute():
        patterns = [str(p)]
    else:
        patterns = [
            str(config_file.parent / p),
            str(PROJECT_ROOT / p),
            str(p),
        ]

    found: list[Path] = []
    seen: set[str] = set()

    for candidate_pattern in patterns:
        for value in glob.glob(candidate_pattern):
            q = Path(value).resolve()
            key = str(q).lower()

            if key not in seen:
                seen.add(key)
                found.append(q)

    return found


def load_city_spec(
    city: str,
) -> tuple[Path, Path, int, list[Path]]:
    if city == "beijing":
        node = BEIJING_NODE.resolve()
        edge = BEIJING_EDGE.resolve()
        theta = BEIJING_THETA
        files = [BEIJING_TRAJECTORY.resolve()]
    else:
        config_file = CONFIG_FILES[city]

        if not config_file.exists():
            raise FileNotFoundError(config_file)

        cfg = json.loads(
            config_file.read_text(
                encoding="utf-8-sig"
            )
        )

        node = resolve_config_path(
            str(cfg["node_file"]),
            config_file,
        )
        edge = resolve_config_path(
            str(cfg["edge_file"]),
            config_file,
        )
        theta = int(
            cfg.get("theta", 64)
        )

        files: list[Path] = []

        for pattern in cfg.get(
            "trajectory_globs",
            [],
        ):
            files.extend(
                expand_glob(
                    str(pattern),
                    config_file,
                )
            )

        unique: list[Path] = []
        seen: set[str] = set()

        for p in files:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        files = sorted(
            unique,
            key=natural_key,
        )

    if not node.exists():
        raise FileNotFoundError(node)

    if not edge.exists():
        raise FileNotFoundError(edge)

    if not files:
        raise FileNotFoundError(
            f"{city}: no trajectory files"
        )

    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)

    return node, edge, theta, files


# ============================================================
# Input loading helpers
# ============================================================

def load_relaxed_json(path: Path):
    """
    Same tolerant behavior used by the trajectory project:
    read the first valid top-level JSON value and tolerate
    trailing unmatched ']'.
    """
    text = path.read_text(
        encoding="utf-8-sig"
    )
    stripped = text.lstrip()

    obj, end = (
        json.JSONDecoder()
        .raw_decode(stripped)
    )

    trailing = stripped[end:].strip()

    if (
        trailing
        and set(trailing) != {"]"}
    ):
        raise ValueError(
            f"unexpected trailing data in {path}: "
            f"{trailing[:100]!r}"
        )

    return obj


def raise_csv_field_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_first_n_json_trajectories(
    trajectory_files: list[Path],
    cutoff: int,
) -> tuple[list, list[str]]:
    """
    Selection/counting is outside the timed Baseline index build.

    No physical subset file is created.

    For each source file:
      remaining = cutoff - selected
      selected.extend(data[:remaining])

    There is no per-trajectory cutoff comparison.
    """
    selected: list = []
    used_files: list[str] = []

    for path in trajectory_files:
        if len(selected) >= cutoff:
            break

        data = load_relaxed_json(path)

        if not isinstance(data, list):
            raise ValueError(
                f"{path}: top-level trajectory data "
                "must be list"
            )

        remaining = cutoff - len(selected)
        take = min(
            len(data),
            remaining,
        )

        if take > 0:
            selected.extend(
                data[:take]
            )
            used_files.append(
                str(path.resolve())
            )

    if len(selected) != cutoff:
        raise RuntimeError(
            f"requested {cutoff} trajectories "
            f"but selected {len(selected)}"
        )

    return selected, used_files


def load_first_n_beijing_rows(
    trajectory_file: Path,
    cutoff: int,
) -> tuple[list[dict], list[str]]:
    """
    Read exactly the first N RAW CSV rows.

    Rows with <2 points still belong to the selected input subset.
    They are represented here and later skipped if they produce
    no road Entry.

    No physical subset CSV is created.
    """
    raise_csv_field_limit()

    selected: list[dict] = []

    with trajectory_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])

        if "points_data" not in fields:
            raise ValueError(
                f"{trajectory_file}: "
                "missing points_data"
            )

        if (
            "source_traj_id" not in fields
            and "traj_id" not in fields
        ):
            raise ValueError(
                f"{trajectory_file}: "
                "missing source_traj_id/traj_id"
            )

        # Exactly N rows; no repeated cutoff check
        # beyond the enumerator bound.
        for row_index, row in enumerate(reader):
            if row_index >= cutoff:
                break

            try:
                points = json.loads(
                    row["points_data"]
                )
            except Exception as exc:
                raise ValueError(
                    f"{trajectory_file}: "
                    f"row {row_index + 2} "
                    "points_data JSON parse failed"
                ) from exc

            selected.append(
                {
                    "row_no": row_index + 2,
                    "source_traj_id": (
                        row.get("source_traj_id")
                        or row.get("traj_id")
                        or ""
                    ).strip(),
                    "points": points,
                }
            )

    if len(selected) != cutoff:
        raise RuntimeError(
            f"requested {cutoff} Beijing rows "
            f"but selected {len(selected)}"
        )

    return (
        selected,
        [str(trajectory_file.resolve())],
    )




# ============================================================
# Memory-safe selected trajectory iterators
# ============================================================

def iter_first_n_json_trajectories(
    trajectory_files: list[Path],
    cutoff: int,
    used_files: list[str],
    selection_stats: dict,
):
    """
    Yield the first N raw trajectories without keeping all N in memory.

    Cutoff logic is evaluated once per source file, not once per trajectory.
    Within a selected chunk we simply yield its trajectories.
    """
    for path in trajectory_files:
        if selection_stats["selected"] >= cutoff:
            break

        data = load_relaxed_json(path)
        if not isinstance(data, list):
            raise ValueError(
                f"{path}: top-level trajectory data must be list"
            )

        remaining = cutoff - selection_stats["selected"]
        take = min(len(data), remaining)
        if take <= 0:
            continue

        used_files.append(str(path.resolve()))
        selection_stats["selected"] += take

        chunk = data if take == len(data) else data[:take]
        for trajectory in chunk:
            yield trajectory


def iter_first_n_beijing_rows(
    trajectory_file: Path,
    cutoff: int,
    used_files: list[str],
    selection_stats: dict,
):
    """
    Yield exactly the first N raw Beijing CSV rows with itertools.islice.
    No physical subset file and no per-row cutoff comparison.
    """
    raise_csv_field_limit()
    used_files.append(str(trajectory_file.resolve()))

    with trajectory_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])

        if "points_data" not in fields:
            raise ValueError(f"{trajectory_file}: missing points_data")
        if "source_traj_id" not in fields and "traj_id" not in fields:
            raise ValueError(
                f"{trajectory_file}: missing source_traj_id/traj_id"
            )

        for row_index, row in enumerate(islice(reader, cutoff)):
            try:
                points = json.loads(row["points_data"])
            except Exception as exc:
                raise ValueError(
                    f"{trajectory_file}: row {row_index + 2} "
                    "points_data JSON parse failed"
                ) from exc

            selection_stats["selected"] += 1
            yield {
                "row_no": row_index + 2,
                "source_traj_id": (
                    row.get("source_traj_id")
                    or row.get("traj_id")
                    or ""
                ).strip(),
                "points": points,
            }


# ============================================================
# Baseline Entry preparation
#
# IMPORTANT:
#   Parsing/EID mapping are OUTSIDE the timed insertion.
#   Only store.insert_by_start(...) is timed.
# ============================================================

def build_store_from_standard_json(
    road,
    trajectories,
    selected_count: int,
):
    from baseline.entry_store import (
        PackedEdgeEntryStore,
    )
    from baseline.trajectory_id import (
        compute_trajectory_id,
    )

    store = PackedEdgeEntryStore(
        road.max_eid
    )

    seen: set[bytes] = set()

    effective_count = 0
    entry_count = 0
    ordered_insert_time = 0.0

    for raw_index, trajectory in enumerate(
        trajectories
    ):
        if (
            not isinstance(trajectory, list)
            or len(trajectory) < 3
        ):
            raise ValueError(
                f"trajectory {raw_index}: "
                "must be list with >=3 entries"
            )

        node_pairs = trajectory[1]
        geometry_segments = trajectory[2]

        if (
            len(node_pairs)
            != len(geometry_segments)
        ):
            raise ValueError(
                f"trajectory {raw_index}: "
                f"{len(node_pairs)} node pairs "
                f"!= {len(geometry_segments)} "
                "geometry segments"
            )

        if not geometry_segments:
            continue

        eids: list[int] = []
        intervals: list[
            tuple[int, int, int]
        ] = []

        # All parsing and EID mapping are outside
        # ordered insertion timing.
        for segment_index, (
            node_pair,
            points,
        ) in enumerate(
            zip(
                node_pairs,
                geometry_segments,
            )
        ):
            if (
                not isinstance(
                    node_pair,
                    (list, tuple),
                )
                or len(node_pair) != 2
            ):
                raise ValueError(
                    f"trajectory {raw_index}, "
                    f"segment {segment_index}: "
                    "invalid node pair"
                )

            if not points:
                raise ValueError(
                    f"trajectory {raw_index}, "
                    f"segment {segment_index}: "
                    "empty geometry segment"
                )

            u = int(node_pair[0])
            v = int(node_pair[1])

            eid = int(
                road.resolve_edge(u, v)
            )

            start = int(
                points[0][0]
            )

            if (
                segment_index + 1
                < len(geometry_segments)
            ):
                next_points = (
                    geometry_segments[
                        segment_index + 1
                    ]
                )

                if not next_points:
                    raise ValueError(
                        f"trajectory {raw_index}, "
                        f"segment {segment_index + 1}: "
                        "empty geometry segment"
                    )

                end = int(
                    next_points[0][0]
                )
            else:
                end = int(
                    points[-1][0]
                )

            # Match the current trajectory parser behavior.
            if end <= start:
                end = start + 1

            eids.append(eid)
            intervals.append(
                (eid, start, end)
            )

        if not eids:
            continue

        trajectory_start = (
            intervals[0][1]
        )
        trajectory_end = (
            intervals[-1][2]
        )

        tid = compute_trajectory_id(
            eids,
            trajectory_start,
            trajectory_end,
        )

        if tid in seen:
            raise ValueError(
                f"duplicate trajectory_id at "
                f"raw trajectory {raw_index}: "
                f"{tid.hex()}"
            )

        seen.add(tid)

        # ----------------------------------------------------
        # INCLUDED:
        # immediate ordered insertion
        # ----------------------------------------------------
        t0 = perf_counter()

        for eid, start, end in intervals:
            store.insert_by_start(
                eid,
                tid,
                start,
                end,
            )

        ordered_insert_time += (
            perf_counter() - t0
        )

        effective_count += 1
        entry_count += len(intervals)

    store.validate_start_order()

    return (
        store,
        selected_count,
        effective_count,
        entry_count,
        ordered_insert_time,
    )


def build_store_from_beijing(
    road,
    rows,
    selected_count: int,
):
    from baseline.entry_store import (
        PackedEdgeEntryStore,
    )
    from baseline.trajectory_id import (
        compute_trajectory_id,
    )

    store = PackedEdgeEntryStore(
        road.max_eid
    )

    seen: set[bytes] = set()

    effective_count = 0
    entry_count = 0
    stationary_count = 0
    short_count = 0
    no_movement_count = 0
    ordered_insert_time = 0.0

    for item in rows:
        row_no = int(
            item["row_no"]
        )
        points = item["points"]

        if (
            not isinstance(points, list)
            or len(points) < 2
        ):
            short_count += 1
            continue

        normalized: list[
            tuple[int, int]
        ] = []

        previous_time = None

        # Parsing outside insertion timing.
        for point_index, point in enumerate(
            points
        ):
            if (
                not isinstance(
                    point,
                    (list, tuple),
                )
                or len(point) < 4
            ):
                raise ValueError(
                    f"row {row_no}, "
                    f"point {point_index}: "
                    "expected "
                    "[node_id,lon,lat,timestamp]"
                )

            node_id = int(point[0])
            timestamp = int(point[3])

            if (
                previous_time is not None
                and timestamp < previous_time
            ):
                raise ValueError(
                    f"row {row_no}: "
                    "timestamp goes backwards: "
                    f"{previous_time} -> "
                    f"{timestamp}"
                )

            normalized.append(
                (node_id, timestamp)
            )
            previous_time = timestamp

        eids: list[int] = []
        intervals: list[
            tuple[int, int, int]
        ] = []

        for i in range(
            len(normalized) - 1
        ):
            u, start = normalized[i]
            v, end = normalized[i + 1]

            if u == v:
                stationary_count += 1
                continue

            if end < start:
                raise ValueError(
                    f"row {row_no}, segment {i}: "
                    "end < start"
                )

            eid = int(
                road.resolve_edge(u, v)
            )

            eids.append(eid)
            intervals.append(
                (
                    eid,
                    int(start),
                    int(end),
                )
            )

        if not eids:
            no_movement_count += 1
            continue

        trajectory_start = (
            intervals[0][1]
        )
        trajectory_end = (
            intervals[-1][2]
        )

        tid = compute_trajectory_id(
            eids,
            trajectory_start,
            trajectory_end,
        )

        if tid in seen:
            raise ValueError(
                f"row {row_no}: "
                "duplicate trajectory_id "
                f"{tid.hex()}"
            )

        seen.add(tid)

        # ----------------------------------------------------
        # INCLUDED:
        # immediate start-time ordered insertion
        # ----------------------------------------------------
        t0 = perf_counter()

        for eid, start, end in intervals:
            store.insert_by_start(
                eid,
                tid,
                start,
                end,
            )

        ordered_insert_time += (
            perf_counter() - t0
        )

        effective_count += 1
        entry_count += len(intervals)

    store.validate_start_order()

    return (
        store,
        selected_count,
        effective_count,
        entry_count,
        ordered_insert_time,
        {
            "stationary_segments": (
                stationary_count
            ),
            "short_trajectories": (
                short_count
            ),
            "no_movement_trajectories": (
                no_movement_count
            ),
        },
    )


# ============================================================
# One worker = one city x one percentage
# ============================================================

def run_worker(
    city: str,
    percent: int,
    result_path: Path,
) -> None:
    from baseline.authenticated_index import (
        BaselineIndex,
        authenticate_spatial_tree,
        build_edge_commitments,
    )
    from baseline.persistence import (
        save_index,
    )
    from baseline.road_network import (
        RoadNetwork,
    )
    from baseline.spatial_tree import (
        BinarySpatialTreeBuilder,
    )

    cutoff = cutoff_for(
        city,
        percent,
    )

    (
        node_file,
        edge_file,
        theta,
        trajectory_files,
    ) = load_city_spec(city)

    print()
    print("=" * 76)
    print(
        f"Baseline | {city} | {percent}% | "
        f"target trajectories={cutoff}"
    )
    print("=" * 76)

    print("Source file order:")
    for i, p in enumerate(
        trajectory_files,
        1,
    ):
        print(
            f"  [{i}] {p.name}"
        )

    # --------------------------------------------------------
    # EXCLUDED: road loading
    # --------------------------------------------------------
    road = RoadNetwork()
    road.load(
        str(node_file),
        str(edge_file),
    )

    # --------------------------------------------------------
    # EXCLUDED:
    # source reading + cutoff selection
    # --------------------------------------------------------
    preparation_t0 = perf_counter()
    used_files: list[str] = []
    selection_stats = {"selected": 0}

    if city == "beijing":
        selected_iter = iter_first_n_beijing_rows(
            trajectory_files[0],
            cutoff,
            used_files,
            selection_stats,
        )

        (
            store,
            selected_count,
            effective_count,
            entry_count,
            ordered_insert_time,
            extra_stats,
        ) = build_store_from_beijing(
            road,
            selected_iter,
            selected_count=cutoff,
        )
    else:
        selected_iter = iter_first_n_json_trajectories(
            trajectory_files,
            cutoff,
            used_files,
            selection_stats,
        )

        (
            store,
            selected_count,
            effective_count,
            entry_count,
            ordered_insert_time,
        ) = build_store_from_standard_json(
            road,
            selected_iter,
            selected_count=cutoff,
        )

        extra_stats = {}

    preparation_wall_time = (
        perf_counter()
        - preparation_t0
    )

    if selection_stats["selected"] != cutoff:
        raise RuntimeError(
            f"{city} {percent}%: "
            f"selected {selection_stats['selected']}, "
            f"expected {cutoff}. Check TOTAL_TRAJECTORIES/input files."
        )

    print(
        f"Selected trajectories: "
        f"{selected_count}/{cutoff}"
    )
    print(
        f"Effective trajectories: "
        f"{effective_count}"
    )
    print(
        f"Entries: {entry_count}"
    )
    print(
        f"Preparation wall time "
        f"(mostly EXCLUDED except separately "
        f"measured ordered insertion): "
        f"{preparation_wall_time:.6f} s"
    )

    # --------------------------------------------------------
    # INCLUDED:
    # post-insertion authenticated Baseline index
    # --------------------------------------------------------
    t0 = perf_counter()

    (
        edge_roots,
        edge_hashes,
        suffix_states,
    ) = build_edge_commitments(
        road,
        store,
    )

    tree = BinarySpatialTreeBuilder(
        road,
        theta,
    ).build()

    root = authenticate_spatial_tree(
        tree,
        edge_hashes,
    )

    post_insert_index_time = (
        perf_counter() - t0
    )

    construction_time = (
        ordered_insert_time
        + post_insert_index_time
    )

    # --------------------------------------------------------
    # EXCLUDED: serialization
    # --------------------------------------------------------
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_file = (
        INDEX_DIR
        / f"{city}_{percent:03d}pct_baseline.dat"
    ).resolve()

    index = BaselineIndex(
        road=road,
        store=store,
        tree=tree,
        edge_entry_roots=edge_roots,
        edge_hashes=edge_hashes,
        edge_suffix_states=suffix_states,
        root_hash=root,
        theta=tree.theta,
        trajectory_files=tuple(
            used_files
        ),
    )

    save_index(
        index,
        index_file,
    )

    size_bytes = (
        index_file.stat().st_size
    )

    result = {
        "method": "baseline",
        "city": city,
        "percent": percent,
        "total_trajectories": (
            TOTAL_TRAJECTORIES[city]
        ),
        "trajectory_count": cutoff,
        "selected_trajectory_count": (
            selected_count
        ),
        "effective_trajectory_count": (
            effective_count
        ),
        "entry_count": entry_count,
        "ordered_insertion_time_s": (
            ordered_insert_time
        ),
        "post_insertion_index_time_s": (
            post_insert_index_time
        ),
        "index_construction_time_s": (
            construction_time
        ),
        "index_size_bytes": size_bytes,
        "index_size_mib": (
            size_bytes
            / 1024.0
            / 1024.0
        ),
        "preparation_wall_time_s": (
            preparation_wall_time
        ),
        "theta": theta,
        "source_files_used": (
            "|".join(used_files)
        ),
        "index_file": str(
            index_file
        ),
        "selection_policy": (
            "first N raw trajectories in natural numeric "
            "file order; cutoff/counting outside Index "
            "Construction Time"
        ),
        "timing_scope": (
            "immediate start-time ordered Entry insertion + "
            "edge authenticated commitments + Binary Spatial "
            "Tree build/authentication; road loading, source "
            "reading, cutoff/counting, raw parsing/EID mapping "
            "and serialization excluded; catalog not counted"
        ),
        **extra_stats,
    }

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== RESULT =====")
    print(
        f"Ordered Entry Insertion Time: "
        f"{ordered_insert_time:.9f} s"
    )
    print(
        f"Post-insertion Index Time: "
        f"{post_insert_index_time:.9f} s"
    )
    print(
        f"Index Construction Time: "
        f"{construction_time:.9f} s"
    )
    print(
        f"Index Size: "
        f"{size_bytes} bytes "
        f"({size_bytes / 1024 / 1024:.3f} MiB)"
    )
    print("root_S:", root.hex())
    print("index:", index_file)


# ============================================================
# Result aggregation
# ============================================================

def collect_results() -> list[dict]:
    if not WORKER_DIR.exists():
        return []

    rows = []

    for p in sorted(
        WORKER_DIR.glob("*.json")
    ):
        rows.append(
            json.loads(
                p.read_text(
                    encoding="utf-8"
                )
            )
        )

    return rows


def write_outputs(
    rows: list[dict],
) -> None:
    if not rows:
        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = sorted(
        rows,
        key=lambda r: (
            CITIES.index(r["city"]),
            int(r["percent"]),
        ),
    )

    fields = [
        "method",
        "city",
        "percent",
        "total_trajectories",
        "trajectory_count",
        "selected_trajectory_count",
        "effective_trajectory_count",
        "entry_count",
        "ordered_insertion_time_s",
        "post_insertion_index_time_s",
        "index_construction_time_s",
        "index_size_bytes",
        "index_size_mib",
        "preparation_wall_time_s",
        "theta",
        "source_files_used",
        "index_file",
        "selection_policy",
        "timing_scope",
        "stationary_segments",
        "short_trajectories",
        "no_movement_trajectories",
    ]

    with RESULT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    lookup = {
        (r["city"], int(r["percent"])): r
        for r in rows
    }

    def write_matrix(
        path: Path,
        field: str,
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                ["city", *PERCENTS]
            )

            for city in CITIES:
                writer.writerow(
                    [city]
                    + [
                        lookup
                        .get(
                            (city, pct),
                            {},
                        )
                        .get(
                            field,
                            "",
                        )
                        for pct in PERCENTS
                    ]
                )

    write_matrix(
        TIME_MATRIX_FILE,
        "index_construction_time_s",
    )
    write_matrix(
        SIZE_MATRIX_FILE,
        "index_size_mib",
    )


# ============================================================
# Parent runner
# ============================================================

def parse_percent_arg(
    value: str,
) -> list[int]:
    if value == "all":
        return list(PERCENTS)

    pct = int(value)

    if pct not in PERCENTS:
        raise ValueError(
            f"percent must be one of "
            f"{PERCENTS}"
        )

    return [pct]


def run_parent(
    city_arg: str,
    percent_arg: str,
    overwrite: bool,
) -> None:
    cities = (
        list(CITIES)
        if city_arg == "all"
        else [city_arg]
    )

    percents = parse_percent_arg(
        percent_arg
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORKER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for city in cities:
        for pct in percents:
            result_path = (
                WORKER_DIR
                / f"{city}_{pct:03d}.json"
            )

            if (
                result_path.exists()
                and not overwrite
            ):
                print(
                    f"skip existing result: "
                    f"{city} {pct}%"
                )
                continue

            cmd = [
                sys.executable,
                str(
                    Path(__file__).resolve()
                ),
                "_worker",
                "--city",
                city,
                "--percent",
                str(pct),
                "--result",
                str(result_path),
            ]

            print()
            print(
                "RUN:",
                " ".join(cmd),
            )

            subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                check=True,
            )

            # Save partial tables after every point.
            write_outputs(
                collect_results()
            )

    write_outputs(
        collect_results()
    )

    print()
    print(
        "BASELINE INDEX SCALING PASS"
    )
    print(
        "results:",
        RESULT_FILE,
    )
    print(
        "time matrix:",
        TIME_MATRIX_FILE,
    )
    print(
        "size matrix:",
        SIZE_MATRIX_FILE,
    )


def show_plan() -> None:
    print(
        "Fixed trajectory totals / cutoffs "
        "(no dataset scan):"
    )

    for city in CITIES:
        values = [
            cutoff_for(city, p)
            for p in PERCENTS
        ]

        print(
            f"{city:8s} total="
            f"{TOTAL_TRAJECTORIES[city]:,} | "
            f"20/40/60/80/100% = "
            f"{[f'{x:,}' for x in values]}"
        )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline index construction-time/size "
            "scaling without physical subset files"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "show",
        help=(
            "show fixed totals/cutoffs only"
        ),
    )

    run = sub.add_parser(
        "run",
        help="run scaling experiment",
    )

    run.add_argument(
        "--city",
        choices=[
            "all",
            *CITIES,
        ],
        default="all",
    )

    run.add_argument(
        "--percent",
        choices=[
            "all",
            *[
                str(p)
                for p in PERCENTS
            ],
        ],
        default="all",
    )

    run.add_argument(
        "--overwrite",
        action="store_true",
    )

    worker = sub.add_parser(
        "_worker",
        help=argparse.SUPPRESS,
    )

    worker.add_argument(
        "--city",
        choices=CITIES,
        required=True,
    )

    worker.add_argument(
        "--percent",
        type=int,
        choices=PERCENTS,
        required=True,
    )

    worker.add_argument(
        "--result",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.command == "show":
        show_plan()
        return

    if args.command == "_worker":
        run_worker(
            args.city,
            args.percent,
            args.result,
        )
        return

    run_parent(
        city_arg=args.city,
        percent_arg=args.percent,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
