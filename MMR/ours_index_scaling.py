from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

# ============================================================
# Ours index scaling experiment
#
# Put this file in:
#   E:\MMR_Trajectory_range_c\ours_index_scaling.py
#
# No subset files are generated.
# No total-count pass is performed.
#
# Data amount:
#   first N trajectories in deterministic original order
#   20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100%
#
# Ours Index Construction Time INCLUDES only:
#   1) Edge tau-MMR
#   2) L0 Spatial Skeleton
#   3) E_h edge-block construction (function name build_all_l1_trees kept for compatibility)
#   4) Authenticated L0
#
# EXCLUDES:
#   road loading
#   trajectory file reading
#   cutoff selection/counting
#   trajectory parsing
#   Entry Store preparation
#   index serialization
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = PROJECT_ROOT / "index_scaling_output" / "ours"
INDEX_DIR = OUTPUT_ROOT / "indexes"
WORKER_DIR = OUTPUT_ROOT / "_worker_results"

RESULT_FILE = OUTPUT_ROOT / "index_scaling_results.csv"
TIME_MATRIX_FILE = OUTPUT_ROOT / "construction_time_matrix_s.csv"
SIZE_MATRIX_FILE = OUTPUT_ROOT / "index_size_matrix_mib.csv"

CITIES = ("chengdu", "xian", "beijing")
PERCENTS = (20, 40, 60, 80, 100)

# Fixed totals used by the current Ours datasets.
# Scaling subsets are prefixes of the original trajectory order.
TOTAL_TRAJECTORIES = {
    "chengdu": 284_608,
    "xian": 45_851,
    "beijing": 31_390,
}

# ============================================================
# Ours dataset paths
#
# No VTRQ_baseline config or files are used.
# Road-network files live in this Ours project; trajectory files
# are read directly from the original trajectory data directories.
# ============================================================

THETA = 64

CHENGDU_NODE = PROJECT_ROOT / "chengdu_nodes.txt"
CHENGDU_EDGE = PROJECT_ROOT / "chengdu_edges.txt"
CHENGDU_TRAJECTORY_DIR = Path(
    r"E:\Graph-Diffusion-Planning-main\chengdu-tra-json"
)
CHENGDU_TRAJECTORY_GLOB = "traj-*.json"

XIAN_NODE = PROJECT_ROOT / "xian_nodes.txt"
XIAN_EDGE = PROJECT_ROOT / "xian_edges.txt"
XIAN_TRAJECTORY_DIR = Path(
    r"E:\Graph-Diffusion-Planning-main\loader\preprocess\mm\sets_data\real2\trajectories"
)
XIAN_TRAJECTORY_GLOB = "traj_mapped_xian_xian10-*.json"

BEIJING_DIR = PROJECT_ROOT / "beijing_preprocessed"
BEIJING_NODE = BEIJING_DIR / "beijing_nodes.txt"
BEIJING_EDGE = BEIJING_DIR / "beijing_edges.txt"
BEIJING_TRAJECTORY = BEIJING_DIR / "beijing_trajectories.csv"


# ============================================================
# Dataset/order helpers
# ============================================================

def cutoff_for(city: str, percent: int) -> int:
    total = TOTAL_TRAJECTORIES[city]
    return total if percent == 100 else total * percent // 100


def natural_key(path: Path):
    """
    Natural numeric order:
      traj-10-1.json
      traj-10-2.json
      ...
      traj-10-10.json
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_city_spec(city: str) -> tuple[Path, Path, int, list[Path]]:
    """
    Resolve input files for the current Ours project only.

    Chengdu:
      PROJECT_ROOT/chengdu_nodes.txt
      PROJECT_ROOT/chengdu_edges.txt
      E:/Graph-Diffusion-Planning-main/chengdu-tra-json/traj-*.json

    Xi'an:
      PROJECT_ROOT/xian_nodes.txt
      PROJECT_ROOT/xian_edges.txt
      E:/Graph-Diffusion-Planning-main/loader/preprocess/mm/sets_data/real2/trajectories/
          traj_mapped_xian_xian10-*.json

    Beijing:
      PROJECT_ROOT/beijing_preprocessed/...
    """

    if city == "chengdu":
        node = CHENGDU_NODE.resolve()
        edge = CHENGDU_EDGE.resolve()
        theta = THETA
        files = sorted(
            CHENGDU_TRAJECTORY_DIR.glob(
                CHENGDU_TRAJECTORY_GLOB
            ),
            key=natural_key,
        )

    elif city == "xian":
        node = XIAN_NODE.resolve()
        edge = XIAN_EDGE.resolve()
        theta = THETA
        files = sorted(
            XIAN_TRAJECTORY_DIR.glob(
                XIAN_TRAJECTORY_GLOB
            ),
            key=natural_key,
        )

    elif city == "beijing":
        node = BEIJING_NODE.resolve()
        edge = BEIJING_EDGE.resolve()
        theta = THETA
        files = [BEIJING_TRAJECTORY.resolve()]

    else:
        raise ValueError(f"Unknown city: {city}")

    if not node.exists():
        raise FileNotFoundError(
            f"{city}: node file not found: {node}"
        )

    if not edge.exists():
        raise FileNotFoundError(
            f"{city}: edge file not found: {edge}"
        )

    if not files:
        if city == "chengdu":
            raise FileNotFoundError(
                f"chengdu: no trajectory files matched "
                f"{CHENGDU_TRAJECTORY_DIR / CHENGDU_TRAJECTORY_GLOB}"
            )
        if city == "xian":
            raise FileNotFoundError(
                f"xian: no trajectory files matched "
                f"{XIAN_TRAJECTORY_DIR / XIAN_TRAJECTORY_GLOB}"
            )
        raise FileNotFoundError(
            f"beijing: trajectory file not found: {BEIJING_TRAJECTORY}"
        )

    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)

    return node, edge, theta, files


# ============================================================
# Limited loaders
#
# Selection/counting is performed BEFORE Ours' timed ADS build.
# ============================================================

class LimitedJSONLoader:
    """
    Wrap the project's normal Chengdu/Xi'an loader.

    There is NO per-trajectory cutoff branch.
    For each source JSON file:
      remaining = cutoff - selected
      data[:remaining]

    Once the target is reached, later files are not read.
    """

    def __init__(self, base_loader, cutoff: int):
        self.base_loader = base_loader
        self.cutoff = int(cutoff)
        self.selected = 0
        self.used_files: list[str] = []

    def __call__(self, file_path):
        if self.selected >= self.cutoff:
            return []

        data = self.base_loader(file_path)

        if not isinstance(data, list):
            raise ValueError(
                f"{file_path}: top-level trajectory data must be list"
            )

        remaining = self.cutoff - self.selected
        take = min(len(data), remaining)

        self.selected += take
        if take > 0:
            self.used_files.append(str(Path(file_path).resolve()))

        if take == len(data):
            return data

        return data[:take]


class LimitedBeijingLoader:
    """
    Read only the first N RAW Beijing CSV trajectory rows.

    This exactly defines the shared subset:
      20% = first N20 raw trajectory rows
      ...
    Rows with <2 points still count toward the raw trajectory cutoff,
    just as they belong to the selected input subset; the adapter
    semantics then skip them from Entry construction.
    """

    def __init__(self, cutoff: int):
        self.cutoff = int(cutoff)
        self.selected = 0
        self.used_files: list[str] = []

    @staticmethod
    def _raise_csv_field_limit() -> None:
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                return
            except OverflowError:
                limit //= 10

    def __call__(self, file_path):
        if self.selected >= self.cutoff:
            return []

        self._raise_csv_field_limit()

        path = Path(file_path)
        trajectories = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])

            if "points_data" not in fields:
                raise ValueError(
                    f"{path}: missing points_data"
                )

            if (
                "source_traj_id" not in fields
                and "traj_id" not in fields
            ):
                raise ValueError(
                    f"{path}: missing source_traj_id/traj_id"
                )

            remaining = self.cutoff - self.selected

            # No repeated "if selected >= cutoff" inside the row loop.
            # We consume exactly at most `remaining` rows.
            for row_index, row in enumerate(reader):
                if row_index >= remaining:
                    break

                self.selected += 1

                source_traj_id = (
                    row.get("source_traj_id")
                    or row.get("traj_id")
                    or ""
                ).strip()

                try:
                    points = json.loads(row["points_data"])
                except Exception as exc:
                    raise ValueError(
                        f"{path}: CSV row {row_index + 2} "
                        "points_data JSON parse failed"
                    ) from exc

                if (
                    not isinstance(points, list)
                    or len(points) < 2
                ):
                    continue

                trajectories.append(
                    {
                        "source_traj_id": source_traj_id,
                        "points": points,
                        "row_no": row_index + 2,
                    }
                )

        if self.selected > 0:
            self.used_files.append(str(path.resolve()))

        return trajectories


# ============================================================
# One worker = one city x one percentage
# ============================================================

def run_worker(
    city: str,
    percent: int,
    result_path: Path,
) -> None:
    cutoff = cutoff_for(city, percent)
    node_file, edge_file, theta, trajectory_files = (
        load_city_spec(city)
    )

    # Beijing adapter MUST be installed before importing edge_entries.
    if city == "beijing":
        from beijing_csv_adapter import install_beijing_adapter
        install_beijing_adapter()

    from road_network import RoadNetwork
    import edge_entries
    from edge_mmr import build_all_edge_mmrs
    from spatial_skeleton import SpatialSkeletonBuilder
    from l1_merkle import build_all_l1_trees
    from l0_authenticated import build_authenticated_l0
    from index_io import PersistedIndex, save_index

    if not hasattr(edge_entries, "load_trajectory_json"):
        raise RuntimeError(
            "edge_entries.py does not expose load_trajectory_json; "
            "cannot install the cutoff loader safely."
        )

    if not hasattr(edge_entries, "parse_trajectory"):
        raise RuntimeError(
            "edge_entries.py does not expose parse_trajectory."
        )

    if city == "beijing":
        from beijing_csv_adapter import parse_beijing_trajectory

        loader = LimitedBeijingLoader(cutoff)
        edge_entries.load_trajectory_json = loader
        edge_entries.parse_trajectory = parse_beijing_trajectory
    else:
        original_loader = edge_entries.load_trajectory_json
        loader = LimitedJSONLoader(
            base_loader=original_loader,
            cutoff=cutoff,
        )
        edge_entries.load_trajectory_json = loader

    print()
    print("=" * 76)
    print(
        f"Ours | {city} | {percent}% | "
        f"target trajectories={cutoff}"
    )
    print("=" * 76)

    print("Source file order:")
    for i, p in enumerate(trajectory_files, 1):
        print(f"  [{i}] {p.name}")

    # --------------------------------------------------------
    # EXCLUDED: road loading
    # --------------------------------------------------------
    road = RoadNetwork()
    road.load(
        node_file=str(node_file),
        edge_file=str(edge_file),
    )

    # --------------------------------------------------------
    # EXCLUDED:
    # trajectory reading + cutoff + parsing + Entry preparation
    # --------------------------------------------------------
    prep_t0 = perf_counter()

    store = edge_entries.build_edge_entries_from_files(
        trajectory_files=[str(p) for p in trajectory_files],
        road=road,
    )

    preparation_wall_time = perf_counter() - prep_t0

    if loader.selected != cutoff:
        raise RuntimeError(
            f"{city} {percent}%: requested {cutoff} trajectories "
            f"but only selected {loader.selected}. "
            "Check TOTAL_TRAJECTORIES or input files."
        )

    print(
        f"Selected trajectories: {loader.selected}/{cutoff}"
    )
    print(
        f"Preparation wall time (EXCLUDED): "
        f"{preparation_wall_time:.6f} s"
    )

    # --------------------------------------------------------
    # INCLUDED: Ours authenticated index construction
    # --------------------------------------------------------
    total_t0 = perf_counter()

    t0 = perf_counter()
    mmr_index = build_all_edge_mmrs(
        store=store,
        road=road,
    )
    mmr_time = perf_counter() - t0

    t0 = perf_counter()
    skeleton = SpatialSkeletonBuilder(
        road=road,
        theta=theta,
    ).build()
    skeleton_time = perf_counter() - t0

    t0 = perf_counter()
    # Current paper-aligned code: this builds flat E_h edge blocks.
    # The function/module name is retained only for compatibility.
    l1_index = build_all_l1_trees(
        skeleton=skeleton,
        mmr_index=mmr_index,
        road=road,
    )
    l1_time = perf_counter() - t0

    t0 = perf_counter()
    l0_index = build_authenticated_l0(
        skeleton=skeleton,
        l1_index=l1_index,
    )
    l0_time = perf_counter() - t0

    construction_time = perf_counter() - total_t0

    # --------------------------------------------------------
    # EXCLUDED: serialization
    # --------------------------------------------------------
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_file = (
        INDEX_DIR
        / f"{city}_{percent:03d}pct_ours.dat"
    ).resolve()

    persisted = PersistedIndex(
        road=road,
        store=store,
        mmr_index=mmr_index,
        skeleton=skeleton,
        l1_index=l1_index,
        l0_index=l0_index,
        theta=theta,
        trajectory_files=tuple(loader.used_files),
    )

    save_index(
        index=persisted,
        file_path=str(index_file),
    )

    size_bytes = index_file.stat().st_size

    result = {
        "method": "ours",
        "city": city,
        "percent": percent,
        "total_trajectories": TOTAL_TRAJECTORIES[city],
        "trajectory_count": cutoff,
        "selected_trajectory_count": loader.selected,
        "index_construction_time_s": construction_time,
        "index_size_bytes": size_bytes,
        "index_size_mib": size_bytes / 1024.0 / 1024.0,
        "mmr_time_s": mmr_time,
        "spatial_skeleton_time_s": skeleton_time,
        "l1_merkle_time_s": l1_time,
        "l0_auth_time_s": l0_time,
        "preparation_wall_time_excluded_s": preparation_wall_time,
        "theta": theta,
        "source_files_used": "|".join(loader.used_files),
        "index_file": str(index_file),
        "selection_policy": (
            "first N raw trajectories in natural numeric file order; "
            "cutoff/counting outside Index Construction Time"
        ),
        "timing_scope": (
            "tau-MMR + L0 spatial skeleton + L1 Merkle + "
            "authenticated L0; road loading, trajectory reading, "
            "cutoff/counting, parsing, Entry preparation and "
            "serialization excluded"
        ),
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
        f"Index Construction Time: "
        f"{construction_time:.9f} s"
    )
    print(
        f"Index Size: {size_bytes} bytes "
        f"({size_bytes / 1024 / 1024:.3f} MiB)"
    )
    print("root_S:", l0_index.root_hash.hex())
    print("index:", index_file)


# ============================================================
# Result aggregation
# ============================================================

def collect_results() -> list[dict]:
    if not WORKER_DIR.exists():
        return []

    rows = []
    for p in sorted(WORKER_DIR.glob("*.json")):
        rows.append(
            json.loads(
                p.read_text(encoding="utf-8")
            )
        )
    return rows


def write_outputs(rows: list[dict]) -> None:
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
        "index_construction_time_s",
        "index_size_bytes",
        "index_size_mib",
        "mmr_time_s",
        "spatial_skeleton_time_s",
        "l1_merkle_time_s",
        "l0_auth_time_s",
        "preparation_wall_time_excluded_s",
        "theta",
        "source_files_used",
        "index_file",
        "selection_policy",
        "timing_scope",
    ]

    with RESULT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

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
                        .get((city, pct), {})
                        .get(field, "")
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

def parse_percent_arg(value: str) -> list[int]:
    if value == "all":
        return list(PERCENTS)

    pct = int(value)
    if pct not in PERCENTS:
        raise ValueError(
            f"percent must be one of {PERCENTS}"
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
                str(Path(__file__).resolve()),
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

            # Save partial table immediately.
            write_outputs(
                collect_results()
            )

    write_outputs(
        collect_results()
    )

    print()
    print("OURS INDEX SCALING PASS")
    print("results:", RESULT_FILE)
    print("time matrix:", TIME_MATRIX_FILE)
    print("size matrix:", SIZE_MATRIX_FILE)


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
            "Ours index construction-time/size scaling "
            "without physical subset files"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "show",
        help="show fixed totals and cutoffs only",
    )

    run = sub.add_parser(
        "run",
        help="run scaling experiment",
    )
    run.add_argument(
        "--city",
        choices=["all", *CITIES],
        default="all",
    )
    run.add_argument(
        "--percent",
        choices=[
            "all",
            *[str(p) for p in PERCENTS],
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
