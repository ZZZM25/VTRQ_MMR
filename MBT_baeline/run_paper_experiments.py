from __future__ import annotations

import argparse
import json
from pathlib import Path

from three_mbt.datasets import load_dataset, write_plaintext_catalog
from three_mbt.experiments import (
    build_index_with_formal_timing,
    run_paper_query_experiment,
    run_scalability_experiment,
    run_update_experiment,
)
from three_mbt.index import ThreeMBTIndex
from three_mbt.disk_index import DiskThreeMBTIndex
from three_mbt.external_build import build_disk_index_from_dataset
from three_mbt.disk_experiments import (
    run_disk_scalability_experiment,
    run_disk_update_experiment,
)
from three_mbt.plaintext_catalog import DiskTrajectoryCatalog

ROOT = Path(__file__).resolve().parent
CITIES = ("chengdu", "xian", "beijing")


def load_config(city: str) -> dict:
    path = ROOT / "configs" / f"{city}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _dataset_args(city: str, cfg: dict):
    input_path = resolve_path(cfg["input"])
    sidecar = resolve_path(cfg.get("legacy_sidecar"))
    if input_path is None:
        raise ValueError("config input is required")
    return input_path, sidecar, int(cfg.get("beijing_subdivisions", 10))


def _bar(done: int, total: int, width: int = 30) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = round(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _build_file_progress(city: str):
    def callback(done: int, total: int, path: Path) -> None:
        pct = 100.0 * done / max(total, 1)
        print(
            f"\r[Load ] {_bar(done, total)} {pct:6.1f}% "
            f"({done}/{total}) {path.name}",
            end="",
            flush=True,
        )
        if done >= total:
            print()

    return callback


def _build_tree_progress(label: str, done: int, total: int) -> None:
    pct = 100.0 * done / max(total, 1)
    print(f"[Build] {_bar(done, total)} {pct:6.1f}% {label}", flush=True)






def _query_progress(label: str, done: int, total: int) -> None:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    pct = 100.0 * done / total
    line = f"[Query] {_bar(done, total)} {pct:6.1f}% {label}"
    live = "CSV scan:" in label or "Exact result verification:" in label
    if live and done < total:
        print("\r" + line, end="", flush=True)
    else:
        if live:
            print("\r" + line, flush=True)
        else:
            print(line, flush=True)

def _build_internal_progress(label: str, phase: str, done: int, total: int) -> None:
    pct = 100.0 * done / max(total, 1)
    # One live line per phase; completion advances to a clean newline.
    print(
        f"\r[{phase:<5}] {label:<8} {_bar(done, total)} {pct:6.1f}% "
        f"({done:,}/{total:,})",
        end="",
        flush=True,
    )
    if done >= total:
        print(flush=True)

def load_points(city: str, cfg: dict, *, show_build_progress: bool = False):
    input_path, sidecar, subdivisions = _dataset_args(city, cfg)
    return load_dataset(
        input_path,
        city,
        beijing_subdivisions=subdivisions,
        legacy_sidecar=sidecar,
        file_progress=_build_file_progress(city) if show_build_progress else None,
    )


def prepare_city(city: str, cfg: dict) -> None:
    input_path, sidecar, subdivisions = _dataset_args(city, cfg)
    catalog_path = resolve_path(cfg["plaintext_catalog"])
    assert catalog_path is not None
    count = write_plaintext_catalog(
        input_path,
        city,
        catalog_path,
        beijing_subdivisions=subdivisions,
        legacy_sidecar=sidecar,
    )
    report = {
        "city": city,
        "plaintext_catalog": str(catalog_path),
        "trajectory_count": count,
        "trajectory_id_definition": "SHA256(road-node sequence || whole-trajectory start || whole-trajectory end), uint32 big-endian",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_city(city: str, cfg: dict) -> None:
    input_path, sidecar, subdivisions = _dataset_args(city, cfg)
    index_path = resolve_path(cfg["index_file"])
    assert index_path is not None

    print(
        f"[Load ] {city}: streaming raw points to disk-backed build store "
        f"(excluded from formal construction time)",
        flush=True,
    )
    idx, construction_time_s = build_disk_index_from_dataset(
        input_path=input_path,
        dataset=city,
        index_path=index_path,
        leaf_capacity=int(cfg.get("leaf_capacity", 128)),
        fanout=int(cfg.get("fanout", 64)),
        beijing_subdivisions=subdivisions,
        legacy_sidecar=sidecar,
        load_progress=_build_file_progress(city),
        build_progress=_build_tree_progress,
        internal_progress=_build_internal_progress,
    )
    print(
        f"[Load ] {city}: streamed {idx.point_count:,} indexed GPS/interpolated points",
        flush=True,
    )

    print(f"[Save ] writing index metadata -> {index_path}", flush=True)
    idx.save(index_path)
    print("[Save ] complete", flush=True)

    index_size_bytes = idx.disk_size_bytes(index_path)
    report_path = index_path.with_suffix(index_path.suffix + ".build.json")
    report = {
        "city": city,
        "trajectory_count": len(idx.trajectory_ids_seen),
        "point_count": idx.point_count,
        "index_construction_time_s": construction_time_s,
        "index_size_bytes": index_size_bytes,
        "index_size_mib": index_size_bytes / (1024.0 * 1024.0),
        "root_lon": idx.mbt_lon.root_hash_hex,
        "root_lat": idx.mbt_lat.root_hash_hex,
        "root_time": idx.mbt_time.root_hash_hex,
        "combined_root": idx.combined_root_hex,
        "storage_mode": "disk-backed packed MBT leaves",
        "timed_scope": (
            "per-dimension external sort + packed MBT leaf/hash build + combined root; "
            "raw I/O/parsing excluded"
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def query_city(city: str, cfg: dict) -> None:
    index_path = resolve_path(cfg["index_file"])
    catalog_path = resolve_path(cfg["plaintext_catalog"])
    assert index_path is not None and catalog_path is not None
    print(f"[Query] loading MBT metadata only; leaf entries remain on disk", flush=True)
    idx = DiskThreeMBTIndex.load(index_path)
    print(
        f"[Query] plaintext lookup uses one sequential CSV scan per candidate set (excluded from formal timing): {catalog_path}",
        flush=True,
    )
    # No auxiliary plaintext index is created; candidate lookup scans the CSV once.
    catalog = DiskTrajectoryCatalog(catalog_path)
    try:
        output_dir = resolve_path(cfg.get("query_output_dir", f"benchmark_output_{city}_three_mbt"))
        assert output_dir is not None
        rows = run_paper_query_experiment(
            idx,
            catalog,
            city,
            output_dir,
            repeats=int(cfg.get("formal_repeats", 30)),
            warmups=int(cfg.get("warmup_repeats", 1)),
            progress=_query_progress,
        )
        print(f"{city}: {len(rows)} query cells -> {output_dir}")
    finally:
        catalog.close()


def scaling_city(city: str, cfg: dict) -> None:
    input_path, sidecar, subdivisions = _dataset_args(city, cfg)
    output_dir = resolve_path(cfg.get("scaling_output_dir", f"index_scaling_output/three_mbt/{city}"))
    assert output_dir is not None
    rows = run_disk_scalability_experiment(
        input_path=input_path,
        dataset=city,
        output_dir=output_dir,
        leaf_capacity=int(cfg.get("leaf_capacity", 128)),
        fanout=int(cfg.get("fanout", 64)),
        beijing_subdivisions=subdivisions,
        legacy_sidecar=sidecar,
        load_progress=_build_file_progress(city),
        build_progress=_build_tree_progress,
        internal_progress=_build_internal_progress,
        status=lambda msg: print(f"[Scale] {city}: {msg}", flush=True),
    )
    print(f"{city}: {len(rows)} scaling points -> {output_dir}")


def update_city(city: str, cfg: dict) -> None:
    input_path, sidecar, subdivisions = _dataset_args(city, cfg)
    output_dir = resolve_path(cfg.get("update_output_dir", f"index_update_output/three_mbt/{city}"))
    assert output_dir is not None
    rows = run_disk_update_experiment(
        input_path=input_path,
        dataset=city,
        output_dir=output_dir,
        leaf_capacity=int(cfg.get("leaf_capacity", 128)),
        fanout=int(cfg.get("fanout", 64)),
        beijing_subdivisions=subdivisions,
        legacy_sidecar=sidecar,
        load_progress=_build_file_progress(city),
        status=lambda msg: print(f"[Update] {city}: {msg}", flush=True),
    )
    print(f"{city}: {len(rows)} update points -> {output_dir}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "build", "query", "scaling", "update"])
    p.add_argument("--city", choices=[*CITIES, "all"], required=True)
    args = p.parse_args()

    cities = CITIES if args.city == "all" else (args.city,)
    for city in cities:
        cfg = load_config(city)
        if args.command == "prepare":
            prepare_city(city, cfg)
        elif args.command == "build":
            build_city(city, cfg)
        elif args.command == "query":
            query_city(city, cfg)
        elif args.command == "scaling":
            scaling_city(city, cfg)
        elif args.command == "update":
            update_city(city, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
