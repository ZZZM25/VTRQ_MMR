from __future__ import annotations

import argparse
import json
from pathlib import Path

from gps_rtree.build import build_index, index_size_bytes
from gps_rtree.experiments import run_query_experiment, run_scaling_experiment, run_update_experiment
from gps_rtree.prepare import prepare_dataset

ROOT = Path(__file__).resolve().parent
CITIES = ("chengdu", "xian", "beijing")


def load_config(city: str) -> dict:
    return json.loads((ROOT / "configs" / f"{city}.json").read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def prepare_city(city: str, cfg: dict) -> None:
    prepare_dataset(
        resolve_path(cfg["input"]),
        city,
        resolve_path(cfg["prepared_dir"]),
        resolve_path(cfg["plaintext_catalog"]),
        beijing_subdivisions=int(cfg.get("beijing_subdivisions", 10)),
    )


def build_city(city: str, cfg: dict) -> None:
    index_dir = resolve_path(cfg["index_dir"])
    meta, elapsed = build_index(
        resolve_path(cfg["prepared_dir"]),
        index_dir,
        trajectory_count=None,
        leaf_capacity=int(cfg.get("leaf_capacity", 128)),
        fanout=int(cfg.get("fanout", 64)),
        sort_chunk_rows=int(cfg.get("sort_chunk_rows", 500000)),
    )
    size = index_size_bytes(index_dir)
    report = {
        "city": city,
        "trajectory_count": meta["trajectory_count"],
        "point_count": meta["point_count"],
        "construction_time_s": elapsed,
        "index_size_bytes": size,
        "index_size_mib": size / (1024.0 * 1024.0),
        "root_hash": meta["root_hash_hex"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def query_city(city: str, cfg: dict) -> None:
    rows = run_query_experiment(
        resolve_path(cfg["index_dir"]),
        resolve_path(cfg["plaintext_catalog"]),
        city,
        resolve_path(cfg["query_output_dir"]),
        repeats=int(cfg.get("formal_repeats", 1)),
        warmups=int(cfg.get("warmup_repeats", 0)),
    )
    print(f"{city}: {len(rows)} query cells complete")


def scaling_city(city: str, cfg: dict) -> None:
    rows = run_scaling_experiment(
        resolve_path(cfg["prepared_dir"]),
        resolve_path(cfg["scaling_output_dir"]),
        city,
        leaf_capacity=int(cfg.get("leaf_capacity", 128)),
        fanout=int(cfg.get("fanout", 64)),
        sort_chunk_rows=int(cfg.get("sort_chunk_rows", 500000)),
    )
    print(f"{city}: {len(rows)} scaling points complete")


def update_city(city: str, cfg: dict) -> None:
    rows = run_update_experiment(
        resolve_path(cfg["prepared_dir"]),
        resolve_path(cfg["update_output_dir"]),
        city,
        leaf_capacity=int(cfg.get("leaf_capacity", 128)),
        fanout=int(cfg.get("fanout", 64)),
        sort_chunk_rows=int(cfg.get("sort_chunk_rows", 500000)),
    )
    print(f"{city}: {len(rows)} update points complete")


def main() -> int:
    p = argparse.ArgumentParser(description="Authenticated GPS-point R-tree baseline")
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
