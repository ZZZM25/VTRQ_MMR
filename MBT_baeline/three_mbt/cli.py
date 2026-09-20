from __future__ import annotations

import argparse
import json

from .datasets import load_dataset, write_plaintext_catalog
from .index import ThreeMBTIndex
from .models import QueryWindow
from .plaintext_catalog import TrajectoryCatalog, run_plaintext_stage


def _dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=["chengdu", "xian", "beijing"])
    p.add_argument("--input", required=True)
    p.add_argument("--legacy-sidecar", default=None)
    p.add_argument("--beijing-subdivisions", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="three-mbt")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    _dataset_args(prep)
    prep.add_argument("--catalog", required=True)

    b = sub.add_parser("build")
    _dataset_args(b)
    b.add_argument("--output", required=True)
    b.add_argument("--leaf-capacity", type=int, default=128)
    b.add_argument("--fanout", type=int, default=64)

    q = sub.add_parser("query")
    q.add_argument("--index", required=True)
    q.add_argument("--catalog", required=True)
    q.add_argument("--min-lon", type=float, required=True)
    q.add_argument("--max-lon", type=float, required=True)
    q.add_argument("--min-lat", type=float, required=True)
    q.add_argument("--max-lat", type=float, required=True)
    q.add_argument("--start-time", type=float, required=True)
    q.add_argument("--end-time", type=float, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "prepare":
        n = write_plaintext_catalog(
            args.input,
            args.dataset,
            args.catalog,
            args.beijing_subdivisions,
            args.legacy_sidecar,
        )
        print(json.dumps({"catalog": args.catalog, "trajectories": n}, indent=2))
        return 0

    if args.cmd == "build":
        points = load_dataset(
            args.input,
            args.dataset,
            args.beijing_subdivisions,
            args.legacy_sidecar,
        )
        idx = ThreeMBTIndex(args.leaf_capacity, args.fanout)
        idx.bulk_build(points)
        idx.save(args.output)
        print(json.dumps(idx.stats(), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "query":
        idx = ThreeMBTIndex.load(args.index)
        catalog = TrajectoryCatalog.load_csv(args.catalog)
        q = QueryWindow(
            args.min_lon,
            args.max_lon,
            args.min_lat,
            args.max_lat,
            args.start_time,
            args.end_time,
        )
        result = idx.query(q)
        verified = idx.verify(result, idx.combined_root)
        plain = run_plaintext_stage(catalog, verified.verified_candidate_trajectory_ids, q)
        print(
            json.dumps(
                {
                    "verified": verified.verified,
                    "lon_trajectory_ids": sorted(result.lon_trajectory_ids),
                    "lat_trajectory_ids": sorted(result.lat_trajectory_ids),
                    "time_trajectory_ids": sorted(result.time_trajectory_ids),
                    "candidate_trajectory_ids": verified.verified_candidate_trajectory_ids,
                    "final_trajectory_ids": plain.final_trajectory_ids,
                    "coarse_query_time_s": result.coarse_query_time_s,
                    "plaintext_lookup_time_s_excluded": plain.lookup_time_s,
                    "fine_filtering_time_s": plain.fine_filtering_time_s,
                    "exact_result_verification_time_s": plain.fine_filtering_time_s,
                    "query_time_s": result.coarse_query_time_s,
                    "mbt_verification_time_s": verified.mbt_verification_time_s,
                    "plaintext_verification_time_s": plain.plaintext_verification_time_s,
                    "verification_time_s": verified.mbt_verification_time_s + plain.fine_filtering_time_s,
                    "trusted_combined_root": idx.combined_root_hex,
                    "reconstructed_combined_root": verified.reconstructed_global_root,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
