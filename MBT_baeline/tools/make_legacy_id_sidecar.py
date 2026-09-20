from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Create Three-MBT ID sidecar from previous-paper outputs."
    )
    p.add_argument("source_csv", help="Previous-method CSV containing trajectory_id")
    p.add_argument("output_csv")
    p.add_argument("--city", required=True, choices=["chengdu", "xian", "beijing"])
    args = p.parse_args()

    source = Path(args.source_csv)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "trajectory_id" not in rows[0]:
        raise ValueError("source CSV has no trajectory_id column")

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        if args.city == "beijing":
            if rows and "source_traj_id" not in rows[0]:
                raise ValueError("Beijing source CSV must contain source_traj_id and trajectory_id")
            w = csv.DictWriter(f, fieldnames=["source_traj_id", "trajectory_id"])
            w.writeheader()
            for row in rows:
                w.writerow({
                    "source_traj_id": row["source_traj_id"],
                    "trajectory_id": row["trajectory_id"],
                })
        else:
            w = csv.DictWriter(f, fieldnames=["ordinal", "trajectory_id"])
            w.writeheader()
            for ordinal, row in enumerate(rows):
                w.writerow({"ordinal": ordinal, "trajectory_id": row["trajectory_id"]})

    print(output)
    if args.city in {"chengdu", "xian"}:
        print("ORDER REQUIREMENT: previous output rows must follow natural numeric JSON-file order, then original trajectory order within each file.")
    else:
        print("Beijing mapping uses source_traj_id -> paper trajectory_id; row order does not matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
