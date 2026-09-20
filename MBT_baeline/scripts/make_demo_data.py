from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def make_chengdu(out: Path) -> None:
    base = 1541260553
    center_lon = 104.0540760857
    center_lat = 30.67289263645
    trajectories = []
    for i in range(20):
        node_path = [1000 + i, 2000 + i, 3000 + i]
        gps = []
        for j in range(6):
            gps.append([
                base + j * 300 + i,
                center_lat + (i - 10) * 0.00008 + (j - 2) * 0.00012,
                center_lon + (i - 10) * 0.00008 + (j - 2) * 0.00012,
            ])
        trajectories.append([node_path, [[node_path[0], node_path[1]], [node_path[1], node_path[2]]], [gps], []])
    (out / "chengdu_demo.json").write_text(json.dumps(trajectories), encoding="utf-8")


def make_beijing(out: Path) -> None:
    path = out / "beijing_demo.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_traj_id", "points_data"])
        w.writeheader()
        for i in range(10):
            points = [
                [100 + i, 116.32 + i * 0.0001, 39.98, 1239679020 + i],
                [200 + i, 116.33 + i * 0.0001, 39.99, 1239679320 + i],
                [300 + i, 116.34 + i * 0.0001, 40.00, 1239679620 + i],
            ]
            w.writerow({"source_traj_id": f"SRC-{i}", "points_data": json.dumps(points, separators=(",", ":"))})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="demo_data")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_chengdu(out)
    make_beijing(out)
    print(out)


if __name__ == "__main__":
    main()
