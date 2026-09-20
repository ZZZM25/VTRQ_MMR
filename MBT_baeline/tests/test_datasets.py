import csv
import json
import tempfile
import unittest
from pathlib import Path

from three_mbt.datasets import iter_beijing_records, iter_chengdu_xian_records
from three_mbt.trajectory_id import compute_paper_trajectory_id


class DatasetTests(unittest.TestCase):
    def test_beijing_real_shape_and_interpolation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bj.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["source_traj_id", "points_data"])
                w.writeheader()
                w.writerow({
                    "source_traj_id": "SRC",
                    "points_data": "[[1,116.0,39.0,100],[2,117.0,40.0,110],[3,118.0,41.0,120]]",
                })
            rec = next(iter_beijing_records(path, subdivisions=10))
            self.assertEqual(len(rec.gps_points), 21)
            self.assertEqual(rec.gps_points[0], (100.0, 39.0, 116.0))
            self.assertEqual(rec.gps_points[-1], (120.0, 41.0, 118.0))
            expected = compute_paper_trajectory_id([1, 2, 3], 100, 120)
            self.assertEqual(rec.trajectory_id, expected)
            self.assertEqual(rec.points_format, "node_lon_lat_time")
            self.assertEqual(rec.plaintext_points[0], [1, 116.0, 39.0, 100])

    def test_multiday_chengdu_natural_order(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # formal shape: [node_path, edge-pairs, GPS-groups, ...]
            (d / "traj-10-10.json").write_text(
                json.dumps([[[3, 4], [[3, 4]], [[[200, 30.0, 104.0], [210, 30.1, 104.1]]], []]]),
                encoding="utf-8",
            )
            (d / "traj-10-7.json").write_text(
                json.dumps([[[1, 2], [[1, 2]], [[[100, 31.0, 105.0], [110, 31.1, 105.1]]], []]]),
                encoding="utf-8",
            )
            recs = list(iter_chengdu_xian_records(d))
            self.assertEqual([r.ordinal for r in recs], [0, 1])
            self.assertEqual(recs[0].road_node_path, [1, 2])
            self.assertEqual(recs[0].gps_points[0][0], 100.0)
            self.assertEqual(
                recs[0].trajectory_id,
                compute_paper_trajectory_id([1, 2], 100, 110),
            )


if __name__ == "__main__":
    unittest.main()
