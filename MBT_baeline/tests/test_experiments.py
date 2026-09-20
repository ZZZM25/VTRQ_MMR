import tempfile
import unittest
from pathlib import Path

from three_mbt.experiments import CITY_QUERY_SETUP, run_paper_query_experiment
from three_mbt.index import ThreeMBTIndex
from three_mbt.models import GPSPoint
from three_mbt.plaintext_catalog import TrajectoryCatalog, TrajectoryPlaintext
from three_mbt.trajectory_id import compute_paper_trajectory_id


class ExperimentTests(unittest.TestCase):
    def test_paper_query_outputs(self):
        s = CITY_QUERY_SETUP["chengdu"]
        start = int(s["base_query_start"])
        end = start + 86400
        tid = compute_paper_trajectory_id([1, 2], start, end)
        pts = [
            GPSPoint(0, tid, start, s["center_lat"], s["center_lon"]),
            GPSPoint(1, tid, end, s["center_lat"] + 0.001, s["center_lon"] + 0.001),
        ]
        catalog = TrajectoryCatalog([
            TrajectoryPlaintext(
                trajectory_id=tid,
                source_traj_id="demo",
                road_node_path=[1, 2],
                points_data=[
                    [start, s["center_lat"], s["center_lon"]],
                    [end, s["center_lat"] + 0.001, s["center_lon"] + 0.001],
                ],
                points_format="time_lat_lon",
            )
        ])
        idx = ThreeMBTIndex(leaf_capacity=2, fanout=2)
        idx.bulk_build(pts)
        with tempfile.TemporaryDirectory() as td:
            rows = run_paper_query_experiment(
                idx, catalog, "chengdu", td, repeats=1, warmups=0
            )
            self.assertEqual(len(rows), 30)
            self.assertTrue((Path(td) / "query_time_matrix_s.csv").exists())
            self.assertTrue((Path(td) / "verification_time_matrix_s.csv").exists())
            self.assertTrue((Path(td) / "plaintext_verification_time_matrix_s.csv").exists())
            self.assertIn("trajectory_id_intersection_time_s", rows[0])
            self.assertIn("plaintext_lookup_time_s_excluded", rows[0])


if __name__ == "__main__":
    unittest.main()
