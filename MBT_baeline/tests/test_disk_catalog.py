import tempfile
import unittest
from pathlib import Path

from three_mbt.plaintext_catalog import (
    DiskTrajectoryCatalog,
    TrajectoryCatalog,
    TrajectoryPlaintext,
)
from three_mbt.trajectory_id import compute_paper_trajectory_id


class DiskCatalogTests(unittest.TestCase):
    def test_offset_catalog_reads_only_requested_rows_and_verifies_plaintext(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            start = 100
            end = 120
            t1 = compute_paper_trajectory_id([1, 2], start, end)
            t2 = compute_paper_trajectory_id([3, 4], start, end)
            rows = [
                TrajectoryPlaintext(
                    trajectory_id=t1,
                    source_traj_id="a",
                    road_node_path=[1, 2],
                    points_data=[[start, 30.0, 104.0], [end, 30.1, 104.1]],
                    points_format="time_lat_lon",
                ),
                TrajectoryPlaintext(
                    trajectory_id=t2,
                    source_traj_id="b",
                    road_node_path=[3, 4],
                    points_data=[[start, 31.0, 105.0], [end, 31.1, 105.1]],
                    points_format="time_lat_lon",
                ),
            ]
            csv_path = td / "catalog.csv"
            TrajectoryCatalog.write_csv(csv_path, rows)
            with DiskTrajectoryCatalog(csv_path) as cat:
                self.assertEqual(len(cat), 2)
                got = cat.get_many([t2, t1])
                self.assertEqual({x.trajectory_id for x in got}, {t1, t2})
                self.assertTrue(all(x.verify_trajectory_id() for x in got))
                self.assertEqual(cat.missing_from([t1, t2]), [])
                self.assertTrue((Path(str(csv_path) + ".offsets.sqlite")).exists())


if __name__ == "__main__":
    unittest.main()
