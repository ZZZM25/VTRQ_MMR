import tempfile
import unittest
from pathlib import Path

from three_mbt.models import QueryWindow
from three_mbt.plaintext_catalog import (
    TrajectoryCatalog,
    TrajectoryPlaintext,
    run_plaintext_stage,
)
from three_mbt.trajectory_id import compute_paper_trajectory_id


class PlaintextTests(unittest.TestCase):
    def test_catalog_roundtrip_verify_and_fine_filter(self):
        tid = compute_paper_trajectory_id([1, 2], 100, 110)
        rec = TrajectoryPlaintext(
            trajectory_id=tid,
            source_traj_id="SRC",
            road_node_path=[1, 2],
            points_data=[[1, 104.0, 30.0, 100], [2, 104.01, 30.01, 110]],
            points_format="node_lon_lat_time",
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "catalog.csv"
            TrajectoryCatalog.write_csv(path, [rec])
            catalog = TrajectoryCatalog.load_csv(path)
            q = QueryWindow(103.9, 104.1, 29.9, 30.1, 90, 120)
            stage = run_plaintext_stage(catalog, [tid], q)
            self.assertEqual(stage.final_trajectory_ids, [tid])

    def test_plaintext_tamper_rejected(self):
        tid = compute_paper_trajectory_id([1, 2], 100, 110)
        rec = TrajectoryPlaintext(
            trajectory_id=tid,
            source_traj_id="SRC",
            road_node_path=[1, 999],
            points_data=[[1, 104.0, 30.0, 100], [999, 104.01, 30.01, 110]],
            points_format="node_lon_lat_time",
        )
        catalog = TrajectoryCatalog([rec])
        q = QueryWindow(103, 105, 29, 31, 90, 120)
        with self.assertRaises(ValueError):
            run_plaintext_stage(catalog, [tid], q)


if __name__ == "__main__":
    unittest.main()
