import json
import tempfile
import unittest
from pathlib import Path

from three_mbt.datasets import load_dataset
from three_mbt.disk_index import DiskThreeMBTIndex
from three_mbt.external_build import build_disk_index_from_dataset
from three_mbt.index import ThreeMBTIndex
from three_mbt.models import QueryWindow


class DiskBuildTests(unittest.TestCase):
    def test_disk_build_matches_in_memory_root_and_vo(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = td / "data"
            data.mkdir()
            items = [
                [[1, 2], [[1, 2]], [[[100, 30.0, 104.0], [110, 30.1, 104.1], [120, 30.2, 104.2]]], []],
                [[3, 4], [[3, 4]], [[[105, 31.0, 105.0], [115, 31.1, 105.1]]], []],
            ]
            (data / "traj-10-7.json").write_text(json.dumps(items), encoding="utf-8")

            points = load_dataset(data, "chengdu")
            mem = ThreeMBTIndex(leaf_capacity=2, fanout=2)
            mem.bulk_build(points)

            index_path = td / "idx.pkl"
            disk, _elapsed = build_disk_index_from_dataset(
                input_path=data,
                dataset="chengdu",
                index_path=index_path,
                leaf_capacity=2,
                fanout=2,
            )
            self.assertEqual(disk.mbt_lon.root_hash_hex, mem.mbt_lon.root_hash_hex)
            self.assertEqual(disk.mbt_lat.root_hash_hex, mem.mbt_lat.root_hash_hex)
            self.assertEqual(disk.mbt_time.root_hash_hex, mem.mbt_time.root_hash_hex)
            self.assertEqual(disk.combined_root_hex, mem.combined_root_hex)

            disk.save(index_path)
            loaded = DiskThreeMBTIndex.load(index_path)
            q = QueryWindow(103.9, 104.15, 29.9, 30.15, 99, 116)
            mem_result = mem.query(q)
            disk_result = loaded.query(q)
            self.assertEqual(mem_result.candidate_trajectory_ids, disk_result.candidate_trajectory_ids)
            self.assertEqual(mem_result.vo_lon, disk_result.vo_lon)
            self.assertTrue(loaded.verify(disk_result, loaded.combined_root).verified)


if __name__ == "__main__":
    unittest.main()
