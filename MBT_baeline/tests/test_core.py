import copy
import unittest

from three_mbt.index import ThreeMBTIndex
from three_mbt.mbt import MerkleBTree
from three_mbt.models import GPSPoint, QueryWindow


class ThreeMBTCoreTests(unittest.TestCase):
    def setUp(self):
        self.points = [
            GPSPoint(0, "A", 100, 50.0, 104.0),
            GPSPoint(1, "A", 200, 30.0, 200.0),
            GPSPoint(2, "B", 100, 60.0, 105.0),
            GPSPoint(3, "B", 200, 61.0, 106.0),
        ]

    def test_trajectory_id_level_coarse_intersection(self):
        idx = ThreeMBTIndex(leaf_capacity=2, fanout=2)
        idx.bulk_build(self.points)
        # A has longitude hit at one GPS point and latitude hit at another GPS point.
        # Therefore trajectory-ID coarse filtering keeps A even though point-ID
        # intersection would not.
        q = QueryWindow(103.9, 104.1, 29.9, 30.1, 90, 210)
        result = idx.query(q)
        self.assertEqual(result.lon_trajectory_ids, {"A"})
        self.assertEqual(result.lat_trajectory_ids, {"A"})
        self.assertEqual(result.time_trajectory_ids, {"A", "B"})
        self.assertEqual(result.candidate_trajectory_ids, ["A"])
        verified = idx.verify(result, idx.combined_root)
        self.assertTrue(verified.verified)
        self.assertEqual(verified.verified_candidate_trajectory_ids, ["A"])
        self.assertEqual(verified.reconstructed_global_root, idx.combined_root_hex)

    def test_tampered_vo_rejected(self):
        idx = ThreeMBTIndex(leaf_capacity=2, fanout=2)
        idx.bulk_build(self.points)
        q = QueryWindow(103.9, 104.1, 29.9, 30.1, 90, 210)
        result = idx.query(q)
        tampered = copy.deepcopy(result)

        def mutate_first_leaf(node):
            if node.get("type") == "leaf":
                node["entries"][0][2] = "EVIL"
                return True
            for c in node.get("children", []):
                if mutate_first_leaf(c):
                    return True
            return False

        self.assertTrue(mutate_first_leaf(tampered.vo_lon["root"]))
        with self.assertRaises(ValueError):
            idx.verify(tampered, idx.combined_root)

    def test_overlapping_hidden_stub_rejected(self):
        idx = ThreeMBTIndex(leaf_capacity=2, fanout=2)
        idx.bulk_build(self.points)
        _tids, proof = idx.mbt_lon.range_query(103.9, 104.1)

        def find_stub(node):
            if node.get("type") == "stub":
                return node
            for c in node.get("children", []):
                hit = find_stub(c)
                if hit:
                    return hit
            return None

        stub = find_stub(proof["root"])
        self.assertIsNotNone(stub)
        stub["min_token"] = 104.0
        stub["max_token"] = 104.0
        with self.assertRaises(ValueError):
            MerkleBTree.reconstruct_range_proof(proof)


if __name__ == "__main__":
    unittest.main()
