import unittest

from baseline.entry_store import PackedEdgeEntryStore
from baseline.road_network import RoadNetwork
from baseline.spatial_tree import BinarySpatialTreeBuilder


def line_road():
    road = RoadNetwork()
    road.max_nid = 3
    road.node_lon = [0.0, 1.0, 2.0, 3.0]
    road.node_lat = [0.0, 0.0, 0.0, 0.0]
    road.node_present = [True] * 4
    road.max_eid = 2
    road.edge_u = [0, 1, 2]
    road.edge_v = [1, 2, 3]
    road.edge_present = [True] * 3
    road.pair_to_eids = {
        road._canonical_pair(0, 1): [0],
        road._canonical_pair(1, 2): [1],
        road._canonical_pair(2, 3): [2],
    }
    return road


class TrajectoryWeightedSpatialTreeTests(unittest.TestCase):
    def test_repeated_traversal_counts_one_trajectory(self):
        store = PackedEdgeEntryStore(1)
        trajectory_a = b"a" * 32
        trajectory_b = b"b" * 32
        store.insert_by_start(0, trajectory_a, 10, 20)
        store.insert_by_start(0, trajectory_a, 30, 40)
        store.insert_by_start(0, trajectory_b, 50, 60)
        self.assertEqual(store.unique_trajectory_counts(), [2, 0])

    def test_split_balances_trajectory_load_instead_of_edge_count(self):
        road = line_road()
        builder = BinarySpatialTreeBuilder(
            road,
            theta=2,
            edge_trajectory_counts=[100, 1, 1],
        )
        left, right = builder._split_vertices(
            road.valid_node_ids(),
            road.valid_edge_ids(),
            (0.0, 0.0, 3.0, 0.0),
        )
        self.assertEqual(left, [0])
        self.assertEqual(right, [1, 2, 3])

    def test_missing_trajectory_weights_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "edge_trajectory_counts"):
            BinarySpatialTreeBuilder(line_road(), theta=2)


if __name__ == "__main__":
    unittest.main()
