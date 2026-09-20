import hashlib
import struct
import unittest

from three_mbt.trajectory_id import compute_paper_trajectory_id


class TrajectoryIDTests(unittest.TestCase):
    def test_node_path_start_end_hash(self):
        nodes = [10, 20, 20, 30]
        start, end = 100, 999
        h = hashlib.sha256()
        for x in nodes:
            h.update(struct.pack(">I", x))
        h.update(struct.pack(">I", start))
        h.update(struct.pack(">I", end))
        self.assertEqual(
            compute_paper_trajectory_id(nodes, start, end),
            h.hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
