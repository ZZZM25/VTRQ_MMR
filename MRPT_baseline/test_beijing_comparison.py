import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import beijing_baseline as beijing


class BeijingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.baseline_dir = root / "baseline"
        self.ours_dir = root / "ours"
        self.baseline_dir.mkdir()
        self.ours_dir.mkdir()
        settings = patch.multiple(
            beijing,
            OUTPUT_DIR=self.baseline_dir,
            OURS_CANDIDATE_MATRIX_FILE=self.ours_dir / "candidate_count_matrix.csv",
            OURS_FINAL_MATRIX_FILE=self.ours_dir / "final_result_count_matrix.csv",
            OURS_RESULT_FILE=self.ours_dir / "cross_range_results.csv",
        )
        settings.start()
        self.addCleanup(settings.stop)
        bounds = beijing.make_square(116.475, 39.92, 2.0)
        self.query = dict(zip(
            ("query_min_lon", "query_min_lat", "query_max_lon", "query_max_lat"),
            bounds,
        ))
        self.query.update(
            spatial_side_km=2.0, temporal_label="1h",
            query_start=1239679020, query_end=1239682620,
        )
        self.write_result(self.baseline_dir, self.query)
        self.write_result(self.ours_dir, self.query)

    def write_result(self, directory, query, candidate=2, final=2):
        with (directory / "cross_range_results.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(query))
            writer.writeheader()
            writer.writerow(query)
        for filename, count in (
            ("candidate_count_matrix.csv", candidate),
            ("final_result_count_matrix.csv", final),
        ):
            with (directory / filename).open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["spatial_side_km", query["temporal_label"]])
                writer.writerow([query["spatial_side_km"], count])

    def compare(self):
        output = io.StringIO()
        with redirect_stdout(output):
            beijing.compare_with_ours_counts()
        return output.getvalue()

    def test_different_center_skips_unrelated_count_mismatch(self):
        query = dict(self.query)
        query["query_min_lon"] -= 0.1
        query["query_max_lon"] -= 0.1
        self.write_result(self.ours_dir, query, candidate=10, final=10)
        output = self.compare()
        self.assertIn("saved query bounds/time differ", output)
        self.assertIn("Count check skipped", output)
        self.assertNotIn("MATCH", output)

    def test_different_time_skips(self):
        query = dict(self.query)
        query["query_start"] += 60
        query["query_end"] += 60
        self.write_result(self.ours_dir, query)
        self.assertIn("saved query bounds/time differ", self.compare())

    def test_coordinate_rounding_allows_comparison(self):
        query = {
            key: f"{value:.10f}" if key.endswith(("_lon", "_lat")) else value
            for key, value in self.query.items()
        }
        self.write_result(self.ours_dir, query)
        self.assertIn("All 1 comparable saved cells match Ours.", self.compare())

    def test_same_query_still_rejects_candidate_or_final_mismatch(self):
        for candidate, final in ((10, 2), (2, 10)):
            with self.subTest(candidate=candidate, final=final):
                self.write_result(self.ours_dir, self.query, candidate, final)
                with self.assertRaisesRegex(RuntimeError, "counts differ"):
                    self.compare()

    def test_missing_count_cell_does_not_report_match(self):
        self.write_result(self.ours_dir, self.query, candidate="")
        output = self.compare()
        self.assertIn("count matrix cell missing", output)
        self.assertNotIn("MATCH", output)

    def test_missing_ours_query_does_not_report_match(self):
        query = dict(self.query, temporal_label="4h", query_end=1239693420)
        self.write_result(self.ours_dir, query)
        output = self.compare()
        self.assertIn("no corresponding Ours query", output)
        self.assertNotIn("MATCH", output)

    def test_compares_saved_cells_after_runtime_ranges_change(self):
        # Standalone compare must use the recorded experiment, not today's config.
        with patch.object(beijing, "SPATIAL_SIDES", [5.0]), patch.object(
            beijing, "TEMPORAL_RANGES", [("24h", 86400)]
        ):
            self.assertIn("All 1 comparable saved cells match Ours.", self.compare())


if __name__ == "__main__":
    unittest.main()
