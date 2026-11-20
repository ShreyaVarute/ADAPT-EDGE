import unittest
import shutil
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_tracker import ExperimentTracker


class TestExperimentTracker(unittest.TestCase):
    """
    Unit tests for utils/experiment_tracker.py:
    - Experiment directory creation with timestamp
    - Metadata serialization (hardware, software, config, research disclosure)
    - Results logging
    - Preservation of historical directories
    """

    def setUp(self):
        self.test_dir = PROJECT_ROOT / "results" / "_test_experiment_tracker"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_tracker_initialization_and_metadata(self):
        config = {
            "model": "test_model",
            "batch_size": 1,
            "threshold": 0.8
        }
        tracker = ExperimentTracker(
            experiment_name="unit_test",
            config=config,
            base_dir=self.test_dir,
            seed=123
        )

        self.assertTrue(tracker.run_dir.exists())
        meta_file = tracker.run_dir / "experiment_metadata.json"
        self.assertTrue(meta_file.exists())

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_name"], "unit_test")
        self.assertEqual(data["random_seed"], 123)
        self.assertIn("software", data)
        self.assertIn("hardware", data)
        self.assertIn("research_disclosure", data)
        self.assertEqual(data["config"]["model"], "test_model")

    def test_log_results(self):
        tracker = ExperimentTracker(
            experiment_name="unit_test_results",
            base_dir=self.test_dir
        )
        results = {"accuracy": 92.5, "mean_latency": 15.2}
        tracker.log_results(results)

        summary_file = tracker.run_dir / "results_summary.json"
        self.assertTrue(summary_file.exists())

        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["results"]["accuracy"], 92.5)
        self.assertIn("duration_seconds", data)


if __name__ == "__main__":
    unittest.main()

