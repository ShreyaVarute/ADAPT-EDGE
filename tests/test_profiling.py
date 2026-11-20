import unittest
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.profiling import time_callable, LatencyStats, benchmark_function, synchronize


class TestProfiling(unittest.TestCase):
    """
    Unit tests for utils/profiling.py:
    - Timing callable execution
    - Statistical metrics (mean, median, p95, stdev, min, max)
    - Benchmark function execution
    - Synchronization call safety
    """

    def test_time_callable(self):
        def work():
            time.sleep(0.01)
            return "done"

        out, elapsed = time_callable(work)
        self.assertEqual(out, "done")
        self.assertGreater(elapsed, 5.0)  # at least ~5ms

    def test_latency_stats(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = LatencyStats(latencies)
        d = stats.to_dict()

        self.assertEqual(d["count"], 5)
        self.assertEqual(d["mean_latency_ms"], 30.0)
        self.assertEqual(d["median_latency_ms"], 30.0)
        self.assertEqual(d["min_latency_ms"], 10.0)
        self.assertEqual(d["max_latency_ms"], 50.0)
        self.assertGreater(d["stdev_latency_ms"], 0.0)
        self.assertAlmostEqual(d["p95_latency_ms"], 48.0, places=1)

    def test_benchmark_function(self):
        counter = {"n": 0}
        def work():
            counter["n"] += 1
            return counter["n"]

        res = benchmark_function(work, warmup_runs=3, measurement_runs=10, track_gpu_memory=False)
        self.assertEqual(counter["n"], 13)
        self.assertEqual(res["warmup_runs"], 3)
        self.assertEqual(res["measurement_runs"], 10)
        self.assertIn("mean_latency_ms", res)
        self.assertIn("p95_latency_ms", res)

    def test_synchronize_safety(self):
        # Should execute safely on any environment (CPU or GPU)
        synchronize()
        synchronize("cpu")


if __name__ == "__main__":
    unittest.main()

