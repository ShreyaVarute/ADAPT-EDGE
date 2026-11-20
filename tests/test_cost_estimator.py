import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.cost_estimator import CostEstimator, DEFAULT_SEGMENT_COSTS_MS


class TestCostEstimator(unittest.TestCase):
    """
    Unit tests for controller/cost_estimator.py:
    - Incremental segment calculation (direct vs compound segments)
    - Cumulative backbone costs respecting sequential dependency
    - Probe evaluation costs (feature transition + head)
    - Completion cost
    - Latency budget evaluation (can_continue)
    - Sequential validity (invalid reverse or zero transitions)
    """

    def setUp(self):
        self.estimator = CostEstimator()

    def test_direct_incremental_segment(self):
        # (0, 4) should match direct calibration
        cost_0_4 = self.estimator.get_incremental_segment_cost(0, 4)
        self.assertAlmostEqual(cost_0_4, DEFAULT_SEGMENT_COSTS_MS[(0, 4)], places=3)

        cost_4_9 = self.estimator.get_incremental_segment_cost(4, 9)
        self.assertAlmostEqual(cost_4_9, DEFAULT_SEGMENT_COSTS_MS[(4, 9)], places=3)

    def test_compound_incremental_segment(self):
        # Moving from block 4 to block 13 requires: (4 -> 9) + (9 -> 13)
        expected = DEFAULT_SEGMENT_COSTS_MS[(4, 9)] + DEFAULT_SEGMENT_COSTS_MS[(9, 13)]
        cost_4_13 = self.estimator.get_incremental_segment_cost(4, 13)
        self.assertAlmostEqual(cost_4_13, expected, places=3)

    def test_cumulative_backbone_cost(self):
        # Cumulative up to 13 = (0->4) + (4->9) + (9->13)
        expected = (
            DEFAULT_SEGMENT_COSTS_MS[(0, 4)]
            + DEFAULT_SEGMENT_COSTS_MS[(4, 9)]
            + DEFAULT_SEGMENT_COSTS_MS[(9, 13)]
        )
        cum_13 = self.estimator.get_cumulative_backbone_cost(13)
        self.assertAlmostEqual(cum_13, expected, places=3)

    def test_probe_evaluation_cost(self):
        # Evaluating probe 9 starting at block 4 = segment (4->9) + head cost (9)
        head_cost_9 = self.estimator.probe_head_costs[9]
        expected = DEFAULT_SEGMENT_COSTS_MS[(4, 9)] + head_cost_9
        eval_cost = self.estimator.get_probe_evaluation_cost(4, 9)
        self.assertAlmostEqual(eval_cost, expected, places=3)

    def test_invalid_transitions_raise(self):
        # End <= start must raise ValueError
        with self.assertRaises(ValueError):
            self.estimator.get_incremental_segment_cost(9, 4)
        with self.assertRaises(ValueError):
            self.estimator.get_incremental_segment_cost(4, 4)

    def test_budget_checking(self):
        check = self.estimator.can_continue(elapsed_cost_ms=2.0, next_stage_cost_ms=1.5, budget_ms=5.0)
        self.assertTrue(check["can_continue"])
        self.assertAlmostEqual(check["remaining_budget_ms"], 3.0, places=2)

        check_exhausted = self.estimator.can_continue(elapsed_cost_ms=4.0, next_stage_cost_ms=1.5, budget_ms=5.0)
        self.assertFalse(check_exhausted["can_continue"])

    def test_get_all_stage_costs(self):
        stages = self.estimator.get_all_stage_costs()
        self.assertIsInstance(stages, list)
        self.assertGreater(len(stages), 0)
        for s in stages:
            self.assertIn("stage", s)
            self.assertIn("incremental_cost_ms", s)
            self.assertIn("cumulative_backbone_ms", s)


if __name__ == "__main__":
    unittest.main()

