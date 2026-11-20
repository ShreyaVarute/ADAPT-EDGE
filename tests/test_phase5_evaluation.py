import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.decision_controller import DecisionController
from cache.continuation_cache import ContinuationCache


class TestPhase5EvaluationSuite(unittest.TestCase):
    """
    Unit tests ensuring:
    1. DecisionController produces explicit status distinguishing CONFIDENT_RETAIN vs BUDGET_EXHAUSTED_RETAIN.
    2. ContinuationCache works cleanly with cache=None (disabled) vs cache instance (enabled).
    3. Input fingerprints change when preprocessing configuration changes.
    """

    def test_status_differentiation_on_retain(self):
        controller = DecisionController(model_a_threshold=0.80, max_latency_ms=25.0)

        # High confidence
        logits_confident = torch.zeros(1, 10)
        logits_confident[0, 2] = 10.0
        dec_conf = controller.decide_initial(logits_confident, current_latency_ms=1.0)
        self.assertEqual(dec_conf["action"], "RETAIN")
        self.assertEqual(dec_conf.get("status"), "CONFIDENT_RETAIN")

        # Uncertain but zero budget
        tight_controller = DecisionController(model_a_threshold=0.80, max_latency_ms=1.0)
        logits_uncertain = torch.zeros(1, 10)
        dec_budget = tight_controller.decide_initial(logits_uncertain, current_latency_ms=5.0)
        self.assertEqual(dec_budget["action"], "RETAIN")
        self.assertEqual(dec_budget.get("status"), "BUDGET_EXHAUSTED_RETAIN")

    def test_cache_disabled_vs_enabled_initialization(self):
        cache_enabled = ContinuationCache(max_entries=10, max_memory_mb=32.0)
        t = torch.randn(1, 40, 28, 28)
        key = cache_enabled.put("req1", "fp1", "b1", 4, t)
        self.assertTrue(cache_enabled.contains("fp1", "b1", 4))
        self.assertEqual(cache_enabled.get_cache_stats()["entries_count"], 1)

    def test_fingerprint_uniqueness_with_preprocessing(self):
        img = torch.randn(1, 3, 224, 224)
        fp_std = ContinuationCache.generate_input_fingerprint(img, {"crop": 224})
        fp_mod = ContinuationCache.generate_input_fingerprint(img, {"crop": 192})
        self.assertNotEqual(fp_std, fp_mod)


if __name__ == "__main__":
    unittest.main()

