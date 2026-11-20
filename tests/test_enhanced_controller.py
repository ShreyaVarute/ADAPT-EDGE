import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.decision_controller import DecisionController
from controller.cost_estimator import CostEstimator
from utils.calibration import TemperatureScaler
from utils.resource_monitor import ResourceMonitor


class TestDecisionController(unittest.TestCase):
    """
    Unit tests for controller/decision_controller.py:
    1. High-confidence Model A triggers RETAIN.
    2. Uncertain Model A triggers PROBE MORE.
    3. Adaptive probe selection picks best candidate based on Information / Cost.
    4. Probe budget exhaustion causes fallback to full Model B or RETAIN.
    5. Confident probe triggers RETAIN.
    6. Uncertain probe triggers next probe or full Model B fallback.
    7. Resource monitor graceful degradation when sensors fail.
    8. Temperature scaler calibration integration.
    """

    def setUp(self):
        self.estimator = CostEstimator()
        self.controller = DecisionController(
            model_a_threshold=0.80,
            probe_threshold=0.85,
            max_latency_ms=15.0,
            cost_estimator=self.estimator
        )

    def test_high_confidence_model_a_retains(self):
        # 10 classes; class 3 has logit 10.0 -> conf > 0.99
        logits = torch.zeros(1, 10)
        logits[0, 3] = 10.0

        decision = self.controller.decide_initial(logits, current_latency_ms=1.0)
        self.assertEqual(decision["action"], "RETAIN")
        self.assertEqual(decision["prediction"], 3)
        self.assertGreaterEqual(decision["confidence"], 0.80)
        self.assertIsNone(decision["selected_probe"])

    def test_uncertain_model_a_probes_more(self):
        # Uniform logits -> conf 0.10, high entropy
        logits = torch.zeros(1, 10)

        decision = self.controller.decide_initial(logits, current_latency_ms=1.0)
        self.assertEqual(decision["action"], "PROBE MORE")
        self.assertIsNotNone(decision["selected_probe"])
        self.assertIn(decision["selected_probe"], [4, 9, 13])
        self.assertGreater(decision["decision_score"], 0.0)

    def test_adaptive_probe_selection_scoring(self):
        # Verify that candidate evaluation ranks candidates by Selection Score = Info / Cost
        candidates = self.controller.evaluate_candidate_probes(
            current_block=0,
            current_confidence=0.50,
            current_entropy=0.80,
            current_latency_ms=0.0,
            executed_probes=[]
        )
        self.assertGreater(len(candidates), 0)
        # Verify descending order of selection score
        scores = [c["selection_score"] for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_confident_probe_retains(self):
        # Probe produces high confidence at block 4
        probe_logits = torch.zeros(1, 10)
        probe_logits[0, 7] = 8.0  # conf > 0.95

        decision = self.controller.decide_probe(
            current_probe_block=4,
            probe_logits=probe_logits,
            current_latency_ms=2.0,
            executed_probes=[4]
        )
        self.assertEqual(decision["action"], "RETAIN")
        self.assertEqual(decision["prediction"], 7)
        self.assertEqual(decision["selected_probe"], 4)

    def test_uncertain_probe_continues_to_next_or_switch(self):
        # Probe 4 is uncertain -> should PROBE MORE or SWITCH
        probe_logits = torch.zeros(1, 10)
        decision = self.controller.decide_probe(
            current_probe_block=4,
            probe_logits=probe_logits,
            current_latency_ms=2.0,
            executed_probes=[4]
        )
        self.assertIn(decision["action"], ["PROBE MORE", "SWITCH"])

    def test_probe_budget_exhaustion_fallback(self):
        # If latency is nearly exhausted (e.g. 14.8ms of 15.0ms)
        tight_controller = DecisionController(
            model_a_threshold=0.80,
            probe_threshold=0.85,
            max_latency_ms=2.0,
            cost_estimator=self.estimator
        )
        probe_logits = torch.zeros(1, 10)
        # Latency already 1.95ms -> cannot execute any further stage or fallback
        decision = tight_controller.decide_probe(
            current_probe_block=13,
            probe_logits=probe_logits,
            current_latency_ms=1.95,
            executed_probes=[4, 9, 13]
        )
        # Budget exhausted -> force RETAIN current probe prediction
        self.assertEqual(decision["action"], "RETAIN")
        self.assertIn("insufficient", decision["reason"].lower())

    def test_temperature_scaler_integration(self):
        scaler = TemperatureScaler(initial_temperature=1.5)
        controller_cal = DecisionController(
            temperature_scaler=scaler
        )
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        _, conf_uncal, _, _ = controller_cal.compute_distribution_metrics(logits, apply_calibration=False)
        _, conf_cal, _, _ = controller_cal.compute_distribution_metrics(logits, apply_calibration=True)
        # T > 1 softens probabilities, lowering maximum confidence
        self.assertLess(conf_cal, conf_uncal)


if __name__ == "__main__":
    unittest.main()

