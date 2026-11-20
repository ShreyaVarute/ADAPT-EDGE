import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare_strategies import run_model_a, run_model_b


class DummyModel(nn.Module):
    def __init__(self, target_class=3, target_conf=0.95):
        super().__init__()
        self.target_class = target_class
        self.target_conf = target_conf

    def forward(self, x):
        # 10 classes
        logits = torch.zeros(x.size(0), 10)
        # Put high logit on target_class
        logits[:, self.target_class] = 10.0
        return logits


class TestTupleUnpackingRegression(unittest.TestCase):
    """
    Regression test for the tuple-unpacking bug in compare_strategies.py.
    Previously:
        prediction, latency = run_model_a(model_a, images)
    caused `prediction` to be a tuple `(pred_class, confidence)` rather than an integer class,
    leading to `(pred_class, conf) == true_label` always evaluating to False (0% accuracy).
    """

    def test_run_model_a_returns_nested_tuple(self):
        model = DummyModel(target_class=3)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        result = run_model_a(model, dummy_input)
        
        # Verify outer structure: (inner_result, latency_ms)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        (pred_class, confidence), latency = result
        
        # Verify types
        self.assertIsInstance(pred_class, int)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(latency, float)
        
        # Verify values
        self.assertEqual(pred_class, 3)
        self.assertGreater(confidence, 0.9)
        self.assertGreaterEqual(latency, 0.0)

    def test_run_model_b_returns_nested_tuple(self):
        model = DummyModel(target_class=7)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        result = run_model_b(model, dummy_input)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        (pred_class, confidence), latency = result
        
        self.assertIsInstance(pred_class, int)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(latency, float)
        
        self.assertEqual(pred_class, 7)
        self.assertGreater(confidence, 0.9)
        self.assertGreaterEqual(latency, 0.0)

    def test_prediction_equality_with_int_target(self):
        """
        Verify that proper unpacking produces an int that correctly matches an int target.
        """
        model = DummyModel(target_class=5)
        dummy_input = torch.randn(1, 3, 224, 224)
        true_label = 5
        
        (prediction, confidence), latency = run_model_a(model, dummy_input)
        
        # Crucial assertion: prediction must equal true_label
        self.assertEqual(prediction, true_label)
        self.assertTrue(prediction == true_label)

        # Illustrate what went wrong previously:
        raw_result_broken = run_model_a(model, dummy_input)
        broken_prediction, _ = raw_result_broken
        # broken_prediction was (5, 0.999...) which is != 5
        self.assertIsInstance(broken_prediction, tuple)
        self.assertFalse(broken_prediction == true_label)


if __name__ == "__main__":
    unittest.main()

