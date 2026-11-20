import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare_strategies import run_conventional_switching


class DeterministicModel(nn.Module):
    def __init__(self, pred_class, confidence_val=0.99):
        super().__init__()
        self.pred_class = pred_class
        self.confidence_val = confidence_val

    def forward(self, x):
        logits = torch.zeros(x.size(0), 10)
        # To get approx confidence_val after softmax:
        # if max logit is L and others are 0, softmax max is exp(L) / (exp(L) + 9)
        # confidence_val = exp(L) / (exp(L) + 9) => exp(L) = 9 * conf / (1 - conf)
        # L = log(9 * conf / (1 - conf))
        target_logit = torch.log(torch.tensor(9.0 * self.confidence_val / (1.0 - self.confidence_val + 1e-7)))
        logits[:, self.pred_class] = target_logit
        return logits


class TestConventionalSwitchingIntegrity(unittest.TestCase):
    """
    Test suite verifying the mathematical and logical invariant of Conventional Switching:
    1. If Model A is confident (>= A_THRESHOLD), conventional switching MUST select Model A,
       its prediction MUST equal Model A's prediction, and used_b MUST be False.
    2. If Model A is uncertain (< A_THRESHOLD), conventional switching MUST select Model B,
       its prediction MUST equal Model B's prediction, and used_b MUST be True.
    3. Under NO circumstances can conventional switching output a prediction that differs from
       the selected model's prediction for that exact image.
    4. Conventional switching cannot produce more correct predictions than the sum of
       the per-image selected predictions.
    """

    def test_conventional_switching_retains_model_a_when_confident(self):
        # A_THRESHOLD is 0.80 in compare_strategies
        model_a = DeterministicModel(pred_class=2, confidence_val=0.90)  # confident
        model_b = DeterministicModel(pred_class=5, confidence_val=0.99)
        dummy_input = torch.randn(1, 3, 224, 224)

        pred_conv, conf_conv, latency_conv, used_b = run_conventional_switching(
            model_a, model_b, dummy_input
        )

        self.assertFalse(used_b, "Should not switch to Model B when Model A is confident")
        self.assertEqual(pred_conv, 2, "Conventional switching prediction must equal Model A prediction")
        self.assertAlmostEqual(conf_conv, 0.90, places=2)

    def test_conventional_switching_switches_to_model_b_when_uncertain(self):
        model_a = DeterministicModel(pred_class=2, confidence_val=0.50)  # uncertain (< 0.80)
        model_b = DeterministicModel(pred_class=5, confidence_val=0.95)
        dummy_input = torch.randn(1, 3, 224, 224)

        pred_conv, conf_conv, latency_conv, used_b = run_conventional_switching(
            model_a, model_b, dummy_input
        )

        self.assertTrue(used_b, "Should switch to Model B when Model A is uncertain")
        self.assertEqual(pred_conv, 5, "Conventional switching prediction must equal Model B prediction")
        self.assertAlmostEqual(conf_conv, 0.95, places=2)

    def test_conventional_switching_prediction_invariant_across_multiple_samples(self):
        """
        Verify across multiple pseudo-samples that:
        pred_conv == (pred_b if used_b else pred_a)
        """
        import random
        random.seed(42)

        for i in range(20):
            a_class = random.randint(0, 9)
            b_class = random.randint(0, 9)
            a_conf = random.uniform(0.3, 0.99)
            b_conf = random.uniform(0.5, 0.99)

            model_a = DeterministicModel(pred_class=a_class, confidence_val=a_conf)
            model_b = DeterministicModel(pred_class=b_class, confidence_val=b_conf)
            dummy_input = torch.randn(1, 3, 224, 224)

            pred_conv, conf_conv, latency_conv, used_b = run_conventional_switching(
                model_a, model_b, dummy_input
            )

            expected_pred = b_class if used_b else a_class
            self.assertEqual(
                pred_conv, expected_pred,
                f"Sample {i}: Pred {pred_conv} != expected {expected_pred} (used_b={used_b})"
            )


if __name__ == "__main__":
    unittest.main()

