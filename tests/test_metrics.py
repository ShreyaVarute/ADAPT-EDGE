import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import compute_metrics, compute_confusion_matrix


class TestMetrics(unittest.TestCase):
    """
    Unit tests for utils/metrics.py:
    - Confusion matrix calculation
    - Overall accuracy
    - Per-class precision, recall, F1, support
    - Macro precision, recall, F1
    - Edge cases: zero samples, unobserved classes, zero division handling
    """

    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        m = compute_metrics(y_true, y_pred, num_classes=10)
        self.assertEqual(m["accuracy"], 100.0)
        self.assertEqual(m["macro_precision"], 1.0)
        self.assertEqual(m["macro_recall"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["correct_predictions"], 10)

    def test_all_incorrect_predictions(self):
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        
        m = compute_metrics(y_true, y_pred, num_classes=2)
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["correct_predictions"], 0)
        # Class 0: TP=0, FP=0, FN=4 -> Recall=0.0, Precision=0.0 (zero division)
        # Class 1: TP=0, FP=4, FN=0 -> Precision=0.0, Recall=0.0
        self.assertEqual(m["macro_precision"], 0.0)
        self.assertEqual(m["macro_recall"], 0.0)
        self.assertEqual(m["macro_f1"], 0.0)

    def test_imbalanced_known_calculation(self):
        # 4 samples:
        # True:  [0, 0, 1, 1]
        # Pred:  [0, 1, 1, 1]
        # Matrix:
        # Class 0: TP=1, FP=0, FN=1 -> Prec=1/1=1.0, Rec=1/2=0.5, F1=2*(1*0.5)/(1.5) = 2/3 = 0.6667
        # Class 1: TP=2, FP=1, FN=0 -> Prec=2/3=0.6667, Rec=2/2=1.0, F1=2*(0.6667*1)/(1.6667) = 0.8
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        
        m = compute_metrics(y_true, y_pred, num_classes=2, class_names=["cat", "dog"])
        self.assertEqual(m["accuracy"], 75.0)
        self.assertEqual(m["correct_predictions"], 3)
        self.assertAlmostEqual(m["class_metrics"]["cat"]["precision"], 1.0, places=3)
        self.assertAlmostEqual(m["class_metrics"]["cat"]["recall"], 0.5, places=3)
        self.assertAlmostEqual(m["class_metrics"]["dog"]["precision"], 0.6667, places=3)
        self.assertAlmostEqual(m["class_metrics"]["dog"]["recall"], 1.0, places=3)

    def test_empty_input_edge_case(self):
        m = compute_metrics([], [], num_classes=5)
        self.assertEqual(m["total_samples"], 0)
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["macro_f1"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_metrics([0, 1], [0], num_classes=2)

    def test_out_of_bounds_label_raises(self):
        with self.assertRaises(ValueError):
            compute_metrics([0, 5], [0, 1], num_classes=3)


if __name__ == "__main__":
    unittest.main()

