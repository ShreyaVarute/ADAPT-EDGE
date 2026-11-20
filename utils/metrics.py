"""
Classification and evaluation metrics for ADAPT-EDGE.
Supports accuracy, per-class and macro precision, recall, F1, and confusion matrix.
Handles edge cases (zero divisions, single-class subsets, unobserved classes).
"""

from typing import Dict, List, Optional, Sequence, Union
import numpy as np


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 10
) -> np.ndarray:
    """
    Compute a square confusion matrix of size (num_classes, num_classes).
    Rows correspond to ground-truth classes; columns correspond to predicted classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1
        else:
            raise ValueError(f"Label out of bounds: true={t}, pred={p} for num_classes={num_classes}")

    return matrix


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 10,
    class_names: Optional[List[str]] = None,
    zero_division: float = 0.0
) -> Dict[str, Union[float, int, Dict[str, float], List[List[int]]]]:
    """
    Compute overall and per-class classification metrics:
      - accuracy
      - macro precision, macro recall, macro F1
      - per-class accuracy, precision, recall, F1, and support
      - confusion matrix (as list of lists for JSON serializability)
    
    Zero-division handling: when precision or recall denominator is 0,
    the metric evaluates to zero_division (default 0.0).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    total_samples = len(y_true)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "macro_precision": zero_division,
            "macro_recall": zero_division,
            "macro_f1": zero_division,
            "class_metrics": {},
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)]
        }

    matrix = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total_correct = int(np.trace(matrix))
    accuracy = float((total_correct / total_samples) * 100.0)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError(f"class_names length ({len(class_names)}) != num_classes ({num_classes})")

    class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        tp = float(matrix[i, i])
        fp = float(np.sum(matrix[:, i]) - tp)
        fn = float(np.sum(matrix[i, :]) - tp)
        support = int(np.sum(matrix[i, :]))

        # Precision = TP / (TP + FP)
        prec_denom = tp + fp
        precision = float(tp / prec_denom) if prec_denom > 0 else float(zero_division)

        # Recall = TP / (TP + FN)
        rec_denom = tp + fn
        recall = float(tp / rec_denom) if rec_denom > 0 else float(zero_division)

        # F1 = 2 * P * R / (P + R)
        f1_denom = precision + recall
        f1 = float(2.0 * precision * recall / f1_denom) if f1_denom > 0 else float(zero_division)

        # Class accuracy (accuracy on samples of this ground truth class)
        cls_acc = float((tp / support) * 100.0) if support > 0 else float(zero_division)

        name = class_names[i]
        class_metrics[name] = {
            "class_id": i,
            "support": support,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "accuracy": round(cls_acc, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "total_samples": total_samples,
        "correct_predictions": total_correct,
        "accuracy": round(accuracy, 2),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "class_metrics": class_metrics,
        "confusion_matrix": matrix.tolist()
    }

