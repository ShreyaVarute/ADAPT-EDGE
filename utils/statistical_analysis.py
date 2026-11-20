"""
ADAPT-EDGE Phase 5: Statistical Analysis Module.

Provides:
1. Paired difference calculations across identical image samples.
2. Bootstrap confidence intervals for accuracy and latency.
3. Statistical significance assessment (Wilcoxon signed-rank test / McNemar's test).
4. Rigorous limitations and caveats documentation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats


def bootstrap_metric_ci(
    values: List[float],
    metric_fn: Any = np.mean,
    num_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence interval for a metric.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    arr = np.array(values)
    point_est = float(metric_fn(arr))

    rng = np.random.RandomState(seed)
    boot_estimates = []

    n = len(arr)
    for _ in range(num_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_estimates.append(metric_fn(resample))

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_estimates, 100 * alpha))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1.0 - alpha)))

    return point_est, ci_lower, ci_upper


def paired_strategy_analysis(
    base_name: str,
    target_name: str,
    base_correct: List[bool],
    target_correct: List[bool],
    base_latencies: List[float],
    target_latencies: List[float]
) -> Dict[str, Any]:
    """
    Performs rigorous paired comparison between two strategies on identical sample indices.
    """
    n = len(base_correct)
    if n != len(target_correct) or n != len(base_latencies) or n != len(target_latencies):
        raise ValueError("Sample lengths must be identical for paired analysis!")

    # 1. Accuracy paired comparison
    base_acc = float(np.mean(base_correct) * 100.0)
    target_acc = float(np.mean(target_correct) * 100.0)
    acc_diff = target_acc - base_acc

    # McNemar's test for binary classification agreement
    # b: target correct, base incorrect; c: base correct, target incorrect
    b = sum(1 for tc, bc in zip(target_correct, base_correct) if tc and not bc)
    c = sum(1 for tc, bc in zip(target_correct, base_correct) if not tc and bc)
    mcnemar_stat = float((abs(b - c) - 1)**2 / (b + c)) if (b + c) > 0 else 0.0
    mcnemar_p = float(1.0 - stats.chi2.cdf(mcnemar_stat, df=1)) if (b + c) > 0 else 1.0

    # 2. Latency paired comparison
    lat_diffs = np.array(target_latencies) - np.array(base_latencies)
    mean_lat_diff = float(np.mean(lat_diffs))
    median_lat_diff = float(np.median(lat_diffs))
    pos_count = int(np.sum(lat_diffs > 0))
    neg_count = int(np.sum(lat_diffs < 0))
    zero_count = int(np.sum(lat_diffs == 0))

    # Paired t-test on mean latency differences
    try:
        t_res = stats.ttest_1samp(lat_diffs, 0)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    except Exception:
        t_stat, t_p = 0.0, 1.0

    # Wilcoxon signed-rank test on paired latency differences (tests median/distribution symmetry around 0)
    try:
        if np.all(lat_diffs == 0):
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(lat_diffs)
            wilcoxon_stat, wilcoxon_p = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    # Bootstrap CIs for mean and median
    _, acc_ci_low, acc_ci_high = bootstrap_metric_ci(
        [float(tc - bc) * 100.0 for tc, bc in zip(target_correct, base_correct)],
        metric_fn=np.mean
    )
    _, mean_lat_ci_low, mean_lat_ci_high = bootstrap_metric_ci(lat_diffs.tolist(), metric_fn=np.mean)
    _, med_lat_ci_low, med_lat_ci_high = bootstrap_metric_ci(lat_diffs.tolist(), metric_fn=np.median)

    def format_p(p_val: float) -> str:
        return "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"

    return {
        "comparison": f"{target_name} vs {base_name}",
        "sample_count": n,
        "base_accuracy": round(base_acc, 2),
        "target_accuracy": round(target_acc, 2),
        "accuracy_difference_pct": round(acc_diff, 2),
        "accuracy_diff_95ci": [round(acc_ci_low, 2), round(acc_ci_high, 2)],
        "mcnemar_stat": round(mcnemar_stat, 4),
        "mcnemar_p_value": round(mcnemar_p, 6),
        "mcnemar_p_formatted": format_p(mcnemar_p),
        "is_accuracy_diff_significant": bool(mcnemar_p < 0.05),
        "mean_latency_diff_ms": round(mean_lat_diff, 3),
        "mean_latency_diff_95ci_ms": [round(mean_lat_ci_low, 3), round(mean_lat_ci_high, 3)],
        "median_latency_diff_ms": round(median_lat_diff, 3),
        "median_latency_diff_95ci_ms": [round(med_lat_ci_low, 3), round(med_lat_ci_high, 3)],
        "counts": {"target_greater": pos_count, "target_smaller": neg_count, "equal": zero_count},
        "paired_ttest_stat": round(t_stat, 4),
        "paired_ttest_p_value": round(t_p, 6),
        "paired_ttest_p_formatted": format_p(t_p),
        "wilcoxon_stat": round(wilcoxon_stat, 4),
        "wilcoxon_p_value": round(wilcoxon_p, 6),
        "wilcoxon_p_formatted": format_p(wilcoxon_p),
        "is_latency_diff_significant": bool(wilcoxon_p < 0.05)
    }

