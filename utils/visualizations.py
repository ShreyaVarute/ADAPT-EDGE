"""
ADAPT-EDGE Phase 5: Publication-Quality Visualization Module.

Generates:
1. accuracy_comparison.png: Bar chart of strategy accuracies.
2. f1_comparison.png: Macro F1 comparison across strategies.
3. latency_comparison.png: Mean, median, and P95 latency comparison.
4. accuracy_vs_latency.png: Pareto frontier trade-off (accuracy vs mean latency).
5. model_b_usage.png: Percentage of full vs partial Model B usage across strategies.
6. probe_distribution.png: Breakdown of probe execution counts (blocks 4, 9, 13).
7. cache_performance.png: Cache hit ratio and computation savings.
8. memory_comparison.png: Peak GPU memory allocation across strategies.
9. confusion_matrices.png: Confusion matrix heatmaps for each evaluated strategy.
10. budget_analysis.png: Latency distribution and budget compliance.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt


def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })


def plot_accuracy_comparison(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    strategies = list(strategy_metrics.keys())
    accuracies = [strategy_metrics[s]["classification"]["accuracy"] for s in strategies]
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#BD10E0", "#9013FE"]

    bars = ax.bar(strategies, accuracies, color=colors[:len(strategies)], width=0.55, edgecolor="#333333", linewidth=1)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy Comparison Across 5 Strategies")
    ax.set_ylim(0, 105)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_f1_comparison(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    strategies = list(strategy_metrics.keys())
    f1s = [strategy_metrics[s]["classification"]["macro_f1"] for s in strategies]
    colors = ["#2E7D32", "#388E3C", "#4CAF50", "#81C784", "#C8E6C9"]

    bars = ax.bar(strategies, f1s, color=colors[:len(strategies)], width=0.55, edgecolor="#2E7D32", linewidth=1)
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Macro F1-Score Comparison Across Strategies")
    ax.set_ylim(0, 1.1)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_latency_comparison(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    strategies = list(strategy_metrics.keys())
    means = [strategy_metrics[s]["latency"]["mean_latency_ms"] for s in strategies]
    medians = [strategy_metrics[s]["latency"]["median_latency_ms"] for s in strategies]
    p95s = [strategy_metrics[s]["latency"]["p95_latency_ms"] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.25

    ax.bar(x - width, means, width, label="Mean Latency", color="#3498DB", edgecolor="#2980B9")
    ax.bar(x, medians, width, label="Median Latency", color="#2ECC71", edgecolor="#27AE60")
    ax.bar(x + width, p95s, width, label="P95 Latency", color="#E74C3C", edgecolor="#C0392B")

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Metrics (Mean, Median, P95)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_vs_latency(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    markers = ["o", "s", "^", "D", "P"]
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#BD10E0", "#9013FE"]

    for i, (name, m) in enumerate(strategy_metrics.items()):
        acc = m["classification"]["accuracy"]
        lat = m["latency"]["mean_latency_ms"]
        p95 = m["latency"]["p95_latency_ms"]
        ax.scatter(lat, acc, s=150, color=colors[i % len(colors)], marker=markers[i % len(markers)], label=name, edgecolors="#222", linewidth=1.5, zorder=4)
        ax.annotate(f" {name}\n ({lat:.1f}ms, {acc:.1f}%)", (lat, acc), textcoords="offset points", xytext=(8, -4), fontsize=9, fontweight="semibold")

    ax.set_xlabel("Mean Latency (ms) [Lower is Better]")
    ax.set_ylabel("Accuracy (%) [Higher is Better]")
    ax.set_title("Pareto Frontier: Accuracy vs. Mean Inference Latency")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_b_usage(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    strategies = list(strategy_metrics.keys())
    full_b = [strategy_metrics[s]["computation"]["full_model_b_pct"] for s in strategies]
    partial_b = [strategy_metrics[s]["computation"]["partial_model_b_pct"] for s in strategies]
    model_a_only = [100.0 - (f + p) for f, p in zip(full_b, partial_b)]

    width = 0.55
    ax.bar(strategies, model_a_only, width, label="Model A Retained", color="#4CAF50", edgecolor="#388E3C")
    ax.bar(strategies, partial_b, width, bottom=model_a_only, label="Partial Model B (Probes)", color="#FF9800", edgecolor="#F57C00")
    ax.bar(strategies, full_b, width, bottom=[a + b for a, b in zip(model_a_only, partial_b)], label="Full Model B Completion", color="#E91E63", edgecolor="#C2185B")

    ax.set_ylabel("Execution Proportion (%)")
    ax.set_title("Computation Path Breakdown Across Strategies")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probe_distribution(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Focus on ADAPT-EDGE strategies
    adapt_strategies = [s for s in strategy_metrics.keys() if "ADAPT-EDGE" in s]
    if not adapt_strategies:
        adapt_strategies = list(strategy_metrics.keys())

    probes = [4, 9, 13]
    x = np.arange(len(probes))
    width = 0.35

    for i, strat in enumerate(adapt_strategies):
        counts = [strategy_metrics[strat]["computation"]["probe_counts"].get(str(p), 0) for p in probes]
        ax.bar(x + (i - 0.5) * width, counts, width, label=strat, edgecolor="#222", linewidth=1)

    ax.set_xlabel("Probe Block Index")
    ax.set_ylabel("Execution Count")
    ax.set_title("Intermediate Probe Execution Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {p}" for p in probes])
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cache_performance(cache_stats: Dict[str, Any], output_path: Path):
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    total = hits + misses

    # 1. Pie chart of Hit vs Miss
    if total > 0:
        ax1.pie([hits, misses], labels=[f"Hits ({hits})", f"Misses ({misses})"], autopct="%1.1f%%", colors=["#2ECC71", "#E74C3C"], startangle=90, explode=(0.08, 0))
    else:
        ax1.text(0.5, 0.5, "No Cache Requests", ha="center", va="center")
    ax1.set_title("Continuation Cache Hit Ratio")

    # 2. Bar of Cache Memory Utilization
    used_mb = cache_stats.get("memory_used_mb", 0.0)
    max_mb = cache_stats.get("max_memory_mb", 64.0)
    ax2.bar(["Memory (MB)"], [used_mb], color="#3498DB", width=0.4, label="Allocated MB")
    ax2.set_ylabel("Memory (MB)")
    ax2.set_ylim(0, max_mb * 1.2)
    ax2.axhline(max_mb, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Max Cap ({max_mb:.0f} MB)")
    ax2.set_title("Cache Memory Footprint")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_memory_comparison(strategy_metrics: Dict[str, Dict[str, Any]], output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    strategies = list(strategy_metrics.keys())
    mems = [strategy_metrics[s]["resource"].get("peak_gpu_memory_mb") or 0.0 for s in strategies]
    colors = ["#34495E", "#2C3E50", "#7F8C8D", "#16A085", "#1ABC9C"]

    bars = ax.bar(strategies, mems, color=colors[:len(strategies)], width=0.55, edgecolor="#222")
    ax.set_ylabel("Peak Allocated GPU Memory (MB)")
    ax.set_title("Peak GPU Memory Consumption Across Strategies")

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.1f} MB", ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrices(strategy_metrics: Dict[str, Dict[str, Any]], class_names: List[str], output_path: Path):
    set_plot_style()
    strategies = list(strategy_metrics.keys())
    n = len(strategies)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), dpi=300)
    if n == 1:
        axes = [axes]

    for i, strat in enumerate(strategies):
        matrix = np.array(strategy_metrics[strat]["classification"]["confusion_matrix"])
        ax = axes[i]
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(strat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("True Class")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels([c[:3] for c in class_names], rotation=45, fontsize=8)
        ax.set_yticklabels([c[:3] for c in class_names], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_budget_analysis(per_image_results: List[Dict[str, Any]], max_latency_ms: float, output_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    # Plot latency distribution across samples for ADAPT-EDGE vs Conventional
    adapt_lats = [r["adapt_edge_cached"]["latency_ms"] for r in per_image_results if "adapt_edge_cached" in r]
    conv_lats = [r["conventional_switching"]["latency_ms"] for r in per_image_results if "conventional_switching" in r]

    bins = np.linspace(0, max_latency_ms * 1.5, 20)
    ax.hist(conv_lats, bins=bins, alpha=0.6, label="Conventional Switching", color="#E67E22", edgecolor="#D35400")
    ax.hist(adapt_lats, bins=bins, alpha=0.6, label="ADAPT-EDGE (Cached)", color="#9B59B6", edgecolor="#8E44AD")

    ax.axvline(max_latency_ms, color="#E74C3C", linestyle="--", linewidth=2, label=f"Max Budget Cap ({max_latency_ms:.1f} ms)")
    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Sample Count")
    ax.set_title("Inference Latency Distribution & Budget Compliance")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

