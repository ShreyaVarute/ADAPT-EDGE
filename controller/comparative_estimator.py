import numpy as np
class ComparativeEstimator:
    """
    Compares the performance of different inference strategies.

    Strategies:
    1. Model A only
    2. Model B only
    3. Conventional switching
    4. ADAPT-EDGE
    """

    def __init__(self):
        self.results = {}

    def add_result(
        self,
        strategy,
        correct,
        latency_ms,
        executed_model,
        executed_probes=None
    ):

        if strategy not in self.results:
            self.results[strategy] = {
                "total": 0,
                "correct": 0,
                "latencies": [],
                "model_b_count": 0,
                "probe_counts": {
                    4: 0,
                    9: 0,
                    13: 0
                }
            }

        result = self.results[strategy]

        result["total"] += 1
        result["correct"] += int(correct)
        result["latencies"].append(latency_ms)

        if executed_model == "B":
            result["model_b_count"] += 1

        if executed_probes:
            for block in executed_probes:
                if block in result["probe_counts"]:
                    result["probe_counts"][block] += 1

    def calculate_metrics(self, strategy):

        if strategy not in self.results:
            raise ValueError(
                f"No results found for {strategy}"
            )

        result = self.results[strategy]

        total = result["total"]

        if total == 0:
            raise ValueError("No samples evaluated")

        latencies = sorted(result["latencies"])

        mean_latency = sum(latencies) / total

        p95_latency = float(np.percentile(latencies, 95))

        accuracy = (
            result["correct"] / total
        ) * 100

        return {
            "strategy": strategy,
            "total_images": total,
            "correct_predictions": result["correct"],
            "accuracy": accuracy,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "model_b_usage_percent": (
                result["model_b_count"] / total
            ) * 100,
            "probe_counts": result["probe_counts"]
        }

    def compare_all(self):

        return {
            strategy: self.calculate_metrics(strategy)
            for strategy in self.results
        }

    def print_comparison(self):

        comparison = self.compare_all()

        print("\n" + "=" * 85)
        print("ADAPT-EDGE STRATEGY COMPARISON")
        print("=" * 85)

        print(
            f"{'Strategy':<25}"
            f"{'Accuracy':>12}"
            f"{'Mean (ms)':>12}"
            f"{'P95 (ms)':>12}"
            f"{'Model B %':>12}"
        )

        print("-" * 85)

        for name, metrics in comparison.items():

            print(
                f"{name:<25}"
                f"{metrics['accuracy']:>11.2f}%"
                f"{metrics['mean_latency_ms']:>12.3f}"
                f"{metrics['p95_latency_ms']:>12.3f}"
                f"{metrics['model_b_usage_percent']:>11.2f}%"
            )

        return comparison
