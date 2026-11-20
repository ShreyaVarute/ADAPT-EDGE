
from controller.comparative_estimator import ComparativeEstimator


def main():

    estimator = ComparativeEstimator()

    # Model A strategy
    estimator.add_result(
        "Model A",
        correct=True,
        latency_ms=3.8,
        executed_model="A"
    )

    estimator.add_result(
        "Model A",
        correct=False,
        latency_ms=4.0,
        executed_model="A"
    )

    # ADAPT-EDGE strategy
    estimator.add_result(
        "ADAPT-EDGE",
        correct=True,
        latency_ms=4.2,
        executed_model="B",
        executed_probes=[4, 9]
    )

    estimator.add_result(
        "ADAPT-EDGE",
        correct=True,
        latency_ms=3.5,
        executed_model="A"
    )

    estimator.print_comparison()


if __name__ == "__main__":
    main()
