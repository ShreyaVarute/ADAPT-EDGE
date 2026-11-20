import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.cost_estimator import CostEstimator


def main():

    estimator = CostEstimator()

    print("\nADAPT-EDGE COST ESTIMATOR")
    print("=" * 50)

    for stage in estimator.get_all_stage_costs():
        print(stage)

    print("\nCan we execute Block 9?")

    decision = estimator.can_continue(
        elapsed_cost_ms=1.0,
        next_stage_cost_ms=1.452,
        budget_ms=5.0
    )

    print(decision)


if __name__ == "__main__":
    main()