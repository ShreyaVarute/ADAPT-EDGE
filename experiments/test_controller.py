
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from controller.decision_controller import DecisionController


def main():

    controller = DecisionController(
        model_a_threshold=0.80,
        probe_threshold=0.85,
        max_latency_ms=5.0
    )

    # Example 1: Model A is confident.
    model_a_logits = torch.tensor([
        [0.1, 0.2, 0.3, 4.0, 0.1,
         0.2, 0.1, 0.1, 0.2, 0.1]
    ])

    result = controller.decide(
        model_a_logits=model_a_logits
    )

    print("\nExample 1: Confident Model A")
    print(result)

    # Example 2: Model A is uncertain.
    model_a_logits = torch.zeros(1, 10)

    result = controller.decide(
        model_a_logits=model_a_logits,
        current_latency_ms=0.0,
        next_probe_cost_ms=1.0
    )

    print("\nExample 2: Uncertain Model A")
    print(result)

    # Example 3: Probe is confident.
    probe_logits = torch.tensor([
        [0.1, 0.2, 0.3, 5.0, 0.1,
         0.2, 0.1, 0.1, 0.2, 0.1]
    ])

    result = controller.decide(
        model_a_logits=torch.zeros(1, 10),
        probe_logits=probe_logits,
        current_latency_ms=2.0,
        next_probe_cost_ms=1.0
    )

    print("\nExample 3: Confident Probe")
    print(result)


if __name__ == "__main__":
    main()
