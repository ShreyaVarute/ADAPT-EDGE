"""
Cost Estimator for ADAPT-EDGE Sequential Partial Execution.

Estimates incremental computation and latency costs across sequential Model B execution stages:
  - Stage 0 -> 4: Initial feature extraction through Block 4.
  - Stage 4 -> 9: Segment feature continuation from Block 5 through Block 9.
  - Stage 9 -> 13: Segment feature continuation from Block 10 through Block 13.
  - Stage 13 -> 16: Feature completion (Blocks 14-16) + Pooling + Classifier.

Probe heads:
  - Probe head 4, Probe head 9, Probe head 13 execution costs.

Research Discipline:
  - Sequential dependency is strictly preserved: To reach Block 9, Blocks 0-4 MUST have been run.
  - Sequential continuation does NOT recompute earlier blocks.
  - Incremental costs reflect true segment execution rather than noisy end-to-end differences.
"""

from typing import Dict, Any, List, Optional, Tuple


# Default calibrated segment latency costs measured on NVIDIA RTX 3050 Laptop GPU (ms)
# Derived from actual sequential partial execution benchmarks
DEFAULT_SEGMENT_COSTS_MS = {
    (0, 4): 0.9845,     # Input to Block 4 features
    (4, 9): 1.4518,     # Block 5 through Block 9 features
    (9, 13): 1.0544,    # Block 10 through Block 13 features
    (13, 16): 0.5044,   # Block 14 through Block 16 + avgpool + classifier
}

# Individual probe head forward pass costs (avgpool + linear)
DEFAULT_PROBE_HEAD_COSTS_MS = {
    4: 0.050,
    9: 0.050,
    13: 0.050
}

# Baseline initial Model A execution cost (ms)
DEFAULT_MODEL_A_COST_MS = 3.8889


class CostEstimator:
    """
    Cost estimator tracking cumulative execution costs and incremental segment costs
    for sequential MobileNetV3-Large execution with intermediate probe heads.
    """

    def __init__(
        self,
        segment_costs_ms: Optional[Dict[Tuple[int, int], float]] = None,
        probe_head_costs_ms: Optional[Dict[int, float]] = None,
        model_a_cost_ms: float = DEFAULT_MODEL_A_COST_MS
    ):
        self.segment_costs = segment_costs_ms or dict(DEFAULT_SEGMENT_COSTS_MS)
        self.probe_head_costs = probe_head_costs_ms or dict(DEFAULT_PROBE_HEAD_COSTS_MS)
        self.model_a_cost_ms = model_a_cost_ms

        # Precompute canonical sequential checkpoints
        # Checkpoints: 0 (input), 4 (probe 1), 9 (probe 2), 13 (probe 3), 16 (final output)
        self.checkpoints = [0, 4, 9, 13, 16]

    def get_incremental_segment_cost(self, start_block: int, end_block: int) -> float:
        """
        Estimate additional latency required to compute features from start_block to end_block.
        If a direct segment is defined, returns its cost.
        Otherwise sums the intermediate subsegments along the sequential backbone.
        """
        if end_block <= start_block:
            raise ValueError(f"end_block ({end_block}) must be greater than start_block ({start_block})")

        key = (start_block, end_block)
        if key in self.segment_costs:
            return float(self.segment_costs[key])

        # Sum constituent sequential segments
        total = 0.0
        cur = start_block
        sorted_cp = sorted([cp for cp in self.checkpoints if start_block <= cp <= end_block])
        if start_block not in sorted_cp:
            sorted_cp.insert(0, start_block)
        if end_block not in sorted_cp:
            sorted_cp.append(end_block)

        for i in range(len(sorted_cp) - 1):
            sub_key = (sorted_cp[i], sorted_cp[i + 1])
            if sub_key in self.segment_costs:
                total += self.segment_costs[sub_key]
            else:
                raise ValueError(f"Undefined sequential segment transition {sub_key}")

        return float(total)

    def get_cumulative_backbone_cost(self, end_block: int) -> float:
        """
        Calculate total feature extraction cost from raw input (block 0) up to end_block.
        Strictly reflects sequential dependency: all blocks 0..end_block must be computed.
        """
        if end_block == 0:
            return 0.0
        return self.get_incremental_segment_cost(0, end_block)

    def get_probe_evaluation_cost(self, current_block: int, target_probe_block: int) -> float:
        """
        Total incremental cost to reach and evaluate a probe head:
          = (incremental feature continuation from current_block to target_probe_block)
            + (probe head classification forward pass)
        """
        feature_cost = self.get_incremental_segment_cost(current_block, target_probe_block)
        head_cost = self.probe_head_costs.get(target_probe_block, 0.05)
        return float(feature_cost + head_cost)

    def get_remaining_completion_cost(self, current_block: int) -> float:
        """
        Remaining incremental cost to finish the entire Model B from current_block (e.g. 13 -> 16).
        """
        if current_block >= 16:
            return 0.0
        return self.get_incremental_segment_cost(current_block, 16)

    def can_continue(
        self,
        elapsed_cost_ms: float,
        next_stage_cost_ms: float,
        budget_ms: float
    ) -> Dict[str, Any]:
        """
        Evaluate if executing next stage is feasible within a latency budget.
        """
        remaining_budget = budget_ms - elapsed_cost_ms
        return {
            "can_continue": (next_stage_cost_ms <= remaining_budget),
            "elapsed_cost_ms": round(elapsed_cost_ms, 4),
            "next_stage_cost_ms": round(next_stage_cost_ms, 4),
            "remaining_budget_ms": round(max(0.0, remaining_budget), 4),
            "budget_ms": round(budget_ms, 4)
        }

    def get_all_stage_costs(self) -> List[Dict[str, Any]]:
        """
        Returns structured overview of all sequential stages, both incremental and cumulative.
        """
        stages = [
            {"from": 0, "to": 4, "type": "backbone_segment", "description": "Blocks 0-4 feature extraction"},
            {"from": 4, "to": 4, "type": "probe_head", "description": "Probe 4 linear classifier"},
            {"from": 4, "to": 9, "type": "backbone_segment", "description": "Blocks 5-9 feature continuation"},
            {"from": 9, "to": 9, "type": "probe_head", "description": "Probe 9 linear classifier"},
            {"from": 9, "to": 13, "type": "backbone_segment", "description": "Blocks 10-13 feature continuation"},
            {"from": 13, "to": 13, "type": "probe_head", "description": "Probe 13 linear classifier"},
            {"from": 13, "to": 16, "type": "backbone_completion", "description": "Blocks 14-16 + pool + classifier"}
        ]

        result = []
        cum_backbone = 0.0

        for s in stages:
            if s["type"].startswith("backbone"):
                inc = self.get_incremental_segment_cost(s["from"], s["to"])
                cum_backbone += inc
                result.append({
                    "stage": f"Blocks {s['from']}->{s['to']}",
                    "type": s["type"],
                    "incremental_cost_ms": round(inc, 4),
                    "cumulative_backbone_ms": round(cum_backbone, 4),
                    "description": s["description"]
                })
            else:
                block = s["from"]
                cost = self.probe_head_costs.get(block, 0.05)
                result.append({
                    "stage": f"Probe {block} Head",
                    "type": s["type"],
                    "incremental_cost_ms": round(cost, 4),
                    "cumulative_backbone_ms": round(cum_backbone, 4),
                    "description": s["description"]
                })

        return result
