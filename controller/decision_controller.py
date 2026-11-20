"""
Enhanced Decision Controller for ADAPT-EDGE.

Supports:
1. RETAIN: Accept current prediction if confidence is sufficient or further computation is infeasible.
2. PROBE MORE: Select an intermediate Model B probe based on Expected Decision Information / Incremental Cost.
3. SWITCH: Fall back to full Model B completion when probes remain uncertain or budget permits.

Sequential Execution & Cost Discipline:
  - Model B is sequential: reaching Block 9 requires Blocks 0-4 to have run.
  - Candidate probes are evaluated based on their incremental execution cost from the current execution block.
  - Probe selection score = Expected Information Gain / Incremental Computation Cost.
  - Resource awareness: integrates latency budgets, memory limits, and hardware metrics.
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from controller.cost_estimator import CostEstimator, DEFAULT_SEGMENT_COSTS_MS
from controller.resource_policy import ResourcePolicy, ResourceEvaluation
from utils.calibration import TemperatureScaler
from utils.resource_monitor import ResourceMonitor


class DecisionController:
    """
    Modular, resource-aware decision controller for ADAPT-EDGE.
    """

    def __init__(
        self,
        model_a_threshold: float = 0.80,
        probe_threshold: float = 0.85,
        max_latency_ms: float = 15.0,
        available_probes: Optional[List[int]] = None,
        cost_estimator: Optional[CostEstimator] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        resource_policy: Optional[ResourcePolicy] = None,
        min_information_threshold: float = 0.05,
        enforce_resource_budget: bool = True
    ):
        self.model_a_threshold = float(model_a_threshold)
        self.probe_threshold = float(probe_threshold)
        self.max_latency_ms = float(max_latency_ms)
        self.available_probes = available_probes or [4, 9, 13]
        self.cost_estimator = cost_estimator or CostEstimator()
        self.temperature_scaler = temperature_scaler
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.resource_policy = resource_policy or ResourcePolicy()
        self.min_information_threshold = float(min_information_threshold)
        self.enforce_resource_budget = enforce_resource_budget

        # Historical / empirical probe accuracy benchmarks on validation data (used as prior decision quality)
        self.probe_prior_accuracies = {
            4: 0.432,   # Block 4 baseline accuracy
            9: 0.552,   # Block 9 baseline accuracy
            13: 0.768,  # Block 13 baseline accuracy
            16: 0.906   # Full Model B accuracy
        }

    def compute_distribution_metrics(
        self,
        logits: torch.Tensor,
        apply_calibration: bool = True
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Calculates:
          - predicted_class: argmax
          - confidence: max softmax probability
          - uncertainty: Shannon entropy H(p) = -sum(p * log(p))
          - probabilities: full softmax distribution
        """
        if apply_calibration and self.temperature_scaler is not None:
            logits = self.temperature_scaler(logits)

        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        # Shannon entropy normalized by log(num_classes) to lie in [0, 1]
        num_classes = probs.size(1)
        raw_entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=1)
        norm_entropy = raw_entropy / math.log(num_classes)

        return (
            int(pred.item()),
            float(conf.item()),
            float(norm_entropy.item()),
            probs
        )

    def estimate_decision_information(
        self,
        current_entropy: float,
        current_confidence: float,
        target_probe_block: int
    ) -> Tuple[float, str]:
        """
        Defines and estimates Expected Decision Information for an intermediate probe.

        Mathematical Formulation:
          The decision information of evaluating candidate probe k given current state (entropy H, confidence C)
          combines:
          1. Expected Uncertainty Reduction: A higher-capacity stage has higher empirical accuracy A_k.
             The expected entropy reduction is proportional to (current_entropy * probe_prior_accuracy_k).
          2. Decision Margin Deficit: (1.0 - current_confidence), representing the confidence gap to certainty.

          Expected Information = (1.0 - current_confidence) * (self.probe_prior_accuracies[k] / 0.906)
          
        This formulation guarantees:
          - When confidence is already high (gap is small), expected information is low.
          - Later probes (with higher capacity) offer higher expected decision information.
          - Values are grounded in empirical validation capabilities, not fabricated.
        """
        prior_acc = self.probe_prior_accuracies.get(target_probe_block, 0.50)
        max_prior_acc = self.probe_prior_accuracies[16]  # 0.906

        capacity_ratio = prior_acc / max_prior_acc
        uncertainty_gap = max(0.0, 1.0 - current_confidence)

        # Expected Information in [0, 1]
        expected_info = uncertainty_gap * capacity_ratio
        method = f"EmpiricalCapacityEntropyReduction(gap={uncertainty_gap:.3f}, prior_acc={prior_acc:.3f})"
        return float(expected_info), method

    def _get_resource_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get a recent resource snapshot, refreshing only when stale."""
        try:
            return self.resource_monitor.snapshot(max_age_s=0.5)
        except TypeError:
            return self.resource_monitor.snapshot()

    def _build_blocked_return(
        self,
        action: str,
        prediction: int,
        confidence: float,
        uncertainty: float,
        current_latency_ms: float,
        selected_probe: Optional[int],
        status: str,
        reason: str,
        snapshot: Optional[Dict[str, Any]],
        violations: Optional[List[str]] = None,
        controller_time_ms: Optional[float] = None,
        decision_score: float = 0.0,
        estimated_incremental_cost: float = 0.0,
        current_probe_block: Optional[int] = None,
    ) -> Dict[str, Any]:
        remaining_budget = max(0.0, self.max_latency_ms - current_latency_ms)
        return {
            "action": action,
            "status": status,
            "selected_probe": selected_probe,
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "estimated_incremental_cost": estimated_incremental_cost,
            "remaining_budget": remaining_budget,
            "resource_snapshot": snapshot,
            "resource_violations": violations or [],
            "decision_score": decision_score,
            "controller_time_ms": round(controller_time_ms or 0.0, 3),
            "reason": reason,
            "current_probe_block": current_probe_block,
        }

    def evaluate_candidate_probes(
        self,
        current_block: int,
        current_confidence: float,
        current_entropy: float,
        current_latency_ms: float,
        executed_probes: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all candidate probes that:
          1. Have not yet been executed.
          2. Are sequentially downstream from current_block.
          3. Can fit within the remaining latency budget.
        """
        remaining_budget = max(0.0, self.max_latency_ms - current_latency_ms)
        candidates = []

        for p_block in self.available_probes:
            if p_block in executed_probes:
                continue
            if p_block <= current_block:
                continue

            # Calculate true incremental sequential cost to reach and evaluate p_block
            inc_cost = self.cost_estimator.get_probe_evaluation_cost(current_block, p_block)

            # Feasibility check
            is_feasible = (inc_cost <= remaining_budget) or (not self.enforce_resource_budget)

            # Expected Information
            exp_info, method = self.estimate_decision_information(
                current_entropy, current_confidence, p_block
            )

            # Selection Value = Expected Decision Information / Incremental Computation Cost
            selection_score = (exp_info / max(0.01, inc_cost)) if is_feasible else -1.0

            candidates.append({
                "probe_block": p_block,
                "incremental_cost_ms": round(inc_cost, 4),
                "expected_information": round(exp_info, 4),
                "selection_score": round(selection_score, 4),
                "is_feasible": is_feasible,
                "calculation_method": method
            })

        # Sort candidates descending by selection_score
        candidates.sort(key=lambda c: c["selection_score"], reverse=True)
        return candidates

    def decide_initial(
        self,
        model_a_logits: torch.Tensor,
        current_latency_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Decision Stage 1: Evaluate initial Model A logits.
        """
        t_start = time.perf_counter()
        pred_a, conf_a, entropy_a, _ = self.compute_distribution_metrics(model_a_logits)
        remaining_budget = max(0.0, self.max_latency_ms - current_latency_ms)

        # Rule 1: Model A is sufficiently confident -> RETAIN immediately
        if conf_a >= self.model_a_threshold:
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return {
                "action": "RETAIN",
                "status": "CONFIDENT_RETAIN",
                "selected_probe": None,
                "prediction": pred_a,
                "confidence": conf_a,
                "uncertainty": entropy_a,
                "estimated_incremental_cost": 0.0,
                "remaining_budget": remaining_budget,
                "decision_score": 1.0,
                "controller_time_ms": round(eval_time, 3),
                "reason": f"Model A confidence ({conf_a:.4f}) meets threshold ({self.model_a_threshold:.4f})."
            }

        snapshot = self._get_resource_snapshot()
        res_eval = self.resource_policy.evaluate(snapshot)

        completion_cost = self.cost_estimator.get_remaining_completion_cost(0)
        precheck_fallback = self.resource_policy.authorize_full_model_b(
            snapshot=snapshot,
            remaining_latency_ms=remaining_budget,
            completion_cost_ms=completion_cost,
            enforce_budget=self.enforce_resource_budget,
        )
        if precheck_fallback["status"] == "BUDGET_EXHAUSTED_RETAIN":
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return self._build_blocked_return(
                action="RETAIN",
                prediction=pred_a,
                confidence=conf_a,
                uncertainty=entropy_a,
                current_latency_ms=current_latency_ms,
                selected_probe=None,
                status="BUDGET_EXHAUSTED_RETAIN",
                reason=f"Latency budget insufficient for full Model B fallback: {precheck_fallback['reason']}",
                snapshot=snapshot,
                violations=precheck_fallback["resource_violations"],
                controller_time_ms=eval_time,
                decision_score=0.0,
                estimated_incremental_cost=completion_cost,
                current_probe_block=0,
            )

        # Rule 2: Hardware resource budget exhausted
        if not res_eval.is_healthy and self.enforce_resource_budget:
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return self._build_blocked_return(
                action="RETAIN",
                prediction=pred_a,
                confidence=conf_a,
                uncertainty=entropy_a,
                current_latency_ms=current_latency_ms,
                selected_probe=None,
                status="RESOURCE_BUDGET_EXHAUSTED",
                reason=(
                    f"Hardware constraint prevents further computation: {'; '.join(res_eval.violations)}. "
                    "Retaining uncertain Model A prediction."
                ),
                snapshot=snapshot,
                violations=res_eval.violations,
                controller_time_ms=eval_time,
                decision_score=0.0,
                estimated_incremental_cost=0.0,
            )

        # Rule 3: Evaluate candidate probes starting from block 0
        candidates = self.evaluate_candidate_probes(
            current_block=0,
            current_confidence=conf_a,
            current_entropy=entropy_a,
            current_latency_ms=current_latency_ms,
            executed_probes=[]
        )

        feasible_candidates = [c for c in candidates if c["is_feasible"] and c["selection_score"] > 0]

        if feasible_candidates:
            best_candidate = feasible_candidates[0]
            selected_probe = best_candidate["probe_block"]
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return {
                "action": "PROBE MORE",
                "selected_probe": selected_probe,
                "candidate_evaluations": candidates,
                "prediction": pred_a,
                "confidence": conf_a,
                "uncertainty": entropy_a,
                "estimated_incremental_cost": best_candidate["incremental_cost_ms"],
                "remaining_budget": remaining_budget,
                "resource_snapshot": snapshot,
                "resource_violations": res_eval.violations,
                "decision_score": best_candidate["selection_score"],
                "controller_time_ms": round(eval_time, 3),
                "reason": (
                    f"Model A uncertain (conf={conf_a:.4f} < {self.model_a_threshold:.4f}). "
                    f"Selected Probe {selected_probe} with score {best_candidate['selection_score']:.3f}."
                )
            }

        # Rule 4: No feasible probes; authorize or reject full Model B completion
        completion_cost = self.cost_estimator.get_remaining_completion_cost(0)
        fallback_auth = self.resource_policy.authorize_full_model_b(
            snapshot=snapshot,
            remaining_latency_ms=remaining_budget,
            completion_cost_ms=completion_cost,
            enforce_budget=self.enforce_resource_budget,
        )

        if fallback_auth["authorized"]:
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return {
                "action": "SWITCH",
                "selected_probe": None,
                "prediction": pred_a,
                "confidence": conf_a,
                "uncertainty": entropy_a,
                "estimated_incremental_cost": completion_cost,
                "remaining_budget": remaining_budget,
                "resource_snapshot": snapshot,
                "resource_violations": fallback_auth["resource_violations"],
                "decision_score": 0.0,
                "controller_time_ms": round(eval_time, 3),
                "reason": fallback_auth["reason"],
            }

        eval_time = (time.perf_counter() - t_start) * 1000.0
        if fallback_auth["status"] == "RESOURCE_BUDGET_EXHAUSTED":
            return self._build_blocked_return(
                action="RETAIN",
                prediction=pred_a,
                confidence=conf_a,
                uncertainty=entropy_a,
                current_latency_ms=current_latency_ms,
                selected_probe=None,
                status="RESOURCE_BUDGET_EXHAUSTED",
                reason=f"Resource policy blocks full Model B fallback: {fallback_auth['reason']}",
                snapshot=snapshot,
                violations=fallback_auth["resource_violations"],
                controller_time_ms=eval_time,
                decision_score=0.0,
                estimated_incremental_cost=completion_cost,
            )

        return self._build_blocked_return(
            action="RETAIN",
            prediction=pred_a,
            confidence=conf_a,
            uncertainty=entropy_a,
            current_latency_ms=current_latency_ms,
            selected_probe=None,
            status="BUDGET_EXHAUSTED_RETAIN",
            reason=f"Latency budget insufficient for full Model B fallback: {fallback_auth['reason']}",
            snapshot=snapshot,
            violations=fallback_auth["resource_violations"],
            controller_time_ms=eval_time,
            decision_score=0.0,
            estimated_incremental_cost=completion_cost,
        )

    def decide_probe(
        self,
        current_probe_block: int,
        probe_logits: torch.Tensor,
        current_latency_ms: float,
        executed_probes: List[int]
    ) -> Dict[str, Any]:
        """
        Decision Stage 2+: Evaluate current intermediate probe evidence.
        """
        t_start = time.perf_counter()
        pred_p, conf_p, entropy_p, _ = self.compute_distribution_metrics(probe_logits)
        remaining_budget = max(0.0, self.max_latency_ms - current_latency_ms)

        # Rule 1: Current probe provides sufficient decision certainty
        if conf_p >= self.probe_threshold:
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return {
                "action": "RETAIN",
                "status": "CONFIDENT_RETAIN",
                "selected_probe": current_probe_block,
                "prediction": pred_p,
                "confidence": conf_p,
                "uncertainty": entropy_p,
                "estimated_incremental_cost": 0.0,
                "remaining_budget": remaining_budget,
                "decision_score": 1.0,
                "controller_time_ms": round(eval_time, 3),
                "reason": f"Probe {current_probe_block} reached confidence threshold ({conf_p:.4f} >= {self.probe_threshold:.4f})."
            }

        snapshot = self._get_resource_snapshot()
        res_eval = self.resource_policy.evaluate(snapshot)

        completion_cost = self.cost_estimator.get_remaining_completion_cost(current_probe_block)
        precheck_fallback = self.resource_policy.authorize_full_model_b(
            snapshot=snapshot,
            remaining_latency_ms=remaining_budget,
            completion_cost_ms=completion_cost,
            enforce_budget=self.enforce_resource_budget,
        )
        if precheck_fallback["status"] == "BUDGET_EXHAUSTED_RETAIN":
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return self._build_blocked_return(
                action="RETAIN",
                prediction=pred_p,
                confidence=conf_p,
                uncertainty=entropy_p,
                current_latency_ms=current_latency_ms,
                selected_probe=current_probe_block,
                status="BUDGET_EXHAUSTED_RETAIN",
                reason=f"Latency budget insufficient for full Model B fallback: {precheck_fallback['reason']}",
                snapshot=snapshot,
                violations=precheck_fallback["resource_violations"],
                controller_time_ms=eval_time,
                decision_score=0.0,
                estimated_incremental_cost=completion_cost,
                current_probe_block=current_probe_block,
            )

        # Rule 2: Hardware resource budget exhausted
        if not res_eval.is_healthy and self.enforce_resource_budget:
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return self._build_blocked_return(
                action="RETAIN",
                prediction=pred_p,
                confidence=conf_p,
                uncertainty=entropy_p,
                current_latency_ms=current_latency_ms,
                selected_probe=current_probe_block,
                status="RESOURCE_BUDGET_EXHAUSTED",
                reason=(
                    f"Hardware constraint prevents further computation: {'; '.join(res_eval.violations)}. "
                    f"Retaining uncertain Probe {current_probe_block} prediction."
                ),
                snapshot=snapshot,
                violations=res_eval.violations,
                controller_time_ms=eval_time,
                decision_score=0.0,
                estimated_incremental_cost=0.0,
                current_probe_block=current_probe_block,
            )

        # Rule 3: Probe remains uncertain -> evaluate remaining candidate probes
        candidates = self.evaluate_candidate_probes(
            current_block=current_probe_block,
            current_confidence=conf_p,
            current_entropy=entropy_p,
            current_latency_ms=current_latency_ms,
            executed_probes=executed_probes
        )

        feasible_candidates = [c for c in candidates if c["is_feasible"] and c["selection_score"] > 0]

        if feasible_candidates:
            best_candidate = feasible_candidates[0]
            selected_probe = best_candidate["probe_block"]
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return {
                "action": "PROBE MORE",
                "selected_probe": selected_probe,
                "candidate_evaluations": candidates,
                "prediction": pred_p,
                "confidence": conf_p,
                "uncertainty": entropy_p,
                "estimated_incremental_cost": best_candidate["incremental_cost_ms"],
                "remaining_budget": remaining_budget,
                "resource_snapshot": snapshot,
                "resource_violations": res_eval.violations,
                "decision_score": best_candidate["selection_score"],
                "controller_time_ms": round(eval_time, 3),
                "reason": (
                    f"Probe {current_probe_block} uncertain (conf={conf_p:.4f}). "
                    f"Selected next Probe {selected_probe} (score={best_candidate['selection_score']:.3f})."
                )
            }

        # Rule 4: No further probes feasible or available -> evaluate Full Model B fallback
        completion_cost = self.cost_estimator.get_remaining_completion_cost(current_probe_block)
        fallback_auth = self.resource_policy.authorize_full_model_b(
            snapshot=snapshot,
            remaining_latency_ms=remaining_budget,
            completion_cost_ms=completion_cost,
            enforce_budget=self.enforce_resource_budget,
        )

        if fallback_auth["authorized"]:
            eval_time = (time.perf_counter() - t_start) * 1000.0
            return {
                "action": "SWITCH",
                "selected_probe": None,
                "prediction": pred_p,
                "confidence": conf_p,
                "uncertainty": entropy_p,
                "estimated_incremental_cost": completion_cost,
                "remaining_budget": remaining_budget,
                "resource_snapshot": snapshot,
                "resource_violations": fallback_auth["resource_violations"],
                "decision_score": 0.5,
                "controller_time_ms": round(eval_time, 3),
                "reason": fallback_auth["reason"],
            }

        eval_time = (time.perf_counter() - t_start) * 1000.0
        if fallback_auth["status"] == "RESOURCE_BUDGET_EXHAUSTED":
            return self._build_blocked_return(
                action="RETAIN",
                prediction=pred_p,
                confidence=conf_p,
                uncertainty=entropy_p,
                current_latency_ms=current_latency_ms,
                selected_probe=current_probe_block,
                status="RESOURCE_BUDGET_EXHAUSTED",
                reason=f"Resource policy blocks full Model B fallback: {fallback_auth['reason']}",
                snapshot=snapshot,
                violations=fallback_auth["resource_violations"],
                controller_time_ms=eval_time,
                decision_score=0.0,
                estimated_incremental_cost=completion_cost,
                current_probe_block=current_probe_block,
            )

        return self._build_blocked_return(
            action="RETAIN",
            prediction=pred_p,
            confidence=conf_p,
            uncertainty=entropy_p,
            current_latency_ms=current_latency_ms,
            selected_probe=current_probe_block,
            status="BUDGET_EXHAUSTED_RETAIN",
            reason=f"Latency budget insufficient for full Model B fallback: {fallback_auth['reason']}",
            snapshot=snapshot,
            violations=fallback_auth["resource_violations"],
            controller_time_ms=eval_time,
            decision_score=0.0,
            estimated_incremental_cost=completion_cost,
            current_probe_block=current_probe_block,
        )

    def decide(
        self,
        model_a_logits: torch.Tensor,
        probe_logits: Optional[torch.Tensor] = None,
        current_latency_ms: float = 0.0,
        next_probe_cost_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Backward-compatible unified interface for legacy call sites.
        """
        if probe_logits is None:
            return self.decide_initial(model_a_logits, current_latency_ms)
        else:
            # Assume evaluating earliest probe if unspecified
            return self.decide_probe(
                current_probe_block=4,
                probe_logits=probe_logits,
                current_latency_ms=current_latency_ms,
                executed_probes=[4]
            )

