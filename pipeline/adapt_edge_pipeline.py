"""
ADAPT-EDGE End-to-End Inference Pipeline.

Integrates:
1. Model A (MobileNetV3-Small)
2. Model B (MobileNetV3-Large with SequentialModelB execution)
3. Probe heads (Blocks 4, 9, 13)
4. DecisionController with empirical information-gain heuristic
5. ResourceMonitor with cached snapshots
6. ResourcePolicy for hardware condition evaluation
7. ContinuationCache with LRU eviction and per-request lifecycle cleanup
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from PIL import features
import torch
import torch.nn.functional as F

from models.model_a import ModelA
from models.model_b import ModelB
from probes.probe_heads import ProbeHead, SequentialModelB
from controller.decision_controller import DecisionController
from controller.cost_estimator import CostEstimator
from controller.resource_policy import ResourcePolicy, ResourcePolicyConfig
from cache.continuation_cache import ContinuationCache
from utils.resource_monitor import ResourceMonitor
from utils.calibration import TemperatureScaler
from utils.profiling import synchronize, measure_peak_gpu_memory_mb, reset_peak_gpu_memory

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROBE_CONFIG = {
    4: 40,
    9: 80,
    13: 160
}


class AdaptEdgePipeline:
    """
    Unified, resource-aware ADAPT-EDGE inference system.
    Supports repeated requests, robust exception handling, and full decision/latency tracing.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_a_threshold: float = 0.80,
        probe_threshold: float = 0.85,
        max_latency_budget_ms: float = 15.0,
        cache_max_entries: int = 20,
        cache_max_memory_mb: float = 64.0,
        enable_cache: bool = True,
        enable_calibration: bool = True,
        enforce_resource_budget: bool = True,
        resource_policy_config: Optional[ResourcePolicyConfig] = None
    ):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.enable_cache = enable_cache
        self.enable_calibration = enable_calibration
        self.enforce_resource_budget = enforce_resource_budget

        # 1. Hardware Monitor & Policy
        self.resource_monitor = ResourceMonitor()
        self.resource_policy = ResourcePolicy(resource_policy_config or ResourcePolicyConfig())

        # 2. Cost Estimator & Calibration
        self.cost_estimator = CostEstimator()
        self.temperature_scaler = TemperatureScaler(initial_temperature=1.0055) if enable_calibration else None

        # 3. Decision Controller
        self.controller = DecisionController(
            model_a_threshold=model_a_threshold,
            probe_threshold=probe_threshold,
            max_latency_ms=max_latency_budget_ms,
            cost_estimator=self.cost_estimator,
            temperature_scaler=self.temperature_scaler,
            resource_monitor=self.resource_monitor,
            resource_policy=self.resource_policy,
            enforce_resource_budget=enforce_resource_budget
        )

        # 4. Continuation Cache
        self.cache = ContinuationCache(
            max_entries=cache_max_entries,
            max_memory_mb=cache_max_memory_mb
        ) if enable_cache else None

        # 5. Models & Checkpoints
        self.model_a: Optional[ModelA] = None
        self.model_b: Optional[ModelB] = None
        self.sequential_b: Optional[SequentialModelB] = None
        self.checkpoint_id: str = "adapt_edge_models_v1"

        self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Load Model A, Model B, and probe checkpoints from outputs/."""
        # Model A
        self.model_a = ModelA(num_classes=10)
        ckpt_a_path = PROJECT_ROOT / "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth"
        ckpt_a = torch.load(ckpt_a_path, map_location=self.device, weights_only=False)
        self.model_a.load_state_dict(ckpt_a["model_state_dict"])
        self.model_a = self.model_a.to(self.device).eval()

        # Model B
        self.model_b = ModelB(num_classes=10)
        ckpt_b_path = PROJECT_ROOT / "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"
        ckpt_b = torch.load(ckpt_b_path, map_location=self.device, weights_only=False)
        self.model_b.load_state_dict(ckpt_b["model_state_dict"])
        self.model_b = self.model_b.to(self.device).eval()

        # Sequential Model B & Probes
        self.sequential_b = SequentialModelB(self.model_b, self.device)
        for block, channels in PROBE_CONFIG.items():
            probe_path = PROJECT_ROOT / f"outputs/probes/block_{block}/best_probe.pth"
            ckpt_probe = torch.load(probe_path, map_location=self.device, weights_only=False)
            head = ProbeHead(channels, num_classes=10).to(self.device)
            head.load_state_dict(ckpt_probe["head_state_dict"])
            head.eval()
            self.sequential_b.register_probe(block, head)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute adaptive inference on an input image tensor (1, 3, 224, 224).
        """
        req_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        decision_trace: List[Dict[str, Any]] = []
        executed_probes: List[int] = []
        cache_hits = 0

        synchronize(self.device)
        start_total = time.perf_counter()

        img_fp = ContinuationCache.generate_input_fingerprint(image) if self.cache is not None else "no_cache"

        # ---------------------------------------------------------
        # Step 1: Model A Inference
        # ---------------------------------------------------------
        start_a = time.perf_counter()
        logits_a = self.model_a(image)
        synchronize(self.device)
        lat_a = (time.perf_counter() - start_a) * 1000.0

        # Initial Decision Stage
        elapsed_so_far = (time.perf_counter() - start_total) * 1000.0
        init_decision = self.controller.decide_initial(logits_a, current_latency_ms=elapsed_so_far)
        decision_trace.append({"stage": "model_a", "decision": init_decision})

        # Check if Model A decision is final (Confident Retain, Resource Exhausted, Budget Exhausted)
        if init_decision["action"] == "RETAIN":
            synchronize(self.device)
            total_lat = (time.perf_counter() - start_total) * 1000.0
            return self._build_result(
                req_id=req_id,
                decision=init_decision,
                action="RETAIN",
                status=init_decision.get("status", "CONFIDENT_RETAIN"),
                executed_probes=[],
                used_full_b=False,
                used_partial_b=False,
                cache_hits=0,
                latencies={"model_a_ms": lat_a, "total_latency_ms": total_lat},
                decision_trace=decision_trace
            )

        # ---------------------------------------------------------
        # Step 2: Adaptive Probing Loop
        # ---------------------------------------------------------
        features = None
        current_block = 0
        if init_decision["action"] == "SWITCH":
            target_probe = None
        else:
            target_probe = init_decision["selected_probe"] or 4
        last_probe_decision = init_decision

        while target_probe is not None:
            # Check Continuation Cache
            if self.cache is not None and current_block > 0:
                cached_feat = self.cache.get(img_fp, self.checkpoint_id, current_block, target_device=self.device)
                if cached_feat is not None:
                    features = cached_feat
                    cache_hits += 1

            # Execute sequential backbone segment
            if current_block == 0:
                features = self.sequential_b.start(image)
            else:
                features = self.sequential_b.continue_to(features, current_block + 1, target_probe)

            current_block = target_probe

            # Store in Continuation Cache
            if self.cache is not None:
                self.cache.put(req_id, img_fp, self.checkpoint_id, current_block, features)

            # Probe inference
            probe_logits = self.sequential_b.predict_probe(features, target_probe)
            executed_probes.append(target_probe)

            synchronize(self.device)
            elapsed_so_far = (time.perf_counter() - start_total) * 1000.0

            probe_decision = self.controller.decide_probe(
                current_probe_block=target_probe,
                probe_logits=probe_logits,
                current_latency_ms=elapsed_so_far,
                executed_probes=executed_probes
            )
            decision_trace.append({"stage": f"probe_{target_probe}", "decision": probe_decision})
            last_probe_decision = probe_decision

            if probe_decision["action"] == "RETAIN":
                # Clean up per-request cache entries
                if self.cache is not None:
                    self.cache.invalidate_request(req_id)

                synchronize(self.device)
                total_lat = (time.perf_counter() - start_total) * 1000.0
                return self._build_result(
                    req_id=req_id,
                    decision=probe_decision,
                    action="RETAIN",
                    status=probe_decision.get("status", "CONFIDENT_RETAIN"),
                    executed_probes=executed_probes,
                    used_full_b=False,
                    used_partial_b=True,
                    cache_hits=cache_hits,
                    latencies={"model_a_ms": lat_a, "total_latency_ms": total_lat},
                    decision_trace=decision_trace
                )
            elif probe_decision["action"] == "PROBE MORE":
                target_probe = probe_decision["selected_probe"]
            else:  # SWITCH
                break

        # ---------------------------------------------------------
        # Step 3: Full Model B Fallback
        # ---------------------------------------------------------
        if self.cache is not None and features is None:
            features = self.cache.get(
                img_fp,
                self.checkpoint_id,
                current_block,
                target_device=self.device
            )

        if features is None:
            features = self.sequential_b.start(image)

            if current_block > 4:
                features = self.sequential_b.continue_to(
                    features, 5, current_block
                )
            else:
                # start() has already executed blocks 0-4.
                current_block = 4

        final_logits = self.sequential_b.finish(
            features,
            current_block=current_block
        )
        probs_b = F.softmax(final_logits, dim=1)
        conf_b, pred_b = probs_b.max(dim=1)
        entropy_b = -torch.sum(probs_b * torch.log(probs_b.clamp(min=1e-12))).item()

        # Clean up per-request cache entries
        if self.cache is not None:
            self.cache.invalidate_request(req_id)

        synchronize(self.device)
        total_lat = (time.perf_counter() - start_total) * 1000.0

        fallback_decision = {
            "prediction": int(pred_b.item()),
            "confidence": float(conf_b.item()),
            "uncertainty": float(entropy_b),
            "reason": "Completed full Model B after probe evidence collection.",
            "decision_score": 0.5
        }
        decision_trace.append({"stage": "fallback_switch", "decision": fallback_decision})

        return self._build_result(
            req_id=req_id,
            decision=fallback_decision,
            action="SWITCH",
            status="FALLBACK_SWITCH",
            executed_probes=executed_probes,
            used_full_b=True,
            used_partial_b=True,
            cache_hits=cache_hits,
            latencies={"model_a_ms": lat_a, "total_latency_ms": total_lat},
            decision_trace=decision_trace
        )

    def _build_result(
        self,
        req_id: str,
        decision: Dict[str, Any],
        action: str,
        status: str,
        executed_probes: List[int],
        used_full_b: bool,
        used_partial_b: bool,
        cache_hits: int,
        latencies: Dict[str, float],
        decision_trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construct structured inference result dictionary."""
        snap = self.resource_monitor.snapshot() if self.resource_monitor is not None else {}
        return {
            "request_id": req_id,
            "prediction": decision["prediction"],
            "confidence": decision["confidence"],
            "uncertainty": decision["uncertainty"],
            "action": action,
            "status": status,
            "executed_probes": executed_probes,
            "used_full_b": used_full_b,
            "used_partial_b": used_partial_b,
            "cache_hits": cache_hits,
            "decision_score": decision.get("decision_score", 0.0),
            "reason": decision.get("reason", ""),
            "resource_violations": decision.get("resource_violations", []),
            "latencies": latencies,
            "latency_ms": latencies["total_latency_ms"],
            "hardware_snapshot": snap,
            "decision_trace": decision_trace
        }
