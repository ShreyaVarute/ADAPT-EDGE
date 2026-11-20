"""
Resource Policy module for ADAPT-EDGE.
Evaluates live hardware state (CPU, RAM, GPU utilization, VRAM, and GPU temperature)
against configurable thresholds to govern adaptive computation.

Design principles:
1. Honest sensor interpretation: missing/None sensor readings are NEVER assumed to be zero.
2. Missing sensors do not falsely trigger resource exhaustion when allow_missing_sensors is True.
3. Distinguishes PyTorch allocated VRAM from total system GPU memory used across processes.
4. Provides deterministic evaluation for both live and mocked hardware snapshots.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ResourcePolicyConfig:
    """
    Configurable thresholds for hardware resource monitoring.
    Sensible defaults suitable for edge AI on mobile GPU (e.g. RTX 3050 Laptop 4GB).
    """
    max_cpu_percent: float = 90.0
    max_ram_percent: float = 95.0
    min_ram_free_mb: float = 500.0
    max_gpu_percent: float = 95.0
    max_gpu_temp_c: float = 85.0
    min_gpu_free_mb: float = 250.0
    max_torch_vram_mb: float = 3500.0
    max_torch_reserved_vram_mb: float = 4000.0

    # Policy enforcement toggles
    enforce_cpu: bool = True
    enforce_ram: bool = True
    enforce_gpu_util: bool = True
    enforce_gpu_temp: bool = True
    enforce_vram: bool = True
    enforce_torch_reserved_vram: bool = True
    allow_missing_sensors: bool = True


@dataclass
class ResourceEvaluation:
    """
    Structured outcome of a resource policy check.
    """
    is_healthy: bool
    can_probe: bool
    can_switch: bool
    status: str  # "HEALTHY", "RESOURCE_CONSTRAINED", "CRITICAL_PRESSURE"
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unmonitored: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "can_probe": self.can_probe,
            "can_switch": self.can_switch,
            "status": self.status,
            "violations": self.violations,
            "warnings": self.warnings,
            "unmonitored": self.unmonitored,
            "summary": self.summary
        }


class ResourcePolicy:
    """
    Evaluates hardware conditions from ResourceMonitor snapshots against configured policy.
    """

    def __init__(self, config: Optional[ResourcePolicyConfig] = None):
        self.config = config or ResourcePolicyConfig()

    def summarize(self, snapshot: Optional[Dict[str, Any]]) -> ResourceEvaluation:
        """Compatibility wrapper around evaluate()."""
        return self.evaluate(snapshot)

    def evaluate(self, snapshot: Optional[Dict[str, Any]]) -> ResourceEvaluation:
        """
        Evaluate a hardware snapshot and return a structured ResourceEvaluation.
        """
        if snapshot is None:
            if self.config.allow_missing_sensors:
                return ResourceEvaluation(
                    is_healthy=True,
                    can_probe=True,
                    can_switch=True,
                    status="HEALTHY",
                    unmonitored=["all_sensors_unavailable"],
                    summary="No resource snapshot provided; proceeding under permissive missing-sensor policy."
                )
            else:
                return ResourceEvaluation(
                    is_healthy=False,
                    can_probe=False,
                    can_switch=False,
                    status="CRITICAL_PRESSURE",
                    violations=["Missing resource snapshot while allow_missing_sensors=False"],
                    summary="Resource snapshot unavailable and policy strictly forbids unmonitored execution."
                )

        violations: List[str] = []
        warnings: List[str] = []
        unmonitored: List[str] = []

        # 1. CPU Utilization
        cpu_pct = snapshot.get("cpu_percent")
        if cpu_pct is not None:
            if self.config.enforce_cpu and cpu_pct > self.config.max_cpu_percent:
                violations.append(f"CPU_PRESSURE: {cpu_pct:.1f}% exceeds limit of {self.config.max_cpu_percent:.1f}%")
            elif cpu_pct > (self.config.max_cpu_percent * 0.85):
                warnings.append(f"CPU_HIGH: {cpu_pct:.1f}% approaching limit")
        else:
            unmonitored.append("cpu_percent")

        # 2. System RAM
        ram_pct = snapshot.get("ram_percent")
        ram_used = snapshot.get("ram_used_mb")
        ram_total = snapshot.get("ram_total_mb")
        ram_available = snapshot.get("ram_available_mb")
        if ram_pct is not None:
            if self.config.enforce_ram and ram_pct > self.config.max_ram_percent:
                violations.append(f"RAM_PRESSURE: {ram_pct:.1f}% exceeds limit of {self.config.max_ram_percent:.1f}%")
        else:
            unmonitored.append("ram_percent")

        if ram_available is not None:
            if self.config.enforce_ram and ram_available < self.config.min_ram_free_mb:
                violations.append(f"LOW_FREE_RAM: {ram_available:.1f} MB below minimum {self.config.min_ram_free_mb:.1f} MB")
        elif ram_total is not None and ram_used is not None:
            free_ram = ram_total - ram_used
            if self.config.enforce_ram and free_ram < self.config.min_ram_free_mb:
                violations.append(f"LOW_FREE_RAM: {free_ram:.1f} MB below minimum {self.config.min_ram_free_mb:.1f} MB")
        elif ram_pct is None:
            unmonitored.append("ram_free_mb")

        # 3. GPU Utilization & Availability
        gpu_avail = snapshot.get("gpu_available", False)
        if gpu_avail:
            gpu_pct = snapshot.get("gpu_utilization_percent")
            if gpu_pct is not None:
                if self.config.enforce_gpu_util and gpu_pct > self.config.max_gpu_percent:
                    violations.append(f"GPU_UTILIZATION_PRESSURE: {gpu_pct:.1f}% exceeds limit of {self.config.max_gpu_percent:.1f}%")
            else:
                unmonitored.append("gpu_utilization_percent")

            # 4. GPU Temperature
            gpu_temp = snapshot.get("gpu_temperature_c")
            if gpu_temp is not None:
                if self.config.enforce_gpu_temp and gpu_temp > self.config.max_gpu_temp_c:
                    violations.append(f"GPU_THERMAL_PRESSURE: {gpu_temp:.1f}C exceeds limit of {self.config.max_gpu_temp_c:.1f}C")
            else:
                unmonitored.append("gpu_temperature_c")

            # 5. VRAM (PyTorch allocated, reserved, and system VRAM)
            torch_alloc = snapshot.get("gpu_torch_allocated_mb")
            if torch_alloc is not None:
                if self.config.enforce_vram and torch_alloc > self.config.max_torch_vram_mb:
                    violations.append(f"PYTORCH_VRAM_EXCEEDED: {torch_alloc:.1f} MB exceeds budget of {self.config.max_torch_vram_mb:.1f} MB")
            else:
                unmonitored.append("gpu_torch_allocated_mb")

            torch_reserved = snapshot.get("gpu_torch_reserved_mb")
            if torch_reserved is not None:
                if self.config.enforce_torch_reserved_vram and torch_reserved > self.config.max_torch_reserved_vram_mb:
                    violations.append(
                        f"PYTORCH_RESERVED_VRAM_EXCEEDED: {torch_reserved:.1f} MB exceeds budget of {self.config.max_torch_reserved_vram_mb:.1f} MB"
                    )
            else:
                unmonitored.append("gpu_torch_reserved_mb")

            total_vram = snapshot.get("total_vram_mb")
            smi_used = snapshot.get("nvidia_smi_used_mb")
            free_vram = snapshot.get("gpu_free_vram_mb")
            if free_vram is None and total_vram is not None and smi_used is not None:
                free_vram = total_vram - smi_used
            if free_vram is not None:
                if self.config.enforce_vram and free_vram < self.config.min_gpu_free_mb:
                    violations.append(f"LOW_TOTAL_VRAM: {free_vram:.1f} MB free below threshold {self.config.min_gpu_free_mb:.1f} MB")
            elif total_vram is None and smi_used is not None:
                unmonitored.append("gpu_free_vram_mb")
        else:
            unmonitored.append("gpu_hardware (running in CPU mode or GPU not present)")

        # Determine health status
        is_healthy = (len(violations) == 0)
        if is_healthy:
            status = "HEALTHY"
            can_probe = True
            can_switch = True
            summary = "Hardware resources healthy; adaptive execution permitted."
        else:
            status = "RESOURCE_CONSTRAINED"
            can_probe = False
            can_switch = False
            summary = f"Resource budget exhausted due to: {'; '.join(violations)}."

        return ResourceEvaluation(
            is_healthy=is_healthy,
            can_probe=can_probe,
            can_switch=can_switch,
            status=status,
            violations=violations,
            warnings=warnings,
            unmonitored=unmonitored,
            summary=summary
        )

    def authorize_full_model_b(self, snapshot: Optional[Dict[str, Any]], remaining_latency_ms: float,
                               completion_cost_ms: float, enforce_budget: Optional[bool] = None) -> Dict[str, Any]:
        """Authorize full Model B execution using the current snapshot, budget, and policy config."""
        eval_result = self.evaluate(snapshot)
        enforce_budget = self.config.allow_missing_sensors if enforce_budget is None else enforce_budget
        budget_ok = remaining_latency_ms >= completion_cost_ms
        if not budget_ok:
            return {
                "authorized": False,
                "status": "BUDGET_EXHAUSTED_RETAIN",
                "reason": (
                    f"Remaining latency budget insufficient for full Model B completion: "
                    f"{remaining_latency_ms:.2f}ms < {completion_cost_ms:.2f}ms."
                ),
                "resource_snapshot": snapshot,
                "resource_violations": eval_result.violations,
                "resource_summary": eval_result.summary,
                "remaining_latency_ms": remaining_latency_ms,
                "completion_cost_ms": completion_cost_ms,
                "budget_ok": False,
                "policy_result": eval_result.to_dict(),
            }
        if not eval_result.is_healthy and (enforce_budget or self.config.allow_missing_sensors):
            return {
                "authorized": False,
                "status": "RESOURCE_BUDGET_EXHAUSTED",
                "reason": (
                    f"Resource policy forbids full Model B execution: {'; '.join(eval_result.violations)}. "
                    f"Remaining latency {remaining_latency_ms:.2f}ms, completion cost {completion_cost_ms:.2f}ms."
                ),
                "resource_snapshot": snapshot,
                "resource_violations": eval_result.violations,
                "resource_summary": eval_result.summary,
                "remaining_latency_ms": remaining_latency_ms,
                "completion_cost_ms": completion_cost_ms,
                "budget_ok": budget_ok,
                "policy_result": eval_result.to_dict(),
            }

        return {
            "authorized": True,
            "status": "RESOURCE_AUTHORIZED",
            "reason": "Resource policy and latency budget allow full Model B completion.",
            "resource_snapshot": snapshot,
            "resource_violations": eval_result.violations,
            "resource_summary": eval_result.summary,
            "remaining_latency_ms": remaining_latency_ms,
            "completion_cost_ms": completion_cost_ms,
            "budget_ok": True,
            "policy_result": eval_result.to_dict(),
        }

