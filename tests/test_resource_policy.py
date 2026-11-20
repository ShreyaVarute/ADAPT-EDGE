"""
Unit and integration tests for ResourcePolicy and Resource-Aware Decision Controller.
Verifies all policy conditions, mocked hardware states, and controller enforcement rules.
"""

import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.resource_policy import ResourcePolicy, ResourcePolicyConfig, ResourceEvaluation
from controller.decision_controller import DecisionController
from utils.resource_monitor import ResourceMonitor


class TestResourcePolicy(unittest.TestCase):
    """Tests evaluating hardware state against configurable thresholds."""

    def setUp(self):
        self.config = ResourcePolicyConfig(
            max_cpu_percent=90.0,
            max_ram_percent=90.0,
            min_ram_free_mb=500.0,
            max_gpu_percent=95.0,
            max_gpu_temp_c=85.0,
            min_gpu_free_mb=250.0,
            max_torch_vram_mb=3500.0
        )
        self.policy = ResourcePolicy(self.config)

    def test_healthy_snapshot(self):
        healthy_snap = {
            "cpu_percent": 25.0,
            "ram_used_mb": 8000.0,
            "ram_total_mb": 16000.0,
            "ram_percent": 50.0,
            "gpu_available": True,
            "gpu_utilization_percent": 30.0,
            "gpu_temperature_c": 55.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 1000.0,
            "gpu_torch_allocated_mb": 150.0
        }
        res = self.policy.evaluate(healthy_snap)
        self.assertTrue(res.is_healthy)
        self.assertTrue(res.can_probe)
        self.assertTrue(res.can_switch)
        self.assertEqual(res.status, "HEALTHY")
        self.assertEqual(len(res.violations), 0)

    def test_cpu_pressure_violation(self):
        snap = {
            "cpu_percent": 96.5,
            "ram_percent": 50.0,
            "gpu_available": False
        }
        res = self.policy.evaluate(snap)
        self.assertFalse(res.is_healthy)
        self.assertFalse(res.can_probe)
        self.assertEqual(res.status, "RESOURCE_CONSTRAINED")
        self.assertTrue(any("CPU_PRESSURE" in v for v in res.violations))

    def test_ram_pressure_and_low_free_ram(self):
        snap = {
            "cpu_percent": 30.0,
            "ram_used_mb": 15800.0,
            "ram_total_mb": 16000.0,
            "ram_percent": 98.7,
            "gpu_available": False
        }
        res = self.policy.evaluate(snap)
        self.assertFalse(res.is_healthy)
        self.assertTrue(any("RAM_PRESSURE" in v for v in res.violations))
        self.assertTrue(any("LOW_FREE_RAM" in v for v in res.violations))

    def test_gpu_thermal_pressure(self):
        snap = {
            "cpu_percent": 20.0,
            "ram_percent": 40.0,
            "gpu_available": True,
            "gpu_temperature_c": 89.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 1000.0
        }
        res = self.policy.evaluate(snap)
        self.assertFalse(res.is_healthy)
        self.assertTrue(any("GPU_THERMAL_PRESSURE" in v for v in res.violations))

    def test_gpu_low_vram(self):
        snap = {
            "cpu_percent": 20.0,
            "ram_percent": 40.0,
            "gpu_available": True,
            "gpu_temperature_c": 60.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 3950.0  # Only ~146 MB free
        }
        res = self.policy.evaluate(snap)
        self.assertFalse(res.is_healthy)
        self.assertTrue(any("LOW_TOTAL_VRAM" in v for v in res.violations))

    def test_missing_sensors_permissive(self):
        # When sensors are None, does NOT report false zero or false violation
        snap = {
            "cpu_percent": None,
            "ram_percent": None,
            "gpu_available": False
        }
        res = self.policy.evaluate(snap)
        self.assertTrue(res.is_healthy)
        self.assertIn("cpu_percent", res.unmonitored)
        self.assertIn("ram_percent", res.unmonitored)

    def test_reserved_vram_pressure(self):
        snap = {
            "cpu_percent": 20.0,
            "ram_percent": 40.0,
            "ram_available_mb": 700.0,
            "gpu_available": True,
            "gpu_temperature_c": 60.0,
            "gpu_torch_reserved_mb": 5000.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 1000.0
        }
        res = self.policy.evaluate(snap)
        self.assertFalse(res.is_healthy)
        self.assertTrue(any("PYTORCH_RESERVED_VRAM_EXCEEDED" in v for v in res.violations))

    def test_none_snapshot_permissive(self):
        res = self.policy.evaluate(None)
        self.assertTrue(res.is_healthy)
        self.assertIn("all_sensors_unavailable", res.unmonitored)


class TestResourceAwareControllerIntegration(unittest.TestCase):
    """Tests DecisionController behavior when resource constraints are active."""

    def setUp(self):
        self.monitor = ResourceMonitor()
        self.policy = ResourcePolicy()
        self.controller = DecisionController(
            model_a_threshold=0.80,
            probe_threshold=0.85,
            max_latency_ms=25.0,
            resource_monitor=self.monitor,
            resource_policy=self.policy,
            enforce_resource_budget=True
        )

    def tearDown(self):
        self.monitor.set_mock_snapshot(None)

    def test_confident_model_a_retains_without_resource_check(self):
        # Even if hardware is under heavy pressure, confident prediction is retained
        self.monitor.set_mock_snapshot({
            "cpu_percent": 99.0,
            "ram_percent": 99.0,
            "gpu_available": True,
            "gpu_temperature_c": 95.0
        })
        logits = torch.zeros(1, 10)
        logits[0, 3] = 10.0  # Confident class 3
        dec = self.controller.decide_initial(logits, current_latency_ms=2.0)
        self.assertEqual(dec["action"], "RETAIN")
        self.assertEqual(dec["status"], "CONFIDENT_RETAIN")

    def test_uncertain_model_a_blocked_by_hardware_pressure(self):
        # Uncertain prediction + thermal pressure -> RESOURCE_BUDGET_EXHAUSTED
        self.monitor.set_mock_snapshot({
            "cpu_percent": 40.0,
            "ram_percent": 50.0,
            "gpu_available": True,
            "gpu_temperature_c": 92.0,  # Overheating
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 1000.0
        })
        logits = torch.ones(1, 10)  # Uniform uncertain logits
        dec = self.controller.decide_initial(logits, current_latency_ms=2.0)
        self.assertEqual(dec["action"], "RETAIN")
        self.assertEqual(dec["status"], "RESOURCE_BUDGET_EXHAUSTED")
        self.assertTrue(len(dec["resource_violations"]) > 0)
        self.assertIn("GPU_THERMAL_PRESSURE", dec["resource_violations"][0])

    def test_uncertain_probe_blocked_by_ram_pressure(self):
        # Uncertain probe + high RAM -> RESOURCE_BUDGET_EXHAUSTED
        self.monitor.set_mock_snapshot({
            "cpu_percent": 25.0,
            "ram_percent": 96.0,
            "ram_used_mb": 15500.0,
            "ram_total_mb": 16000.0,
            "gpu_available": True,
            "gpu_temperature_c": 50.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 800.0
        })
        probe_logits = torch.ones(1, 10)
        dec = self.controller.decide_probe(
            current_probe_block=4,
            probe_logits=probe_logits,
            current_latency_ms=4.0,
            executed_probes=[4]
        )
        self.assertEqual(dec["action"], "RETAIN")
        self.assertEqual(dec["status"], "RESOURCE_BUDGET_EXHAUSTED")
        self.assertEqual(dec["selected_probe"], 4)

    def test_enforce_resource_budget_false_allows_probing_under_pressure(self):
        # Disabling budget enforcement allows probing despite pressure
        unconstrained_controller = DecisionController(
            model_a_threshold=0.80,
            max_latency_ms=25.0,
            resource_monitor=self.monitor,
            resource_policy=self.policy,
            enforce_resource_budget=False
        )
        self.monitor.set_mock_snapshot({
            "cpu_percent": 99.0,
            "ram_percent": 99.0,
            "gpu_available": True,
            "gpu_temperature_c": 95.0
        })
        logits = torch.ones(1, 10)
        dec = unconstrained_controller.decide_initial(logits, current_latency_ms=2.0)
        self.assertEqual(dec["action"], "PROBE MORE")

    def test_latency_exhaustion_vs_resource_exhaustion_differentiation(self):
        # Healthy hardware, but latency budget exhausted -> BUDGET_EXHAUSTED_RETAIN
        self.monitor.set_mock_snapshot({
            "cpu_percent": 20.0,
            "ram_percent": 40.0,
            "gpu_available": True,
            "gpu_temperature_c": 50.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 500.0
        })
        logits = torch.ones(1, 10)
        dec = self.controller.decide_initial(logits, current_latency_ms=24.5)  # only 0.5ms left
        self.assertEqual(dec["action"], "RETAIN")
        self.assertEqual(dec["status"], "BUDGET_EXHAUSTED_RETAIN")


if __name__ == "__main__":
    unittest.main()

