
"""
Integration tests for AdaptEdgePipeline.
Verifies complete end-to-end inference, repeated requests, resource policy triggering,
cache isolation, and lifecycle cleanup.
"""

import unittest
import torch
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.adapt_edge_pipeline import AdaptEdgePipeline
from controller.resource_policy import ResourcePolicyConfig


class TestAdaptEdgePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        cls.pipeline = AdaptEdgePipeline(
            device=cls.device,
            max_latency_budget_ms=30.0
        )

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline.model_a)
        self.assertIsNotNone(self.pipeline.model_b)
        self.assertIsNotNone(self.pipeline.sequential_b)
        self.assertIsNotNone(self.pipeline.controller)
        self.assertIsNotNone(self.pipeline.cache)

    def test_single_inference_request(self):
        img = torch.randn(
            1, 3, 224, 224, device=self.device
        )

        res = self.pipeline.predict(
            img, request_id="test_req_01"
        )

        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], list(range(10)))
        self.assertIn("action", res)
        self.assertIn(
            res["action"],
            ["RETAIN", "PROBE MORE", "SWITCH"]
        )
        self.assertIn("status", res)
        self.assertIn("latencies", res)
        self.assertGreater(res["latency_ms"], 0.0)
        self.assertIn("decision_trace", res)
        self.assertGreaterEqual(
            len(res["decision_trace"]), 1
        )

    def test_repeated_inference_requests_clean_cache(self):
        # Run 5 consecutive requests and verify cache invalidation.
        img = torch.randn(
            1, 3, 224, 224, device=self.device
        )

        for i in range(5):
            res = self.pipeline.predict(
                img, request_id=f"rep_req_{i}"
            )
            self.assertIsNotNone(res["prediction"])

        # Active cache entries should be zero after each request.
        stats = self.pipeline.cache.get_cache_stats()

        self.assertEqual(stats["entries_count"], 0)
        self.assertEqual(stats["memory_used_mb"], 0.0)

    def test_resource_exhaustion_in_pipeline(self):
        # Force hardware violation while making Model A uncertain.
        mock_hot_gpu = {
            "cpu_percent": 30.0,
            "ram_percent": 45.0,
            "gpu_available": True,
            "gpu_temperature_c": 98.0,
            "total_vram_mb": 4096.0,
            "nvidia_smi_used_mb": 1000.0
        }

        self.pipeline.resource_monitor.set_mock_snapshot(
            mock_hot_gpu
        )

        try:
            original_a = self.pipeline.model_a
            original_thresh = (
                self.pipeline.controller.model_a_threshold
            )

            self.pipeline.controller.model_a_threshold = 0.9999

            def uncertain_logits(image):
                return torch.ones(
                    1, 10, device=image.device
                ) / 10.0

            self.pipeline.model_a = uncertain_logits

            img = torch.randn(
                1, 3, 224, 224, device=self.device
            )

            res = self.pipeline.predict(
                img, request_id="thermal_req"
            )

            self.assertEqual(res["action"], "RETAIN")
            self.assertEqual(
                res["status"],
                "RESOURCE_BUDGET_EXHAUSTED"
            )
            self.assertTrue(
                len(res["resource_violations"]) > 0
            )
            self.assertIn(
                "GPU_THERMAL_PRESSURE",
                res["resource_violations"][0]
            )

        finally:
            self.pipeline.resource_monitor.set_mock_snapshot(None)
            self.pipeline.controller.model_a_threshold = original_thresh
            self.pipeline.model_a = original_a

    def test_direct_switch_executes_full_model_b(self):
        img = torch.randn(
            1, 3, 224, 224, device=self.device
        )

        # Calculate the expected prediction using full Model B.
        with torch.inference_mode():
            expected_logits = self.pipeline.model_b(img)
            expected_prediction = (
                expected_logits.argmax(dim=1).item()
            )

        # Force the initial controller decision to SWITCH.
        forced_decision = {
            "action": "SWITCH",
            "status": "DIRECT_SWITCH_TEST",
            "selected_probe": None,
            "prediction": 0,
            "confidence": 0.10,
            "uncertainty": 1.0,
            "decision_score": 0.0,
            "reason": "Testing direct Model B fallback."
        }

        with patch.object(
            self.pipeline.controller,
            "decide_initial",
            return_value=forced_decision
        ):
            result = self.pipeline.predict(
                img,
                request_id="direct_switch_test"
            )

        self.assertEqual(result["action"], "SWITCH")
        self.assertEqual(result["executed_probes"], [])
        self.assertTrue(result["used_full_b"])
        self.assertEqual(
            result["prediction"],
            expected_prediction
        )


if __name__ == "__main__":
    unittest.main()
