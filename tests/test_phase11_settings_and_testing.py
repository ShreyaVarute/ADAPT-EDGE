"""
Phase 11 - Automated Test Suite: Settings, Hardware Devices, In-RAM Training & Session Lifecycle,
Inference Validation, Session Cleanup, and Protected Checkpoint Integrity Verification.
"""

import io
import sys
import json
from pathlib import Path
import unittest
import torch
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.app import app
from server.core import ServiceManager
from server.api.training import CANDIDATE_MODELS, CANDIDATE_MODELS_LOCK


class TestPhase11SettingsAndWorkbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manager = ServiceManager.get_instance()
        if not cls.manager._initialized:
            cls.manager.initialize()

    def test_01_device_and_gpu_configuration(self):
        """Test GET /api/settings and POST /api/settings/device."""
        # 1. Fetch system settings
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("device", data)
        self.assertIn("training_defaults", data)
        self.assertIn("resource_budget", data)
        self.assertIn("backend", data)
        self.assertIn("session_storage", data)
        self.assertIn("active_device", data["device"])

        # 2. Switch device to CPU explicitly
        switch_res = self.client.post("/api/settings/device", json={"device": "cpu", "cudnn_benchmark": False})
        self.assertEqual(switch_res.status_code, 200)
        switch_data = switch_res.json()
        self.assertTrue(switch_data["success"])
        self.assertEqual(switch_data["active_device"], "cpu")
        self.assertEqual(str(self.manager.device), "cpu")

    def test_02_default_training_parameters(self):
        """Test configuring default training parameters via POST /api/settings/training-defaults."""
        payload = {
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "img_size": 224,
            "val_split": 0.25
        }
        res = self.client.post("/api/settings/training-defaults", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        defaults = data["training_defaults"]
        self.assertEqual(defaults["batch_size"], 8)
        self.assertEqual(defaults["epochs"], 10)
        self.assertEqual(defaults["learning_rate"], 0.0005)
        self.assertEqual(defaults["optimizer"], "AdamW")
        self.assertEqual(defaults["val_split"], 0.25)

        # Check that GET /api/settings reflects this update
        settings_res = self.client.get("/api/settings")
        self.assertEqual(settings_res.json()["training_defaults"]["optimizer"], "AdamW")

    def test_03_resource_budget_configuration(self):
        """Test configuring resource budgets via POST /api/settings/resource-budget."""
        payload = {
            "latency_budget_ms": 30.0,
            "ram_budget_mb": 4096.0,
            "vram_budget_mb": 2048.0,
            "cpu_throttle_threshold": 90.0,
            "enforce_resource_budget": True,
            "threshold_a": 0.88,
            "threshold_probe": 0.82
        }
        res = self.client.post("/api/settings/resource-budget", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        budget = data["resource_budget"]
        self.assertEqual(budget["latency_budget_ms"], 30.0)
        self.assertEqual(budget["threshold_a"], 0.88)
        self.assertEqual(budget["threshold_probe"], 0.82)
        self.assertTrue(budget["enforce_resource_budget"])

        # Verify active_pair was synchronized
        self.assertEqual(self.manager.active_pair["latency_budget_ms"], 30.0)
        self.assertEqual(self.manager.active_pair["threshold_a"], 0.88)

    def test_04_backend_settings(self):
        """Test updating backend telemetry settings via POST /api/settings/backend."""
        payload = {
            "telemetry_poll_interval_ms": 1500,
            "cache_retention_limit": 150,
            "simulation_mode": False
        }
        res = self.client.post("/api/settings/backend", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        backend = data["backend"]
        self.assertEqual(backend["telemetry_poll_interval_ms"], 1500)
        self.assertEqual(backend["cache_retention_limit"], 150)

    def test_05_training_and_inference_with_ram_candidate(self):
        """
        Verify training and custom inference lifecycle:
        1. Register candidate model in RAM.
        2. Verify candidate appears in /api/training/custom-models.
        3. Execute custom inference using the candidate model.
        4. Configure Model A / Model B pair using this candidate.
        """
        run_id = "test-ram-candidate-p11"
        dummy_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 2)
        )
        candidate = {
            "model": dummy_model,
            "metadata": {
                "run_id": run_id,
                "architecture": "squeezenet",
                "dataset_id": "cifar-tiny",
                "class_names": ["airplane", "automobile"],
                "num_classes": 2,
                "epochs": 3,
                "batch_size": 4,
                "status": "COMPLETED",
                "preprocessing": {
                    "resize": [32, 32],
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225]
                }
            },
            "metrics": {
                "accuracy": 88.5,
                "f1_score": 0.88,
                "precision": 0.89,
                "recall": 0.88,
                "latency_ms": 8.4,
                "p95_latency_ms": 11.2,
                "peak_ram_mb": 145.0,
                "params": 500000,
                "model_size_mb": 2.0
            },
            "history": [
                {"epoch": 1, "train_loss": 0.65, "val_loss": 0.55, "train_acc": 70.0, "val_acc": 78.0},
                {"epoch": 2, "train_loss": 0.45, "val_loss": 0.38, "train_acc": 82.0, "val_acc": 88.5}
            ]
        }

        with CANDIDATE_MODELS_LOCK:
            CANDIDATE_MODELS[run_id] = candidate

        # Verify listing
        models_res = self.client.get("/api/training/custom-models")
        self.assertEqual(models_res.status_code, 200)
        models_data = models_res.json()["models"]
        self.assertTrue(any(m["run_id"] == run_id for m in models_data))

        # Test custom inference via uploaded image
        test_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        upload_res = self.client.post(
            f"/api/inference/upload?model_mode=custom&custom_run_id={run_id}",
            files={"file": ("sample.png", img_bytes, "image/png")}
        )
        self.assertEqual(upload_res.status_code, 200)
        upload_data = upload_res.json()
        self.assertTrue(upload_data["success"])
        self.assertEqual(upload_data["model_mode"], "custom")
        self.assertIn(upload_data["primary"]["prediction_class"], ["airplane", "automobile"])

        # Test configuring this candidate as Model A in ADAPT-EDGE pair
        pair_res = self.client.post("/api/inference/pair", json={
            "model_a_id": run_id,
            "model_b_id": "mobilenet_v3_large_baseline",
            "threshold_a": 0.85,
            "threshold_probe": 0.80,
            "latency_budget_ms": 25.0
        })
        self.assertEqual(pair_res.status_code, 200)
        pair_data = pair_res.json()["active_pair"]
        self.assertEqual(pair_data["model_a_id"], run_id)
        # Model B is MobileNetV3-Large, so probes remain compatible
        self.assertTrue(pair_data["probes_compatible"])

    def test_06_session_cleanup_testing(self):
        """
        Verify that POST /api/settings/cleanup completely resets RAM candidate models,
        caches, and memory without leaving lingering files.
        """
        # Ensure at least one candidate is in RAM
        self.assertGreater(len(CANDIDATE_MODELS), 0)

        # Call cleanup endpoint
        res = self.client.post("/api/settings/cleanup")
        self.assertEqual(res.status_code, 200)
        cleanup_data = res.json()
        self.assertTrue(cleanup_data["success"])
        self.assertEqual(cleanup_data["remaining_candidate_models_in_ram"], 0)

        # Verify RAM dictionary is empty
        with CANDIDATE_MODELS_LOCK:
            self.assertEqual(len(CANDIDATE_MODELS), 0)

        # Verify custom-models endpoint returns 0 candidates
        models_res = self.client.get("/api/training/custom-models")
        self.assertEqual(len(models_res.json()["models"]), 0)

        # Verify zero candidate files on disk
        rogue_files = list(PROJECT_ROOT.glob("outputs/**/test-ram-candidate*"))
        self.assertEqual(len(rogue_files), 0, "No candidate weights may exist on disk.")

    def test_07_protected_checkpoint_integrity_verification(self):
        """Verify GET /api/settings/verify-integrity checks outputs/ and confirms clean research isolation."""
        res = self.client.get("/api/settings/verify-integrity")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("checkpoints", data)
        self.assertIn("research_data_isolation", data)

        isolation = data["research_data_isolation"]
        self.assertTrue(isolation["clean_isolation"])
        self.assertEqual(len(isolation["rogue_candidate_checkpoints"]), 0)

        # Confirm all 5 expected checkpoints are audited
        ckpts = data["checkpoints"]
        self.assertIn("model_a", ckpts)
        self.assertIn("model_b", ckpts)
        self.assertIn("probe_4", ckpts)
        self.assertIn("probe_9", ckpts)
        self.assertIn("probe_13", ckpts)


if __name__ == "__main__":
    unittest.main()
