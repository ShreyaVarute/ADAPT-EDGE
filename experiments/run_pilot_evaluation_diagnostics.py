"""
ADAPT-EDGE: 500-Image Diagnostic Pilot Evaluation

Compares:
1. Model A only
2. Model B only
3. Conventional switching
4. Actual integrated ADAPT-EDGE pipeline

Records:
- Accuracy, precision, recall, macro F1
- Mean, median and P95 latency
- CPU and RAM usage
- GPU utilization and temperature
- VRAM usage and peak allocated memory
- Probe execution and fallback decisions
- Resource-policy violations
- Resource-monitor timing and cache information
- Controller timing and decision traces
- Internal versus external pipeline latency

Existing checkpoints and previous experiment files are read-only.
"""

import sys
import csv
import json
import time
import threading
import statistics
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# ============================================================
# 1. PROJECT PATHS AND IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_a import ModelA
from models.model_b import ModelB
from pipeline.adapt_edge_pipeline import AdaptEdgePipeline


# ============================================================
# 2. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

EVAL_COUNT = 500
EVAL_START = 0

BATCH_SIZE = 1
NUM_WARMUP = 5

MODEL_A_THRESHOLD = 0.80
PROBE_THRESHOLD = 0.85
LATENCY_BUDGET_MS = 25.0

MONITOR_INTERVAL_SECONDS = 0.5

TIMESTAMP = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "pilot_evaluation"
RUN_DIR = RESULTS_ROOT / f"run_diagnostics_{TIMESTAMP}"

RUN_DIR.mkdir(parents=True, exist_ok=False)


# ============================================================
# 3. HARDWARE MONITORING
# ============================================================

class HardwareSampler:
    """
    Periodically records hardware snapshots.

    The sampler uses the same ResourceMonitor instance
    as the integrated ADAPT-EDGE pipeline.
    """

    def __init__(self, monitor, interval=0.5):

        self.monitor = monitor
        self.interval = interval

        self.samples = []
        self.running = False
        self.thread = None

        self.lock = threading.Lock()

    def _sample_loop(self):

        while self.running:

            try:
                snapshot = self.monitor.snapshot(
                    force_refresh=True
                )

                with self.lock:
                    self.samples.append(snapshot)

            except Exception as error:

                with self.lock:
                    self.samples.append({
                        "monitoring_error": str(error)
                    })

            time.sleep(self.interval)

    def start(self):

        self.samples = []
        self.running = True

        self.thread = threading.Thread(
            target=self._sample_loop,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def get_samples(self):

        with self.lock:
            return list(self.samples)


# ============================================================
# 4. DATASET
# ============================================================

def load_test_dataset():

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    ])

    dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data" / "cifar10"),
        train=False,
        download=False,
        transform=transform
    )

    end_index = min(
        EVAL_START + EVAL_COUNT,
        len(dataset)
    )

    if EVAL_START >= end_index:
        raise ValueError("Invalid evaluation range.")

    indices = list(range(EVAL_START, end_index))

    print("\nDataset:", "CIFAR-10")
    print("Total test images:", len(dataset))
    print("Images selected:", len(indices))
    print(
        "Selected range:",
        EVAL_START,
        "to",
        end_index - 1
    )

    return dataset, indices


# ============================================================
# 5. MODEL LOADING
# ============================================================

def load_baseline_models():

    print("\nLoading baseline models...")

    model_a = ModelA(
        num_classes=10,
        pretrained=False
    )

    model_b = ModelB(
        num_classes=10,
        pretrained=False
    )

    path_a = (
        PROJECT_ROOT
        / "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth"
    )

    path_b = (
        PROJECT_ROOT
        / "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"
    )

    checkpoint_a = torch.load(
        path_a,
        map_location=DEVICE,
        weights_only=False
    )

    checkpoint_b = torch.load(
        path_b,
        map_location=DEVICE,
        weights_only=False
    )

    model_a.load_state_dict(
        checkpoint_a["model_state_dict"]
    )

    model_b.load_state_dict(
        checkpoint_b["model_state_dict"]
    )

    model_a = model_a.to(DEVICE).eval()
    model_b = model_b.to(DEVICE).eval()

    print("Model A loaded.")
    print("Model B loaded.")

    return model_a, model_b


# ============================================================
# 6. TIMING UTILITIES
# ============================================================

def synchronize():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def reset_gpu_peak():

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)


def get_gpu_peak_mb():

    if DEVICE.type == "cuda":

        return round(
            torch.cuda.max_memory_allocated(DEVICE)
            / (1024 ** 2),
            3
        )

    return None


def timed_call(function):

    synchronize()

    start = time.perf_counter()

    result = function()

    synchronize()

    latency = (
        time.perf_counter() - start
    ) * 1000.0

    return result, latency


@torch.inference_mode()
def get_prediction(model, image):

    logits = model(image)

    probabilities = F.softmax(
        logits,
        dim=1
    )

    confidence, prediction = probabilities.max(
        dim=1
    )

    return (
        int(prediction.item()),
        float(confidence.item())
    )