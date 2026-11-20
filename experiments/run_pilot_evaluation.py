
"""
ADAPT-EDGE: 500-Image Pilot Evaluation

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

Existing checkpoints and experiment files are read-only.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_a import ModelA
from models.model_b import ModelB
from pipeline.adapt_edge_pipeline import AdaptEdgePipeline


# ============================================================
# 1. CONFIGURATION
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
RUN_DIR = RESULTS_ROOT / f"run_{TIMESTAMP}"

RUN_DIR.mkdir(parents=True, exist_ok=False)


# ============================================================
# 2. HARDWARE MONITORING
# ============================================================

class HardwareSampler:
    """
    Periodically records real hardware snapshots during inference.

    Sensor values that are unavailable remain None.
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
# 3. DATASET
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
    print("Selected range:", EVAL_START, "to", end_index - 1)

    return dataset, indices


# ============================================================
# 4. MODEL LOADING
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
# 5. TIMING UTILITIES
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

    probabilities = F.softmax(logits, dim=1)

    confidence, prediction = probabilities.max(dim=1)

    return (
        int(prediction.item()),
        float(confidence.item())
    )


# ============================================================
# 6. STRATEGY 1: MODEL A ONLY
# ============================================================

@torch.inference_mode()
def run_model_a(model_a, image):

    return timed_call(
        lambda: get_prediction(model_a, image)
    )


# ============================================================
# 7. STRATEGY 2: MODEL B ONLY
# ============================================================

@torch.inference_mode()
def run_model_b(model_b, image):

    return timed_call(
        lambda: get_prediction(model_b, image)
    )


# ============================================================
# 8. STRATEGY 3: CONVENTIONAL SWITCHING
# ============================================================

@torch.inference_mode()
def run_conventional_switching(
    model_a,
    model_b,
    image
):

    start = time.perf_counter()

    pred_a, conf_a = get_prediction(model_a, image)

    if conf_a >= MODEL_A_THRESHOLD:

        synchronize()

        latency = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "prediction": pred_a,
            "confidence": conf_a,
            "selected_model": "A",
            "used_full_b": False
        }, latency

    pred_b, conf_b = get_prediction(model_b, image)

    synchronize()

    latency = (
        time.perf_counter() - start
    ) * 1000.0

    return {
        "prediction": pred_b,
        "confidence": conf_b,
        "selected_model": "B",
        "used_full_b": True
    }, latency


# ============================================================
# 9. ACTUAL INTEGRATED ADAPT-EDGE
# ============================================================

@torch.inference_mode()
def run_integrated_adapt_edge(pipeline, image, request_id):

    synchronize()

    start = time.perf_counter()

    result = pipeline.predict(
        image,
        request_id=request_id
    )

    synchronize()

    external_latency = (
        time.perf_counter() - start
    ) * 1000.0

    return result, external_latency


# ============================================================
# 10. METRIC CALCULATION
# ============================================================

def calculate_classification_metrics(records, prediction_key):

    labels = [r["true_label"] for r in records]

    predictions = [
        r[prediction_key]
        for r in records
    ]

    classes = range(10)

    precisions = []
    recalls = []
    f1_scores = []

    for class_id in classes:

        tp = sum(
            1 for y, p in zip(labels, predictions)
            if y == class_id and p == class_id
        )

        fp = sum(
            1 for y, p in zip(labels, predictions)
            if y != class_id and p == class_id
        )

        fn = sum(
            1 for y, p in zip(labels, predictions)
            if y == class_id and p != class_id
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0 else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0 else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    correct = sum(
        1 for y, p in zip(labels, predictions)
        if y == p
    )

    return {
        "total_images": len(records),
        "correct_predictions": correct,
        "accuracy_percent": round(
            100.0 * correct / len(records), 3
        ),
        "macro_precision_percent": round(
            100.0 * statistics.mean(precisions), 3
        ),
        "macro_recall_percent": round(
            100.0 * statistics.mean(recalls), 3
        ),
        "macro_f1_percent": round(
            100.0 * statistics.mean(f1_scores), 3
        )
    }


def latency_metrics(latencies):

    values = np.asarray(latencies, dtype=float)

    return {
        "mean_ms": round(float(np.mean(values)), 3),
        "median_ms": round(float(np.median(values)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "min_ms": round(float(np.min(values)), 3),
        "max_ms": round(float(np.max(values)), 3)
    }


# ============================================================
# 11. HARDWARE SUMMARY
# ============================================================

def numeric_values(samples, key):

    return [
        float(s[key])
        for s in samples
        if isinstance(s, dict)
        and isinstance(s.get(key), (int, float))
    ]


def hardware_summary(samples):

    metrics = [
        "cpu_percent",
        "ram_percent",
        "ram_used_mb",
        "ram_available_mb",
        "gpu_utilization_percent",
        "gpu_temperature_c",
        "gpu_used_vram_mb",
        "gpu_torch_allocated_mb",
        "gpu_torch_reserved_mb",
        "gpu_torch_peak_mb"
    ]

    summary = {
        "sample_count": len(samples),
        "unavailable_metrics": {}
    }

    for metric in metrics:

        values = numeric_values(samples, metric)

        if values:

            summary[metric] = {
                "average": round(
                    statistics.mean(values), 3
                ),
                "maximum": round(max(values), 3),
                "minimum": round(min(values), 3)
            }

        else:

            summary[metric] = None

            summary["unavailable_metrics"][metric] = (
                "No numeric hardware readings were captured."
            )

    return summary


# ============================================================
# 12. CSV EXPORT
# ============================================================

def save_per_image_csv(records):

    path = RUN_DIR / "per_image_results.csv"

    columns = [
        "image_index",
        "dataset_index",
        "true_label",

        "model_a_prediction",
        "model_a_confidence",
        "model_a_latency_ms",

        "model_b_prediction",
        "model_b_confidence",
        "model_b_latency_ms",

        "conventional_prediction",
        "conventional_selected_model",
        "conventional_latency_ms",

        "adapt_prediction",
        "adapt_confidence",
        "adapt_action",
        "adapt_status",
        "adapt_latency_ms",
        "adapt_pipeline_latency_ms",
        "adapt_executed_probes",
        "adapt_used_full_b",
        "adapt_cache_hits",
        "adapt_resource_violations"
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                key: (
                    json.dumps(record.get(key))
                    if isinstance(record.get(key), (list, dict))
                    else record.get(key)
                )
                for key in columns
            })

    return path


def save_hardware_csv(samples):

    path = RUN_DIR / "hardware_metrics.csv"

    columns = [
        "timestamp",
        "cpu_percent",
        "ram_percent",
        "ram_used_mb",
        "ram_available_mb",
        "gpu_utilization_percent",
        "gpu_temperature_c",
        "total_vram_mb",
        "gpu_used_vram_mb",
        "gpu_free_vram_mb",
        "gpu_torch_allocated_mb",
        "gpu_torch_reserved_mb",
        "gpu_torch_peak_mb",
        "monitoring_time_ms"
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        for sample in samples:

            writer.writerow({
                key: sample.get(key)
                for key in columns
            })

    return path


# ============================================================
# 13. GRAPH GENERATION
# ============================================================

def generate_graphs(summary):

    strategies = [
        "Model A Only",
        "Model B Only",
        "Conventional Switching",
        "ADAPT-EDGE"
    ]

    colors = None

    # Accuracy graph
    accuracies = [
        summary["strategies"][s]["accuracy_percent"]
        for s in strategies
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(strategies, accuracies)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_comparison.png", dpi=200)
    plt.close()

    # Latency graph
    latencies = [
        summary["strategies"][s]["latency"]["mean_ms"]
        for s in strategies
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(strategies, latencies)
    plt.ylabel("Mean Latency (ms)")
    plt.title("Mean Inference Latency Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "latency_comparison.png", dpi=200)
    plt.close()

    # Hardware graph
    hardware = summary["hardware"]

    names = [
        "CPU",
        "RAM",
        "GPU"
    ]

    values = [
        hardware.get("cpu_percent"),
        hardware.get("ram_percent"),
        hardware.get("gpu_utilization_percent")
    ]

    plot_names = []
    plot_values = []

    for name, metric in zip(names, values):

        if metric is not None:

            plot_names.append(name)
            plot_values.append(metric["average"])

    if plot_names:

        plt.figure(figsize=(8, 5))
        plt.bar(plot_names, plot_values)
        plt.ylabel("Utilization (%)")
        plt.title("Average Hardware Utilization")
        plt.tight_layout()
        plt.savefig(RUN_DIR / "resource_usage.png", dpi=200)
        plt.close()


# ============================================================
# 14. MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("ADAPT-EDGE 100-IMAGE PILOT EVALUATION")
    print("=" * 65)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(DEVICE)
        )

    dataset, indices = load_test_dataset()

    model_a, model_b = load_baseline_models()

    # The integrated pipeline loads its own checkpoints.
    # All checkpoint loading is read-only.
    pipeline = AdaptEdgePipeline(
        device=DEVICE,
        model_a_threshold=MODEL_A_THRESHOLD,
        probe_threshold=PROBE_THRESHOLD,
        max_latency_budget_ms=LATENCY_BUDGET_MS,
        enable_cache=True,
        enable_calibration=True,
        enforce_resource_budget=True
    )

    print("\nIntegrated ADAPT-EDGE pipeline loaded.")

    print("\nWarming up models...")

    warmup_image, _ = dataset[indices[0]]
    warmup_image = warmup_image.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        for _ in range(NUM_WARMUP):

            model_a(warmup_image)
            model_b(warmup_image)

            pipeline.predict(
                warmup_image,
                request_id=f"warmup_{_}"
            )

    synchronize()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    print("Warm-up completed.")

    sampler = HardwareSampler(
        pipeline.resource_monitor,
        interval=MONITOR_INTERVAL_SECONDS
    )

    sampler.start()

    records = []

    decision_counts = {
        "RETAIN": 0,
        "PROBE MORE": 0,
        "SWITCH": 0
    }

    status_counts = {}
    probe_counts = {
        4: 0,
        9: 0,
        13: 0
    }

    full_b_count = 0
    cache_hit_count = 0
    resource_violation_count = 0

    print("\nStarting evaluation...\n")

    try:

        for position, dataset_index in enumerate(indices):

            image, label = dataset[dataset_index]

            image = image.unsqueeze(0).to(DEVICE)
            label = int(label)

            record = {
                "image_index": position,
                "dataset_index": dataset_index,
                "true_label": label
            }

            # ------------------------------------------------
            # Model A
            # ------------------------------------------------

            reset_gpu_peak()

            (pred_a, conf_a), lat_a = run_model_a(
                model_a,
                image
            )

            record.update({
                "model_a_prediction": pred_a,
                "model_a_confidence": conf_a,
                "model_a_latency_ms": lat_a,
                "model_a_peak_vram_mb": get_gpu_peak_mb()
            })

            # ------------------------------------------------
            # Model B
            # ------------------------------------------------

            reset_gpu_peak()

            (pred_b, conf_b), lat_b = run_model_b(
                model_b,
                image
            )

            record.update({
                "model_b_prediction": pred_b,
                "model_b_confidence": conf_b,
                "model_b_latency_ms": lat_b,
                "model_b_peak_vram_mb": get_gpu_peak_mb()
            })

            # ------------------------------------------------
            # Conventional Switching
            # ------------------------------------------------

            reset_gpu_peak()

            conventional, lat_conv = run_conventional_switching(
                model_a,
                model_b,
                image
            )

            record.update({
                "conventional_prediction": conventional["prediction"],
                "conventional_selected_model": conventional["selected_model"],
                "conventional_latency_ms": lat_conv,
                "conventional_used_full_b": conventional["used_full_b"],
                "conventional_peak_vram_mb": get_gpu_peak_mb()
            })

            # ------------------------------------------------
            # Actual Integrated ADAPT-EDGE
            # ------------------------------------------------

            reset_gpu_peak()

            adapt_result, lat_adapt = run_integrated_adapt_edge(
                pipeline,
                image,
                request_id=f"pilot_{dataset_index}"
            )

            prediction_adapt = int(adapt_result["prediction"])

            action = adapt_result.get("action", "UNKNOWN")
            status = adapt_result.get("status", "UNKNOWN")

            probes = adapt_result.get("executed_probes", [])
            violations = adapt_result.get("resource_violations", [])

            for probe in probes:

                if probe in probe_counts:
                    probe_counts[probe] += 1

            if action in decision_counts:
                decision_counts[action] += 1

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            used_full_b = bool(
                adapt_result.get("used_full_b", False)
            )

            full_b_count += int(used_full_b)

            cache_hit_count += int(
                adapt_result.get("cache_hits", 0)
            )

            resource_violation_count += len(violations)

            record.update({
                "adapt_prediction": prediction_adapt,
                "adapt_confidence": adapt_result.get("confidence"),
                "adapt_action": action,
                "adapt_status": status,
                "adapt_latency_ms": lat_adapt,
                "adapt_pipeline_latency_ms": adapt_result.get("latency_ms"),
                "adapt_executed_probes": probes,
                "adapt_used_full_b": used_full_b,
                "adapt_used_partial_b": adapt_result.get("used_partial_b"),
                "adapt_cache_hits": adapt_result.get("cache_hits"),
                "adapt_resource_violations": violations,
                "adapt_decision_trace": adapt_result.get("decision_trace"),
                "adapt_peak_vram_mb": get_gpu_peak_mb()
            })

            records.append(record)

            if (position + 1) % 10 == 0:

                print(
                    f"Processed {position + 1}/{len(indices)} images"
                )

    finally:

        sampler.stop()

    hardware_samples = sampler.get_samples()

    # ========================================================
    # 15. CALCULATE STRATEGY METRICS
    # ========================================================

    strategy_definitions = {
        "Model A Only": (
            "model_a_prediction",
            "model_a_latency_ms"
        ),
        "Model B Only": (
            "model_b_prediction",
            "model_b_latency_ms"
        ),
        "Conventional Switching": (
            "conventional_prediction",
            "conventional_latency_ms"
        ),
        "ADAPT-EDGE": (
            "adapt_prediction",
            "adapt_latency_ms"
        )
    }

    strategy_metrics = {}

    for strategy, (prediction_key, latency_key) in strategy_definitions.items():

        strategy_metrics[strategy] = (
            calculate_classification_metrics(
                records,
                prediction_key
            )
        )

        strategy_metrics[strategy]["latency"] = (
            latency_metrics([
                r[latency_key]
                for r in records
            ])
        )

    # ========================================================
    # 16. COMPUTATIONAL COST ACCOUNTING
    # ========================================================

    conventional_b_count = sum(
        1 for r in records
        if r["conventional_used_full_b"]
    )

    strategy_metrics["Conventional Switching"]["model_b_usage_percent"] = (
        100.0 * conventional_b_count / len(records)
    )

    strategy_metrics["ADAPT-EDGE"]["model_b_usage_percent"] = (
        100.0 * sum(
            1 for r in records
            if r["adapt_executed_probes"] or r["adapt_used_full_b"]
        ) / len(records)
    )

    # Model-stage counts are execution indicators, NOT FLOPs.
    strategy_metrics["ADAPT-EDGE"]["probe_counts"] = probe_counts

    strategy_metrics["ADAPT-EDGE"]["full_model_b_fallback_count"] = (
        full_b_count
    )

    strategy_metrics["ADAPT-EDGE"]["cache_hits"] = cache_hit_count

    strategy_metrics["ADAPT-EDGE"]["resource_violation_count"] = (
        resource_violation_count
    )

    # ========================================================
    # 17. HARDWARE SUMMARY
    # ========================================================

    hw_summary = hardware_summary(hardware_samples)

    # ========================================================
    # 18. SAVE REPORTS
    # ========================================================

    summary = {
        "experiment": "ADAPT-EDGE 100-Image Pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "gpu_name": (
            torch.cuda.get_device_name(DEVICE)
            if DEVICE.type == "cuda" else None
        ),
        "dataset": "CIFAR-10",
        "evaluation_start": EVAL_START,
        "evaluation_count": len(records),
        "batch_size": BATCH_SIZE,
        "warmup_iterations": NUM_WARMUP,
        "configuration": {
            "model_a_threshold": MODEL_A_THRESHOLD,
            "probe_threshold": PROBE_THRESHOLD,
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "cache_enabled": True,
            "resource_budget_enforced": True
        },
        "strategies": strategy_metrics,
        "decision_counts": decision_counts,
        "status_counts": status_counts,
        "hardware": hw_summary,
        "hardware_sampling_interval_seconds": MONITOR_INTERVAL_SECONDS,
        "hardware_samples": len(hardware_samples),
        "notes": [
            "ADAPT-EDGE uses the actual integrated AdaptEdgePipeline.",
            "Baseline and conventional strategies are independently evaluated.",
            "External wall-clock latency includes Python and pipeline overhead.",
            "Hardware statistics are sampled during the complete experiment.",
            "Hardware averages are observational and not isolated per strategy.",
            "Peak VRAM is measured per inference using PyTorch CUDA counters.",
            "Probe counts and model usage are execution indicators, not FLOPs.",
            "No model retraining or checkpoint modification was performed."
        ]
    }

    summary_path = RUN_DIR / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    per_image_path = save_per_image_csv(records)

    hardware_path = save_hardware_csv(hardware_samples)

    generate_graphs(summary)

    # ========================================================
    # 19. FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 75)
    print("PILOT EVALUATION SUMMARY")
    print("=" * 75)

    print(
        f"{'Strategy':<28}"
        f"{'Accuracy':>12}"
        f"{'Mean ms':>12}"
        f"{'P95 ms':>12}"
    )

    print("-" * 75)

    for strategy, metrics in strategy_metrics.items():

        print(
            f"{strategy:<28}"
            f"{metrics['accuracy_percent']:>11.2f}%"
            f"{metrics['latency']['mean_ms']:>12.3f}"
            f"{metrics['latency']['p95_ms']:>12.3f}"
        )

    print("\nADAPT-EDGE decisions:", decision_counts)
    print("Probe counts:", probe_counts)
    print("Full Model B fallbacks:", full_b_count)
    print("Resource violations:", resource_violation_count)

    print("\nResults saved in:")
    print(RUN_DIR)

    print("\nGenerated files:")
    print(summary_path)
    print(per_image_path)
    print(hardware_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
