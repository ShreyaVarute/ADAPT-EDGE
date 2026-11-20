import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in str(__import__('sys').path):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.resource_policy import ResourcePolicy
from pipeline.adapt_edge_pipeline import AdaptEdgePipeline


REPORT_DIR = PROJECT_ROOT / "results" / "live_resource_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_cifar10_samples(limit: int = 3) -> List[Tuple[torch.Tensor, int, int]]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR10(root=str(PROJECT_ROOT / "data" / "cifar10"), train=False, download=False, transform=transform)
    samples: List[Tuple[torch.Tensor, int, int]] = []
    for index in range(min(limit, len(dataset))):
        image, label = dataset[index]
        samples.append((image.unsqueeze(0), int(label), index))
    return samples


def snapshot_summary(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if snapshot is None:
        return {"available": False, "reason": "no_snapshot"}
    return {
        "timestamp": snapshot.get("timestamp"),
        "snapshot_age_s": snapshot.get("snapshot_age_s"),
        "monitoring_time_ms": snapshot.get("monitoring_time_ms"),
        "cpu_percent": snapshot.get("cpu_percent"),
        "ram_percent": snapshot.get("ram_percent"),
        "ram_available_mb": snapshot.get("ram_available_mb"),
        "gpu_available": snapshot.get("gpu_available"),
        "gpu_utilization_percent": snapshot.get("gpu_utilization_percent"),
        "gpu_temperature_c": snapshot.get("gpu_temperature_c"),
        "total_vram_mb": snapshot.get("total_vram_mb"),
        "gpu_free_vram_mb": snapshot.get("gpu_free_vram_mb"),
        "gpu_torch_allocated_mb": snapshot.get("gpu_torch_allocated_mb"),
        "gpu_torch_reserved_mb": snapshot.get("gpu_torch_reserved_mb"),
        "is_mocked": snapshot.get("is_mocked", False),
        "diagnostics": snapshot.get("diagnostics", {}),
    }


@torch.inference_mode()
def measure_stage_latencies(pipeline: AdaptEdgePipeline, image: torch.Tensor) -> Dict[str, float]:
    """Measure actual Model A, probe-head, and full Model B timings without retraining or mocking."""
    image = image.to(pipeline.device)
    device = pipeline.device
    stages: Dict[str, float] = {}
    snapshot = pipeline.resource_monitor.snapshot(force_refresh=True)
    policy_eval = pipeline.resource_policy.evaluate(snapshot)

    t0 = time.perf_counter()
    logits_a = pipeline.model_a(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stages["model_a_ms"] = (time.perf_counter() - t0) * 1000.0

    if not policy_eval.is_healthy:
        stages["probe_4_ms"] = 0.0
        stages["probe_9_ms"] = 0.0
        stages["probe_13_ms"] = 0.0
        stages["full_model_b_ms"] = 0.0
        return stages

    t0 = time.perf_counter()
    features_4 = pipeline.sequential_b.start(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stages["probe_4_ms"] = (time.perf_counter() - t0) * 1000.0
    _ = pipeline.sequential_b.predict_probe(features_4, 4)

    t0 = time.perf_counter()
    features_9 = pipeline.sequential_b.continue_to(features_4, 5, 9)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stages["probe_9_ms"] = (time.perf_counter() - t0) * 1000.0
    _ = pipeline.sequential_b.predict_probe(features_9, 9)

    t0 = time.perf_counter()
    features_13 = pipeline.sequential_b.continue_to(features_9, 10, 13)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stages["probe_13_ms"] = (time.perf_counter() - t0) * 1000.0
    _ = pipeline.sequential_b.predict_probe(features_13, 13)

    t0 = time.perf_counter()
    _ = pipeline.sequential_b.finish(features_13)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stages["full_model_b_ms"] = (time.perf_counter() - t0) * 1000.0
    return stages


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipeline = AdaptEdgePipeline(
        device=device,
        max_latency_budget_ms=25.0,
        enable_cache=True,
        enforce_resource_budget=True,
    )

    policy = pipeline.resource_policy
    monitor = pipeline.resource_monitor
    controller = pipeline.controller

    samples = load_cifar10_samples(limit=3)
    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "hardware_summary": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": None,
        },
        "resource_policy": {
            "config": policy.config.__dict__,
        },
        "samples": [],
        "decision_counts": {"RETAIN": 0, "PROBE MORE": 0, "SWITCH": 0},
        "resource_pressure_observed": False,
        "resource_pressure_caused_decision": False,
        "notes": [],
    }

    if torch.cuda.is_available():
        report["hardware_summary"]["gpu_name"] = torch.cuda.get_device_name(device)

    for index, (image, label, dataset_index) in enumerate(samples):
        snapshot_before = monitor.snapshot(force_refresh=True)
        policy_eval_before = policy.evaluate(snapshot_before)

        start = time.perf_counter()
        result = pipeline.predict(image, request_id=f"live_scan_{dataset_index}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        snapshot_after = monitor.snapshot(force_refresh=True)
        policy_eval_after = policy.evaluate(snapshot_after)

        stage_latencies = measure_stage_latencies(pipeline, image)
        decision = result.get("action")
        if decision in report["decision_counts"]:
            report["decision_counts"][decision] += 1

        resource_violations = list(result.get("resource_violations", []))
        if resource_violations:
            report["resource_pressure_observed"] = True
        if result.get("status") in {"RESOURCE_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED_RETAIN"} and resource_violations:
            report["resource_pressure_caused_decision"] = True

        sample_report = {
            "sample_index": dataset_index,
            "image_label": label,
            "decision": decision,
            "status": result.get("status"),
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "uncertainty": result.get("uncertainty"),
            "reason": result.get("reason"),
            "resource_violations": resource_violations,
            "execution_latency_ms": round(elapsed_ms, 3),
            "result_latency_ms": round(float(result.get("latency_ms", 0.0)), 3),
            "model_a_ms": result.get("latencies", {}).get("model_a_ms"),
            "probe_ms": stage_latencies.get("probe_seed_ms", 0.0),
            "full_model_b_ms": stage_latencies.get("full_model_b_ms", 0.0),
            "hardware_before": snapshot_summary(snapshot_before),
            "hardware_after": snapshot_summary(snapshot_after),
            "policy_before": {
                "status": policy_eval_before.status,
                "is_healthy": policy_eval_before.is_healthy,
                "violations": policy_eval_before.violations,
                "unmonitored": policy_eval_before.unmonitored,
            },
            "policy_after": {
                "status": policy_eval_after.status,
                "is_healthy": policy_eval_after.is_healthy,
                "violations": policy_eval_after.violations,
                "unmonitored": policy_eval_after.unmonitored,
            },
            "decision_trace": result.get("decision_trace", []),
        }
        report["samples"].append(sample_report)

    if not report["resource_pressure_observed"]:
        report["notes"].append(
            "No resource-policy violations were observed in the sampled live hardware readouts; this indicates the system was not naturally operating under a constrained state during validation."
        )
    else:
        report["notes"].append(
            "Live resource-policy violations were observed in the sampled hardware readouts; all decisions are tied to the captured readings, not to mocked sensors."
        )

    report_path = REPORT_DIR / f"resource_awareness_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"Saved live diagnostic report to: {report_path}")
    print(json.dumps({
        "device": str(device),
        "decision_counts": report["decision_counts"],
        "resource_pressure_observed": report["resource_pressure_observed"],
        "resource_pressure_caused_decision": report["resource_pressure_caused_decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
