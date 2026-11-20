import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.model_a import ModelA
from models.model_b import ModelB
from probes.probe_heads import ProbeHead, SequentialModelB
from controller.decision_controller import DecisionController
from controller.cost_estimator import CostEstimator
from cache.continuation_cache import ContinuationCache
from utils.resource_monitor import ResourceMonitor
from utils.profiling import synchronize
from utils.calibration import TemperatureScaler
from utils.experiment_tracker import ExperimentTracker


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

PROBES = {
    4: 40,
    9: 80,
    13: 160
}

MODEL_A_THRESHOLD = 0.80
PROBE_THRESHOLD = 0.85
MAX_LATENCY_MS = 25.0


def load_model(model_class, checkpoint_path):
    model = model_class(pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(DEVICE).eval()


def load_sequential_model(cache=None):
    backbone = load_model(
        ModelB,
        "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"
    )
    sequential = SequentialModelB(backbone, DEVICE, cache=cache)

    for block, channels in PROBES.items():
        checkpoint = torch.load(
            f"outputs/probes/block_{block}/best_probe.pth",
            map_location="cpu",
            weights_only=False
        )
        head = ProbeHead(channels, num_classes=10)
        head.load_state_dict(checkpoint["head_state_dict"])
        sequential.register_probe(block, head)

    return sequential


def load_test_images(start_index=1000, count=10):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=False,
        download=False,
        transform=transform
    )

    return [
        (dataset[i][0].unsqueeze(0), dataset[i][1], i)
        for i in range(start_index, start_index + count)
    ]


@torch.inference_mode()
def run_adapt_edge_sample(
    image: torch.Tensor,
    request_id: str,
    model_a: ModelA,
    sequential_b: SequentialModelB,
    controller: DecisionController,
    cache: Optional[ContinuationCache] = None,
    checkpoint_id: str = "model_b_large_v1"
) -> Dict[str, Any]:
    """
    Executes ADAPT-EDGE decision-selective computation with optional continuation caching.
    """
    synchronize(DEVICE)
    start_time = time.perf_counter()

    # Input fingerprint for safe continuation cache
    img_fingerprint = ContinuationCache.generate_input_fingerprint(image)

    # Stage 1: Model A
    logits_a = model_a(image)
    synchronize(DEVICE)
    elapsed_a_ms = (time.perf_counter() - start_time) * 1000.0

    init_decision = controller.decide_initial(logits_a, current_latency_ms=elapsed_a_ms)

    if init_decision["action"] == "RETAIN":
        synchronize(DEVICE)
        total_latency = (time.perf_counter() - start_time) * 1000.0
        return {
            "action": "RETAIN",
            "prediction": init_decision["prediction"],
            "confidence": init_decision["confidence"],
            "uncertainty": init_decision["uncertainty"],
            "executed_probes": [],
            "used_full_b": False,
            "cache_hits": 0,
            "decision_score": init_decision["decision_score"],
            "reason": init_decision["reason"],
            "latency_ms": total_latency
        }

    # Model A is uncertain -> Proceed to Sequential Model B Probing
    executed_probes = []
    features = None
    current_block = 0
    cache_hits = 0

    target_probe = init_decision["selected_probe"] or 4

    while target_probe is not None:
        # Check cache for existing features at current_block if applicable
        if cache is not None and current_block > 0:
            cached_feat = cache.get(img_fingerprint, checkpoint_id, current_block, target_device=DEVICE)
            if cached_feat is not None:
                features = cached_feat
                cache_hits += 1

        # Execute missing feature segment up to target_probe
        if current_block == 0:
            features = sequential_b.start(image)
        else:
            features = sequential_b.continue_to(features, current_block + 1, target_probe)

        current_block = target_probe

        # Cache newly computed features
        if cache is not None:
            cache.put(request_id, img_fingerprint, checkpoint_id, current_block, features)

        # Evaluate probe head
        probe_logits = sequential_b.predict_probe(features, target_probe)
        executed_probes.append(target_probe)

        synchronize(DEVICE)
        current_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        probe_decision = controller.decide_probe(
            current_probe_block=target_probe,
            probe_logits=probe_logits,
            current_latency_ms=current_elapsed_ms,
            executed_probes=executed_probes
        )

        if probe_decision["action"] == "RETAIN":
            total_latency = (time.perf_counter() - start_time) * 1000.0
            return {
                "action": "RETAIN",
                "prediction": probe_decision["prediction"],
                "confidence": probe_decision["confidence"],
                "uncertainty": probe_decision["uncertainty"],
                "executed_probes": executed_probes,
                "used_full_b": False,
                "cache_hits": cache_hits,
                "decision_score": probe_decision["decision_score"],
                "reason": probe_decision["reason"],
                "latency_ms": total_latency
            }

        elif probe_decision["action"] == "PROBE MORE":
            target_probe = probe_decision["selected_probe"]

        else:  # SWITCH
            break

    # Final Model B completion fallback (Blocks 14..16 + Classifier)
    # Check if features at current_block are cached or in memory
    if cache is not None and features is None:
        features = cache.get(img_fingerprint, checkpoint_id, current_block, target_device=DEVICE)

    if features is None:
        # Safety fallback: recompute from start if cache was absent
        features = sequential_b.start(image)
        if current_block > 4:
            features = sequential_b.continue_to(features, 5, current_block)

    final_logits = sequential_b.finish(features)
    probs_b = F.softmax(final_logits, dim=1)
    final_conf, final_pred = probs_b.max(dim=1)

    synchronize(DEVICE)
    total_latency = (time.perf_counter() - start_time) * 1000.0

    return {
        "action": "SWITCH",
        "prediction": final_pred.item(),
        "confidence": final_conf.item(),
        "uncertainty": -torch.sum(probs_b * torch.log(probs_b.clamp(min=1e-12))).item(),
        "executed_probes": executed_probes,
        "used_full_b": True,
        "cache_hits": cache_hits,
        "decision_score": 0.5,
        "reason": "Completed full Model B after probe evidence collection.",
        "latency_ms": total_latency
    }


def main():
    print("=" * 65)
    print("ADAPT-EDGE PHASE 3 & 4 EVALUATION")
    print("=" * 65)
    print("Device:", DEVICE)

    model_a = load_model(ModelA, "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth")
    
    # Initialize continuation cache
    cache = ContinuationCache(max_entries=20, max_memory_mb=64.0)
    sequential_b = load_sequential_model(cache=cache)

    cost_estimator = CostEstimator()
    monitor = ResourceMonitor()

    controller = DecisionController(
        model_a_threshold=MODEL_A_THRESHOLD,
        probe_threshold=PROBE_THRESHOLD,
        max_latency_ms=MAX_LATENCY_MS,
        cost_estimator=cost_estimator,
        resource_monitor=monitor
    )

    tracker = ExperimentTracker(
        experiment_name="adapt_edge_phase3_phase4",
        config={
            "model_a_threshold": MODEL_A_THRESHOLD,
            "probe_threshold": PROBE_THRESHOLD,
            "max_latency_ms": MAX_LATENCY_MS,
            "device": str(DEVICE),
            "probes": list(PROBES.keys())
        }
    )

    samples = load_test_images(start_index=1000, count=10)
    results = []

    for idx, (image, label, ds_idx) in enumerate(samples):
        image = image.to(DEVICE)
        req_id = f"req_{ds_idx}_{idx}"

        res = run_adapt_edge_sample(
            image=image,
            request_id=req_id,
            model_a=model_a,
            sequential_b=sequential_b,
            controller=controller,
            cache=cache
        )

        res["image_index"] = idx
        res["dataset_index"] = ds_idx
        res["true_label"] = label
        res["true_class"] = CLASS_NAMES[label]
        res["predicted_class"] = CLASS_NAMES[res["prediction"]]
        res["correct"] = (res["prediction"] == label)

        # Invalidate request entries after completion
        cache.invalidate_request(req_id)

        print(f"Sample {idx+1}: True={res['true_class']}, Pred={res['predicted_class']}, "
              f"Action={res['action']}, Correct={res['correct']}, Latency={res['latency_ms']:.2f}ms")

        results.append(res)

    cache_stats = cache.get_cache_stats()
    print("\nCache Stats:", cache_stats)

    acc = (sum(r["correct"] for r in results) / len(results)) * 100.0
    print(f"\nFinal Accuracy on 10 Diagnostic Samples: {acc:.2f}%")

    # Save to timestamped experiment run directory
    tracker.log_results({
        "accuracy": acc,
        "sample_count": len(results),
        "cache_stats": cache_stats,
        "results": results
    })

    print(f"All outputs safely saved to: {tracker.run_dir}")


if __name__ == "__main__":
    main()
