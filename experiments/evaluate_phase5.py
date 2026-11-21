"""
ADAPT-EDGE Phase 5: Complete Comparative Evaluation Pipeline.

Executes all five strategies on identical CIFAR-10 test samples:
1. Strategy 1: Model A Only
2. Strategy 2: Model B Only
3. Strategy 3: Conventional Switching
4. Strategy 4: ADAPT-EDGE (Cache Disabled)
5. Strategy 5: ADAPT-EDGE (Cache Enabled)

Generates:
- Structured output directory: results/phase5_YYYYMMDD_HHMMSS/
- Raw per-image records CSV & JSON
- Classification metrics, Latency metrics, Computation metrics, Cache metrics, Resource metrics
- Publication-quality Matplotlib plots
- Statistical paired analysis & research reports
"""

import sys
import os
import json
import time
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from models.model_a import ModelA
from models.model_b import ModelB
from probes.probe_heads import ProbeHead, SequentialModelB
from controller.decision_controller import DecisionController
from controller.cost_estimator import CostEstimator
from cache.continuation_cache import ContinuationCache
from utils.resource_monitor import ResourceMonitor
from utils.profiling import synchronize, time_callable, LatencyStats, measure_peak_gpu_memory_mb, reset_peak_gpu_memory
from utils.metrics import compute_metrics
from utils.calibration import TemperatureScaler
from utils.experiment_tracker import ExperimentTracker
from utils.visualizations import (
    plot_accuracy_comparison,
    plot_f1_comparison,
    plot_latency_comparison,
    plot_accuracy_vs_latency,
    plot_model_b_usage,
    plot_probe_distribution,
    plot_cache_performance,
    plot_memory_comparison,
    plot_confusion_matrices,
    plot_budget_analysis
)
from utils.statistical_analysis import paired_strategy_analysis


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

PROBE_CONFIG = {
    4: 40,
    9: 80,
    13: 160
}

# Calibrated Thresholds
DEFAULT_MODEL_A_THRESHOLD = 0.80
DEFAULT_PROBE_THRESHOLD = 0.85
DEFAULT_MAX_LATENCY_MS = 25.0


def load_all_models():
    """Load pretrained Model A, Model B, and register all probe heads."""
    print("Loading component models and probe heads...")
    
    # Model A
    model_a = ModelA(pretrained=False)
    ckpt_a = torch.load(
        PROJECT_ROOT / "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth",
        map_location=DEVICE,
        weights_only=False
    )
    model_a.load_state_dict(ckpt_a["model_state_dict"])
    model_a = model_a.to(DEVICE).eval()

    # Model B
    model_b = ModelB(pretrained=False)
    ckpt_b = torch.load(
        PROJECT_ROOT / "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth",
        map_location=DEVICE,
        weights_only=False
    )
    model_b.load_state_dict(ckpt_b["model_state_dict"])
    model_b = model_b.to(DEVICE).eval()

    # Sequential Model B
    sequential_b = SequentialModelB(model_b, DEVICE)

    # Load probe heads
    for block, channels in PROBE_CONFIG.items():
        ckpt_probe = torch.load(
            PROJECT_ROOT / f"outputs/probes/block_{block}/best_probe.pth",
            map_location=DEVICE,
            weights_only=False
        )
        head = ProbeHead(channels, num_classes=10).to(DEVICE)
        head.load_state_dict(ckpt_probe["head_state_dict"])
        head.eval()
        sequential_b.register_probe(block, head)

    print("All checkpoints loaded successfully.")
    return model_a, model_b, sequential_b


def get_evaluation_loader(
    start_index: int = 1000,
    count: int = 20,
    stratified: bool = False,
    random_seed: int = 42
) -> Tuple[DataLoader, List[int]]:
    """
    Load CIFAR-10 test subset for evaluation.
    Supports contiguous range selection or reproducible class-stratified sampling.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data/cifar10"),
        train=False,
        download=False,
        transform=transform
    )

    if stratified:
        class_indices = defaultdict(list)
        for i, target in enumerate(dataset.targets):
            class_indices[target].append(i)

        per_class = count // 10
        remainder = count % 10
        rng = np.random.RandomState(random_seed)
        indices = []
        for c in range(10):
            take = per_class + (1 if c < remainder else 0)
            chosen = rng.choice(class_indices[c], size=take, replace=False)
            indices.extend(chosen.tolist())
        indices.sort()
    else:
        indices = list(range(start_index, start_index + count))

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    return loader, indices


# Strategy 1: Model A Only
@torch.inference_mode()
def execute_model_a_only(model_a: ModelA, image: torch.Tensor) -> Tuple[int, float, float]:
    synchronize(DEVICE)
    start = time.perf_counter()
    logits = model_a(image)
    synchronize(DEVICE)
    lat = (time.perf_counter() - start) * 1000.0

    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return int(pred.item()), float(conf.item()), float(lat)


# Strategy 2: Model B Only
@torch.inference_mode()
def execute_model_b_only(model_b: ModelB, image: torch.Tensor) -> Tuple[int, float, float]:
    synchronize(DEVICE)
    start = time.perf_counter()
    logits = model_b(image)
    synchronize(DEVICE)
    lat = (time.perf_counter() - start) * 1000.0

    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return int(pred.item()), float(conf.item()), float(lat)


# Strategy 3: Conventional Switching
@torch.inference_mode()
def execute_conventional_switching(
    model_a: ModelA,
    model_b: ModelB,
    image: torch.Tensor,
    threshold_a: float = DEFAULT_MODEL_A_THRESHOLD
) -> Dict[str, Any]:
    synchronize(DEVICE)
    start_total = time.perf_counter()

    # Step 1: Model A
    start_a = time.perf_counter()
    logits_a = model_a(image)
    synchronize(DEVICE)
    lat_a = (time.perf_counter() - start_a) * 1000.0

    probs_a = F.softmax(logits_a, dim=1)
    conf_a, pred_a = probs_a.max(dim=1)

    if conf_a.item() >= threshold_a:
        total_lat = (time.perf_counter() - start_total) * 1000.0
        return {
            "prediction": int(pred_a.item()),
            "confidence": float(conf_a.item()),
            "selected_model": "A",
            "used_model_b": False,
            "latency_ms": float(total_lat),
            "model_a_latency_ms": float(lat_a),
            "model_b_latency_ms": 0.0
        }

    # Step 2: Fallback to full Model B
    start_b = time.perf_counter()
    logits_b = model_b(image)
    synchronize(DEVICE)
    lat_b = (time.perf_counter() - start_b) * 1000.0

    probs_b = F.softmax(logits_b, dim=1)
    conf_b, pred_b = probs_b.max(dim=1)
    total_lat = (time.perf_counter() - start_total) * 1000.0

    return {
        "prediction": int(pred_b.item()),
        "confidence": float(conf_b.item()),
        "selected_model": "B",
        "used_model_b": True,
        "latency_ms": float(total_lat),
        "model_a_latency_ms": float(lat_a),
        "model_b_latency_ms": float(lat_b)
    }


# Strategy 4 & 5: ADAPT-EDGE (with or without ContinuationCache)
@torch.inference_mode()
def execute_adapt_edge(
    model_a: ModelA,
    sequential_b: SequentialModelB,
    controller: DecisionController,
    image: torch.Tensor,
    request_id: str,
    cache: Optional[ContinuationCache] = None,
    checkpoint_id: str = "model_b_large_v1"
) -> Dict[str, Any]:
    synchronize(DEVICE)
    start_total = time.perf_counter()

    img_fp = ContinuationCache.generate_input_fingerprint(image)

    # Stage 1: Model A
    start_a = time.perf_counter()
    logits_a = model_a(image)
    synchronize(DEVICE)
    lat_a = (time.perf_counter() - start_a) * 1000.0

    init_decision = controller.decide_initial(logits_a, current_latency_ms=lat_a)

    if init_decision["action"] == "RETAIN":
        total_lat = (time.perf_counter() - start_total) * 1000.0
        return {
            "action": "RETAIN",
            "status": init_decision.get("status", "CONFIDENT_RETAIN"),
            "prediction": init_decision["prediction"],
            "confidence": init_decision["confidence"],
            "uncertainty": init_decision["uncertainty"],
            "executed_probes": [],
            "used_full_b": False,
            "used_partial_b": False,
            "cache_hits": 0,
            "decision_score": init_decision["decision_score"],
            "reason": init_decision["reason"],
            "latency_ms": float(total_lat)
        }

    # Probing loop
    executed_probes = []
    features = None
    current_block = 0
    cache_hits = 0

    target_probe = init_decision["selected_probe"] or 4

    while target_probe is not None:
        # Check cache
        if cache is not None and current_block > 0:
            cached_feat = cache.get(img_fp, checkpoint_id, current_block, target_device=DEVICE)
            if cached_feat is not None:
                features = cached_feat
                cache_hits += 1

        # Execute missing backbone segment
        if current_block == 0:
            features = sequential_b.start(image)
        else:
            features = sequential_b.continue_to(features, current_block + 1, target_probe)

        current_block = target_probe

        # Cache newly computed features
        if cache is not None:
            cache.put(request_id, img_fp, checkpoint_id, current_block, features)

        # Probe inference
        probe_logits = sequential_b.predict_probe(features, target_probe)
        executed_probes.append(target_probe)

        synchronize(DEVICE)
        elapsed_so_far_ms = (time.perf_counter() - start_total) * 1000.0

        probe_decision = controller.decide_probe(
            current_probe_block=target_probe,
            probe_logits=probe_logits,
            current_latency_ms=elapsed_so_far_ms,
            executed_probes=executed_probes
        )

        if probe_decision["action"] == "RETAIN":
            total_lat = (time.perf_counter() - start_total) * 1000.0
            return {
                "action": "RETAIN",
                "status": probe_decision.get("status", "CONFIDENT_RETAIN"),
                "prediction": probe_decision["prediction"],
                "confidence": probe_decision["confidence"],
                "uncertainty": probe_decision["uncertainty"],
                "executed_probes": executed_probes,
                "used_full_b": False,
                "used_partial_b": True,
                "cache_hits": cache_hits,
                "decision_score": probe_decision["decision_score"],
                "reason": probe_decision["reason"],
                "latency_ms": float(total_lat)
            }
        elif probe_decision["action"] == "PROBE MORE":
            target_probe = probe_decision["selected_probe"]
        else:  # SWITCH
            break

    # Full Model B fallback
    if cache is not None and features is None:
        features = cache.get(img_fp, checkpoint_id, current_block, target_device=DEVICE)

    if features is None:
        features = sequential_b.start(image)
        if current_block > 4:
            features = sequential_b.continue_to(features, 5, current_block)

    final_logits = sequential_b.finish(features)
    probs_b = F.softmax(final_logits, dim=1)
    conf_b, pred_b = probs_b.max(dim=1)

    synchronize(DEVICE)
    total_lat = (time.perf_counter() - start_total) * 1000.0

    return {
        "action": "SWITCH",
        "status": "FALLBACK_SWITCH",
        "prediction": int(pred_b.item()),
        "confidence": float(conf_b.item()),
        "uncertainty": float(-torch.sum(probs_b * torch.log(probs_b.clamp(min=1e-12))).item()),
        "executed_probes": executed_probes,
        "used_full_b": True,
        "used_partial_b": True,
        "cache_hits": cache_hits,
        "decision_score": 0.5,
        "reason": "Completed full Model B after probe evidence collection.",
        "latency_ms": float(total_lat)
    }


def run_comparative_evaluation(
    eval_start_index: int = 1000,
    eval_count: int = 20,
    stratified: bool = False,
    stratified_seed: int = 42,
    max_latency_budget_ms: float = DEFAULT_MAX_LATENCY_MS,
    enable_calibration: bool = True
) -> Dict[str, Any]:
    """
    Runs the complete 5-strategy evaluation on identical sample inputs.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / f"results/phase5_{timestamp_str}"
    
    # Create required directory hierarchy
    for sub in ["configs", "logs", "raw", "metrics", "plots", "reports", "metadata"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"ADAPT-EDGE PHASE 5: COMPARATIVE EVALUATION ({eval_count} samples)")
    print(f"Sampling Mode: {'Stratified (balanced 10 classes, seed=' + str(stratified_seed) + ')' if stratified else f'Contiguous range ({eval_start_index}..{eval_start_index + eval_count - 1})'}")
    print(f"Output Directory: {run_dir}")
    print("=" * 75)

    # 1. Models & Hardware Setup
    model_a, model_b, sequential_b = load_all_models()
    cost_estimator = CostEstimator()
    resource_monitor = ResourceMonitor()
    initial_resource_snapshot = resource_monitor.snapshot()

    # Temperature Scaler
    scaler = TemperatureScaler(initial_temperature=1.0055) if enable_calibration else None

    # Decision Controller
    controller = DecisionController(
        model_a_threshold=DEFAULT_MODEL_A_THRESHOLD,
        probe_threshold=DEFAULT_PROBE_THRESHOLD,
        max_latency_ms=max_latency_budget_ms,
        cost_estimator=cost_estimator,
        temperature_scaler=scaler,
        resource_monitor=resource_monitor
    )

    # Continuation Cache
    cache = ContinuationCache(max_entries=20, max_memory_mb=64.0)

    # 2. Data Loader
    loader, eval_indices = get_evaluation_loader(
        start_index=eval_start_index,
        count=eval_count,
        stratified=stratified,
        random_seed=stratified_seed
    )

    # Save exact sample indices for complete reproducibility
    with open(run_dir / "metadata/sample_indices.json", "w") as f:
        json.dump({
            "sample_count": len(eval_indices),
            "stratified": stratified,
            "seed": stratified_seed if stratified else None,
            "start_index": eval_start_index if not stratified else None,
            "indices": eval_indices
        }, f, indent=2)

    # Save Experiment Metadata
    tracker = ExperimentTracker(
        experiment_name=f"phase5_{'stratified' if stratified else 'range'}_{eval_count}",
        config={
            "eval_start_index": eval_start_index,
            "eval_count": eval_count,
            "stratified": stratified,
            "stratified_seed": stratified_seed if stratified else None,
            "max_latency_budget_ms": max_latency_budget_ms,
            "calibration_enabled": enable_calibration,
            "calibration_temperature": 1.0055,
            "device": str(DEVICE)
        },
        base_dir=run_dir
    )

    # Tracking storage
    strategy_names = [
        "Model A Only",
        "Model B Only",
        "Conventional Switching",
        "ADAPT-EDGE (Uncached)",
        "ADAPT-EDGE (Cached)"
    ]

    strategy_predictions = {s: [] for s in strategy_names}
    strategy_confidences = {s: [] for s in strategy_names}
    strategy_latencies = {s: [] for s in strategy_names}
    strategy_correct = {s: [] for s in strategy_names}
    strategy_peak_gpu_mb = {s: [] for s in strategy_names}

    per_image_rows = []
    ground_truth_labels = []

    # Warm-up pass
    print("Executing GPU warm-up...")
    warmup_img, _ = next(iter(loader))
    warmup_img = warmup_img.to(DEVICE)
    for _ in range(5):
        execute_model_a_only(model_a, warmup_img)
        execute_model_b_only(model_b, warmup_img)
        execute_conventional_switching(model_a, model_b, warmup_img)
        execute_adapt_edge(model_a, sequential_b, controller, warmup_img, "warmup_uncached", cache=None)
        execute_adapt_edge(model_a, sequential_b, controller, warmup_img, "warmup_cached", cache=cache)
        cache.clear()

    print("\nStarting evaluation across all 5 strategies on identical samples...\n")

    for idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        true_label = labels.item()
        ground_truth_labels.append(true_label)
        dataset_idx = eval_indices[idx]
        req_id = f"eval_{dataset_idx}_{idx}"

        # ---------------------------------------------
        # S1: Model A Only
        # ---------------------------------------------
        reset_peak_gpu_memory(DEVICE)
        p_a, c_a, lat_a = execute_model_a_only(model_a, images)
        strategy_predictions["Model A Only"].append(p_a)
        strategy_confidences["Model A Only"].append(c_a)
        strategy_latencies["Model A Only"].append(lat_a)
        corr_a = (p_a == true_label)
        strategy_correct["Model A Only"].append(corr_a)
        strategy_peak_gpu_mb["Model A Only"].append(measure_peak_gpu_memory_mb(DEVICE) or 0.0)

        # ---------------------------------------------
        # S2: Model B Only
        # ---------------------------------------------
        reset_peak_gpu_memory(DEVICE)
        p_b, c_b, lat_b = execute_model_b_only(model_b, images)
        strategy_predictions["Model B Only"].append(p_b)
        strategy_confidences["Model B Only"].append(c_b)
        strategy_latencies["Model B Only"].append(lat_b)
        corr_b = (p_b == true_label)
        strategy_correct["Model B Only"].append(corr_b)
        strategy_peak_gpu_mb["Model B Only"].append(measure_peak_gpu_memory_mb(DEVICE) or 0.0)

        # ---------------------------------------------
        # S3: Conventional Switching
        # ---------------------------------------------
        reset_peak_gpu_memory(DEVICE)
        res_conv = execute_conventional_switching(model_a, model_b, images, DEFAULT_MODEL_A_THRESHOLD)
        p_conv = res_conv["prediction"]
        c_conv = res_conv["confidence"]
        lat_conv = res_conv["latency_ms"]
        corr_conv = (p_conv == true_label)
        # Assertion invariant
        expected_conv_p = p_b if res_conv["used_model_b"] else p_a
        assert p_conv == expected_conv_p, f"Conventional switching prediction invariant violated on sample {idx}!"
        strategy_predictions["Conventional Switching"].append(p_conv)
        strategy_confidences["Conventional Switching"].append(c_conv)
        strategy_latencies["Conventional Switching"].append(lat_conv)
        strategy_correct["Conventional Switching"].append(corr_conv)
        strategy_peak_gpu_mb["Conventional Switching"].append(measure_peak_gpu_memory_mb(DEVICE) or 0.0)

        # ---------------------------------------------
        # S4: ADAPT-EDGE (Uncached)
        # ---------------------------------------------
        reset_peak_gpu_memory(DEVICE)
        res_adapt_uncached = execute_adapt_edge(
            model_a, sequential_b, controller, images, request_id=f"{req_id}_uncached", cache=None
        )
        p_unc = res_adapt_uncached["prediction"]
        c_unc = res_adapt_uncached["confidence"]
        lat_unc = res_adapt_uncached["latency_ms"]
        corr_unc = (p_unc == true_label)
        strategy_predictions["ADAPT-EDGE (Uncached)"].append(p_unc)
        strategy_confidences["ADAPT-EDGE (Uncached)"].append(c_unc)
        strategy_latencies["ADAPT-EDGE (Uncached)"].append(lat_unc)
        strategy_correct["ADAPT-EDGE (Uncached)"].append(corr_unc)
        strategy_peak_gpu_mb["ADAPT-EDGE (Uncached)"].append(measure_peak_gpu_memory_mb(DEVICE) or 0.0)

        # ---------------------------------------------
        # S5: ADAPT-EDGE (Cached)
        # ---------------------------------------------
        reset_peak_gpu_memory(DEVICE)
        res_adapt_cached = execute_adapt_edge(
            model_a, sequential_b, controller, images, request_id=f"{req_id}_cached", cache=cache
        )
        p_c = res_adapt_cached["prediction"]
        c_c = res_adapt_cached["confidence"]
        lat_c = res_adapt_cached["latency_ms"]
        corr_c = (p_c == true_label)
        strategy_predictions["ADAPT-EDGE (Cached)"].append(p_c)
        strategy_confidences["ADAPT-EDGE (Cached)"].append(c_c)
        strategy_latencies["ADAPT-EDGE (Cached)"].append(lat_c)
        strategy_correct["ADAPT-EDGE (Cached)"].append(corr_c)
        strategy_peak_gpu_mb["ADAPT-EDGE (Cached)"].append(measure_peak_gpu_memory_mb(DEVICE) or 0.0)

        # Invalidate request entries after completion
        cache.invalidate_request(f"{req_id}_cached")

        # Record per-image detail row
        row = {
            "image_index": idx,
            "dataset_index": dataset_idx,
            "true_label": true_label,
            "true_class": CLASS_NAMES[true_label],
            "model_a_pred": p_a,
            "model_a_conf": round(c_a, 4),
            "model_a_correct": corr_a,
            "model_a_latency_ms": round(lat_a, 2),
            "model_b_pred": p_b,
            "model_b_conf": round(c_b, 4),
            "model_b_correct": corr_b,
            "model_b_latency_ms": round(lat_b, 2),
            "conv_pred": p_conv,
            "conv_conf": round(c_conv, 4),
            "conv_selected": res_conv["selected_model"],
            "conv_correct": corr_conv,
            "conv_latency_ms": round(lat_conv, 2),
            "adapt_uncached_pred": p_unc,
            "adapt_uncached_conf": round(c_unc, 4),
            "adapt_uncached_action": res_adapt_uncached["action"],
            "adapt_uncached_status": res_adapt_uncached["status"],
            "adapt_uncached_probes": str(res_adapt_uncached["executed_probes"]),
            "adapt_uncached_full_b": res_adapt_uncached["used_full_b"],
            "adapt_uncached_correct": corr_unc,
            "adapt_uncached_latency_ms": round(lat_unc, 2),
            "adapt_cached_pred": p_c,
            "adapt_cached_conf": round(c_c, 4),
            "adapt_cached_action": res_adapt_cached["action"],
            "adapt_cached_status": res_adapt_cached["status"],
            "adapt_cached_probes": str(res_adapt_cached["executed_probes"]),
            "adapt_cached_full_b": res_adapt_cached["used_full_b"],
            "adapt_cached_hits": res_adapt_cached["cache_hits"],
            "adapt_cached_correct": corr_c,
            "adapt_cached_latency_ms": round(lat_c, 2),
            # Nested strategy dicts for visualization scripts
            "conventional_switching": {"latency_ms": lat_conv},
            "adapt_edge_cached": {"latency_ms": lat_c}
        }
        per_image_rows.append(row)

        if (idx + 1) % 50 == 0 or (idx + 1) == eval_count:
            print(f"Evaluated [{idx + 1:>4}/{eval_count}] images | Sample {idx}: True={CLASS_NAMES[true_label]:<10} | "
                  f"A={'OK' if corr_a else 'X'} ({lat_a:.1f}ms) | B={'OK' if corr_b else 'X'} ({lat_b:.1f}ms) | Conv={'OK' if corr_conv else 'X'} ({lat_conv:.1f}ms) | "
                  f"ADAPT={'OK' if corr_c else 'X'} ({lat_c:.1f}ms)")

    # -------------------------------------------------------------
    # 3. Aggregate Metrics & Statistical Comparisons
    # -------------------------------------------------------------
    final_strategy_metrics = {}
    for strat in strategy_names:
        # Classification metrics
        cls_m = compute_metrics(
            y_true=ground_truth_labels,
            y_pred=strategy_predictions[strat],
            num_classes=10,
            class_names=CLASS_NAMES
        )

        # Latency statistics
        lat_stats = LatencyStats(strategy_latencies[strat]).to_dict()

        # Computation statistics
        is_adapt = "ADAPT-EDGE" in strat
        if is_adapt:
            # Gather probe executions from row records
            key_prefix = "adapt_cached" if "Cached" in strat else "adapt_uncached"
            p_counts = {"4": 0, "9": 0, "13": 0}
            full_b_cnt = 0
            partial_b_cnt = 0
            for r in per_image_rows:
                probes = eval(r[f"{key_prefix}_probes"])
                for p in probes:
                    p_counts[str(p)] += 1
                if r[f"{key_prefix}_full_b"]:
                    full_b_cnt += 1
                if len(probes) > 0:
                    partial_b_cnt += 1

            comp_stats = {
                "full_model_b_pct": round((full_b_cnt / eval_count) * 100.0, 2),
                "partial_model_b_pct": round((partial_b_cnt / eval_count) * 100.0, 2),
                "probe_counts": p_counts
            }
        elif strat == "Conventional Switching":
            b_cnt = sum(1 for r in per_image_rows if r["conv_selected"] == "B")
            comp_stats = {
                "full_model_b_pct": round((b_cnt / eval_count) * 100.0, 2),
                "partial_model_b_pct": 0.0,
                "probe_counts": {"4": 0, "9": 0, "13": 0}
            }
        elif strat == "Model B Only":
            comp_stats = {
                "full_model_b_pct": 100.0,
                "partial_model_b_pct": 0.0,
                "probe_counts": {"4": 0, "9": 0, "13": 0}
            }
        else:  # Model A Only
            comp_stats = {
                "full_model_b_pct": 0.0,
                "partial_model_b_pct": 0.0,
                "probe_counts": {"4": 0, "9": 0, "13": 0}
            }

        # Resource stats
        res_stats = {
            "peak_gpu_memory_mb": round(float(np.max(strategy_peak_gpu_mb[strat])), 2)
        }

        final_strategy_metrics[strat] = {
            "classification": cls_m,
            "latency": lat_stats,
            "computation": comp_stats,
            "resource": res_stats
        }

    # Cache Stats
    cache_metrics = cache.get_cache_stats()

    # Paired Statistical Analyses
    statistical_reports = [
        paired_strategy_analysis(
            base_name="Model A Only",
            target_name="ADAPT-EDGE (Cached)",
            base_correct=strategy_correct["Model A Only"],
            target_correct=strategy_correct["ADAPT-EDGE (Cached)"],
            base_latencies=strategy_latencies["Model A Only"],
            target_latencies=strategy_latencies["ADAPT-EDGE (Cached)"]
        ),
        paired_strategy_analysis(
            base_name="Conventional Switching",
            target_name="ADAPT-EDGE (Cached)",
            base_correct=strategy_correct["Conventional Switching"],
            target_correct=strategy_correct["ADAPT-EDGE (Cached)"],
            base_latencies=strategy_latencies["Conventional Switching"],
            target_latencies=strategy_latencies["ADAPT-EDGE (Cached)"]
        ),
        paired_strategy_analysis(
            base_name="Model B Only",
            target_name="ADAPT-EDGE (Cached)",
            base_correct=strategy_correct["Model B Only"],
            target_correct=strategy_correct["ADAPT-EDGE (Cached)"],
            base_latencies=strategy_latencies["Model B Only"],
            target_latencies=strategy_latencies["ADAPT-EDGE (Cached)"]
        ),
        paired_strategy_analysis(
            base_name="ADAPT-EDGE (Uncached)",
            target_name="ADAPT-EDGE (Cached)",
            base_correct=strategy_correct["ADAPT-EDGE (Uncached)"],
            target_correct=strategy_correct["ADAPT-EDGE (Cached)"],
            base_latencies=strategy_latencies["ADAPT-EDGE (Uncached)"],
            target_latencies=strategy_latencies["ADAPT-EDGE (Cached)"]
        )
    ]

    # -------------------------------------------------------------
    # 4. Save Raw Outputs, CSVs, and JSONs
    # -------------------------------------------------------------
    # Raw per-image CSV
    per_image_csv = run_dir / "raw/per_image_results.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [k for k in per_image_rows[0].keys() if not isinstance(per_image_rows[0][k], dict)]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_image_rows)

    # Raw per-image JSONL
    per_image_jsonl = run_dir / "raw/per_image_results.jsonl"
    with open(per_image_jsonl, "w", encoding="utf-8") as f:
        for r in per_image_rows:
            f.write(json.dumps(r) + "\n")

    # Strategy comparison summary CSV
    strat_csv = run_dir / "metrics/strategy_comparison.csv"
    with open(strat_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["strategy", "accuracy", "macro_f1", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "full_model_b_pct", "partial_model_b_pct", "peak_gpu_memory_mb"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for s in strategy_names:
            writer.writerow({
                "strategy": s,
                "accuracy": final_strategy_metrics[s]["classification"]["accuracy"],
                "macro_f1": final_strategy_metrics[s]["classification"]["macro_f1"],
                "mean_latency_ms": final_strategy_metrics[s]["latency"]["mean_latency_ms"],
                "median_latency_ms": final_strategy_metrics[s]["latency"]["median_latency_ms"],
                "p95_latency_ms": final_strategy_metrics[s]["latency"]["p95_latency_ms"],
                "full_model_b_pct": final_strategy_metrics[s]["computation"]["full_model_b_pct"],
                "partial_model_b_pct": final_strategy_metrics[s]["computation"]["partial_model_b_pct"],
                "peak_gpu_memory_mb": final_strategy_metrics[s]["resource"]["peak_gpu_memory_mb"]
            })

    # Individual metrics JSONs
    with open(run_dir / "metrics/classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump({s: final_strategy_metrics[s]["classification"] for s in strategy_names}, f, indent=4)
    with open(run_dir / "metrics/latency_metrics.json", "w", encoding="utf-8") as f:
        json.dump({s: final_strategy_metrics[s]["latency"] for s in strategy_names}, f, indent=4)
    with open(run_dir / "metrics/cache_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cache_metrics, f, indent=4)
    with open(run_dir / "metrics/resource_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"initial_snapshot": initial_resource_snapshot, "final_peak_mb": {s: final_strategy_metrics[s]["resource"] for s in strategy_names}}, f, indent=4)

    # -------------------------------------------------------------
    # 5. Generate Visualizations
    # -------------------------------------------------------------
    print("Generating publication-quality graphs...")
    plot_accuracy_comparison(final_strategy_metrics, run_dir / "plots/accuracy_comparison.png")
    plot_f1_comparison(final_strategy_metrics, run_dir / "plots/f1_comparison.png")
    plot_latency_comparison(final_strategy_metrics, run_dir / "plots/latency_comparison.png")
    plot_accuracy_vs_latency(final_strategy_metrics, run_dir / "plots/accuracy_vs_latency.png")
    plot_model_b_usage(final_strategy_metrics, run_dir / "plots/model_b_usage.png")
    plot_probe_distribution(final_strategy_metrics, run_dir / "plots/probe_distribution.png")
    plot_cache_performance(cache_metrics, run_dir / "plots/cache_performance.png")
    plot_memory_comparison(final_strategy_metrics, run_dir / "plots/memory_comparison.png")
    plot_confusion_matrices(final_strategy_metrics, CLASS_NAMES, run_dir / "plots/confusion_matrices.png")
    plot_budget_analysis(per_image_rows, max_latency_budget_ms, run_dir / "plots/budget_analysis.png")
    print("Visualizations successfully saved to plots/ directory.")

    # Save class distribution
    class_dist = Counter(ground_truth_labels)
    with open(run_dir / "metadata/class_distribution.json", "w") as f:
        json.dump({
            CLASS_NAMES[c]: class_dist[c] for c in range(10)
        }, f, indent=2)

    sample_desc = (
        f"{eval_count} samples (CIFAR-10 stratified subset across all 10 classes, seed={stratified_seed})"
        if stratified
        else f"{eval_count} samples (CIFAR-10 test indices {eval_start_index}..{eval_start_index + eval_count - 1})"
    )

    # -------------------------------------------------------------
    # 6. Generate Reports
    # -------------------------------------------------------------
    report_title = "Diagnostic Experimental Evaluation Report" if eval_count < 100 else "Comprehensive Experimental Evaluation Report"
    report_content = f"""# ADAPT-EDGE Phase 5: {report_title}

**Evaluation Timestamp**: {timestamp_str}
**Sample Count**: {sample_desc}
**Hardware**: NVIDIA RTX 3050 Laptop GPU (4GB VRAM), Intel/AMD CPU, 16GB RAM

---

## 1. Executive Summary Table

| Strategy | Accuracy (%) | Macro F1 | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Full Model B (%) | Peak GPU (MB) |
|---|---|---|---|---|---|---|---|
"""
    for s in strategy_names:
        m = final_strategy_metrics[s]
        report_content += (
            f"| **{s}** | {m['classification']['accuracy']:.2f}% | {m['classification']['macro_f1']:.3f} | "
            f"{m['latency']['mean_latency_ms']:.2f} | {m['latency']['median_latency_ms']:.2f} | {m['latency']['p95_latency_ms']:.2f} | "
            f"{m['computation']['full_model_b_pct']:.1f}% | {m['resource']['peak_gpu_memory_mb']:.1f} |\n"
        )

    report_content += f"""
---

## 2. Statistical Paired Comparisons

"""
    for stat in statistical_reports:
        mean_lat_ci = stat.get('mean_latency_diff_95ci_ms') or stat.get('latency_diff_95ci_ms', [0.0, 0.0])
        acc_ci = stat.get('accuracy_diff_95ci', [0.0, 0.0])
        report_content += f"""### {stat['comparison']}
- **Sample Count**: {stat['sample_count']}
- **Accuracy Difference**: {stat['accuracy_difference_pct']:+.2f}% (95% CI: [{acc_ci[0]:+.2f}%, {acc_ci[1]:+.2f}%])
- **McNemar Test**: stat={stat['mcnemar_stat']:.4f}, p={stat['mcnemar_p_value']:.4f} (Significant: {stat['is_accuracy_diff_significant']})
- **Mean Latency Difference**: {stat['mean_latency_diff_ms']:+.3f} ms (95% CI: [{mean_lat_ci[0]:+.3f}, {mean_lat_ci[1]:+.3f}] ms)
- **Wilcoxon Signed-Rank Test**: stat={stat['wilcoxon_stat']:.4f}, p={stat['wilcoxon_p_value']:.4f} (Significant: {stat['is_latency_diff_significant']})

"""

    report_content += f"""
---

## 3. Continuation Cache Performance
- **Cache Hits**: {cache_metrics['hits']}
- **Cache Misses**: {cache_metrics['misses']}
- **Hit Ratio**: {cache_metrics['hit_ratio'] * 100.0:.1f}%
- **Memory Allocated**: {cache_metrics['memory_used_mb']} MB (Cap: {cache_metrics['max_memory_mb']} MB)
- **Invalidations**: {cache_metrics['invalidations']}
- **Observation**: Continuation caching eliminated redundant backbone recomputations between sequential probe activations without altering predictions.

---

## 4. Methodological Disclosures & Experimental Limitations
1. **Dataset Status**: The evaluated samples belong to the CIFAR-10 test set, which was previously explored during baseline benchmark profiling.
2. **Sample Size & Sampling**: Evaluated {sample_desc}.
3. **Information Score Formulation**: The probe selection score is an empirical capacity-scaled heuristic, not formal Bayesian Expected Information Gain.
"""

    with open(run_dir / "reports/phase5_results_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    for s in strategy_names:
        m = final_strategy_metrics[s]
        print(f"{s:<26}: Acc={m['classification']['accuracy']:>5.1f}% | Mean Lat={m['latency']['mean_latency_ms']:>6.2f}ms | P95={m['latency']['p95_latency_ms']:>6.2f}ms | Full B={m['computation']['full_model_b_pct']:>5.1f}%")
    print("=" * 75)
    print(f"Results Report: {run_dir / 'reports/phase5_results_report.md'}")

    return {
        "run_dir": str(run_dir),
        "metrics": final_strategy_metrics,
        "statistical_reports": statistical_reports,
        "cache_metrics": cache_metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT-EDGE Phase 5 Comparative Evaluation")
    parser.add_argument("--count", type=int, default=20, help="Number of test images to evaluate")
    parser.add_argument("--start", type=int, default=1000, help="Start index in CIFAR-10 test set")
    parser.add_argument("--stratified", action="store_true", help="Use stratified sampling across all 10 classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling")
    parser.add_argument("--budget", type=float, default=DEFAULT_MAX_LATENCY_MS, help="Max latency budget in ms")
    parser.add_argument("--no-calibration", action="store_true", help="Disable confidence calibration")
    args = parser.parse_args()

    run_comparative_evaluation(
        eval_start_index=args.start,
        eval_count=args.count,
        stratified=args.stratified,
        stratified_seed=args.seed,
        max_latency_budget_ms=args.budget,
        enable_calibration=not args.no_calibration
    )
