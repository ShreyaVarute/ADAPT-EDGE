"""
Profiling and latency measurement utilities for ADAPT-EDGE.
Provides proper CUDA synchronization, CPU/GPU timing, warm-up runs,
and statistical aggregation (mean, median, p95, stdev, min, max, peak memory).
"""

import time
import math
from typing import Callable, Any, Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
import torch


def synchronize(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Synchronize the specified device. If CUDA device is available,
    calls torch.cuda.synchronize() to ensure asynchronous GPU kernels have completed.
    """
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            dev = torch.device(device)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)


def measure_peak_gpu_memory_mb(device: Optional[Union[str, torch.device]] = None) -> Optional[float]:
    """
    Returns peak GPU memory allocated in megabytes (MB) since last reset,
    or None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    try:
        dev = torch.device(device) if device is not None else None
        bytes_allocated = torch.cuda.max_memory_allocated(device=dev)
        return float(bytes_allocated / (1024.0 ** 2))
    except Exception:
        return None


def reset_peak_gpu_memory(device: Optional[Union[str, torch.device]] = None) -> None:
    """Reset peak memory tracking if CUDA is available."""
    if torch.cuda.is_available():
        try:
            dev = torch.device(device) if device is not None else None
            torch.cuda.reset_peak_memory_stats(device=dev)
        except Exception:
            pass


def time_callable(
    fn: Callable[[], Any],
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[Any, float]:
    """
    Execute a callable once, measuring wall-clock duration in milliseconds (ms)
    with proper device synchronization before and after.
    Returns (output, elapsed_ms).
    """
    synchronize(device)
    start = time.perf_counter()
    output = fn()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


class LatencyStats:
    """
    Statistical aggregator for repeated latency measurements in milliseconds.
    """

    def __init__(self, latencies_ms: Sequence[float]):
        if not latencies_ms:
            raise ValueError("latencies_ms must not be empty")

        self.raw = list(latencies_ms)
        sorted_vals = sorted(self.raw)
        n = len(sorted_vals)

        self.count = n
        self.mean = float(np.mean(sorted_vals))
        self.median = float(np.median(sorted_vals))
        self.p95 = float(np.percentile(sorted_vals, 95))
        self.stdev = float(np.std(sorted_vals, ddof=1)) if n > 1 else 0.0
        self.min = float(sorted_vals[0])
        self.max = float(sorted_vals[-1])

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_latency_ms": round(self.mean, 4),
            "median_latency_ms": round(self.median, 4),
            "p95_latency_ms": round(self.p95, 4),
            "stdev_latency_ms": round(self.stdev, 4),
            "min_latency_ms": round(self.min, 4),
            "max_latency_ms": round(self.max, 4)
        }


def benchmark_function(
    fn: Callable[[], Any],
    warmup_runs: int = 5,
    measurement_runs: int = 20,
    device: Optional[Union[str, torch.device]] = None,
    track_gpu_memory: bool = True
) -> Dict[str, Any]:
    """
    Benchmark a function across repeated executions:
      1. Performs warmup_runs without recording latencies.
      2. Resets peak memory stats (if on CUDA and track_gpu_memory=True).
      3. Performs measurement_runs with per-run timing.
      4. Aggregates mean, median, p95, stdev, min, max, peak memory.
    """
    # 1. Warm-up
    for _ in range(warmup_runs):
        fn()
    synchronize(device)

    # 2. Reset peak memory if requested
    if track_gpu_memory:
        reset_peak_gpu_memory(device)

    # 3. Measurement
    latencies: List[float] = []
    last_output = None

    for _ in range(measurement_runs):
        out, elapsed = time_callable(fn, device=device)
        latencies.append(elapsed)
        last_output = out

    # 4. Aggregation
    stats = LatencyStats(latencies)
    result = stats.to_dict()
    result["warmup_runs"] = warmup_runs
    result["measurement_runs"] = measurement_runs

    if track_gpu_memory and torch.cuda.is_available():
        result["peak_gpu_memory_mb"] = measure_peak_gpu_memory_mb(device)
    else:
        result["peak_gpu_memory_mb"] = None

    return result
