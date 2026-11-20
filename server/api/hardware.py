"""
Hardware & System Overview API Router for ADAPT-EDGE.
Provides live system telemetry, GPU stats, and KPI overview.
"""

from pathlib import Path
from fastapi import APIRouter
import torch

from server.core import ServiceManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
router = APIRouter(prefix="/api/system", tags=["System & Hardware"])


@router.get("/hardware")
def get_hardware_telemetry():
    """Returns live hardware metrics collected via ctypes, CUDA, and nvidia-smi."""
    mgr = ServiceManager.get_instance()
    snap = mgr.resource_monitor.snapshot(force_refresh=True)

    # Determine status tags
    cpu_status = "normal"
    if snap.get("cpu_percent") is not None:
        if snap["cpu_percent"] > 90.0:
            cpu_status = "critical"
        elif snap["cpu_percent"] > 75.0:
            cpu_status = "warning"

    ram_status = "normal"
    if snap.get("ram_percent") is not None:
        if snap["ram_percent"] > 90.0:
            ram_status = "critical"
        elif snap["ram_percent"] > 80.0:
            ram_status = "warning"

    gpu_status = "normal"
    if snap.get("gpu_available"):
        gpu_pct = snap.get("gpu_utilization_percent")
        temp = snap.get("gpu_temperature_c")
        if (temp is not None and temp > 85.0) or (gpu_pct is not None and gpu_pct > 95.0):
            gpu_status = "critical"
        elif (temp is not None and temp > 75.0) or (gpu_pct is not None and gpu_pct > 80.0):
            gpu_status = "warning"
    else:
        gpu_status = "unavailable"

    return {
        "success": True,
        "telemetry": snap,
        "status": {
            "cpu": cpu_status,
            "ram": ram_status,
            "gpu": gpu_status
        }
    }


@router.get("/overview")
def get_overview_dashboard():
    """Returns high-level research platform KPIs and system health."""
    mgr = ServiceManager.get_instance()
    pipe = mgr.pipeline
    snap = mgr.resource_monitor.snapshot()

    # Model and checkpoint statuses
    model_a_loaded = pipe is not None and pipe.model_a is not None
    model_b_loaded = pipe is not None and pipe.model_b is not None
    probes_loaded = pipe is not None and pipe.sequential_b is not None and len(getattr(pipe.sequential_b, 'probe_heads', {})) == 3

    # Scan existing experiment count
    results_dir = PROJECT_ROOT / "results"
    total_experiments = 0
    latest_acc = None
    latest_lat = None
    latest_run_id = None

    if results_dir.exists():
        runs = [d for d in results_dir.iterdir() if d.is_dir() and "phase5_" in d.name]
        total_experiments = len(runs)
        if runs:
            runs.sort(key=lambda d: d.name, reverse=True)
            latest_run_id = runs[0].name
            metrics_file = runs[0] / "metrics/strategy_comparison.csv"
            if metrics_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(metrics_file)
                    adapt_row = df[df["strategy"] == "ADAPT-EDGE (Cached)"]
                    if not adapt_row.empty:
                        latest_acc = float(adapt_row["accuracy"].values[0])
                        latest_lat = float(adapt_row["mean_latency_ms"].values[0])
                except Exception:
                    pass

    return {
        "success": True,
        "backend_online": True,
        "active_dataset": "CIFAR-10 (10 classes, 10k test set)",
        "models": {
            "model_a": {
                "name": "MobileNetV3-Small",
                "loaded": model_a_loaded,
                "checkpoint": "outputs/baseline/mobilenet_v3_small/best_model.pth"
            },
            "model_b": {
                "name": "MobileNetV3-Large",
                "loaded": model_b_loaded,
                "checkpoint": "outputs/baseline/mobilenet_v3_large/best_model.pth"
            },
            "probes": {
                "loaded": probes_loaded,
                "registered_blocks": [4, 9, 13] if probes_loaded else []
            }
        },
        "hardware": {
            "device": str(mgr.device),
            "gpu_name": snap.get("gpu_name") or "CPU Only",
            "cpu_percent": snap.get("cpu_percent"),
            "ram_percent": snap.get("ram_percent"),
            "ram_used_mb": snap.get("ram_used_mb"),
            "ram_total_mb": snap.get("ram_total_mb"),
            "gpu_utilization": snap.get("gpu_utilization_percent"),
            "gpu_temp_c": snap.get("gpu_temperature_c"),
            "torch_vram_allocated_mb": snap.get("gpu_torch_allocated_mb"),
            "total_vram_mb": snap.get("total_vram_mb")
        },
        "experiments": {
            "total_runs": total_experiments,
            "latest_run_id": latest_run_id,
            "latest_accuracy_pct": latest_acc,
            "latest_latency_ms": latest_lat
        },
        "recent_activity": mgr.recent_activity[:15]
    }
