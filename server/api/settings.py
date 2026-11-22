"""
Settings, Device Configuration, and System Integrity API Router for ADAPT-EDGE.
Provides centralized endpoints for hardware devices, training defaults, resource budgets,
session cleanup, and cryptographic verification of protected research checkpoints.
"""

import gc
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import torch

from server.core import ServiceManager, PROJECT_ROOT

router = APIRouter(prefix="/api/settings", tags=["Settings & Final Verification"])

# Expected SHA-256 checksums for audited research assets
AUDITED_CHECKPOINTS = {
    "model_a": {
        "name": "MobileNetV3-Small (Baseline)",
        "rel_path": "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth",
        "expected_sha256": "bef9caa8b19ce9e8d2e6b130e9cbfc371308ecab130c776eae29ae612966e6c1",
        "role": "Model A (Lightweight Initial Classifier)"
    },
    "model_b": {
        "name": "MobileNetV3-Large (Baseline)",
        "rel_path": "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth",
        "expected_sha256": "f02bade7248cf5d5ce9baf746cf6176e68a1c16d08d561b713c73501af4265e6",
        "role": "Model B (High-Capacity Feature Extractor)"
    },
    "probe_4": {
        "name": "Probe Head Block 4",
        "rel_path": "outputs/probes/block_4/best_probe.pth",
        "expected_sha256": "db49f173feda64aadaf314a345eb4303afc8de79e3ba84a50c7b0997f3fa7943",
        "role": "Early Exit Probe (Block 4)"
    },
    "probe_9": {
        "name": "Probe Head Block 9",
        "rel_path": "outputs/probes/block_9/best_probe.pth",
        "expected_sha256": "fa14f00bb35669391727e84745b940c5458111e6094c2c8e32c639315ef62995",
        "role": "Mid-Stage Probe (Block 9)"
    },
    "probe_13": {
        "name": "Probe Head Block 13",
        "rel_path": "outputs/probes/block_13/best_probe.pth",
        "expected_sha256": "96c0ae373e8bd2651e9aae4d693e1ee68976e1c409bb77e088f1babdc203d65b",
        "role": "Deep Probe (Block 13)"
    }
}


class DeviceConfigRequest(BaseModel):
    device: str = Field(..., description="Target device string, e.g. 'cpu' or 'cuda' or 'cuda:0'")
    cudnn_benchmark: Optional[bool] = Field(None, description="Enable torch.backends.cudnn.benchmark for convolutions")


class TrainingDefaultsRequest(BaseModel):
    batch_size: Optional[int] = Field(None, ge=1, le=128)
    epochs: Optional[int] = Field(None, ge=1, le=100)
    learning_rate: Optional[float] = Field(None, gt=0, le=1.0)
    optimizer: Optional[str] = Field(None, pattern="^(Adam|AdamW|SGD)$")
    img_size: Optional[int] = Field(None, ge=32, le=512)
    val_split: Optional[float] = Field(None, ge=0.05, le=0.5)
    device: Optional[str] = None
    architecture: Optional[str] = None


class ResourceBudgetConfigRequest(BaseModel):
    latency_budget_ms: Optional[float] = Field(None, ge=1.0, le=500.0)
    ram_budget_mb: Optional[float] = Field(None, ge=256.0, le=65536.0)
    vram_budget_mb: Optional[float] = Field(None, ge=128.0, le=32768.0)
    cpu_throttle_threshold: Optional[float] = Field(None, ge=10.0, le=100.0)
    enforce_resource_budget: Optional[bool] = None
    threshold_a: Optional[float] = Field(None, ge=0.1, le=1.0)
    threshold_probe: Optional[float] = Field(None, ge=0.1, le=1.0)


class BackendSettingsRequest(BaseModel):
    telemetry_poll_interval_ms: Optional[int] = Field(None, ge=100, le=10000)
    cache_retention_limit: Optional[int] = Field(None, ge=10, le=500)
    simulation_mode: Optional[bool] = None
    debug_logging: Optional[bool] = None


@router.get("")
def get_all_settings():
    """Return all system settings, hardware telemetry, defaults, and budget configurations."""
    mgr = ServiceManager.get_instance()
    cuda_avail = torch.cuda.is_available()
    
    device_info = {
        "active_device": str(mgr.device),
        "cuda_available": cuda_avail,
        "available_devices": ["cpu"] + ([f"cuda:{i}" for i in range(torch.cuda.device_count())] if cuda_avail else []),
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU Host",
        "cudnn_benchmark": torch.backends.cudnn.benchmark if hasattr(torch.backends, "cudnn") else False,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }
    
    if cuda_avail:
        device_info["cuda_version"] = torch.version.cuda
        device_info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
        device_info["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
        device_info["gpu_total_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)

    from server.api.training import CANDIDATE_MODELS
    
    continuation_cache_count = len(mgr.pipeline.cache._cache) if (mgr.pipeline and mgr.pipeline.cache) else 0

    return {
        "success": True,
        "device": device_info,
        "training_defaults": mgr.default_training_params,
        "resource_budget": mgr.resource_budget,
        "backend": mgr.backend_settings,
        "active_pair": mgr.active_pair,
        "session_storage": {
            "candidate_models_in_ram": len(CANDIDATE_MODELS),
            "candidate_keys": list(CANDIDATE_MODELS.keys()),
            "ui_sample_cache_count": len(mgr.sample_cache),
            "continuation_cache_entries": continuation_cache_count,
            "persistence_mode": "Strict In-Memory RAM Only"
        }
    }


@router.post("/device")
def configure_device(req: DeviceConfigRequest):
    """Switch active hardware compute device (e.g., cpu vs cuda)."""
    mgr = ServiceManager.get_instance()
    target = req.device.strip().lower()
    
    try:
        active = mgr.set_device(target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if req.cudnn_benchmark is not None and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = req.cudnn_benchmark

    return {
        "success": True,
        "active_device": active,
        "cudnn_benchmark": torch.backends.cudnn.benchmark if hasattr(torch.backends, "cudnn") else False,
        "message": f"Execution device successfully changed to {active}"
    }


@router.post("/training-defaults")
def update_training_defaults(req: TrainingDefaultsRequest):
    """Configure default parameters for the model training workbench."""
    mgr = ServiceManager.get_instance()
    req_dict = req.model_dump(exclude_unset=True)
    for k, v in req_dict.items():
        if v is not None:
            mgr.default_training_params[k] = v
            
    mgr.log_activity("Training Defaults Updated", f"Workbench defaults updated: {list(req_dict.keys())}", "info")
    return {
        "success": True,
        "training_defaults": mgr.default_training_params,
        "message": "Training defaults updated successfully."
    }


@router.post("/resource-budget")
def update_resource_budget(req: ResourceBudgetConfigRequest):
    """Configure resource limits and confidence thresholds."""
    mgr = ServiceManager.get_instance()
    req_dict = req.model_dump(exclude_unset=True)
    for k, v in req_dict.items():
        if v is not None:
            mgr.resource_budget[k] = v
            if k in mgr.active_pair:
                mgr.active_pair[k] = v
                
    if mgr.pipeline and mgr.pipeline.controller:
        if req.latency_budget_ms is not None:
            mgr.pipeline.controller.max_latency_budget_ms = req.latency_budget_ms
        if req.enforce_resource_budget is not None:
            mgr.pipeline.controller.enforce_resource_budget = req.enforce_resource_budget

    mgr.log_activity("Resource Budget Configured", f"Budget updated: {list(req_dict.keys())}", "info")
    return {
        "success": True,
        "resource_budget": mgr.resource_budget,
        "active_pair": mgr.active_pair,
        "message": "Resource budget parameters successfully saved."
    }


@router.post("/backend")
def update_backend_settings(req: BackendSettingsRequest):
    """Update API telemetry and platform behavior."""
    mgr = ServiceManager.get_instance()
    req_dict = req.model_dump(exclude_unset=True)
    for k, v in req_dict.items():
        if v is not None:
            mgr.backend_settings[k] = v

    mgr.log_activity("Backend Settings Configured", f"Updated backend config: {list(req_dict.keys())}", "info")
    return {
        "success": True,
        "backend": mgr.backend_settings,
        "message": "Backend settings updated successfully."
    }


@router.post("/cleanup")
def session_cleanup():
    """
    Perform complete session cleanup:
    1. Flushes all candidate models and metadata in RAM.
    2. Resets training status to idle.
    3. Clears intermediate continuation cache.
    4. Clears UI sample cache.
    5. Forces garbage collection and CUDA cache flush.
    """
    mgr = ServiceManager.get_instance()
    from server.api.training import (
        CANDIDATE_MODELS, CANDIDATE_MODELS_LOCK, TRAINER, TRAINING_JOBS, TRAINING_JOBS_LOCK,
    )

    with TRAINING_JOBS_LOCK:
        for job_id, trainer in list(TRAINING_JOBS.items()):
            if trainer.status != "RUNNING":
                del TRAINING_JOBS[job_id]

    with CANDIDATE_MODELS_LOCK:
        models_cleared = len(CANDIDATE_MODELS)
        CANDIDATE_MODELS.clear()

    # Reset trainer to idle if not actively training
    if TRAINER.status != "RUNNING":
        with TRAINER.lock:
            TRAINER.status = "IDLE"
            TRAINER.current_epoch = 0
            TRAINER.total_epochs = 0
            TRAINER.history = []
            TRAINER.logs = []
            TRAINER.error_message = None
            TRAINER.candidate_model_id = None

    continuation_cleared = 0
    if mgr.pipeline and mgr.pipeline.cache:
        continuation_cleared = len(mgr.pipeline.cache._cache)
        mgr.pipeline.cache.clear()

    ui_cleared = len(mgr.sample_cache)
    mgr.sample_cache.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mgr.log_activity(
        "Session Cleanup Complete",
        f"Purged {models_cleared} candidate models, {continuation_cleared} cached features, and {ui_cleared} UI previews.",
        "success"
    )

    return {
        "success": True,
        "candidate_models_cleared": models_cleared,
        "continuation_cache_cleared": continuation_cleared,
        "ui_samples_cleared": ui_cleared,
        "remaining_candidate_models_in_ram": 0,
        "status": "Session state completely reset. Zero lingering candidate weights or cache entries."
    }


@router.get("/verify-integrity")
def verify_protected_checkpoints():
    """
    Cryptographic SHA-256 integrity audit for all baseline and probe checkpoints.
    Guarantees baseline research weights in outputs/ have not been modified or corrupted.
    """
    results = {}
    all_verified = True
    any_found = False

    for key, info in AUDITED_CHECKPOINTS.items():
        path = PROJECT_ROOT / info["rel_path"]
        exists = path.is_file()
        actual_sha = None
        status = "MISSING"

        if exists:
            any_found = True
            content = path.read_bytes()
            actual_sha = hashlib.sha256(content).hexdigest()
            if actual_sha == info["expected_sha256"]:
                status = "VERIFIED"
            else:
                status = "MISMATCH"
                all_verified = False
        else:
            status = "MISSING_ON_DISK"

        results[key] = {
            "name": info["name"],
            "role": info["role"],
            "rel_path": info["rel_path"],
            "file_exists": exists,
            "file_size_bytes": path.stat().st_size if exists else 0,
            "status": status,
            "expected_sha256": info["expected_sha256"],
            "actual_sha256": actual_sha
        }

    # Verify no rogue candidate model files exist in outputs/
    rogue_candidates = []
    outputs_dir = PROJECT_ROOT / "outputs"
    if outputs_dir.exists():
        for p in outputs_dir.rglob("*.pth"):
            rel_str = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            known_paths = [info["rel_path"] for info in AUDITED_CHECKPOINTS.values()]
            if rel_str not in known_paths:
                rogue_candidates.append(rel_str)

    clean_isolation = len(rogue_candidates) == 0

    return {
        "success": True,
        "all_audited_checkpoints_verified": all_verified,
        "any_checkpoint_present": any_found,
        "checkpoints": results,
        "research_data_isolation": {
            "clean_isolation": clean_isolation,
            "rogue_candidate_checkpoints": rogue_candidates,
            "description": "Candidate models are strictly in-RAM only; no candidate checkpoints pollute outputs/."
        }
    }
