"""
Custom PyTorch training and custom-model discovery.
Supports dynamic architecture selection via the unified model factory.
Supports multiple concurrent training jobs via per-job TrainingState instances.
"""

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.core import PROJECT_ROOT, ServiceManager
from models.architectures import (
    ARCHITECTURE_REGISTRY,
    VALID_MODEL_TYPES,
    build_model,
    get_architecture_info,
    resolve_model_type,
)

router = APIRouter(prefix="/api/training", tags=["Model Training"])
CANDIDATE_MODELS: Dict[str, Dict[str, Any]] = {}
CANDIDATE_MODELS_LOCK = threading.Lock()

# ── Multi-job registry ─────────────────────────────────────────────────────
# Maps job_id -> TrainingState
TRAINING_JOBS: Dict[str, "TrainingState"] = {}
TRAINING_JOBS_LOCK = threading.Lock()


class TrainingConfigRequest(BaseModel):
    model_type: str = "mobilenet_v3_small"
    dataset_id: str = "cifar10"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.001
    optimizer_type: str = "adam"
    device_selection: str = "auto"
    val_split: float = 0.15
    train_limit: Optional[int] = 500
    image_size: Optional[int] = None


class TrainingState:
    def __init__(self, job_id: str = "default", config: Optional["TrainingConfigRequest"] = None):
        self.job_id = job_id
        self.config_snapshot: Dict = config.model_dump() if config else {}
        self.lock = threading.Lock()
        self.status = "IDLE"
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.train_loss = None
        self.val_loss = None
        self.train_acc = None
        self.val_acc = None
        self.elapsed_seconds = 0.0
        self.history: List[Dict] = []
        self.logs: List[str] = []
        self.candidate_model_id = None
        self.error_message = None
        self.stop_requested = False
        self.device_name = None
        self.gpu_memory_mb = None
        self.thread = None
        self.created_at: str = time.strftime("%Y-%m-%d %H:%M:%S")

    def reset(self, epochs: int):
        with self.lock:
            self.status = "RUNNING"
            self.current_epoch = 0
            self.total_epochs = epochs
            self.current_step = 0
            self.total_steps = 0
            self.train_loss = self.val_loss = None
            self.train_acc = self.val_acc = None
            self.elapsed_seconds = 0.0
            self.history = []
            self.logs = []
            self.candidate_model_id = None
            self.error_message = None
            self.stop_requested = False
            self.device_name = None
            self.gpu_memory_mb = None

    def add_log(self, message: str):
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message}")
            self.logs = self.logs[-150:]

    def snapshot(self) -> Dict:
        with self.lock:
            progress = round(self.current_epoch / self.total_epochs * 100.0, 1) if self.total_epochs else 0.0
            return {
                "job_id": self.job_id,
                "config": self.config_snapshot,
                "created_at": self.created_at,
                "status": self.status,
                "stop_requested": self.stop_requested,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress_pct": progress,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "train_acc": self.train_acc,
                "val_acc": self.val_acc,
                "elapsed_seconds": round(self.elapsed_seconds, 1),
                "history": list(self.history),
                "logs": list(self.logs),
                "candidate_model_id": self.candidate_model_id,
                "error_message": self.error_message,
                "device": self.device_name,
                "gpu_memory_mb": self.gpu_memory_mb,
            }


# Legacy single-trainer kept for backward-compat with settings.py cleanup endpoint
TRAINER = TrainingState(job_id="legacy")


def _validate_config(config: TrainingConfigRequest):
    """Shared validation for training configs."""
    if config.model_type not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"model_type {config.model_type!r} is not supported. "
                   f"Valid options: {sorted(VALID_MODEL_TYPES)}",
        )
    if config.image_size is not None and not (32 <= config.image_size <= 512):
        raise HTTPException(status_code=400, detail="image_size must be between 32 and 512.")
    if not 1 <= config.epochs <= 100:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 100.")
    if not 1 <= config.batch_size <= 32:
        raise HTTPException(status_code=400, detail="batch_size must be between 1 and 32.")
    if config.learning_rate <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    if not 0.05 <= config.val_split < 0.9:
        raise HTTPException(status_code=400, detail="val_split must be between 0.05 and 0.89.")
    if config.device_selection == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA was requested but is unavailable.")
    if config.dataset_id != "cifar10":
        _safe_custom_dataset(config.dataset_id)


@router.get("/recommendations")
def get_hardware_recommendations():
    mgr = ServiceManager.get_instance()
    snap = mgr.resource_monitor.snapshot(force_refresh=True)
    cuda_available = torch.cuda.is_available()
    total_vram = snap.get("total_vram_mb")
    if cuda_available and total_vram and total_vram <= 4096:
        tier = "Edge/Laptop GPU (4 GB VRAM)"
        batch = 4
        device = "cuda"
        warning = "Conservative settings recommended: batch size 4 and num_workers=0."
    elif cuda_available:
        tier = "GPU"
        batch = 4
        device = "cuda"
        warning = "Use batch size 4 initially; increase only after a successful run."
    else:
        tier = "CPU Only"
        batch = 4
        device = "cpu"
        warning = "CUDA is unavailable. CPU training is supported with a low sample limit."
    return {
        "success": True,
        "hardware_tier": tier,
        "device_name": snap.get("gpu_name") or "System CPU",
        "cuda_available": cuda_available,
        "total_vram_mb": total_vram,
        "total_ram_mb": snap.get("ram_total_mb"),
        "recommended_batch_size": batch,
        "recommended_device": device,
        "recommended_num_workers": 0,
        "hardware_advisory": warning,
    }


@router.get("/status")
def get_training_status():
    return {"success": True, "telemetry": TRAINER.snapshot()}


@router.post("/stop")
def stop_training():
    if TRAINER.status not in {"RUNNING"}:
        return {"success": False, "message": "No active training job."}
    TRAINER.stop_requested = True
    TRAINER.add_log("Stop requested. The training loop will stop after the current batch.")
    return {"success": True, "message": "Training stop signal received."}


def _safe_custom_dataset(dataset_id: str) -> Path:
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or ".." in dataset_id:
        raise HTTPException(status_code=400, detail="Invalid dataset ID.")
    root = (PROJECT_ROOT / "data" / "custom_uploads" / dataset_id).resolve()
    try:
        root.relative_to((PROJECT_ROOT / "data" / "custom_uploads").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset path.")
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Custom dataset not found.")
    return root


def _training_worker(config: TrainingConfigRequest, trainer: TrainingState):
    start_time = time.time()
    device = torch.device(
        "cuda:0" if (config.device_selection in {"auto", "cuda"} and torch.cuda.is_available())
        else "cpu"
    )
    with trainer.lock:
        trainer.device_name = str(device)

    try:
        if config.dataset_id == "cifar10":
            dataset_root = PROJECT_ROOT / "data" / "cifar10"
            if not dataset_root.exists():
                raise FileNotFoundError("CIFAR-10 data is not available.")
            class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]
            full_dataset = datasets.CIFAR10(
                root=str(dataset_root), train=True, download=False,
                transform=None,
            )
            train_limit = config.train_limit or len(full_dataset)
            indices = list(range(min(train_limit, len(full_dataset))))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            dataset_root = _safe_custom_dataset(config.dataset_id)
            raw_dataset = datasets.ImageFolder(root=str(dataset_root))
            class_names = list(raw_dataset.classes)
            train_limit = config.train_limit or len(raw_dataset)
            indices = list(range(min(train_limit, len(raw_dataset))))
            full_dataset = torch.utils.data.Subset(raw_dataset, indices)

        arch_info = get_architecture_info(config.model_type)
        input_size: int = config.image_size or arch_info["input_size"]
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if config.dataset_id == "cifar10":
            train_base = datasets.CIFAR10(root=str(dataset_root), train=True, download=False, transform=train_transform)
            val_base = datasets.CIFAR10(root=str(dataset_root), train=True, download=False, transform=val_transform)
            pool_size = len(train_base)
            pool_indices = list(range(min(config.train_limit or pool_size, pool_size)))
        else:
            train_base = datasets.ImageFolder(root=str(dataset_root), transform=train_transform)
            val_base = datasets.ImageFolder(root=str(dataset_root), transform=val_transform)
            pool_size = len(train_base)
            pool_indices = list(range(min(config.train_limit or pool_size, pool_size)))

        total = len(pool_indices)
        val_size = max(1, min(total - 1, int(total * config.val_split)))
        train_size = total - val_size
        if train_size < 1:
            raise ValueError("Training split has no images.")

        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(pool_indices, [train_size, val_size], generator=generator)
        train_subset = torch.utils.data.Subset(train_base, list(train_indices))
        val_subset = torch.utils.data.Subset(val_base, list(val_indices))

        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        model, input_size = build_model(
            model_type=config.model_type,
            num_classes=len(class_names),
            pretrained=True,
        )
        model = model.to(device)
        canonical_key = resolve_model_type(config.model_type)
        arch_info = get_architecture_info(canonical_key)

        criterion = nn.CrossEntropyLoss()
        if config.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        elif config.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        run_id = f"candidate_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        preprocessing = {
            "resize": [input_size, input_size],
            "interpolation": "bilinear",
            "to_tensor": True,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "rgb": True,
            "training_random_horizontal_flip": True,
        }

        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

        best_val_acc = -1.0
        best_state = None
        best_val_predictions = []
        best_val_targets = []
        peak_vram_mb = 0.0
        val_latencies_ms = []

        with trainer.lock:
            trainer.total_steps = len(train_loader)

        for epoch in range(1, config.epochs + 1):
            if trainer.stop_requested:
                trainer.add_log("Training stopped by request.")
                break

            model.train()
            running_loss = correct = total = 0
            for step, (images, labels) in enumerate(train_loader, start=1):
                if trainer.stop_requested:
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
                with trainer.lock:
                    trainer.current_step = step
                    trainer.elapsed_seconds = time.time() - start_time
                    if torch.cuda.is_available() and device.type == "cuda":
                        cur_mem = round(torch.cuda.memory_allocated(device) / 1024**2, 1)
                        trainer.gpu_memory_mb = cur_mem
                        if cur_mem > peak_vram_mb:
                            peak_vram_mb = cur_mem

            if trainer.stop_requested:
                break

            model.eval()
            val_loss = val_correct = val_total = 0
            epoch_val_preds = []
            epoch_val_targets = []
            epoch_latencies = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    t_eval_start = time.perf_counter()
                    outputs = model(images)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_eval_end = time.perf_counter()
                    epoch_latencies.append((t_eval_end - t_eval_start) * 1000.0 / max(1, images.size(0)))

                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    epoch_val_preds.extend(preds.cpu().tolist())
                    epoch_val_targets.extend(labels.cpu().tolist())

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1) * 100.0
            val_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1) * 100.0
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_acc": round(train_acc, 3),
                "val_acc": round(val_acc, 3),
                "elapsed_seconds": round(time.time() - start_time, 1),
            }
            with trainer.lock:
                trainer.current_epoch = epoch
                trainer.train_loss = record["train_loss"]
                trainer.val_loss = record["val_loss"]
                trainer.train_acc = record["train_acc"]
                trainer.val_acc = record["val_acc"]
                trainer.history.append(record)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_predictions = list(epoch_val_preds)
                best_val_targets = list(epoch_val_targets)
                val_latencies_ms = list(epoch_latencies)
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            trainer.add_log(
                f"Epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f}, "
                f"train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )

        with trainer.lock:
            status = "STOPPED" if trainer.stop_requested else "COMPLETED"
            history = list(trainer.history)

            # Calculate parameter count, model size, and FLOPs estimation
            param_count = sum(p.numel() for p in model.parameters())
            trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2), 2)
            flops_m = round((param_count * 2) / 1e6, 2)

            # Inference Latency metrics
            avg_latency_ms = round(float(np.mean(val_latencies_ms)), 2) if val_latencies_ms else None
            p95_latency_ms = round(float(np.percentile(val_latencies_ms, 95)), 2) if val_latencies_ms else None

            # Advanced Classification Evaluation
            eval_metrics = {}
            cm_list = []
            clf_report = {}
            if best_val_targets and best_val_predictions:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    best_val_targets, best_val_predictions, average="weighted", zero_division=0
                )
                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    best_val_targets, best_val_predictions, average="macro", zero_division=0
                )
                labels_in_scope = list(range(len(class_names)))
                cm = confusion_matrix(best_val_targets, best_val_predictions, labels=labels_in_scope)
                cm_list = cm.tolist()
                clf_report = classification_report(
                    best_val_targets, best_val_predictions,
                    labels=labels_in_scope,
                    target_names=class_names, output_dict=True, zero_division=0
                )
                eval_metrics = {
                    "accuracy": round(best_val_acc, 2),
                    "precision_weighted": round(float(prec) * 100.0, 2),
                    "recall_weighted": round(float(rec) * 100.0, 2),
                    "f1_score_weighted": round(float(f1) * 100.0, 2),
                    "precision_macro": round(float(macro_p) * 100.0, 2),
                    "recall_macro": round(float(macro_r) * 100.0, 2),
                    "f1_score_macro": round(float(macro_f1) * 100.0, 2),
                }

            metrics = {
                "best_val_accuracy": round(best_val_acc, 3) if best_val_acc >= 0 else None,
                "final_train_loss": trainer.train_loss,
                "final_val_loss": trainer.val_loss,
                "final_train_accuracy": trainer.train_acc,
                "final_val_accuracy": trainer.val_acc,
                "duration_seconds": round(time.time() - start_time, 1),
                "parameter_count": param_count,
                "trainable_parameters": trainable_param_count,
                "model_size_mb": model_size_mb,
                "estimated_mflops": flops_m,
                "average_inference_latency_ms": avg_latency_ms,
                "p95_inference_latency_ms": p95_latency_ms,
                "peak_memory_mb": peak_vram_mb if peak_vram_mb > 0 else round(float(param_count * 4 / (1024**2)), 1),
                "evaluation": eval_metrics,
                "confusion_matrix": cm_list,
                "classification_report": clf_report,
            }

        if best_state is not None and status in {"COMPLETED", "STOPPED"}:
            model.load_state_dict(best_state)
            model = model.to("cpu").eval()
            metadata = {
                "run_id": run_id,
                "status": status,
                "dataset_id": config.dataset_id,
                "architecture": canonical_key,
                "architecture_display": arch_info["display_name"],
                "class_names": class_names,
                "num_classes": len(class_names),
                "preprocessing": preprocessing,
                "device": str(device),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with CANDIDATE_MODELS_LOCK:
                CANDIDATE_MODELS[run_id] = {
                    "model": model,
                    "metadata": metadata,
                    "metrics": metrics,
                    "history": history,
                }
            with trainer.lock:
                trainer.candidate_model_id = run_id
                trainer.status = status
                trainer.elapsed_seconds = time.time() - start_time
        else:
            with trainer.lock:
                trainer.status = status
                trainer.elapsed_seconds = time.time() - start_time

        trainer.add_log(f"Training final status: {status}.")

    except Exception as exc:
        message = str(exc)
        if isinstance(exc, RuntimeError) and "out of memory" in message.lower():
            message = "CUDA out of memory. Reduce batch size or use CPU."
        trainer.add_log(f"ERROR: {message}")
        with trainer.lock:
            trainer.error_message = message
            trainer.elapsed_seconds = time.time() - start_time
            trainer.status = "FAILED"


# ── Multi-job endpoints ────────────────────────────────────────────────────

@router.post("/jobs/start")
def start_training_job(config: TrainingConfigRequest):
    """Launch a new parallel training job. Returns a unique job_id for tracking."""
    _validate_config(config)

    job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    trainer = TrainingState(job_id=job_id, config=config)

    resolved = resolve_model_type(config.model_type)
    display = get_architecture_info(resolved)["display_name"]
    trainer.reset(config.epochs)
    trainer.add_log(f"[{job_id}] Starting: {display} on {config.dataset_id}.")

    thread = threading.Thread(target=_training_worker, args=(config, trainer), daemon=True)
    trainer.thread = thread

    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = trainer

    try:
        thread.start()
    except RuntimeError:
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS.pop(job_id, None)
        raise HTTPException(status_code=503, detail="Could not start a training worker.")

    return {"success": True, "job_id": job_id, "status": "RUNNING", "message": f"Job {job_id} started."}


@router.get("/jobs")
def list_training_jobs(include_finished: bool = False):
    """List active jobs and optionally their session history."""
    with TRAINING_JOBS_LOCK:
        snapshots = [trainer.snapshot() for trainer in TRAINING_JOBS.values()]
    jobs = snapshots if include_finished else [
        job for job in snapshots if job["status"] == "RUNNING"
    ]
    return {"success": True, "jobs": jobs, "count": len(jobs)}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get telemetry for a specific active training job."""
    with TRAINING_JOBS_LOCK:
        trainer = TRAINING_JOBS.get(job_id)
    if trainer is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"success": True, "telemetry": trainer.snapshot()}


@router.post("/jobs/{job_id}/stop")
def stop_training_job(job_id: str):
    """Request a specific job to stop after the current batch."""
    with TRAINING_JOBS_LOCK:
        trainer = TRAINING_JOBS.get(job_id)
    if trainer is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    with trainer.lock:
        if trainer.status != "RUNNING":
            return {"success": False, "message": f"Job '{job_id}' is not running (status: {trainer.status})."}
        if trainer.stop_requested:
            return {"success": True, "message": f"Stop already requested for job '{job_id}'."}
        trainer.stop_requested = True
    trainer.add_log(f"Stop requested for job {job_id}.")
    return {"success": True, "message": f"Stop signal sent to job '{job_id}'."}


# ── Legacy single-job endpoints (kept for backward-compat) ─────────────────

@router.post("/start")
def start_training(config: TrainingConfigRequest):
    if TRAINER.status in {"RUNNING"}:
        raise HTTPException(status_code=409, detail="A training job is already running.")
    _validate_config(config)

    resolved = resolve_model_type(config.model_type)
    display = get_architecture_info(resolved)["display_name"]
    TRAINER.reset(config.epochs)
    TRAINER.add_log(f"Starting training: {display} on dataset {config.dataset_id}.")
    thread = threading.Thread(target=_training_worker, args=(config, TRAINER), daemon=True)
    TRAINER.thread = thread
    thread.start()
    return {"success": True, "status": "RUNNING", "message": "Training started."}


@router.get("/architectures")
def list_architectures():
    """Return the full catalogue of supported candidate model architectures."""
    return {
        "success": True,
        "architectures": [
            {**info, "model_type": key}
            for key, info in ARCHITECTURE_REGISTRY.items()
        ],
    }


@router.get("/custom-models")
def list_custom_models():
    with CANDIDATE_MODELS_LOCK:
        result = [
            {
                "run_id": run_id,
                "architecture": candidate["metadata"]["architecture"],
                "architecture_display": candidate["metadata"].get("architecture_display", candidate["metadata"]["architecture"]),
                "class_names": candidate["metadata"]["class_names"],
                "status": candidate["metadata"]["status"],
                "metrics": candidate["metrics"],
                "history": candidate["history"],
                "dataset_id": candidate["metadata"]["dataset_id"],
            }
            for run_id, candidate in CANDIDATE_MODELS.items()
        ]
    return {"success": True, "models": result}
