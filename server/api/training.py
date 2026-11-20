"""
Custom PyTorch training and custom-model discovery.
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
from models.model_a import ModelA
from models.model_b import ModelB

router = APIRouter(prefix="/api/training", tags=["Model Training"])
CANDIDATE_MODELS: Dict[str, Dict[str, Any]] = {}
CANDIDATE_MODELS_LOCK = threading.Lock()


class TrainingConfigRequest(BaseModel):
    model_type: str = "small"
    dataset_id: str = "cifar10"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.001
    optimizer_type: str = "adam"
    device_selection: str = "auto"
    val_split: float = 0.15
    train_limit: Optional[int] = 500


class TrainingState:
    def __init__(self):
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
                "status": self.status,
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


TRAINER = TrainingState()


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


def _training_worker(config: TrainingConfigRequest):
    start_time = time.time()
    device = torch.device(
        "cuda:0" if (config.device_selection in {"auto", "cuda"} and torch.cuda.is_available())
        else "cpu"
    )
    with TRAINER.lock:
        TRAINER.device_name = str(device)

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

        input_size = 224
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

        if config.model_type == "large":
            model = ModelB(pretrained=True, num_classes=len(class_names)).to(device)
        else:
            model = ModelA(pretrained=True, num_classes=len(class_names)).to(device)

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

        best_val_acc = -1.0
        best_state = None
        with TRAINER.lock:
            TRAINER.total_steps = len(train_loader)

        for epoch in range(1, config.epochs + 1):
            if TRAINER.stop_requested:
                TRAINER.status = "STOPPED"
                TRAINER.add_log("Training stopped by request.")
                break

            model.train()
            running_loss = correct = total = 0
            for step, (images, labels) in enumerate(train_loader, start=1):
                if TRAINER.stop_requested:
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
                with TRAINER.lock:
                    TRAINER.current_step = step
                    TRAINER.elapsed_seconds = time.time() - start_time
                    if torch.cuda.is_available() and device.type == "cuda":
                        TRAINER.gpu_memory_mb = round(torch.cuda.memory_allocated(device) / 1024**2, 1)

            if TRAINER.stop_requested:
                TRAINER.status = "STOPPED"
                break

            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

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
            with TRAINER.lock:
                TRAINER.current_epoch = epoch
                TRAINER.train_loss = record["train_loss"]
                TRAINER.val_loss = record["val_loss"]
                TRAINER.train_acc = record["train_acc"]
                TRAINER.val_acc = record["val_acc"]
                TRAINER.history.append(record)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            TRAINER.add_log(
                f"Epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f}, "
                f"train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )

        with TRAINER.lock:
            status = "STOPPED" if TRAINER.status == "STOPPED" else "COMPLETED"
            history = list(TRAINER.history)
            metrics = {
                "best_val_accuracy": round(best_val_acc, 3) if best_val_acc >= 0 else None,
                "final_train_loss": TRAINER.train_loss,
                "final_val_loss": TRAINER.val_loss,
                "final_train_accuracy": TRAINER.train_acc,
                "final_val_accuracy": TRAINER.val_acc,
                "duration_seconds": round(time.time() - start_time, 1),
            }

        if best_state is not None and status in {"COMPLETED", "STOPPED"}:
            model.load_state_dict(best_state)
            model = model.to("cpu").eval()
            metadata = {
                "run_id": run_id,
                "status": status,
                "dataset_id": config.dataset_id,
                "architecture": "mobilenet_v3_large" if config.model_type == "large" else "mobilenet_v3_small",
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
            with TRAINER.lock:
                TRAINER.candidate_model_id = run_id
                TRAINER.status = status
                TRAINER.elapsed_seconds = time.time() - start_time
        else:
            with TRAINER.lock:
                TRAINER.status = status
                TRAINER.elapsed_seconds = time.time() - start_time

        TRAINER.add_log(f"Training final status: {status}.")
    except RuntimeError as exc:
        TRAINER.status = "FAILED"
        TRAINER.error_message = ("CUDA out of memory. Reduce batch size or use CPU." if "out of memory" in str(exc).lower() else str(exc))
        TRAINER.add_log(f"ERROR: {TRAINER.error_message}")
    except Exception as exc:
        TRAINER.status = "FAILED"
        TRAINER.error_message = str(exc)
        TRAINER.add_log(f"ERROR: {exc}")


@router.post("/start")
def start_training(config: TrainingConfigRequest):
    if TRAINER.status in {"RUNNING"}:
        raise HTTPException(status_code=409, detail="A training job is already running.")
    if config.model_type not in {"small", "large"}:
        raise HTTPException(status_code=400, detail="model_type must be small or large.")
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

    TRAINER.reset(config.epochs)
    TRAINER.add_log(f"Starting real {config.model_type} training on dataset {config.dataset_id}.")
    thread = threading.Thread(target=_training_worker, args=(config,), daemon=True)
    TRAINER.thread = thread
    thread.start()
    return {"success": True, "status": "RUNNING", "message": "Training started."}


@router.get("/custom-models")
def list_custom_models():
    with CANDIDATE_MODELS_LOCK:
        models = [
            {
                "run_id": run_id,
                "architecture": candidate["metadata"]["architecture"],
                "class_names": candidate["metadata"]["class_names"],
                "status": candidate["metadata"]["status"],
                "metrics": candidate["metrics"],
                "history": candidate["history"],
                "dataset_id": candidate["metadata"]["dataset_id"],
            }
            for run_id, candidate in CANDIDATE_MODELS.items()
        ]
    return {"success": True, "models": models}
