"""
Core service manager for ADAPT-EDGE Web Platform.
Maintains singleton instances of AdaptEdgePipeline, ResourceMonitor,
and provides sample caching and helper inference utilities.
"""

import sys
import io
import time
import base64
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.adapt_edge_pipeline import AdaptEdgePipeline
from utils.resource_monitor import ResourceMonitor
from controller.resource_policy import ResourcePolicyConfig
from utils.profiling import synchronize

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Standard normalization for MobileNet backbones
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ServiceManager:
    """
    Manages global resources, models, and execution for the FastAPI backend.
    """
    _instance: Optional['ServiceManager'] = None

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[ServiceManager] Initializing on device: {self.device}")
        
        # Singleton pipeline
        self.pipeline: Optional[AdaptEdgePipeline] = None
        self.resource_monitor = ResourceMonitor()
        self.cifar10_dataset: Optional[datasets.CIFAR10] = None
        self.sample_cache: List[Dict[str, Any]] = []
        self.recent_activity: List[Dict[str, Any]] = []
        self.is_initialized = False
        self._initialized = False
        self.running_jobs: Dict[str, Dict[str, Any]] = {}
        self.last_sample_indices: List[int] = []

        # User-configured Pair Configuration for ADAPT-EDGE
        self.active_pair = {
            "model_a_id": "mobilenet_v3_small_baseline",
            "model_a_name": "MobileNetV3-Small (Baseline)",
            "model_a_type": "mobilenet_v3_small",
            "model_b_id": "mobilenet_v3_large_baseline",
            "model_b_name": "MobileNetV3-Large (Baseline)",
            "model_b_type": "mobilenet_v3_large",
            "threshold_a": 0.85,
            "threshold_probe": 0.80,
            "latency_budget_ms": 25.0,
            "enforce_resource_budget": True,
            "probes_compatible": True,  # True only if Model B is MobileNetV3-Large with matching probe heads
            "probe_blocks": [4, 9, 13]
        }

        # Global Settings: Default Training Parameters
        self.default_training_params = {
            "batch_size": 4,
            "epochs": 5,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "img_size": 224,
            "val_split": 0.2,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "architecture": "mobilenet_v3_small"
        }

        # Global Settings: Resource Budget Configuration
        self.resource_budget = {
            "latency_budget_ms": 25.0,
            "ram_budget_mb": 2048.0,
            "vram_budget_mb": 1024.0,
            "cpu_throttle_threshold": 85.0,
            "enforce_resource_budget": True,
            "threshold_a": 0.85,
            "threshold_probe": 0.80
        }

        # Global Settings: Backend & API Telemetry
        self.backend_settings = {
            "telemetry_poll_interval_ms": 1000,
            "cache_retention_limit": 100,
            "simulation_mode": False,
            "debug_logging": False
        }

    def set_device(self, new_device_str: str) -> str:
        """Switch device between CPU and CUDA."""
        if "cuda" in new_device_str.lower() and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this host.")
        new_device = torch.device(new_device_str)
        self.device = new_device
        if self.pipeline is not None:
            self.pipeline.device = new_device
            if self.pipeline.model_a is not None:
                self.pipeline.model_a = self.pipeline.model_a.to(new_device)
            if self.pipeline.model_b is not None:
                self.pipeline.model_b = self.pipeline.model_b.to(new_device)
            if self.pipeline.sequential_b is not None:
                self.pipeline.sequential_b.device = new_device
                for head in self.pipeline.sequential_b.probe_heads.values():
                    head.to(new_device)
        self.log_activity("Device Changed", f"Execution device changed to {new_device}", "info")
        return str(self.device)

    @classmethod
    def get_instance(cls) -> 'ServiceManager':
        if cls._instance is None:
            cls._instance = ServiceManager()
        return cls._instance

    def initialize(self) -> None:
        """Load pipeline and preload sample dataset."""
        try:
            print("[ServiceManager] Loading AdaptEdgePipeline...")
            self.pipeline = AdaptEdgePipeline(device=self.device)
            print("[ServiceManager] Pipeline successfully loaded with verified checkpoints.")
            self.log_activity("System Initialized", "All checkpoints and models loaded into memory.")
        except Exception as e:
            print(f"[ServiceManager] Warning: Error loading pipeline: {e}")
            self.log_activity("Initialization Warning", f"Error loading pipeline: {str(e)}")

        try:
            cifar_path = PROJECT_ROOT / "data/cifar10"
            if cifar_path.exists():
                print("[ServiceManager] Loading CIFAR-10 test set for live sample selection...")
                self.cifar10_dataset = datasets.CIFAR10(
                    root=str(cifar_path),
                    train=False,
                    download=False
                )
                self._preload_samples(count=30)
                print(f"[ServiceManager] Preloaded {len(self.sample_cache)} sample images.")
            else:
                self._generate_synthetic_samples(count=30)
                print(f"[ServiceManager] Preloaded {len(self.sample_cache)} synthetic CIFAR-10 research samples.")
        except Exception as e:
            print(f"[ServiceManager] Warning: CIFAR-10 loading error: {e}")
            if not self.sample_cache:
                self._generate_synthetic_samples(count=30)

        self.is_initialized = True
        self._initialized = True

    def log_activity(self, title: str, details: str, event_type: str = "info") -> None:
        """Append to live recent activity feed."""
        entry = {
            "id": f"act_{len(self.recent_activity) + 1}",
            "timestamp": time.strftime("%H:%M:%S"),
            "title": title,
            "details": details,
            "type": event_type
        }
        self.recent_activity.insert(0, entry)
        if len(self.recent_activity) > 50:
            self.recent_activity.pop()

    def _generate_synthetic_samples(self, count: int = 30) -> None:
        """Generate high-contrast CIFAR-10 sample thumbnails for immediate evaluation when dataset files are absent."""
        self.sample_cache.clear()
        palette = [
            (220, 50, 50), (50, 180, 50), (50, 80, 220), (220, 160, 30), (160, 50, 220),
            (50, 200, 200), (220, 100, 180), (120, 120, 120), (180, 140, 80), (70, 70, 70)
        ]
        for idx in range(count):
            label = idx % 10
            base_col = palette[label]
            img = Image.new("RGB", (64, 64), color=base_col)
            # Add simple geometric gradient pattern
            for x in range(16, 48):
                for y in range(16, 48):
                    img.putpixel((x, y), ((base_col[0] + 50) % 256, (base_col[1] + 50) % 256, (base_col[2] + 50) % 256))
            buffered = io.BytesIO()
            preview = img.resize((128, 128), Image.Resampling.NEAREST)
            preview.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            self.sample_cache.append({
                "index": idx,
                "dataset_index": idx,
                "label": label,
                "label_id": label,
                "class_name": CIFAR10_CLASSES[label],
                "image_b64": f"data:image/png;base64,{img_b64}",
                "image_base64": f"data:image/png;base64,{img_b64}"
            })

    def _preload_samples(self, count: int = 30) -> None:
        """Cache first N test samples with base64 PNG thumbnails for instant UI rendering."""
        if self.cifar10_dataset is None:
            self._generate_synthetic_samples(count)
            return

        self.sample_cache.clear()
        step = max(1, len(self.cifar10_dataset) // count)
        indices = [i * step for i in range(count)]

        for idx in indices:
            raw_img, label = self.cifar10_dataset[idx]
            buffered = io.BytesIO()
            preview_img = raw_img.resize((128, 128), Image.Resampling.NEAREST)
            preview_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            self.sample_cache.append({
                "index": idx,
                "dataset_index": idx,
                "label": label,
                "label_id": label,
                "class_name": CIFAR10_CLASSES[label],
                "image_b64": f"data:image/png;base64,{img_b64}",
                "image_base64": f"data:image/png;base64,{img_b64}"
            })

    def get_sample_image(self, dataset_index: int) -> Tuple[torch.Tensor, int, str]:
        """Fetch raw image tensor and base64 string for given CIFAR-10 test index."""
        if self.cifar10_dataset is not None and 0 <= dataset_index < len(self.cifar10_dataset):
            pil_img, label = self.cifar10_dataset[dataset_index]
        elif len(self.sample_cache) > 0:
            cached = self.sample_cache[dataset_index % len(self.sample_cache)]
            label = cached["label_id"]
            b64_data = cached["image_b64"].split(",", 1)[-1]
            pil_img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        else:
            self._generate_synthetic_samples(30)
            cached = self.sample_cache[0]
            label = cached["label_id"]
            b64_data = cached["image_b64"].split(",", 1)[-1]
            pil_img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")

        tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(self.device)
        buffered = io.BytesIO()
        preview_img = pil_img.resize((128, 128), Image.Resampling.NEAREST)
        preview_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return tensor, label, f"data:image/png;base64,{img_b64}"

    def refresh_sample_cache(self, count: int = 12) -> List[Dict[str, Any]]:
        """Select a fresh deterministic-randomized set from the real CIFAR-10 test set."""
        if self.cifar10_dataset is None:
            if not self.sample_cache:
                self._generate_synthetic_samples(count)
            return self.sample_cache[:count]
        count = max(1, min(count, len(self.cifar10_dataset)))
        indices = list(range(len(self.cifar10_dataset)))
        random.shuffle(indices)
        selected = indices[:count]
        if set(selected) == set(self.last_sample_indices) and len(indices) > count:
            random.shuffle(indices)
            selected = indices[:count]
        self.last_sample_indices = list(selected)
        refreshed = []
        for idx in selected:
            raw_img, label = self.cifar10_dataset[idx]
            buffered = io.BytesIO()
            preview_img = raw_img.resize((128, 128), Image.Resampling.BILINEAR)
            preview_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            refreshed.append({
                "index": idx,
                "dataset_index": idx,
                "label": label,
                "label_id": label,
                "class_name": CIFAR10_CLASSES[label],
                "image_b64": f"data:image/png;base64,{img_b64}",
                "image_base64": f"data:image/png;base64,{img_b64}",
            })
        self.sample_cache = refreshed
        return refreshed

    def process_uploaded_image(self, image_bytes: bytes) -> Tuple[torch.Tensor, str]:
        """Process arbitrary user-uploaded image bytes into PyTorch tensor and preview string."""
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        buffered = io.BytesIO()
        preview_img = pil_img.resize((128, 128), Image.Resampling.BILINEAR)
        preview_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return tensor, f"data:image/png;base64,{img_b64}"
