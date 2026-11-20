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
        except Exception as e:
            print(f"[ServiceManager] Warning: CIFAR-10 loading error: {e}")

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

    def _preload_samples(self, count: int = 30) -> None:
        """Cache first N test samples with base64 PNG thumbnails for instant UI rendering."""
        if self.cifar10_dataset is None:
            return

        self.sample_cache.clear()
        # Sample across diverse classes
        step = max(1, len(self.cifar10_dataset) // count)
        indices = [i * step for i in range(count)]

        for idx in indices:
            raw_img, label = self.cifar10_dataset[idx]
            # Convert PIL to base64
            buffered = io.BytesIO()
            # Upscale slightly for crisp UI display
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
        if self.cifar10_dataset is None:
            raise ValueError("CIFAR-10 dataset not loaded.")
        if dataset_index < 0 or dataset_index >= len(self.cifar10_dataset):
            raise IndexError(f"Index {dataset_index} out of bounds (0..{len(self.cifar10_dataset)-1})")

        pil_img, label = self.cifar10_dataset[dataset_index]
        tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        buffered = io.BytesIO()
        preview_img = pil_img.resize((128, 128), Image.Resampling.NEAREST)
        preview_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return tensor, label, f"data:image/png;base64,{img_b64}"

    def refresh_sample_cache(self, count: int = 12) -> List[Dict[str, Any]]:
        """Select a fresh deterministic-randomized set from the real CIFAR-10 test set."""
        if self.cifar10_dataset is None:
            return []
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
