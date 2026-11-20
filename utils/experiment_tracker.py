"""
Experiment metadata and execution tracking for ADAPT-EDGE.
Records hardware specifications, software versions, random seeds, dataset splits,
checkpoints used, threshold configurations, and research protocol disclosures.
Manages isolated timestamped experiment directories to preserve all historical outputs.
"""

import sys
import os
import json
import time
import platform
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch

from utils.resource_monitor import ResourceMonitor


# Default root directory for new experimental runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


class ExperimentTracker:
    """
    Manages experiment metadata, hardware provenance, configuration logging,
    and output directory isolation.
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        config: Optional[Dict[str, Any]] = None,
        base_dir: Optional[Path] = None,
        seed: int = 42
    ):
        self.experiment_name = experiment_name
        self.seed = seed
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"

        base = base_dir or RESULTS_ROOT
        self.run_dir = Path(base) / f"run_{self.experiment_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.start_time = time.time()

        # Capture environmental and hardware provenance
        self.metadata = self._build_metadata()
        self._save_metadata()

    def _get_file_hash(self, path: Union[str, Path]) -> Optional[str]:
        p = Path(path)
        if not p.is_file():
            return None
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                # Read 64KB chunks
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]  # Short 16-char fingerprint
        except Exception:
            return None

    def _build_metadata(self) -> Dict[str, Any]:
        monitor = ResourceMonitor()
        snapshot = monitor.snapshot()

        # Software stack
        software = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        }

        # Hardware stack
        hardware = {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "total_ram_mb": snapshot.get("ram_total_mb"),
            "gpu_name": snapshot.get("gpu_name"),
            "gpu_vram_mb": snapshot.get("ram_total_mb") if snapshot.get("gpu_available") else None,
            "gpu_temperature_c": snapshot.get("gpu_temperature_c"),
        }

        # Known checkpoint paths and hashes if configured
        checkpoints_info = {}
        if "checkpoints" in self.config:
            for name, path in self.config["checkpoints"].items():
                p = Path(path)
                checkpoints_info[name] = {
                    "path": str(path),
                    "exists": p.is_file(),
                    "sha256_prefix": self._get_file_hash(p) if p.is_file() else None
                }

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "random_seed": self.seed,
            "software": software,
            "hardware": hardware,
            "checkpoints": checkpoints_info,
            "config": self.config,
            "research_disclosure": {
                "dataset": "CIFAR-10",
                "cifar10_test_set_status": "Used in prior exploratory experiments; recorded limitation.",
                "model_a": "MobileNetV3-Small (reused pretrained checkpoint, frozen weights)",
                "model_b": "MobileNetV3-Large (reused pretrained checkpoint, frozen weights)",
                "probes": "Trained intermediate heads at blocks 4, 9, 13 (frozen backbone)",
                "evaluation_protocol": "Sequential partial execution with decision-selective computation."
            }
        }

    def _save_metadata(self) -> None:
        path = self.run_dir / "experiment_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Record final results and calculate elapsed experiment duration.
        """
        duration = time.time() - self.start_time
        summary = {
            "experiment_id": self.experiment_id,
            "duration_seconds": round(duration, 2),
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        path = self.run_dir / "results_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
