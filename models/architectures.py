"""
ADAPT-EDGE Model Architecture Factory
Provides a unified factory for all supported candidate model architectures.
Each architecture is genuinely instantiated from torchvision with pretrained
ImageNet weights (when available) and a remapped classification head for the
requested number of output classes.

Supported model_type keys:
    mobilenet_v3_small  — MobileNetV3-Small   (~2.5 M params)
    mobilenet_v3_large  — MobileNetV3-Large   (~5.4 M params)
    shufflenet_v2       — ShuffleNetV2-x1.0   (~2.3 M params)
    resnet18            — ResNet-18            (~11.7 M params)
    resnet50            — ResNet-50            (~25.6 M params)
    efficientnet_b0     — EfficientNet-B0      (~5.3 M params)
    efficientnet_b3     — EfficientNet-B3      (~12.2 M params)
    squeezenet          — SqueezeNet-1.0       (~1.2 M params)

Legacy aliases (kept for backwards compatibility with existing sessions):
    small  → mobilenet_v3_small
    large  → mobilenet_v3_large
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from typing import Tuple


# ---------------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------------

ARCHITECTURE_REGISTRY: dict[str, dict] = {
    "mobilenet_v3_small": {
        "display_name": "MobileNetV3-Small",
        "category": "Lightweight",
        "params_m": 2.5,
        "description": "Fast and compact. Optimised for edge inference with low VRAM footprint.",
        "input_size": 224,
        "recommended_batch": 16,
        "tags": ["lightweight", "edge"],
    },
    "mobilenet_v3_large": {
        "display_name": "MobileNetV3-Large",
        "category": "Efficient",
        "params_m": 5.4,
        "description": "Larger MobileNet variant. Better accuracy with moderate compute cost.",
        "input_size": 224,
        "recommended_batch": 8,
        "tags": ["efficient", "baseline"],
    },
    "shufflenet_v2": {
        "display_name": "ShuffleNetV2",
        "category": "Lightweight",
        "params_m": 2.3,
        "description": "Channel-shuffle architecture. Very fast on CPU with minimal parameters.",
        "input_size": 224,
        "recommended_batch": 16,
        "tags": ["lightweight", "edge"],
    },
    "resnet18": {
        "display_name": "ResNet-18",
        "category": "Residual CNN",
        "params_m": 11.7,
        "description": "Classic residual network. Reliable accuracy, well understood behaviour.",
        "input_size": 224,
        "recommended_batch": 8,
        "tags": ["residual", "classic"],
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "category": "Deeper CNN",
        "params_m": 25.6,
        "description": "Deeper residual network. Higher accuracy, requires more compute and VRAM.",
        "input_size": 224,
        "recommended_batch": 4,
        "tags": ["residual", "deep"],
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "category": "Efficient CNN",
        "params_m": 5.3,
        "description": "Compound-scaled network. Strong accuracy-to-parameter ratio.",
        "input_size": 224,
        "recommended_batch": 8,
        "tags": ["efficient", "scalable"],
    },
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "category": "Larger CNN",
        "params_m": 12.2,
        "description": "Larger EfficientNet variant. Best accuracy in this family, more VRAM needed.",
        "input_size": 300,
        "recommended_batch": 4,
        "tags": ["efficient", "scalable", "large"],
    },
    "squeezenet": {
        "display_name": "SqueezeNet",
        "category": "Compact CNN",
        "params_m": 1.2,
        "description": "Fire-module architecture. Fewest parameters; suitable for very constrained devices.",
        "input_size": 224,
        "recommended_batch": 16,
        "tags": ["compact", "edge"],
    },
}

# Legacy key aliases
_ALIASES: dict[str, str] = {
    "small": "mobilenet_v3_small",
    "large": "mobilenet_v3_large",
}

VALID_MODEL_TYPES: set[str] = set(ARCHITECTURE_REGISTRY) | set(_ALIASES)


def resolve_model_type(model_type: str) -> str:
    """Resolve a model_type key (including legacy aliases) to its canonical key."""
    return _ALIASES.get(model_type, model_type)


def get_architecture_info(model_type: str) -> dict:
    """Return registry metadata for a given model_type (or alias)."""
    key = resolve_model_type(model_type)
    if key not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type!r}. Valid choices: {sorted(VALID_MODEL_TYPES)}")
    return {**ARCHITECTURE_REGISTRY[key], "model_type": key}


def build_model(model_type: str, num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Instantiate and return a torchvision model with remapped classifier head.

    Returns:
        (model, input_size)  — PyTorch module in train mode and the expected
                               square input resolution.
    """
    key = resolve_model_type(model_type)
    info = get_architecture_info(key)
    input_size: int = info["input_size"]

    # ── MobileNetV3-Small ────────────────────────────────────────────────────
    if key == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)

    # ── MobileNetV3-Large ────────────────────────────────────────────────────
    elif key == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)

    # ── ShuffleNetV2-x1.0 ───────────────────────────────────────────────────
    elif key == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        m = models.shufflenet_v2_x1_0(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    # ── ResNet-18 ────────────────────────────────────────────────────────────
    elif key == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    # ── ResNet-50 ────────────────────────────────────────────────────────────
    elif key == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    # ── EfficientNet-B0 ──────────────────────────────────────────────────────
    elif key == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)

    # ── EfficientNet-B3 ──────────────────────────────────────────────────────
    elif key == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b3(weights=weights)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)

    # ── SqueezeNet-1.0 ───────────────────────────────────────────────────────
    elif key == "squeezenet":
        weights = models.SqueezeNet1_0_Weights.DEFAULT if pretrained else None
        m = models.squeezenet1_0(weights=weights)
        # SqueezeNet uses a Conv2d as final classifier — replace with num_classes channels
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        m.num_classes = num_classes

    else:
        raise ValueError(f"build_model: unhandled key {key!r}")

    return m, input_size
