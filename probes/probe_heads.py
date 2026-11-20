import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any


class ProbeHead(nn.Module):

    def __init__(self, in_channels, num_classes=10):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            in_channels,
            num_classes
        )

    def forward(self, x):

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)


class SequentialModelB:
    """
    Sequential execution wrapper for MobileNetV3-Large.
    Supports partial execution to intermediate probe locations (4, 9, 13)
    and optional integration with ContinuationCache for intermediate feature reuse.
    """

    def __init__(self, model_b, device, cache=None):

        self.backbone = model_b.model
        self.features = self.backbone.features
        self.classifier = self.backbone.classifier
        self.device = device
        self.cache = cache

        self.probe_heads = {}

        self.probe_blocks = [4, 9, 13]

        self.channels = {
            4: 40,
            9: 80,
            13: 160
        }

    def register_probe(self, block, head):

        if block not in self.probe_blocks:
            raise ValueError(f"Invalid probe block: {block}. Must be one of {self.probe_blocks}")

        self.probe_heads[block] = head.to(self.device).eval()

    @torch.inference_mode()
    def start(self, images):
        """Execute from raw image through Block 4 (indices 0..4)."""
        x = images
        for index in range(5):
            x = self.features[index](x)
        return x

    @torch.inference_mode()
    def continue_to(self, features, start_block, end_block):
        """Execute feature blocks from start_block through end_block inclusive."""
        x = features
        for index in range(start_block, end_block + 1):
            x = self.features[index](x)
        return x

    @torch.inference_mode()
    def predict_probe(self, features, block):
        """Obtain classification logits from the registered probe head at block."""
        if block not in self.probe_heads:
            raise ValueError(
                f"Probe head for block {block} is not registered"
            )

        return self.probe_heads[block](features)

    @torch.inference_mode()
    def finish(self, features, current_block=13):
        """
        Continue sequential Model B execution from the current block
        through block 16, then perform final classification.
        """
        x = features

        if current_block not in [4, 9, 13]:
            raise ValueError(
                f"Invalid current block: {current_block}. "
                "Expected 4, 9, or 13."
            )

        # Continue from the block immediately after current_block.
        for index in range(current_block + 1, 17):
            x = self.features[index](x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)