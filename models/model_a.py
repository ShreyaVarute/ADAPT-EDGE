
import torch.nn as nn
from torchvision import models


class ModelA(nn.Module):

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )

        self.model = models.mobilenet_v3_small(weights=weights)

        input_features = self.model.classifier[3].in_features

        self.model.classifier[3] = nn.Linear(
            input_features, num_classes
        )

    def forward(self, x):
        return self.model(x)
