"""
Confidence calibration utilities for ADAPT-EDGE.
Implements Temperature Scaling (Guo et al., 2017) calibrated on validation data only.
Does NOT tune on the final evaluation / test set.
Can be enabled or disabled configurably.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to logits:
        calibrated_logits = logits / temperature
    A single learned scalar T > 0 softens or sharpens the softmax distribution
    to minimize Negative Log-Likelihood (cross-entropy) on validation data without altering predictions.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([float(initial_temperature)]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature.to(logits.device), min=0.01, max=10.0)
        return logits / temp

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature parameter using L-BFGS optimizer on validation logits/labels.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = F.cross_entropy(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())

    def get_temperature(self) -> float:
        return float(self.temperature.item())

    def state_dict_cal(self) -> Dict[str, float]:
        return {"temperature": self.get_temperature()}

    def load_state_dict_cal(self, state: Dict[str, float]) -> None:
        if "temperature" in state:
            with torch.no_grad():
                self.temperature.copy_(torch.tensor([float(state["temperature"])]))

