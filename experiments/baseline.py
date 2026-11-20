
import time
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_a = models.mobilenet_v3_small(weights="DEFAULT")
model_b = models.mobilenet_v3_large(weights="DEFAULT")

model_a.classifier[3] = nn.Linear(model_a.classifier[3].in_features, 10)
model_b.classifier[3] = nn.Linear(model_b.classifier[3].in_features, 10)

models_to_test = [
    ("Model A: MobileNetV3-Small", model_a.to(device).eval()),
    ("Model B: MobileNetV3-Large", model_b.to(device).eval()),
]

x = torch.randn(1, 3, 224, 224, device=device)

for name, model in models_to_test:
    with torch.inference_mode():
        # Warm-up: initializes CUDA kernels and other runtime components
        for _ in range(10):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(50):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        avg_ms = (time.perf_counter() - start) * 1000 / 50

    print(f"\n{name}")
    print("Output shape:", tuple(model(x).shape))
    print(f"Average latency: {avg_ms:.2f} ms/image")
