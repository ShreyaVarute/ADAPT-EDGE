
import json
import time
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.model_b import ModelB
from probes.probe_heads import ProbeHead


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROBES = {
    4: 40,
    9: 80,
    13: 160
}

WARMUP = 20
MEASUREMENTS = 100


def load_backbone():

    model = ModelB(pretrained=False)

    checkpoint = torch.load(
        "outputs/baseline/mobilenet_v3_large/"
        "checkpoints/best_model.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(DEVICE).eval()


def load_probe(block, channels):

    path = Path(
        f"outputs/probes/block_{block}/best_probe.pth"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    head = ProbeHead(channels, 10)
    head.load_state_dict(checkpoint["head_state_dict"])

    return head.to(DEVICE).eval()


def load_sample():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=False,
        download=False,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    image, label = next(iter(loader))

    return image.to(DEVICE), label.to(DEVICE)


def synchronize():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def profile_probe(backbone, head, image, block):

    latencies = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():

        for _ in range(WARMUP):

            x = image

            for index in range(block + 1):
                x = backbone.model.features[index](x)

            head(x)

        synchronize()

        for _ in range(MEASUREMENTS):

            synchronize()
            start = time.perf_counter()

            x = image

            for index in range(block + 1):
                x = backbone.model.features[index](x)

            logits = head(x)

            synchronize()

            elapsed = (time.perf_counter() - start) * 1000

            latencies.append(elapsed)

    latencies.sort()

    accuracy = None

    # Prediction for the measured sample
    with torch.inference_mode():

        x = image

        for index in range(block + 1):
            x = backbone.model.features[index](x)

        logits = head(x)

        accuracy = logits.argmax(dim=1).item()

    peak_memory = None

    if DEVICE.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    return {
        "probe_block": block,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))],
        "peak_gpu_memory_mb": peak_memory,
        "sample_prediction": accuracy
    }


def profile_full_model(backbone, image):

    latencies = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():

        for _ in range(WARMUP):
            backbone(image)

        synchronize()

        for _ in range(MEASUREMENTS):

            synchronize()
            start = time.perf_counter()

            backbone(image)

            synchronize()

            latencies.append(
                (time.perf_counter() - start) * 1000
            )

    latencies.sort()

    memory = None

    if DEVICE.type == "cuda":
        memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "model": "Full MobileNetV3-Large",
        "mean_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))],
        "peak_gpu_memory_mb": memory
    }


def main():

    print("Device:", DEVICE)

    backbone = load_backbone()

    image, label = load_sample()

    results = []

    for block, channels in PROBES.items():

        print(f"\nProfiling probe after Block {block}")

        head = load_probe(block, channels)

        result = profile_probe(
            backbone,
            head,
            image,
            block
        )

        result["true_label"] = label.item()

        results.append(result)

        print(result)

    print("\nProfiling complete Model B")

    full_result = profile_full_model(backbone, image)

    results.append(full_result)

    print(full_result)

    output = Path("results/probe_profiling_results.json")

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to:", output)


if __name__ == "__main__":
    main()
