
import json
import time
from pathlib import Path

import torch
from torchvision import transforms

from models.model_a import ModelA
from models.model_b import ModelB


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WARMUP = 30
NUM_RUNS = 200

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model(model_class, checkpoint_path):

    model = model_class(pretrained=False)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE).eval()

    return model


def profile_model(model, model_name):

    # One image, matching our CIFAR-10 preprocessing
    x = torch.randn(1, 3, 224, 224, device=DEVICE)

    latencies = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():

        # Warm-up
        for _ in range(NUM_WARMUP):
            model(x)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # Repeated measurements
        for _ in range(NUM_RUNS):

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(x)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000

            latencies.append(elapsed)

    latencies.sort()

    results = {
        "model": model_name,
        "device": torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda" else "CPU",
        "batch_size": 1,
        "input_shape": [1, 3, 224, 224],
        "warmup_runs": NUM_WARMUP,
        "measurement_runs": NUM_RUNS,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))],
    }

    if DEVICE.type == "cuda":

        results["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    print("\n", "=" * 40)
    print(model_name)
    print("=" * 40)

    for key, value in results.items():
        print(f"{key}: {value}")

    return results


def main():

    print("Device:", DEVICE)

    model_a = load_model(
        ModelA,
        "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth"
    )

    model_b = load_model(
        ModelB,
        "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"
    )

    results = []

    results.append(
        profile_model(model_a, "MobileNetV3-Small")
    )

    results.append(
        profile_model(model_b, "MobileNetV3-Large")
    )

    output_path = OUTPUT_DIR / "single_image_profiling.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nProfiling completed.")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
