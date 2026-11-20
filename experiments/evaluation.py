
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report

from models.model_a import ModelA
from models.model_b import ModelB


# -----------------------------------------
# Configuration
# -----------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# Load test dataset
# -----------------------------------------

def load_test_data():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return loader


# -----------------------------------------
# Evaluate model
# -----------------------------------------

def evaluate_model(model, loader, checkpoint_path, model_name):

    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    # Warm-up
    dummy = torch.randn(
        BATCH_SIZE, 3, 224, 224, device=DEVICE
    )

    with torch.inference_mode():
        for _ in range(5):
            model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total

    avg_latency = (elapsed * 1000) / total

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    class_accuracy = {}

    for i, class_name in enumerate(CLASS_NAMES):
        class_total = matrix[i].sum()
        class_correct = matrix[i, i]

        class_accuracy[class_name] = (
            100.0 * class_correct / class_total
            if class_total else 0.0
        )

    results = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "test_images": total,
        "test_accuracy": accuracy,
        "test_loss": avg_loss,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "total_inference_seconds": elapsed,
        "average_latency_ms_per_image": avg_latency,
        "class_accuracy": class_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }

    print("\n" + "=" * 45)
    print(model_name)
    print("=" * 45)

    print(f"Test images: {total}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Average latency: {avg_latency:.3f} ms/image")
    print(f"Total inference time: {elapsed:.2f} seconds")

    print("\nClass-wise accuracy:")

    for name, value in class_accuracy.items():
        print(f"{name}: {value:.2f}%")

    return results


# -----------------------------------------
# Main
# -----------------------------------------

def main():

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_loader = load_test_data()

    print("Test dataset:", len(test_loader.dataset))

    experiments = [
        {
            "name": "MobileNetV3-Small",
            "model": ModelA(pretrained=False),
            "checkpoint": (
                "outputs/baseline/mobilenet_v3_small/"
                "checkpoints/best_model.pth"
            )
        },
        {
            "name": "MobileNetV3-Large",
            "model": ModelB(pretrained=False),
            "checkpoint": (
                "outputs/baseline/mobilenet_v3_large/"
                "checkpoints/best_model.pth"
            )
        }
    ]

    all_results = []

    for experiment in experiments:

        results = evaluate_model(
            model=experiment["model"],
            loader=test_loader,
            checkpoint_path=experiment["checkpoint"],
            model_name=experiment["name"]
        )

        all_results.append(results)

    # Save detailed JSON results
    json_path = RESULT_DIR / "baseline_test_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # Save comparison CSV
    csv_path = RESULT_DIR / "baseline_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        fields = [
            "model",
            "test_accuracy",
            "test_loss",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "average_latency_ms_per_image",
            "total_inference_seconds"
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in all_results:
            writer.writerow({
                key: result[key] for key in fields
            })

    print("\nEvaluation completed.")
    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
