
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from models.model_a import ModelA
from models.model_b import ModelB
from probes.probe_heads import ProbeHead


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
TRAIN_LIMIT = 5000

PROBES = {
    4: 40,
    9: 80,
    13: 160
}


def load_model(model_class, checkpoint_path):

    model = model_class(pretrained=False)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(DEVICE).eval()


def load_validation_data():

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
        train=True,
        download=False,
        transform=transform
    )

    dataset = torch.utils.data.Subset(
        dataset,
        range(TRAIN_LIMIT)
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


def load_probe(block, channels):

    checkpoint_path = (
        f"outputs/probes/block_{block}/best_probe.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    head = ProbeHead(channels, num_classes=10)

    head.load_state_dict(checkpoint["head_state_dict"])

    return head.to(DEVICE).eval()


def run_probe(backbone, head, images, block):

    x = images

    with torch.inference_mode():

        for index in range(block + 1):
            x = backbone.model.features[index](x)

        logits = head(x)

    return logits


def evaluate_model(name, loader, prediction_function):

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = prediction_function(images)

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    result = {
        "model": name,
        "accuracy": round(accuracy * 100, 2),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "confusion_matrix": matrix.tolist()
    }

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)
    print(f"Accuracy: {result['accuracy']}%")
    print(f"Macro Precision: {result['macro_precision']}")
    print(f"Macro Recall: {result['macro_recall']}")
    print(f"Macro F1: {result['macro_f1']}")

    return result


def main():

    print("Device:", DEVICE)

    loader = load_validation_data()

    print("Validation images:", len(loader.dataset))

    model_a = load_model(
        ModelA,
        "outputs/baseline/mobilenet_v3_small/"
        "checkpoints/best_model.pth"
    )

    model_b = load_model(
        ModelB,
        "outputs/baseline/mobilenet_v3_large/"
        "checkpoints/best_model.pth"
    )

    results = []

    # Model A
    results.append(
        evaluate_model(
            "MobileNetV3-Small",
            loader,
            lambda x: model_a(x)
        )
    )

    # Probe heads
    for block, channels in PROBES.items():

        head = load_probe(block, channels)

        results.append(
            evaluate_model(
                f"Probe after Block {block}",
                loader,
                lambda x, h=head, b=block:
                    run_probe(model_b, h, x, b)
            )
        )

    # Complete Model B
    results.append(
        evaluate_model(
            "MobileNetV3-Large",
            loader,
            lambda x: model_b(x)
        )
    )

    output_path = Path(
        "results/probe_accuracy_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)

    for result in results:
        print(
            f"{result['model']:<30} "
            f"{result['accuracy']:>7.2f}% "
            f"F1: {result['macro_f1']:.4f}"
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
