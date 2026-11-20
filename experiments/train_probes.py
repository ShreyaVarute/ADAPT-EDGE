
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from models.model_b import ModelB
from probes.probe_heads import ProbeHead


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = (
    "outputs/baseline/mobilenet_v3_large/"
    "checkpoints/best_model.pth"
)

PROBE_CONFIG = {
    4: 40,
    9: 80,
    13: 160
}

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001


def load_backbone():
    print("Loading trained Model B...")

    model = ModelB(pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()

    # Freeze the entire backbone.
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_data(train_limit=0):

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

    if train_limit > 0:
        dataset = torch.utils.data.Subset(
            dataset,
            range(min(train_limit, len(dataset)))
        )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader


def extract_features(backbone, images, probe_block):

    x = images

    with torch.no_grad():
        for index in range(probe_block + 1):
            x = backbone.model.features[index](x)

    return x.detach()


def evaluate(head, backbone, loader, probe_block):

    head.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            features = extract_features(
                backbone,
                images,
                probe_block
            )

            logits = head(features)

            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


def train_probe(
    backbone,
    train_loader,
    val_loader,
    probe_block,
    in_channels,
    epochs
):

    print("\n" + "=" * 55)
    print(f"Training Probe after Block {probe_block}")
    print("=" * 55)

    head = ProbeHead(
        in_channels=in_channels,
        num_classes=10
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    output_dir = Path(
        f"outputs/probes/block_{probe_block}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(epochs):

        head.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Extract frozen backbone features.
            features = extract_features(
                backbone,
                images,
                probe_block
            )

            optimizer.zero_grad()

            logits = head(features)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / total
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate(
            head,
            backbone,
            val_loader,
            probe_block
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

        history.append(epoch_result)

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
            f"\nTrain Loss: {train_loss:.4f}"
            f"\nTrain Accuracy: {train_accuracy:.2f}%"
            f"\nVal Loss: {val_loss:.4f}"
            f"\nVal Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                {
                    "probe_block": probe_block,
                    "in_channels": in_channels,
                    "head_state_dict": head.state_dict(),
                    "best_val_accuracy": best_accuracy
                },
                output_dir / "best_probe.pth"
            )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(
        f"\nBest validation accuracy for Block "
        f"{probe_block}: {best_accuracy:.2f}%"
    )

    return {
        "probe_block": probe_block,
        "best_val_accuracy": best_accuracy,
        "checkpoint": str(output_dir / "best_probe.pth")
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=5000
    )

    args = parser.parse_args()

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    backbone = load_backbone()

    train_loader, val_loader = load_data(
        args.train_limit
    )

    print("\nTraining samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    all_results = []

    start_time = time.time()

    for block, channels in PROBE_CONFIG.items():

        result = train_probe(
            backbone,
            train_loader,
            val_loader,
            block,
            channels,
            args.epochs
        )

        all_results.append(result)

    output_path = Path("results/probe_training_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 55)
    print("ALL PROBES TRAINED")
    print("=" * 55)

    for result in all_results:
        print(
            f"Block {result['probe_block']}: "
            f"{result['best_val_accuracy']:.2f}%"
        )

    print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
    print("Results saved:", output_path)


if __name__ == "__main__":
    main()
