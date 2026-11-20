
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from models.model_a import ModelA
from models.model_b import ModelB
from utils.training_logger import TrainingLogger


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_name):
    if model_name == "small":
        return ModelA(pretrained=True)

    if model_name == "large":
        return ModelB(pretrained=True)

    raise ValueError("Model must be small or large")


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

def prepare_data(batch_size, train_limit):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    full_dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=False,
        transform=transform
    )

    # Optional smaller subset for initial experiments
    if train_limit > 0:
        indices = list(range(min(train_limit, len(full_dataset))))
        full_dataset = Subset(full_dataset, indices)

    # 90% training, 10% validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


# --------------------------------------------------
# 3. Training
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# --------------------------------------------------
# 4. Validation
# --------------------------------------------------

def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# --------------------------------------------------
# 5. Main training function
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["small", "large"],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument(
        "--train-limit",
        type=int,
        default=5000,
        help="Number of training images; use 0 for all images"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_name = (
        "mobilenet_v3_small"
        if args.model == "small"
        else "mobilenet_v3_large"
    )

    print("\nDevice:", device)
    print("Model:", model_name)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Create experiment-specific logger
    experiment_name = f"baseline/{model_name}"

    logger = TrainingLogger(
        experiment_name=experiment_name,
        config={
            "model": model_name,
            "dataset": "CIFAR-10",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_limit": args.train_limit,
            "image_size": 224,
            "seed": 42
        }
    )

    try:

        # Load dataset
        train_loader, val_loader = prepare_data(
            args.batch_size,
            args.train_limit
        )

        logger.log(
            f"Training images: {len(train_loader.dataset)}"
        )

        logger.log(
            f"Validation images: {len(val_loader.dataset)}"
        )

        # Initialize model
        model = get_model(args.model).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )

        start_epoch = 1
        best_val_accuracy = 0.0

        # Resume previous training if requested
        if args.resume:

            completed_epoch, best_val_accuracy = (
                logger.load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
            )

            start_epoch = completed_epoch + 1

        if start_epoch > args.epochs:
            logger.log("Requested training is already completed.")
            return

        logger.log("Training started.")

        # Training loop
        for epoch in range(start_epoch, args.epochs + 1):

            start_time = time.perf_counter()

            logger.log(
                f"Starting epoch {epoch}/{args.epochs}"
            )

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            val_loss, val_accuracy = validate(
                model,
                val_loader,
                criterion,
                device
            )

            scheduler.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            epoch_time = time.perf_counter() - start_time

            is_best = val_accuracy > best_val_accuracy

            if is_best:
                best_val_accuracy = val_accuracy

            # Save metrics after each epoch
            logger.log_epoch({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_time
            })

            # Save checkpoint after each epoch
            logger.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_accuracy=best_val_accuracy,
                extra={"is_best": is_best}
            )

            print(
                f"\nEpoch {epoch}/{args.epochs}"
                f"\nTrain Loss: {train_loss:.4f}"
                f"\nTrain Accuracy: {train_accuracy:.2f}%"
                f"\nValidation Loss: {val_loss:.4f}"
                f"\nValidation Accuracy: {val_accuracy:.2f}%"
                f"\nEpoch Time: {epoch_time:.2f} seconds"
            )

        logger.log("Training completed successfully.")

    except KeyboardInterrupt:

        logger.log(
            "Training manually interrupted. "
            "Resume from the last completed epoch using --resume."
        )

        raise

    except Exception as error:

        logger.log_error(error)
        raise


if __name__ == "__main__":
    main()
