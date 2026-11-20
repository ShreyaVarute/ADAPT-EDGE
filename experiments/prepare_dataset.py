
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Keep the first experiment small and GPU-friendly
BATCH_SIZE = 16
DATA_DIR = "./data/cifar10"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

print("Downloading/loading CIFAR-10...")

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

images, labels = next(iter(train_loader))

print("\nDataset loaded successfully")
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Image batch shape:", tuple(images.shape))
print("Label batch shape:", tuple(labels.shape))
print("Number of classes:", len(train_dataset.classes))
print("Classes:", train_dataset.classes)
