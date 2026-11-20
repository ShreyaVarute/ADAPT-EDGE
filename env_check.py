
import sys
import importlib

print("=" * 50)
print("ADAPT-EDGE ENVIRONMENT CHECK")
print("=" * 50)

print("\nPython executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

packages = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "PIL",
    "tqdm",
    "datasets",
]

print("\nChecking libraries:")

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "Version unavailable")
        print(f"[OK] {package}: {version}")
    except Exception as error:
        print(f"[ERROR] {package}: {error}")

print("\nChecking CUDA:")

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )
    else:
        print("GPU not detected by PyTorch.")

except Exception as error:
    print("CUDA check failed:", error)

print("\nEnvironment check completed.")
