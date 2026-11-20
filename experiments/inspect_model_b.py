import torch
from models.model_b import ModelB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = ModelB(pretrained=False)

    checkpoint = torch.load(
        "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()

    print("\nMODEL B STRUCTURE")
    print("=" * 60)

    for name, module in model.model.features.named_children():
        print(f"\nBlock {name}: {module}")

    print("\nCLASSIFIER")
    print(model.model.classifier)

    print("\nTotal feature blocks:",
          len(model.model.features))

if __name__ == "__main__":
    main()