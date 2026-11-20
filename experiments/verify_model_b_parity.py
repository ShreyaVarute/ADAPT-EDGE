
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_b import ModelB
from probes.probe_heads import SequentialModelB


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the existing trained checkpoint without downloading weights.
    model_b = ModelB(num_classes=10, pretrained=False)

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_b.load_state_dict(checkpoint["model_state_dict"])
    model_b = model_b.to(device).eval()

    sequential = SequentialModelB(model_b, device)

    # Deterministic test input.
    torch.manual_seed(42)
    images = torch.randn(1, 3, 224, 224).to(device)

    with torch.inference_mode():
        # Reference: original complete Model B.
        reference_logits = model_b(images)

        print("\nReference Model B prediction:",
              reference_logits.argmax(dim=1).item())

        # Test completion from each probe location.
        for block in [4, 9, 13]:

            features = sequential.start(images)

            if block >= 9:
                features = sequential.continue_to(
                    features, 5, 9
                )

            if block >= 13:
                features = sequential.continue_to(
                    features, 10, 13
                )

            sequential_logits = sequential.finish(
                features,
                current_block=block
            )

            max_difference = (
                reference_logits - sequential_logits
            ).abs().max().item()

            is_close = torch.allclose(
                reference_logits,
                sequential_logits,
                rtol=1e-4,
                atol=1e-5
            )

            print(f"\nStarting from block {block}")
            print("Sequential prediction:",
                  sequential_logits.argmax(dim=1).item())
            print("Maximum logit difference:",
                  max_difference)
            print("Logits match:", is_close)

            if not is_close:
                raise AssertionError(
                    f"Parity failed from block {block}"
                )

    print("\nSUCCESS: All sequential fallback paths match full Model B.")


if __name__ == "__main__":
    main()
