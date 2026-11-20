
import sys
import json
import time
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_a import ModelA
from models.model_b import ModelB
from probes.probe_heads import ProbeHead, SequentialModelB
from controller.comparative_estimator import ComparativeEstimator


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_A_PATH = (
    PROJECT_ROOT
    / "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth"
)

MODEL_B_PATH = (
    PROJECT_ROOT
    / "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"
)

PROBE_PATHS = {
    4: PROJECT_ROOT / "outputs/probes/block_4/best_probe.pth",
    9: PROJECT_ROOT / "outputs/probes/block_9/best_probe.pth",
    13: PROJECT_ROOT / "outputs/probes/block_13/best_probe.pth"
}

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Experiment configuration
BATCH_SIZE = 1
NUM_WORKERS = 0

EVAL_START = 1000
EVAL_COUNT = 10

A_THRESHOLD = 0.80
PROBE_THRESHOLD = 0.85

PROBE_BLOCKS = [4, 9, 13]

WARMUP_ITERATIONS = 20


# ============================================================
# 2. DEVICE SYNCHRONIZATION AND TIMING
# ============================================================

def synchronize():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def timed_inference(function):

    synchronize()

    start_time = time.perf_counter()

    output = function()

    synchronize()

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return output, elapsed_ms


# ============================================================
# 3. CHECKPOINT LOADING
# ============================================================

def load_models():

    print("\nLoading models and probe heads...")

    # --------------------------------------------------------
    # Model A
    # --------------------------------------------------------

    model_a = ModelA(
        num_classes=10,
        pretrained=False
    )

    checkpoint_a = torch.load(
        MODEL_A_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model_a.load_state_dict(
        checkpoint_a["model_state_dict"]
    )

    model_a = model_a.to(DEVICE).eval()

    print("Model A loaded successfully.")

    # --------------------------------------------------------
    # Model B
    # --------------------------------------------------------

    model_b = ModelB(
        num_classes=10,
        pretrained=False
    )

    checkpoint_b = torch.load(
        MODEL_B_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model_b.load_state_dict(
        checkpoint_b["model_state_dict"]
    )

    model_b = model_b.to(DEVICE).eval()

    print("Model B loaded successfully.")

    # --------------------------------------------------------
    # Sequential Model B
    # --------------------------------------------------------

    sequential_b = SequentialModelB(
        model_b,
        DEVICE
    )

    probe_channels = {
        4: 40,
        9: 80,
        13: 160
    }

    # --------------------------------------------------------
    # Load all probe heads
    # --------------------------------------------------------

    for block, channels in probe_channels.items():

        head = ProbeHead(
            in_channels=channels,
            num_classes=10
        ).to(DEVICE)

        checkpoint = torch.load(
            PROBE_PATHS[block],
            map_location=DEVICE,
            weights_only=False
        )

        # Your probe checkpoints use head_state_dict.
        head.load_state_dict(
            checkpoint["head_state_dict"]
        )

        head.eval()

        sequential_b.register_probe(
            block,
            head
        )

        print(
            f"Probe {block} loaded successfully | "
            f"Best validation accuracy: "
            f"{checkpoint['best_val_accuracy']:.2f}%"
        )

    print("\nAll models and probes loaded successfully.")

    return model_a, model_b, sequential_b


# ============================================================
# 4. LOAD CIFAR-10 EVALUATION DATA
# ============================================================

def load_evaluation_data():

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data/cifar10"),
        train=False,
        download=False,
        transform=transform
    )

    end_index = min(
        EVAL_START + EVAL_COUNT,
        len(dataset)
    )

    if EVAL_START >= end_index:
        raise ValueError(
            "Invalid evaluation range. "
            "Check EVAL_START and EVAL_COUNT."
        )

    indices = range(
        EVAL_START,
        end_index
    )

    subset = Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\nDataset loaded successfully.")
    print("Evaluation images:", len(subset))

    return loader


# ============================================================
# 5. MODEL PREDICTION
# ============================================================

@torch.inference_mode()
def predict_model(model, images):

    logits = model(images)

    probabilities = F.softmax(
        logits,
        dim=1
    )

    confidence, prediction = probabilities.max(
        dim=1
    )

    return (
        prediction.item(),
        confidence.item()
    )


# ============================================================
# 6. STRATEGY 1: MODEL A ONLY
# ============================================================

@torch.inference_mode()
def run_model_a(model_a, images):

    return timed_inference(
        lambda: predict_model(
            model_a,
            images
        )
    )


# ============================================================
# 7. STRATEGY 2: MODEL B ONLY
# ============================================================

@torch.inference_mode()
def run_model_b(model_b, images):

    return timed_inference(
        lambda: predict_model(
            model_b,
            images
        )
    )


# ============================================================
# 8. STRATEGY 3: CONVENTIONAL SWITCHING
# ============================================================

@torch.inference_mode()
def run_conventional_switching(
    model_a,
    model_b,
    images
):

    # First execute Model A.

    (prediction_a, confidence_a), latency_a = (
        run_model_a(
            model_a,
            images
        )
    )

    # Retain Model A prediction if confidence is sufficient.

    if confidence_a >= A_THRESHOLD:

        return (
            prediction_a,
            confidence_a,
            latency_a,
            False
        )

    # Otherwise execute the complete Model B.

    (prediction_b, confidence_b), latency_b = (
        run_model_b(
            model_b,
            images
        )
    )

    total_latency = latency_a + latency_b

    return (
        prediction_b,
        confidence_b,
        total_latency,
        True
    )


# ============================================================
# 9. STRATEGY 4: ADAPT-EDGE
# ============================================================

@torch.inference_mode()
def run_adapt_edge(
    model_a,
    sequential_b,
    images
):

    # --------------------------------------------------------
    # Step 1: Execute Model A
    # --------------------------------------------------------

    (prediction_a, confidence_a), latency_a = (
        run_model_a(
            model_a,
            images
        )
    )

    # Retain Model A if confidence is sufficient.

    if confidence_a >= A_THRESHOLD:

        return (
            prediction_a,
            confidence_a,
            latency_a,
            False,
            []
        )

    # --------------------------------------------------------
    # Step 2: Start sequential Model B execution
    # --------------------------------------------------------

    synchronize()

    start_time = time.perf_counter()

    features = sequential_b.start(images)

    executed_probes = []

    # --------------------------------------------------------
    # Step 3: Execute intermediate probes
    # --------------------------------------------------------

    for block in PROBE_BLOCKS:

        # Continue from the previous probe.

        if block != PROBE_BLOCKS[0]:

            previous_block = executed_probes[-1]

            features = sequential_b.continue_to(
                features,
                previous_block + 1,
                block
            )

        # Obtain intermediate prediction.

        probe_logits = sequential_b.predict_probe(
            features,
            block
        )

        probabilities = F.softmax(
            probe_logits,
            dim=1
        )

        confidence, prediction = probabilities.max(
            dim=1
        )

        executed_probes.append(block)

        # ----------------------------------------------------
        # Step 4: Check probe confidence
        # ----------------------------------------------------

        if confidence.item() >= PROBE_THRESHOLD:

            synchronize()

            probe_latency = (
                time.perf_counter() - start_time
            ) * 1000

            total_latency = (
                latency_a + probe_latency
            )

            # Decision: retain the probe prediction.

            return (
                prediction.item(),
                confidence.item(),
                total_latency,
                False,
                executed_probes
            )

    # --------------------------------------------------------
    # Step 5: Complete Model B if probes remain uncertain
    # --------------------------------------------------------

    final_logits = sequential_b.finish(
        features
    )

    final_probabilities = F.softmax(final_logits, dim=1)
    final_confidence, final_prediction = final_probabilities.max(dim=1)

    synchronize()

    probe_latency = (
        time.perf_counter() - start_time
    ) * 1000

    total_latency = (
        latency_a + probe_latency
    )

    # Decision: switch to the complete Model B prediction.

    return (
        final_prediction.item(),
        final_confidence.item(),
        total_latency,
        True,
        executed_probes
    )


# ============================================================
# 10. WARM-UP
# ============================================================

@torch.inference_mode()
def warmup_models(
    model_a,
    model_b,
    sequential_b,
    images
):

    print("\nWarming up models...")

    for _ in range(WARMUP_ITERATIONS):

        model_a(images)
        model_b(images)

    # Warm up sequential execution and all probes.

    for _ in range(5):

        features = sequential_b.start(images)

        features = sequential_b.continue_to(
            features,
            5,
            9
        )

        features = sequential_b.continue_to(
            features,
            10,
            13
        )

        sequential_b.predict_probe(
            features,
            13
        )

        sequential_b.finish(features)

    synchronize()

    print("Warm-up completed.")


# ============================================================
# 11. MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("ADAPT-EDGE STRATEGY COMPARISON")
    print("=" * 65)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(DEVICE)
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    model_a, model_b, sequential_b = load_models()

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    loader = load_evaluation_data()

    # --------------------------------------------------------
    # Initialize comparative estimator
    # --------------------------------------------------------

    estimator = ComparativeEstimator()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_images, _ = next(iter(loader))

    warmup_images = warmup_images.to(DEVICE)

    warmup_models(
        model_a,
        model_b,
        sequential_b,
        warmup_images
    )

    # --------------------------------------------------------
    # Run four strategies
    # --------------------------------------------------------

    total = 0
    per_image_logs = []

    print("\nStarting strategy comparison...\n")

    for images, labels in loader:

        images = images.to(DEVICE)

        true_label = labels.item()

        # ====================================================
        # STRATEGY 1: MODEL A ONLY
        # ====================================================

        (pred_a, conf_a), latency_a = run_model_a(
            model_a,
            images
        )
        correct_a = (pred_a == true_label)

        estimator.add_result(
            strategy="Model A Only",
            correct=correct_a,
            latency_ms=latency_a,
            executed_model="A"
        )

        # ====================================================
        # STRATEGY 2: MODEL B ONLY
        # ====================================================

        (pred_b, conf_b), latency_b = run_model_b(
            model_b,
            images
        )
        correct_b = (pred_b == true_label)

        estimator.add_result(
            strategy="Model B Only",
            correct=correct_b,
            latency_ms=latency_b,
            executed_model="B"
        )

        # ====================================================
        # STRATEGY 3: CONVENTIONAL SWITCHING
        # ====================================================

        pred_conv, conf_conv, latency_conv, used_b_conv = (
            run_conventional_switching(
                model_a,
                model_b,
                images
            )
        )
        correct_conv = (pred_conv == true_label)

        # Strict assertion: Conventional switching prediction MUST equal
        # the prediction from the model that was selected (A or B)
        selected_model = "B" if used_b_conv else "A"
        expected_pred = pred_b if used_b_conv else pred_a
        assert pred_conv == expected_pred, (
            f"Conventional switching prediction {pred_conv} does not match selected model "
            f"{selected_model} prediction {expected_pred} for image {total}!"
        )

        estimator.add_result(
            strategy="Conventional Switching",
            correct=correct_conv,
            latency_ms=latency_conv,
            executed_model=selected_model
        )

        # ====================================================
        # STRATEGY 4: ADAPT-EDGE
        # ====================================================

        (
            pred_adapt,
            conf_adapt,
            latency_adapt,
            used_full_b_adapt,
            probes_adapt
        ) = run_adapt_edge(
            model_a,
            sequential_b,
            images
        )
        correct_adapt = (pred_adapt == true_label)

        # Count Model B usage even when only partial
        # Model B execution through probes was required.
        used_model_b_adapt = (
            used_full_b_adapt or len(probes_adapt) > 0
        )

        estimator.add_result(
            strategy="ADAPT-EDGE",
            correct=correct_adapt,
            latency_ms=latency_adapt,
            executed_model="B" if used_model_b_adapt else "A",
            executed_probes=probes_adapt
        )

        # Record per-image detailed record
        img_record = {
            "image_index": total,
            "dataset_index": EVAL_START + total,
            "true_label": true_label,
            "model_a": {
                "prediction": pred_a,
                "confidence": conf_a,
                "correct": correct_a,
                "latency_ms": latency_a
            },
            "model_b": {
                "prediction": pred_b,
                "confidence": conf_b,
                "correct": correct_b,
                "latency_ms": latency_b
            },
            "conventional_switching": {
                "prediction": pred_conv,
                "confidence": conf_conv,
                "selected_model": selected_model,
                "correct": correct_conv,
                "latency_ms": latency_conv
            },
            "adapt_edge": {
                "prediction": pred_adapt,
                "confidence": conf_adapt,
                "executed_probes": probes_adapt,
                "used_full_b": used_full_b_adapt,
                "correct": correct_adapt,
                "latency_ms": latency_adapt
            }
        }
        per_image_logs.append(img_record)

        total += 1

        if total % 100 == 0:

            print(
                f"Processed {total} / "
                f"{len(loader.dataset)} images"
            )

    # ========================================================
    # 12. CALCULATE RESULTS
    # ========================================================

    print("\nExperiment completed.")

    print("\nGenerating comparison metrics...")

    comparison = estimator.print_comparison()

    # --------------------------------------------------------
    # Create Timestamped Run Directory to preserve history
    # --------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Save per-image logs (JSONL and JSON)
    # --------------------------------------------------------
    per_image_jsonl = run_dir / "per_image_diagnostics.jsonl"
    with open(per_image_jsonl, "w", encoding="utf-8") as f:
        for record in per_image_logs:
            f.write(json.dumps(record) + "\n")

    per_image_json = run_dir / "per_image_diagnostics.json"
    with open(per_image_json, "w", encoding="utf-8") as f:
        json.dump(per_image_logs, f, indent=4)

    # Per-image CSV for easy tabular inspection
    per_image_csv = run_dir / "per_image_diagnostics.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "image_index", "dataset_index", "true_label",
            "model_a_pred", "model_a_conf", "model_a_correct", "model_a_latency_ms",
            "model_b_pred", "model_b_conf", "model_b_correct", "model_b_latency_ms",
            "conv_pred", "conv_conf", "conv_selected", "conv_correct", "conv_latency_ms",
            "adapt_pred", "adapt_conf", "adapt_probes", "adapt_full_b", "adapt_correct", "adapt_latency_ms"
        ])
        for r in per_image_logs:
            csv_writer.writerow([
                r["image_index"], r["dataset_index"], r["true_label"],
                r["model_a"]["prediction"], f"{r['model_a']['confidence']:.4f}", r["model_a"]["correct"], f"{r['model_a']['latency_ms']:.2f}",
                r["model_b"]["prediction"], f"{r['model_b']['confidence']:.4f}", r["model_b"]["correct"], f"{r['model_b']['latency_ms']:.2f}",
                r["conventional_switching"]["prediction"], f"{r['conventional_switching']['confidence']:.4f}",
                r["conventional_switching"]["selected_model"], r["conventional_switching"]["correct"], f"{r['conventional_switching']['latency_ms']:.2f}",
                r["adapt_edge"]["prediction"], f"{r['adapt_edge']['confidence']:.4f}",
                str(r["adapt_edge"]["executed_probes"]), r["adapt_edge"]["used_full_b"], r["adapt_edge"]["correct"], f"{r['adapt_edge']['latency_ms']:.2f}"
            ])

    # --------------------------------------------------------
    # Save Strategy Comparison JSON & CSV in Run Directory
    # --------------------------------------------------------

    json_path = run_dir / "strategy_comparison.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4
        )

    csv_path = run_dir / "strategy_comparison.csv"

    csv_columns = [
        "strategy",
        "total_images",
        "correct_predictions",
        "accuracy",
        "mean_latency_ms",
        "p95_latency_ms",
        "model_b_usage_percent",
        "probe_counts"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_columns
        )

        writer.writeheader()

        for strategy, metrics in comparison.items():

            row = metrics.copy()

            # Store probe counts as readable JSON text.

            row["probe_counts"] = json.dumps(
                row["probe_counts"]
            )

            writer.writerow(row)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)

    print("Total images evaluated:", total)
    print("Run output directory:", run_dir)
    print("JSON results:", json_path)
    print("CSV results:", csv_path)
    print("Per-image diagnostics:", per_image_jsonl)

    print("\nAll four strategies evaluated successfully.")


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
