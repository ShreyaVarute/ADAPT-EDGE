
# ADAPT-EDGE

## Decision-Selective Computation for Resource-Bounded Edge AI Model Switching

ADAPT-EDGE is a research project that investigates how edge AI systems can reduce unnecessary computation when a lightweight model is uncertain about its prediction.

Instead of always executing a larger model completely, the system investigates whether selected intermediate computations from the larger model can provide useful evidence for deciding what to do next.

## 1. Objectives

- Compare a lightweight model with a larger model.
- Investigate partial execution of the larger model.
- Use intermediate probe outputs to gather additional evidence.
- Decide whether to retain the lightweight prediction, execute another probe, or complete the larger model.
- Evaluate accuracy, latency, computational cost, and resource usage.

## 2. Technology Stack

- Python 3.10
- PyTorch
- Torchvision
- CUDA
- MobileNetV3-Small
- MobileNetV3-Large
- CIFAR-10
- NumPy and Pandas
- Scikit-learn
- Matplotlib

## 3. Hardware

Development hardware:
- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB GPU memory

Experiments should begin with small batches to limit GPU memory usage and heat.

## 4. Project Structure

```text
ADAPT-EDGE/
├── README.md
├── requirements.txt
├── .gitignore
├── env_check.py
├── data/
│   └── cifar10/
├── models/
│   ├── model_a.py
│   ├── model_b.py
│   └── checkpoints/
├── probes/
│   ├── probe_heads.py
│   └── probe_data.py
├── controller/
│   ├── comparative_estimator.py
│   ├── cost_estimator.py
│   └── decision_controller.py
├── cache/
│   └── continuation_cache.py
├── experiments/
│   ├── baseline.py
│   ├── prepare_dataset.py
│   ├── probe_experiment.py
│   └── evaluation.py
└── utils/
    ├── metrics.py
    └── profiling.py
```

## 5. Environment Setup

Windows CMD (recommended for this repo):

```cmd
cd /d "D:\Semester\7th sem\IPR\ADAPT-EDGE"
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python env_check.py
```

PowerShell equivalent:

```powershell
cd "D:\Semester\7th sem\IPR\ADAPT-EDGE"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python env_check.py
```

PyTorch and Torchvision are installed separately with CUDA support when available.

## 6. Inference Usage

From the project root, run one real CIFAR-10 image through the integrated resource-aware ADAPT-EDGE pipeline:

```cmd
cd /d "D:\Semester\7th sem\IPR\ADAPT-EDGE"
.\.venv\Scripts\python.exe -c "from pipeline.adapt_edge_pipeline import AdaptEdgePipeline; from torchvision import datasets, transforms; import torch; transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]); ds = datasets.CIFAR10(root='data/cifar10', train=False, download=False, transform=transform); img, label = ds[0]; pipeline = AdaptEdgePipeline(device=(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')), max_latency_budget_ms=25.0, enable_cache=True, enforce_resource_budget=True); result = pipeline.predict(img.unsqueeze(0), request_id='cmd_smoke_test'); print({'prediction': result['prediction'], 'confidence': round(float(result['confidence']), 4), 'action': result['action'], 'status': result.get('status'), 'latency_ms': round(float(result['latency_ms']), 2), 'resource_violations': result.get('resource_violations', [])})"
```

This uses the actual Model A + ResourceMonitor + DecisionController + continuation cache path in the real inference pipeline. It does not retrain the models or overwrite the checkpoint files.

## 7. Dataset

The initial benchmark uses CIFAR-10:

- 50,000 training images
- 10,000 test images
- 10 image classes

Dataset location:

```text
data/cifar10/
```

The dataset is downloaded only when required files are missing. Once downloaded successfully, it can be reused across experiments.

For the initial experiment, use a small subset if training takes too long. Later experiments can use more or all of the training data.

## 7. Model Definitions

### Model A
MobileNetV3-Small is the lightweight model.

### Model B
MobileNetV3-Large is the larger model.

Both models must be adapted and trained or fine-tuned for the selected dataset before meaningful accuracy comparisons can be made.

## 8. Proposed Execution Workflow

1. Verify the Python and CUDA environment.
2. Load and prepare CIFAR-10.
3. Implement and train Model A.
4. Implement and train Model B.
5. Measure standalone model accuracy and latency.
6. Divide Model B into explicitly executable segments.
7. Add intermediate probe heads.
8. Estimate the benefit and cost of further computation.
9. Implement the decision controller.
10. Evaluate the complete ADAPT-EDGE method against baselines.

## 9. Evaluation Metrics

Record the following measurements:

- Classification accuracy
- End-to-end latency
- Additional computation required by probes
- Total computation cost
- GPU memory usage
- Model switching frequency
- Accuracy and cost trade-offs

Use repeated measurements and a consistent hardware setup when comparing models.

## 10. Research Note

The proposed method is a research hypothesis that must be evaluated experimentally. Improved accuracy, reduced latency, and reduced computation must not be assumed before measurement.

Random input tests can verify execution but cannot establish meaningful classification accuracy.

## 11. Current Status

- Python virtual environment: configured
- PyTorch and CUDA: verified
- GPU detection: verified
- Model A and Model B: basic inference tested
- CIFAR-10: downloading/preparation in progress
- Model training: pending
- Probe mechanism: pending
- Decision controller: pending
- Final evaluation: pending
