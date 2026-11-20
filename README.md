# ADAPT-EDGE
## Decision-Selective Computation for Resource-Bounded Edge AI Model Switching

ADAPT-EDGE is a research project that investigates adaptive computation for AI inference on resource-constrained edge devices.

Instead of always executing a larger model completely when a lightweight model produces an uncertain prediction, ADAPT-EDGE investigates whether selected intermediate computations can provide sufficient additional evidence.

The system uses three execution decisions:

- **RETAIN:** Keep the lightweight model's prediction.
- **PROBE:** Execute a selected intermediate probe in the larger model.
- **SWITCH:** Execute the complete larger model.

The decision controller considers prediction uncertainty, expected decision information, computational cost and available hardware resources.

## 1. Objectives

- Compare lightweight and larger deep learning models.
- Investigate partial execution of a larger model.
- Use intermediate probe outputs to gather additional evidence.
- Implement RETAIN, PROBE and SWITCH decisions.
- Monitor hardware resources during inference.
- Investigate continuation caching to reduce repeated computation.
- Evaluate accuracy, latency, computational cost and resource usage.
- Support training and evaluation of candidate models through the application interface.

## 2. Technology Stack

**Backend and AI**
- Python 3.10
- PyTorch
- Torchvision
- CUDA
- FastAPI
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- tqdm

**Frontend**
- React
- Vite
- JavaScript
- CSS

**Research models and dataset**
- MobileNetV3-Small
- MobileNetV3-Large
- CIFAR-10
- Additional compatible candidate architectures, subject to implementation and evaluation.

## 3. Development Hardware

Development hardware:
- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB dedicated GPU memory
- Windows 11
- Python 3.10

For GPU-constrained environments, begin with small batch sizes and monitor GPU memory and temperature.

## 4. Project Structure

```text
ADAPT-EDGE/
│
├── README.md
├── requirements.txt
├── .gitignore
├── env_check.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── pages/
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── server/
│   ├── api/
│   │   ├── datasets.py
│   │   ├── diagnostics.py
│   │   ├── experiments.py
│   │   ├── hardware.py
│   │   ├── inference.py
│   │   ├── reports.py
│   │   └── training.py
│   ├── app.py
│   └── core.py
│
├── models/
│   ├── model_a.py
│   └── model_b.py
│
├── pipeline/
│   └── adapt_edge_pipeline.py
│
├── controller/
│   ├── comparative_estimator.py
│   ├── cost_estimator.py
│   ├── decision_controller.py
│   └── resource_policy.py
│
├── probes/
│   ├── probe_heads.py
│   └── probe_data.py
│
├── cache/
│   └── continuation_cache.py
│
├── experiments/
│
├── tests/
│
├── utils/
│
├── start_dev.bat
└── start_platform.bat
```

This is a representative structure. Refer to the actual repository for the complete list of files.

## 5. Environment Setup

### Prerequisites

- Python 3.10
- Node.js and npm
- Git
- NVIDIA GPU and compatible CUDA-enabled PyTorch installation (optional)

### Clone the repository

```cmd
git clone https://github.com/ShreyaVarute/ADAPT-EDGE.git
cd ADAPT-EDGE
```

### Create a Python virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Install Python dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install PyTorch with CUDA support

Install the PyTorch and Torchvision versions compatible with your CUDA environment using the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Verify the environment:

```cmd
python env_check.py
```

### Install frontend dependencies

```cmd
cd frontend
npm install
cd ..
```

## 6. Running the Application

ADAPT-EDGE provides two Windows startup scripts.

### Development mode

Run:

```cmd
start_dev.bat
```

This starts:
- FastAPI backend: http://127.0.0.1:8000
- React frontend: http://localhost:3000
- API documentation: http://127.0.0.1:8000/docs

Development mode supports backend and frontend hot reloading.

### Demonstration mode

Run:

```cmd
start_platform.bat
```

This starts the FastAPI application and opens the dashboard at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## 7. Dataset Management

The initial research benchmark uses CIFAR-10:

- 50,000 training images
- 10,000 test images
- 10 image classes

CIFAR-10 contains:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

The application also supports custom image dataset upload through its dataset-management workflow.

Custom datasets should be organized into class-specific folders before being packaged into a ZIP archive.

Dataset files are not included in the GitHub source repository.

## 8. Model Training and Selection

The application supports a session-based candidate model training workflow.

Users can:
- Upload and inspect datasets.
- Configure training parameters.
- Select available candidate architectures.
- Train candidate models.
- View training progress and metrics.
- Evaluate candidate models.
- Compare their performance.
- Use compatible trained models for inference.

Model A represents the lightweight model.

Model B represents the more computationally capable model.

Model selection should consider accuracy, parameter count, FLOPs, inference latency and memory consumption.

## 9. Temporary Candidate Model Storage

Newly trained experimental candidate models use temporary backend memory storage.

- Trained candidate weights remain in RAM during the active backend session.
- Training metrics and evaluation results remain available during that session.
- Candidate models can be used for inference without permanently saving their checkpoints.
- Candidate models and session results are cleared when the backend restarts.
- Candidate models must be trained again after a backend restart.

Existing baseline and probe research checkpoints are maintained separately and are not modified by this temporary candidate-training workflow.

## 10. ADAPT-EDGE Execution Workflow

The proposed execution process is:

1. Receive an input image.
2. Preprocess the image.
3. Generate an initial prediction using Model A.
4. Evaluate prediction uncertainty.
5. Assess expected decision information and computational cost.
6. Check available hardware resources.
7. Select RETAIN, PROBE or SWITCH.
8. Execute the selected computation.
9. Produce the final prediction.
10. Record inference performance metrics.

Intermediate probes must be compatible with the selected Model B architecture and its feature representations.

## 11. Evaluation Metrics

The research evaluation considers:

- Classification accuracy
- Precision
- Recall
- F1-score
- Training loss
- Validation loss
- Parameter count
- FLOPs
- Model size
- Average inference latency
- P95 inference latency
- GPU memory usage
- Resource violations
- Model switching frequency
- Computational cost

Use consistent datasets, hardware configurations and measurement conditions when comparing approaches.

## 12. Research Comparisons

The experimental framework can compare:

- Model A independently.
- Model B independently.
- Conventional Model A to Model B switching.
- ADAPT-EDGE without caching.
- ADAPT-EDGE with caching.
- Compatible pretrained models.
- External black-box AI services, where supported.

Local and remote inference measurements should be reported separately.

## 13. Research Limitations

ADAPT-EDGE is an experimental research approach.

Performance improvements in accuracy, latency, computational cost and resource consumption must be established through actual experiments.

A larger model does not automatically guarantee higher accuracy. Similarly, caching does not automatically guarantee faster inference.

## 14. Current Implementation

The application includes:
- FastAPI backend.
- React and Vite frontend.
- Dataset management.
- Candidate model training workflow.
- Session-based candidate model storage.
- Model inference.
- Results and comparison interface.
- Hardware monitoring.
- ADAPT-EDGE pipeline and decision controller.
- RETAIN, PROBE and SWITCH execution logic.
- ContinuationCache.
- Automated tests.

Some capabilities, such as support for arbitrary model architectures, compatible probe generation and external black-box benchmarking, depend on their individual implementation and validation.

## 15. Repository

GitHub: https://github.com/ShreyaVarute/ADAPT-EDGE

The repository contains application source code and research implementation files. Datasets, virtual environments, secrets and trained model weights are excluded from the source repository.