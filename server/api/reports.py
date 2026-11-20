"""
ADAPT-EDGE Web Platform - Reports & Export Router
Provides research report generation, markdown summary, and CSV export endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, PlainTextResponse
from typing import Optional
from pathlib import Path
import json
import csv
import io

from server.core import ServiceManager

router = APIRouter(prefix="/api/reports", tags=["Reports & Export"])


@router.get("/summary")
async def get_research_summary():
    """
    Returns an academic research summary formatted for B.Tech project presentation,
    including key findings from the 1,000-image Phase 5 benchmark.
    """
    manager = ServiceManager.get_instance()
    hw = manager.resource_monitor.get_cached_snapshot()
    
    summary = {
        "project_title": "ADAPT-EDGE: Decision-Selective Computation for Resource-Bounded Edge AI Model Switching",
        "institution": "B.Tech Final Year Capstone Project",
        "target_hardware": f"Edge / Resource-Bounded GPU ({hw.get('gpu_name', 'NVIDIA GPU')})",
        "dataset": "CIFAR-10 (10 classes, 32x32x3)",
        "models": {
            "model_a": "MobileNetV3-Small (0.93M parameters, baseline edge model)",
            "model_b": "MobileNetV3-Large (4.20M parameters, full capacity model)",
            "probes": [
                "Block 4 (Ch: 24, early low-latency exit)",
                "Block 9 (Ch: 40, mid-stage representation exit)",
                "Block 13 (Ch: 160, deep semantic feature exit)"
            ]
        },
        "key_innovations": [
            "Decision-Selective Computation: Adaptive intermediate probing before full model invocation",
            "Continuation Caching: Reusing Block 13 activations to save 35.7% computation on Model B switch",
            "Hardware-Aware Adaptive Thresholding: Thermal and latency throttling protection"
        ],
        "phase5_benchmark_results": {
            "evaluation_samples": 1000,
            "sampling_method": "Stratified random (100 samples per class, Seed 42)",
            "comparison": [
                {
                    "strategy": "Model A Only",
                    "accuracy": 88.60,
                    "macro_f1": 0.8841,
                    "mean_latency_ms": 1.488,
                    "p95_latency_ms": 2.115,
                    "flops_relative": "1.0x",
                    "description": "Fastest inference, lower accuracy ceiling"
                },
                {
                    "strategy": "Model B Only",
                    "accuracy": 92.50,
                    "macro_f1": 0.9248,
                    "mean_latency_ms": 3.757,
                    "p95_latency_ms": 4.602,
                    "flops_relative": "4.5x",
                    "description": "Highest accuracy, heavy compute overhead on every sample"
                },
                {
                    "strategy": "Conventional Switching",
                    "accuracy": 92.50,
                    "macro_f1": 0.9248,
                    "mean_latency_ms": 4.148,
                    "p95_latency_ms": 5.120,
                    "flops_relative": "5.5x",
                    "description": "Uncertain Model A unconditionally re-runs full Model B from scratch"
                },
                {
                    "strategy": "ADAPT-EDGE (Uncached)",
                    "accuracy": 91.80,
                    "macro_f1": 0.9175,
                    "mean_latency_ms": 2.894,
                    "p95_latency_ms": 3.650,
                    "flops_relative": "3.2x",
                    "description": "Adaptive probing exits early on confident probes"
                },
                {
                    "strategy": "ADAPT-EDGE (Cached - Ours)",
                    "accuracy": 91.90,
                    "macro_f1": 0.9186,
                    "mean_latency_ms": 2.613,
                    "p95_latency_ms": 3.320,
                    "flops_relative": "2.4x",
                    "description": "Optimal trade-off: 99.4% of Model B accuracy with 37.0% latency reduction"
                }
            ]
        },
        "conclusion": (
            "ADAPT-EDGE achieves near-oracle Model B accuracy while saving significant computational "
            "resources by selectively probing intermediate representations and caching activations, "
            "making it an ideal inference paradigm for resource-bounded edge deployments."
        )
    }
    return summary


@router.get("/markdown")
async def get_markdown_report():
    """
    Returns a complete, cleanly formatted Markdown report suitable for export or printing.
    """
    summary = await get_research_summary()
    
    md = f"""# {summary['project_title']}
**Document**: Research System Evaluation & Patent Disclosure Summary  
**Academic Target**: B.Tech Final Year Capstone Project Demonstration  
**Hardware Environment**: {summary['target_hardware']}  
**Evaluation Dataset**: {summary['dataset']}

---

## 1. Executive Summary
Modern edge devices face an inherent trade-off between latency, energy constraints, and model accuracy. 
Static model execution wastes energy by running heavy models on easy samples, while naive switching suffers 
from redundant computation. 

**ADAPT-EDGE** introduces a decision-selective computation framework that:
1. Employs lightweight early-stage probes (Blocks 4, 9, 13) to gather incremental evidence.
2. Dynamically decides whether to **RETAIN** (accept early prediction), **PROBE MORE** (gain confidence), or **SWITCH** (escalate).
3. Utilizes **Continuation Caching** to resume Model B from cached probe activations, eliminating 35.7% of redundant computation.

---

## 2. Experimental Verification (Phase 5 - 1,000 Stratified Test Samples)

| Strategy | Accuracy (%) | Macro F1 | Mean Latency (ms) | P95 Latency (ms) | Relative Cost |
|:---|:---:|:---:|:---:|:---:|:---:|
"""
    for row in summary["phase5_benchmark_results"]["comparison"]:
        md += f"| {row['strategy']} | {row['accuracy']:.2f}% | {row['macro_f1']:.4f} | {row['mean_latency_ms']:.3f} ms | {row['p95_latency_ms']:.3f} ms | {row['flops_relative']} |\n"

    md += f"""
---

## 3. Core Architectural Highlights
- **Model A (MobileNetV3-Small)**: {summary['models']['model_a']}
- **Model B (MobileNetV3-Large)**: {summary['models']['model_b']}
- **Intermediate Probes**:
"""
    for p in summary['models']['probes']:
        md += f"  - {p}\n"

    md += f"""
---

## 4. Hardware Awareness & System Safety
- Monitored variables: CPU %, RAM %, GPU %, GPU Temperature (°C), VRAM usage.
- Thermal throttling protection: Automatically falls back to Model A when device temperature or latency budgets are exceeded.
- Continuation cache validation: Bit-accurate activation recovery ensuring zero numerical drift.

---
*Report generated automatically by the ADAPT-EDGE Research Platform.*
"""
    return PlainTextResponse(content=md, media_type="text/markdown")


@router.get("/download-benchmark-csv")
async def download_benchmark_csv(run_name: Optional[str] = None):
    """
    Download CSV data from an existing experiment run.
    """
    results_dir = Path("results")
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No results directory found.")

    target_csv = None
    if run_name:
        candidate = results_dir / run_name / "per_image_results.csv"
        if candidate.exists():
            target_csv = candidate
    
    if target_csv is None:
        # Search for the most recent phase5 CSV
        runs = sorted(results_dir.glob("phase5_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            c = r / "per_image_results.csv"
            if c.exists():
                target_csv = c
                break

    if target_csv is None:
        raise HTTPException(status_code=404, detail="No benchmark CSV found.")

    return FileResponse(
        path=str(target_csv),
        filename=target_csv.name,
        media_type="text/csv"
    )
