"""
Experiments & Benchmarking API Router for ADAPT-EDGE.
Provides access to historical evaluation runs, decision traces, and live benchmark execution.
"""

import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
import pandas as pd

from server.core import ServiceManager, PROJECT_ROOT

router = APIRouter(prefix="/api/experiments", tags=["Experiments & Benchmarks"])


class RunBenchmarkRequest(BaseModel):
    sample_count: int = 10
    start_index: int = 1000
    stratified: bool = True
    threshold_a: float = 0.80
    threshold_probe: float = 0.85
    latency_budget_ms: float = 15.0
    enforce_resource_budget: bool = True
    enable_cache: bool = True


@router.get("/history")
def get_experiment_history():
    """Lists all historical experiment runs found in results/ directory."""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return {"success": True, "experiments": []}

    experiments = []
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and "phase5_" in d.name], reverse=True)

    for r_dir in run_dirs:
        summary_file = r_dir / "metrics/strategy_comparison.csv"
        report_file = r_dir / "reports/phase5_results_report.md"
        meta_file = r_dir / "metadata/sample_indices.json"

        exp_data: Dict[str, Any] = {
            "run_id": r_dir.name,
            "timestamp": r_dir.name.replace("phase5_", ""),
            "has_metrics": summary_file.exists(),
            "has_report": report_file.exists(),
            "strategies": []
        }

        # Parse sample count if available
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    m = json.load(f)
                    exp_data["sample_count"] = m.get("sample_count", "unknown")
                    exp_data["stratified"] = m.get("stratified", False)
            except Exception:
                pass

        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                for _, row in df.iterrows():
                    exp_data["strategies"].append({
                        "name": str(row["strategy"]),
                        "accuracy": float(row["accuracy"]),
                        "macro_f1": float(row["macro_f1"]),
                        "mean_latency_ms": float(row["mean_latency_ms"]),
                        "p95_latency_ms": float(row["p95_latency_ms"]),
                        "full_model_b_pct": float(row["full_model_b_pct"])
                    })
            except Exception:
                pass

        experiments.append(exp_data)

    return {"success": True, "experiments": experiments}


@router.get("/{run_id}")
def get_experiment_details(run_id: str, limit_traces: int = 50, filter_action: Optional[str] = None):
    """Returns complete details, comparison table, and decision traces for a specific run."""
    run_dir = PROJECT_ROOT / f"results/{run_id}"
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{run_id}' not found.")

    summary_file = run_dir / "metrics/strategy_comparison.csv"
    per_image_file = run_dir / "raw/per_image_results.csv"
    report_file = run_dir / "reports/phase5_results_report.md"
    cache_file = run_dir / "metrics/cache_metrics.json"

    strategies = []
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file)
        strategies = df_summary.to_dict(orient="records")

    cache_metrics = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache_metrics = json.load(f)

    traces = []
    total_traces_count = 0
    if per_image_file.exists():
        df_traces = pd.read_csv(per_image_file)
        total_traces_count = len(df_traces)

        if filter_action:
            df_traces = df_traces[df_traces["adapt_cached_action"] == filter_action]

        traces = df_traces.head(limit_traces).to_dict(orient="records")

    report_md = ""
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            report_md = f.read()

    return {
        "success": True,
        "run_id": run_id,
        "strategies": strategies,
        "cache_metrics": cache_metrics,
        "total_traces": total_traces_count,
        "traces": traces,
        "report_markdown": report_md
    }


@router.post("/run")
def run_live_benchmark(req: RunBenchmarkRequest):
    """
    Execute a live benchmark evaluation across all 4 strategies on N identical samples.
    Runs quickly for small sample counts (e.g. 5, 10, 20 images) to provide real live results.
    """
    mgr = ServiceManager.get_instance()
    pipe = mgr.pipeline
    if pipe is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized.")

    if req.sample_count > 50:
        raise HTTPException(status_code=400, detail="Max interactive sample count is 50 for live demo. Use CLI for large evaluations.")

    mgr.log_activity("Live Benchmark Started", f"Evaluating {req.sample_count} images across all 4 strategies...")

    from experiments.evaluate_phase5 import run_comparative_evaluation

    try:
        results = run_comparative_evaluation(
            eval_start_index=req.start_index,
            eval_count=req.sample_count,
            stratified=req.stratified,
            max_latency_budget_ms=req.latency_budget_ms,
            enable_calibration=True
        )

        run_id = Path(results["run_dir"]).name
        mgr.log_activity("Live Benchmark Completed", f"Run {run_id} finished across {req.sample_count} samples.")

        # Return the summary metrics
        summary_csv = Path(results["run_dir"]) / "metrics/strategy_comparison.csv"
        df_res = pd.read_csv(summary_csv)
        summary_table = df_res.to_dict(orient="records")

        return {
            "success": True,
            "run_id": run_id,
            "sample_count": req.sample_count,
            "summary": summary_table,
            "cache_metrics": results.get("cache_metrics", {})
        }
    except Exception as e:
        mgr.log_activity("Benchmark Error", str(e), event_type="error")
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


def _safe_run_id(run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid run ID.")
    run_dir = (PROJECT_ROOT / "results" / run_id).resolve()
    try:
        run_dir.relative_to((PROJECT_ROOT / "results").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID.")
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Experiment run not found.")
    return run_dir


@router.get("/results")
@router.get("/runs")
def list_result_runs():
    results_root = PROJECT_ROOT / "results"
    runs = []
    if not results_root.exists():
        return {"success": True, "runs": runs}
    for run_dir in sorted(results_root.iterdir(), key=lambda p: p.name, reverse=True):
        if not run_dir.is_dir():
            continue
        metric_files = [
            run_dir / "metrics/strategy_comparison.csv",
            run_dir / "strategy_comparison.csv",
            run_dir / "baseline_test_results.json",
        ]
        if not any(path.exists() for path in metric_files):
            continue
        runs.append({
            "run_id": run_dir.name,
            "path": str(run_dir.relative_to(PROJECT_ROOT)),
            "historical": True,
            "has_metrics": True,
        })
    return {"success": True, "runs": runs}


@router.get("/results/compare")
def compare_result_runs(run_ids: str):
    ids = [value.strip() for value in run_ids.split(",") if value.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="Provide at least one run_id.")
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="A maximum of 10 runs can be compared.")
    comparisons = []
    for run_id in ids:
        run_dir = _safe_run_id(run_id)
        csv_path = run_dir / "metrics/strategy_comparison.csv"
        if not csv_path.exists():
            csv_path = run_dir / "strategy_comparison.csv"
        strategies = []
        if csv_path.exists():
            strategies = pd.read_csv(csv_path).to_dict(orient="records")
        comparisons.append({"run_id": run_id, "strategies": strategies})
    return {"success": True, "comparisons": comparisons}


@router.get("/results/export/{run_id}")
def export_result_run(run_id: str, format: str = "csv"):
    run_dir = _safe_run_id(run_id)
    csv_path = run_dir / "metrics/strategy_comparison.csv"
    if not csv_path.exists():
        csv_path = run_dir / "strategy_comparison.csv"
    if format.lower() == "csv" and csv_path.exists():
        return FileResponse(str(csv_path), media_type="text/csv", filename=f"{run_id}.csv")
    if format.lower() == "json":
        if csv_path.exists():
            data = pd.read_csv(csv_path).to_dict(orient="records")
            return JSONResponse(content={"run_id": run_id, "historical": True, "strategies": data})
        json_candidates = list(run_dir.glob("*.json"))
        if json_candidates:
            return JSONResponse(content=json.loads(json_candidates[0].read_text(encoding="utf-8")))
        raise HTTPException(status_code=404, detail="No JSON or CSV metrics found.")
    raise HTTPException(status_code=404, detail="Requested export format is unavailable.")

