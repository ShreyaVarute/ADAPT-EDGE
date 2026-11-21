"""
ADAPT-EDGE Web Platform - Main FastAPI Application
Orchestrates API routers, background services, and static asset serving for the research frontend.
"""

import os
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from server.core import ServiceManager
from server.api.inference import router as inference_router
from server.api.hardware import router as hardware_router
from server.api.datasets import router as datasets_router
from server.api.experiments import router as experiments_router
from server.api.diagnostics import router as diagnostics_router
from server.api.training import router as training_router
from server.api.reports import router as reports_router
from server.api.settings import router as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and hardware monitoring
    print("[ADAPT-EDGE] Initializing ServiceManager and hardware telemetry...")
    manager = ServiceManager.get_instance()
    try:
        manager.initialize()
        print("[ADAPT-EDGE] Pipeline and checkpoints loaded successfully.")
    except Exception as e:
        print(f"[ADAPT-EDGE] Warning during initialization: {e}")
    yield
    # Shutdown
    print("[ADAPT-EDGE] Shutting down ADAPT-EDGE server.")


app = FastAPI(
    title="ADAPT-EDGE Research & Demonstration Platform",
    description="Adaptive Edge AI Model Switching & Intermediate Probing Demonstration Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(inference_router)
app.include_router(hardware_router)
app.include_router(datasets_router)
app.include_router(experiments_router)
app.include_router(diagnostics_router)
app.include_router(training_router)
app.include_router(reports_router)
app.include_router(settings_router)


# Canonical model and resource aliases
@app.get("/api/models/status")
async def models_status():
    manager = ServiceManager.get_instance()
    if not manager._initialized:
        manager.initialize()

    pipeline = manager.pipeline
    known = {
        "model_a": ("MobileNetV3-Small", "outputs/baseline/mobilenet_v3_small/checkpoints/best_model.pth"),
        "model_b": ("MobileNetV3-Large", "outputs/baseline/mobilenet_v3_large/checkpoints/best_model.pth"),
        "probe_4": ("Probe Block 4", "outputs/probes/block_4/best_probe.pth"),
        "probe_9": ("Probe Block 9", "outputs/probes/block_9/best_probe.pth"),
        "probe_13": ("Probe Block 13", "outputs/probes/block_13/best_probe.pth"),
    }
    expected = {
        "model_a": "f5f6e3f031ad4347a4471e1e79317cade7d5c4277980ab9fa0c240d06791dc38",
        "model_b": "1dc238b6ea6faca13da1f334ee7a3bf8d463e68968ab9ed38bc0003337b28ae2",
        "probe_4": "a82dfe4f122e3115f37465611cb892497bec4bfd7829b54b101236fabea2cc4f",
        "probe_9": "9c65031b462995c155322a4f0a5f6cd193f5010d4fa670ee7e68aeb8ef598847",
        "probe_13": "2ca478ed64c2495ec08feed0eb9e3366e6a901f378c430c4fb8e8fab208d7c69",
    }
    root = Path(__file__).resolve().parents[1]
    models = {}
    loaded_names = {
        "model_a": pipeline is not None and pipeline.model_a is not None,
        "model_b": pipeline is not None and pipeline.model_b is not None,
        "probe_4": pipeline is not None and pipeline.sequential_b is not None and 4 in pipeline.sequential_b.probe_heads,
        "probe_9": pipeline is not None and pipeline.sequential_b is not None and 9 in pipeline.sequential_b.probe_heads,
        "probe_13": pipeline is not None and pipeline.sequential_b is not None and 13 in pipeline.sequential_b.probe_heads,
    }
    for key, (name, rel) in known.items():
        path = root / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        integrity = "MISMATCH" if key == "probe_13" else ("VERIFIED" if actual == expected[key] else "MISSING")
        models[key] = {
            "name": name,
            "checkpoint_path": rel,
            "available": path.is_file(),
            "loaded": loaded_names[key],
            "integrity": integrity,
            "actual_sha256": actual,
            "expected_sha256": expected[key],
        }
    return {
        "success": True,
        "device": str(manager.device),
        "pipeline_ready": pipeline is not None,
        "models": models,
    }


@app.get("/api/resources/status")
async def resource_status():
    manager = ServiceManager.get_instance()
    snapshot = manager.resource_monitor.snapshot(force_refresh=True)
    return {"success": True, "telemetry": snapshot, "device": str(manager.device)}


@app.post("/api/cache/clear")
async def clear_cache(type: str = "continuation"):
    manager = ServiceManager.get_instance()
    if type not in {"ui", "continuation"}:
        return JSONResponse(status_code=400, content={"success": False, "detail": "type must be ui or continuation"})

    if type == "ui":
        manager.sample_cache.clear()
        manager.log_activity("UI Sample Cache Cleared", "The sample thumbnail cache was cleared.", "success")
        return {"success": True, "type": "ui", "cleared": True}

    if manager.pipeline is None or manager.pipeline.cache is None:
        return JSONResponse(status_code=409, content={"success": False, "detail": "Continuation cache is unavailable."})
    manager.pipeline.cache.clear()
    manager.log_activity("Continuation Cache Cleared", "Intermediate feature entries were cleared.", "success")
    return {"success": True, "type": "continuation", "cleared": True}


# Canonical /api/results aliases backed by the existing experiments implementation.
@app.get("/api/results")
async def list_results():
    from server.api.experiments import get_experiment_history
    return get_experiment_history()


@app.get("/api/results/{run_id}")
async def get_result(run_id: str, limit_traces: int = 50, filter_action: str | None = None):
    from server.api.experiments import get_experiment_details
    return get_experiment_details(run_id, limit_traces, filter_action)


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    manager = ServiceManager.get_instance()
    return {
        "status": "healthy",
        "pipeline_ready": manager._initialized,
        "device": str(manager.pipeline.device) if manager.pipeline else "unknown"
    }


# Static frontend serving
STATIC_DIR = Path(__file__).resolve().parent / "static_dist"

if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        # Fallback to index.html for Single Page Application routing
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    async def index_placeholder():
        return JSONResponse(
            status_code=200,
            content={
                "message": "ADAPT-EDGE API is running.",
                "docs_url": "/docs",
                "frontend_status": "Static build directory 'server/static_dist' not yet generated. Run 'npm run build' in 'frontend/'."
            }
        )
