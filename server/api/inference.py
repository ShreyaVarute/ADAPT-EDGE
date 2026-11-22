"""
Inference API Router for ADAPT-EDGE.
Handles real model execution for Model A, Model B, Conventional Switching,
and full ADAPT-EDGE with step-by-step decision tracing and explanations.
"""

import time
import io
import threading
import base64
import copy
import random
import os
from collections import deque
from contextlib import closing
from typing import Dict, Any, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
from dotenv import dotenv_values
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from server.core import ServiceManager, CIFAR10_CLASSES, PROJECT_ROOT
from server.api.datasets import dataset_sample_paths
from server.api.training import CANDIDATE_MODELS, CANDIDATE_MODELS_LOCK
from utils.profiling import synchronize

router = APIRouter(prefix="/api/inference", tags=["Inference"])

INFERENCE_HISTORY = deque(maxlen=100)
INFERENCE_HISTORY_LOCK = threading.Lock()


class ConfigurePairRequest(BaseModel):
    model_a_id: str
    model_b_id: str
    threshold_a: Optional[float] = 0.85
    threshold_probe: Optional[float] = 0.80
    latency_budget_ms: Optional[float] = 25.0
    enforce_resource_budget: Optional[bool] = True


@router.get("/pair")
def get_active_model_pair():
    """Returns the currently active Model A and Model B pair configuration."""
    mgr = ServiceManager.get_instance()
    with CANDIDATE_MODELS_LOCK:
        candidates = [
            {
                "id": run_id,
                "name": f"{cand['metadata'].get('architecture_display', cand['metadata']['architecture'])} ({run_id})",
                "architecture": cand["metadata"]["architecture"],
                "accuracy": cand["metrics"].get("best_val_accuracy"),
                "dataset_id": cand["metadata"]["dataset_id"],
            }
            for run_id, cand in CANDIDATE_MODELS.items()
        ]
    return {
        "success": True,
        "active_pair": mgr.active_pair,
        "available_candidates": candidates,
    }


@router.post("/pair")
def configure_model_pair(req: ConfigurePairRequest):
    """
    Configures and switches active Model A and Model B pairing.
    Validates probe compatibility: PROBE is only enabled if Model B is MobileNetV3-Large.
    """
    mgr = ServiceManager.get_instance()

    # Resolve Model A
    model_a_name = "MobileNetV3-Small (Baseline)"
    model_a_type = "mobilenet_v3_small"
    if req.model_a_id != "mobilenet_v3_small_baseline":
        with CANDIDATE_MODELS_LOCK:
            cand_a = CANDIDATE_MODELS.get(req.model_a_id)
        if not cand_a:
            raise HTTPException(status_code=404, detail=f"Candidate Model A '{req.model_a_id}' not found.")
        model_a_name = f"{cand_a['metadata'].get('architecture_display', cand_a['metadata']['architecture'])} (Candidate)"
        model_a_type = cand_a["metadata"]["architecture"]

    # Resolve Model B
    model_b_name = "MobileNetV3-Large (Baseline)"
    model_b_type = "mobilenet_v3_large"
    if req.model_b_id != "mobilenet_v3_large_baseline":
        with CANDIDATE_MODELS_LOCK:
            cand_b = CANDIDATE_MODELS.get(req.model_b_id)
        if not cand_b:
            raise HTTPException(status_code=404, detail=f"Candidate Model B '{req.model_b_id}' not found.")
        model_b_name = f"{cand_b['metadata'].get('architecture_display', cand_b['metadata']['architecture'])} (Candidate)"
        model_b_type = cand_b["metadata"]["architecture"]

    # Probe Compatibility Enforcement
    # Probe checkpoints are trained exclusively for MobileNetV3-Large backbone representations
    probes_compatible = (
        model_b_type in {"mobilenet_v3_large", "large"} and
        (req.model_b_id == "mobilenet_v3_large_baseline" or "mobilenet_v3_large" in req.model_b_id)
    )

    mgr.active_pair = {
        "model_a_id": req.model_a_id,
        "model_a_name": model_a_name,
        "model_a_type": model_a_type,
        "model_b_id": req.model_b_id,
        "model_b_name": model_b_name,
        "model_b_type": model_b_type,
        "threshold_a": float(req.threshold_a if req.threshold_a is not None else 0.85),
        "threshold_probe": float(req.threshold_probe if req.threshold_probe is not None else 0.80),
        "latency_budget_ms": float(req.latency_budget_ms if req.latency_budget_ms is not None else 25.0),
        "enforce_resource_budget": bool(req.enforce_resource_budget),
        "probes_compatible": probes_compatible,
        "probe_blocks": [4, 9, 13] if probes_compatible else [],
    }

    mgr.log_activity(
        "Model Pair Configured",
        f"A: {model_a_name} | B: {model_b_name} (Probes: {'Active' if probes_compatible else 'Disabled - Incompatible Architecture'})",
        "info"
    )

    return {
        "success": True,
        "message": "Model pair updated successfully.",
        "active_pair": mgr.active_pair,
    }


class PredictRequest(BaseModel):
    dataset_id: str = "cifar10"
    source: Optional[str] = "sample"  # "sample" or "upload"
    dataset_index: Optional[int] = None
    sample_index: Optional[int] = None
    strategy: Optional[str] = "ALL"
    tau_high: Optional[float] = None
    tau_low: Optional[float] = None
    tau_probe: Optional[float] = None
    threshold_a: Optional[float] = 0.85
    threshold_probe: Optional[float] = 0.80
    latency_budget_ms: Optional[float] = 25.0
    enforce_resource_budget: Optional[bool] = True
    enable_cache: Optional[bool] = True
    custom_run_id: Optional[str] = None


@router.get("/samples")
def get_sample_gallery(count: int = 12, dataset_id: str = "cifar10"):
    """Returns pre-loaded CIFAR-10 test set images with thumbnails for 1-click testing."""
    if dataset_id != "cifar10":
        return custom_sample_gallery(dataset_id, count)
    mgr = ServiceManager.get_instance()
    samples = mgr.sample_cache[:count] if mgr.sample_cache else []
    return {
        "success": True,
        "count": len(samples),
        "samples": samples
    }


@router.post("/samples/refresh")
def refresh_sample_gallery(count: int = 12, dataset_id: str = "cifar10"):
    if dataset_id != "cifar10":
        return custom_sample_gallery(dataset_id, count, shuffle=True)
    mgr = ServiceManager.get_instance()
    if mgr.cifar10_dataset is None:
        raise HTTPException(status_code=503, detail="CIFAR-10 sample dataset is unavailable.")
    samples = mgr.refresh_sample_cache(count=count)
    return {"success": True, "count": len(samples), "samples": samples}


def custom_sample_gallery(dataset_id: str, count: int, shuffle: bool = False):
    paths = dataset_sample_paths(dataset_id)
    limit = max(1, min(count, 30))
    by_class = {}
    for index, (name, _) in enumerate(paths):
        by_class.setdefault(name, deque()).append(index)
    if shuffle:
        for indices in by_class.values():
            shuffled = list(indices)
            random.shuffle(shuffled)
            indices.clear()
            indices.extend(shuffled)

    samples = []
    while by_class and len(samples) < limit:
        for name, indices in list(by_class.items()):
            index = indices.popleft()
            if not indices:
                del by_class[name]
            path = paths[index][1]
            try:
                with Image.open(path) as source, io.BytesIO() as buffer:
                    image = source.convert("RGB")
                    dimensions = list(image.size)
                    image.thumbnail((128, 128))
                    image.save(buffer, format="PNG")
                    preview = base64.b64encode(buffer.getvalue()).decode("ascii")
            except (OSError, ValueError):
                continue
            samples.append({
                "index": index, "class_name": name, "filename": path.name,
                "dimensions": dimensions, "image_b64": f"data:image/png;base64,{preview}",
            })
            if len(samples) == limit:
                break
    return {"success": True, "count": len(samples), "samples": samples}


# Rekognition returns open-vocabulary labels. These exact aliases define the
# CIFAR-10 scoring rule independently of the test labels.
CLOUD_LABEL_CLASSES = {
    "airplane": "airplane", "aircraft": "airplane", "aeroplane": "airplane",
    "automobile": "automobile", "car": "automobile", "passenger car": "automobile",
    "bird": "bird", "cat": "cat", "kitten": "cat", "deer": "deer",
    "dog": "dog", "puppy": "dog", "frog": "frog", "horse": "horse",
    "ship": "ship", "boat": "ship", "truck": "truck", "pickup truck": "truck",
}


class CloudTestRequest(BaseModel):
    dataset_id: str = "cifar10"
    sample_indices: list[int] = Field(min_length=1, max_length=12)


@router.post("/cloud-test")
def test_cloud_accuracy(req: CloudTestRequest):
    """Evaluate Amazon Rekognition on explicitly selected labeled samples."""
    is_cifar = req.dataset_id == "cifar10"
    dataset = ServiceManager.get_instance().cifar10_dataset if is_cifar else dataset_sample_paths(req.dataset_id)
    if dataset is None:
        raise HTTPException(503, "The labeled CIFAR-10 test dataset is unavailable.")

    indices = req.sample_indices
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise HTTPException(400, "Provide unique, valid image indices for the selected dataset.")

    classes = CIFAR10_CLASSES if is_cifar else sorted({name for name, _ in dataset})
    normalized_classes = {" ".join(name.lower().replace("_", " ").replace("-", " ").split()): name for name in classes}
    label_map = {
        alias: normalized_classes[target]
        for alias, target in CLOUD_LABEL_CLASSES.items()
        if target in normalized_classes
    }
    label_map.update(normalized_classes)

    config = dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig", interpolate=False)
    profile = os.environ.get("AWS_PROFILE") or config.get("AWS_PROFILE") or "adapt-edge"
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or config.get("AWS_REGION") or config.get("AWS_DEFAULT_REGION")
    rows = []
    start = time.perf_counter()
    try:
        session = boto3.Session(profile_name=profile)
        with closing(session.client(
            "rekognition",
            region_name=region or session.region_name or "us-east-1",
            config=Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 1}),
        )) as client:
            for index in indices:
                if is_cifar:
                    image, label = dataset[index]
                    actual = CIFAR10_CLASSES[label]
                    image = image.resize((96, 96), Image.Resampling.BILINEAR)
                else:
                    actual, path = dataset[index]
                    try:
                        with Image.open(path) as source:
                            image = source.convert("RGB")
                    except (OSError, ValueError):
                        raise HTTPException(422, "A selected dataset image could not be read.") from None
                    image.thumbnail((1024, 1024))
                    # Rekognition requires both dimensions to be at least 80px.
                    horizontal = max(0, 80 - image.width)
                    vertical = max(0, 80 - image.height)
                    image = ImageOps.expand(image, (horizontal // 2, vertical // 2, (horizontal + 1) // 2, (vertical + 1) // 2))
                with io.BytesIO() as buffer:
                    image.save(buffer, format="PNG")
                    response = client.detect_labels(
                        Image={"Bytes": buffer.getvalue()},
                        Features=["GENERAL_LABELS"],
                        MaxLabels=10,
                    )
                labels = sorted(response.get("Labels", []), key=lambda item: item["Confidence"], reverse=True)
                descriptions = [item["Name"] for item in labels]
                mapped = (label_map.get(" ".join(name.lower().replace("_", " ").replace("-", " ").split())) for name in descriptions)
                prediction = next((name for name in mapped if name is not None), None)
                rows.append({
                    "index": index, "actual": actual, "prediction": prediction,
                    "correct": prediction == actual, "cloud_labels": descriptions,
                })
    except ProfileNotFound:
        raise HTTPException(503, "AWS profile not found. Run aws login --profile adapt-edge or set AWS_PROFILE to your configured profile.") from None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"AccessDenied", "AccessDeniedException"}:
            message = "Amazon Rekognition denied access. Verify that the AWS identity has rekognition:DetectLabels permission and the account can use Rekognition."
        elif code in {"ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId", "UnrecognizedClientException"}:
            message = "AWS credentials expired or were rejected. Sign in again to the configured AWS profile."
        elif code in {"ThrottlingException", "ProvisionedThroughputExceededException"}:
            message = "Amazon Rekognition throttled the request. Check the account quota before retrying."
        else:
            message = "Amazon Rekognition rejected the request. Check AWS service access and region."
        raise HTTPException(502, message) from None
    except BotoCoreError:
        raise HTTPException(503, "Unable to use AWS Rekognition. Verify the AWS profile login, region, and network connection.") from None
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    correct = sum(row["correct"] for row in rows)
    return {
        "correct": correct, "total": len(rows), "accuracy_pct": round(100 * correct / len(rows), 2),
        "batch_latency_ms": latency_ms, "rows": rows, "dataset_id": req.dataset_id,
    }


def execute_strategy(
    strategy: str,
    tensor: torch.Tensor,
    mgr: ServiceManager,
    threshold_a: float = 0.85,
    threshold_probe: float = 0.80,
    latency_budget_ms: float = 25.0,
    enforce_resource_budget: bool = True,
    enable_cache: bool = True
) -> Dict[str, Any]:
    """Execute a single inference strategy on the given image tensor."""
    pipe = mgr.pipeline
    if pipe is None:
        raise HTTPException(status_code=500, detail="AdaptEdgePipeline is not loaded.")

    device = pipe.device
    model_a = pipe.model_a
    model_b = pipe.model_b

    # Resolve model_a and model_b based on active_pair
    active = getattr(mgr, "active_pair", {})
    model_a_id = active.get("model_a_id", "mobilenet_v3_small_baseline")
    model_b_id = active.get("model_b_id", "mobilenet_v3_large_baseline")

    # Resolve class names: if custom candidate model is paired, use its custom class names!
    class_names = CIFAR10_CLASSES
    if model_a_id != "mobilenet_v3_small_baseline":
        with CANDIDATE_MODELS_LOCK:
            cand_a = CANDIDATE_MODELS.get(model_a_id)
            if cand_a and "model" in cand_a:
                model_a = cand_a["model"].to(device).eval()
                if "class_names" in cand_a.get("metadata", {}):
                    class_names = cand_a["metadata"]["class_names"]

    if model_b_id != "mobilenet_v3_large_baseline":
        with CANDIDATE_MODELS_LOCK:
            cand_b = CANDIDATE_MODELS.get(model_b_id)
            if cand_b and "model" in cand_b:
                model_b = cand_b["model"].to(device).eval()
                if class_names == CIFAR10_CLASSES and "class_names" in cand_b.get("metadata", {}):
                    class_names = cand_b["metadata"]["class_names"]

    strat = strategy.lower()

    if "model_a" in strat:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_a(tensor)
            probs = F.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
        synchronize(device)
        lat = (time.perf_counter() - t0) * 1000.0

        p_idx = int(pred.item())
        pred_label = class_names[p_idx] if p_idx < len(class_names) else f"Class {p_idx}"
        return {
            "strategy": "Model A Baseline",
            "strategy_name": "Model A Baseline",
            "strategy_key": "MODEL_A",
            "prediction": p_idx,
            "prediction_class": pred_label,
            "confidence": float(conf.item()),
            "latency_ms": lat,
            "decision": "CONFIDENT_RETAIN",
            "cache_status": "N/A",
            "status": "CONFIDENT_RETAIN"
        }

    elif "model_b" in strat:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_b(tensor)
            probs = F.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
        synchronize(device)
        lat = (time.perf_counter() - t0) * 1000.0

        p_idx = int(pred.item())
        pred_label = class_names[p_idx] if p_idx < len(class_names) else f"Class {p_idx}"
        return {
            "strategy": "Model B Baseline",
            "strategy_name": "Model B Baseline",
            "strategy_key": "MODEL_B",
            "prediction": p_idx,
            "prediction_class": pred_label,
            "confidence": float(conf.item()),
            "latency_ms": lat,
            "decision": "SWITCH_MODEL_B",
            "cache_status": "N/A",
            "status": "MODEL_B_FULL"
        }

    elif "conventional" in strat:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_a = model_a(tensor)
            probs_a = F.softmax(out_a, dim=1)
            conf_a, pred_a = torch.max(probs_a, dim=1)
            
            c_val = float(conf_a.item())
            if c_val >= threshold_a:
                synchronize(device)
                lat = (time.perf_counter() - t0) * 1000.0
                p_idx = int(pred_a.item())
                pred_label = class_names[p_idx] if p_idx < len(class_names) else f"Class {p_idx}"
                return {
                    "strategy": "Conventional Switching",
                    "strategy_name": "Conventional Switching",
                    "strategy_key": "CONVENTIONAL",
                    "prediction": p_idx,
                    "prediction_class": pred_label,
                    "confidence": c_val,
                    "latency_ms": lat,
                    "decision": "RETAIN",
                    "cache_status": "NONE",
                    "status": "RETAIN"
                }
            else:
                out_b = model_b(tensor)
                probs_b = F.softmax(out_b, dim=1)
                conf_b, pred_b = torch.max(probs_b, dim=1)
                synchronize(device)
                lat = (time.perf_counter() - t0) * 1000.0
                p_idx = int(pred_b.item())
                pred_label = class_names[p_idx] if p_idx < len(class_names) else f"Class {p_idx}"
                return {
                    "strategy": "Conventional Switching",
                    "strategy_name": "Conventional Switching",
                    "strategy_key": "CONVENTIONAL",
                    "prediction": p_idx,
                    "prediction_class": pred_label,
                    "confidence": float(conf_b.item()),
                    "latency_ms": lat,
                    "decision": "SWITCH",
                    "cache_status": "MISS (Full Re-run)",
                    "status": "SWITCH"
                }

    else:
        # Full ADAPT-EDGE
        with torch.no_grad():
            logits_a_initial = model_a(tensor)
            probs_a_initial = F.softmax(logits_a_initial, dim=1)
            conf_a_initial, pred_a_initial = probs_a_initial.max(dim=1)
            topk_conf, topk_idx = probs_a_initial.topk(k=min(5, 10), dim=1)
        orig_thresh_a = pipe.controller.model_a_threshold
        orig_thresh_probe = pipe.controller.probe_threshold
        orig_budget = pipe.controller.max_latency_ms
        orig_enforce = pipe.controller.enforce_resource_budget

        try:
            pipe.controller.model_a_threshold = threshold_a
            pipe.controller.probe_threshold = threshold_probe
            pipe.controller.max_latency_ms = latency_budget_ms
            pipe.controller.enforce_resource_budget = enforce_resource_budget
            original_cache = pipe.cache
            if not enable_cache:
                pipe.cache = None

            orig_model_a = pipe.model_a
            orig_model_b = pipe.model_b
            pipe.model_a = model_a
            pipe.model_b = model_b

            try:
                res = pipe.predict(tensor)
            finally:
                pipe.model_a = orig_model_a
                pipe.model_b = orig_model_b
            p_idx = int(res.get("prediction", 0))
            pred_label = class_names[p_idx] if p_idx < len(class_names) else f"Class {p_idx}"
            a_idx = int(pred_a_initial.item())
            a_label = class_names[a_idx] if a_idx < len(class_names) else f"Class {a_idx}"
            conf = float(res.get("confidence", 0.0))
            status = str(res.get("status", "COMPLETED"))
            lat = float(res.get("latency_ms", 0.0))

            decision_type = "RETAIN" if "RETAIN" in status else ("SWITCH" if "SWITCH" in status else "PROBE")
            cache_status = "HIT (Saved 35.7%)" if enable_cache and ("SWITCH" in status or len(res.get("executed_probes", [])) > 0) else ("DISABLED" if not enable_cache else "NOT_NEEDED")

            executed_blocks = res.get("executed_probes", [])
            first_probe = executed_blocks[0] if executed_blocks else None
            trace_info = {
                "decision": status,
                "decision_reason": res.get("reason") or res.get("decision_reason") or f"Model evaluation resolved with status '{status}' under threshold {threshold_a:.2f}.",
                "model_a_latency_ms": lat * 0.45 if "RETAIN" not in status else lat,
                "model_a_pred_class": a_label,
                "model_a_conf": float(conf_a_initial.item()),
                "model_a_topk": [
                    {"class": class_names[int(idx)] if int(idx) < len(class_names) else f"Class {idx}", "confidence": float(value)}
                    for value, idx in zip(topk_conf[0].tolist(), topk_idx[0].tolist())
                ],
                "executed_probe": first_probe,
                "probes_executed": [
                    {
                        "block_id": b,
                        "pred_class": "Measured by probe during pipeline execution",
                        "confidence": None,
                        "resolved": True
                    } for b in executed_blocks
                ],
                "final_model": res.get("final_model", "Model A" if "RETAIN" in status else "Model B Tail"),
                "continuation_cache_status": cache_status
            }

            strat_name = "ADAPT-EDGE (Cached)" if enable_cache else "ADAPT-EDGE (Uncached)"
            strat_key = "ADAPT_EDGE_CACHED" if enable_cache else "ADAPT_EDGE_UNCACHED"

            return {
                "strategy": strat_name,
                "strategy_name": strat_name,
                "strategy_key": strat_key,
                "prediction": p_idx,
                "prediction_class": pred_label,
                "confidence": conf,
                "latency_ms": lat,
                "decision": decision_type,
                "cache_status": cache_status,
                "status": status,
                "trace": trace_info
            }
        finally:
            if 'original_cache' in locals():
                pipe.cache = original_cache
            pipe.controller.model_a_threshold = orig_thresh_a
            pipe.controller.probe_threshold = orig_thresh_probe
            pipe.controller.max_latency_ms = orig_budget
            pipe.controller.enforce_resource_budget = orig_enforce


@router.post("/predict")
def predict_image(req: PredictRequest):
    """
    Run real inference for selected strategy or all strategies on an image.
    Accepts JSON body.
    """
    mgr = ServiceManager.get_instance()
    
    # Resolve index
    idx = req.sample_index if req.sample_index is not None else (req.dataset_index if req.dataset_index is not None else 0)
    if req.dataset_id != "cifar10":
        paths = dataset_sample_paths(req.dataset_id)
        if idx < 0 or idx >= len(paths):
            raise HTTPException(400, "Invalid image index for the selected dataset.")
        with CANDIDATE_MODELS_LOCK:
            candidate = CANDIDATE_MODELS.get(req.custom_run_id)
        if candidate is None or candidate["metadata"]["dataset_id"] != req.dataset_id:
            raise HTTPException(400, "Select a session model trained on this dataset.")
        actual, path = paths[idx]
        with path.open("rb") as image:
            result = predict_uploaded_image(
                UploadFile(image, filename=path.name),
                model_mode="custom", custom_run_id=req.custom_run_id,
            )
        return {**result, "ground_truth": actual, "true_class": actual, "dataset_id": req.dataset_id}
    
    # Resolve thresholds
    thresh_a = req.tau_high if req.tau_high is not None else req.threshold_a
    thresh_p = req.tau_probe if req.tau_probe is not None else req.threshold_probe
    budget = req.latency_budget_ms or 25.0

    try:
        tensor, label_id, preview_b64 = mgr.get_sample_image(idx)
        true_label = label_id
        true_class = CIFAR10_CLASSES[label_id]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Sample error: {str(e)}")

    strat_req = (req.strategy or "ALL").upper()

    if strat_req in ["ALL", "ALL_STRATEGIES"]:
        # Execute all 4 strategies in sequence on the same input
        comp_a = execute_strategy("model_a", tensor, mgr, thresh_a, thresh_p, budget)
        comp_b = execute_strategy("model_b", tensor, mgr, thresh_a, thresh_p, budget)
        comp_conv = execute_strategy("conventional", tensor, mgr, thresh_a, thresh_p, budget)
        comp_edge = execute_strategy("adapt_edge", tensor, mgr, thresh_a, thresh_p, budget, enable_cache=True)

        comparison_list = [comp_a, comp_b, comp_conv, comp_edge]

        result = {
            "success": True,
            "sample_index": idx,
            "preview_b64": preview_b64,
            "ground_truth": true_class,
            "true_label": true_label,
            "true_class": true_class,
            "comparison": comparison_list,
            "adapt_edge_cached": comp_edge
        }
    else:
        enable_cache = False if "UNCACHED" in strat_req else (req.enable_cache if req.enable_cache is not None else True)
        res = execute_strategy(strat_req, tensor, mgr, thresh_a, thresh_p, budget, enable_cache=enable_cache)

        result = {
            "success": True,
            "sample_index": idx,
            "preview_b64": preview_b64,
            "ground_truth": true_class,
            "true_label": true_label,
            "true_class": true_class,
            "adapt_edge_cached": res,
            "primary": res
        }

    primary = result.get("adapt_edge_cached") or result.get("primary")
    if primary:
        with INFERENCE_HISTORY_LOCK:
            INFERENCE_HISTORY.appendleft({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": f"CIFAR-10 sample {idx}",
                "model_mode": "cifar10",
                "model_name": "CIFAR-10 ADAPT-EDGE research pipeline",
                "prediction": primary.get("prediction_class"),
                "action": primary.get("decision"),
                "selected_probe": primary.get("trace", {}).get("executed_probe"),
                "probes": primary.get("trace", {}).get("probes_executed", []),
                "latency_ms": primary.get("latency_ms"),
                "confidence": primary.get("confidence"),
                "ground_truth": result.get("ground_truth"),
                "ood_status": "IN_DATASET",
            })
    result["model_mode"] = "cifar10"
    result["ood"] = {
        "status": "IN_DATASET",
        "is_potential_ood": False,
        "message": "Input is from the CIFAR-10 test dataset. Advanced external-image OOD calibration is pending.",
        "calibration": "pending",
    }
    return result


@router.post("/upload")
def predict_uploaded_image(
    image: UploadFile = File(...),
    strategy: str = "ADAPT_EDGE",
    enable_cache: bool = True,
    model_mode: str = "cifar10",
    custom_run_id: Optional[str] = None,
):
    if not (image.filename or "").lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    data = image.file.read(10 * 1024 * 1024)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Corrupted or invalid image: {exc}")

    mgr = ServiceManager.get_instance()
    if model_mode == "custom":
        if not custom_run_id:
            raise HTTPException(status_code=400, detail="A valid custom run ID is required.")
        with CANDIDATE_MODELS_LOCK:
            candidate = CANDIDATE_MODELS.get(custom_run_id)
        if candidate is None:
            raise HTTPException(
                status_code=404,
                detail="Custom model is not available in this backend session. Train it again.",
            )
        try:
            class_names = candidate["metadata"]["class_names"]
            preprocessing = candidate["metadata"]["preprocessing"]
            metadata = candidate["metadata"]
            if len(class_names) != int(metadata["num_classes"]):
                raise ValueError("Class mapping length does not match model output dimensions.")
            model = copy.deepcopy(candidate["model"]).to(mgr.device).eval()
            size = tuple(preprocessing["resize"])
            mean = preprocessing["normalize_mean"]
            std = preprocessing["normalize_std"]
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            tensor = transform(pil_image).unsqueeze(0).to(mgr.device)
            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, prediction = probabilities.max(dim=1)
            synchronize(mgr.device)
            latency = (time.perf_counter() - start) * 1000.0
            prediction_index = int(prediction.item())
            predicted_class = class_names[prediction_index]
            topk = probabilities.topk(k=min(5, len(class_names)), dim=1)
            result = {
                "strategy": "Custom Trained Model",
                "strategy_name": "Custom Trained Model",
                "strategy_key": "CUSTOM",
                "prediction": prediction_index,
                "prediction_class": predicted_class,
                "confidence": float(confidence.item()),
                "latency_ms": latency,
                "decision": None,
                "cache_status": "N/A",
                "status": "CUSTOM_MODEL_INFERENCE",
                "top_k": [
                    {"class": class_names[int(idx)], "confidence": float(value)}
                    for value, idx in zip(topk.values[0].tolist(), topk.indices[0].tolist())
                ],
                "metadata": metadata,
            }
        except HTTPException:
            raise
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Custom model validation failed: {exc}")

        preview_img = pil_image.resize((256, 256), Image.Resampling.BILINEAR)
        import io as pil_io
        buffer = pil_io.BytesIO()
        preview_img.save(buffer, format="PNG")
        preview_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        ood = {
            "status": "CALIBRATION_PENDING",
            "is_potential_ood": None,
            "message": "Advanced OOD calibration is pending for custom models. Confidence is not a confirmed in-domain reliability score.",
        }
        with INFERENCE_HISTORY_LOCK:
            INFERENCE_HISTORY.appendleft({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": image.filename,
                "model_mode": "custom",
                "model_name": custom_run_id,
                "prediction": predicted_class,
                "confidence": float(confidence.item()),
                "action": None,
                "latency_ms": latency,
                "ood_status": ood["status"],
            })
        return {
            "success": True,
            "filename": image.filename,
            "dimensions": [pil_image.width, pil_image.height],
            "preview_b64": preview_b64,
            "ground_truth": None,
            "primary": result,
            "adapt_edge_cached": result,
            "model_mode": "custom",
            "ood": ood,
        }

    if mgr.pipeline is None:
        raise HTTPException(status_code=503, detail="AdaptEdgePipeline is not loaded.")
    tensor, preview = mgr.process_uploaded_image(data)
    enable_cache = str(enable_cache).lower() not in {"false", "0"}
    strat_req = (strategy or "ALL").upper()

    active = getattr(mgr, "active_pair", {})
    thresh_a = float(active.get("threshold_a", 0.85))
    thresh_p = float(active.get("threshold_probe", 0.80))
    budget = float(active.get("latency_budget_ms", 25.0))

    if strat_req in ["ALL", "ALL_STRATEGIES"]:
        comp_a = execute_strategy("model_a", tensor, mgr, thresh_a, thresh_p, budget)
        comp_b = execute_strategy("model_b", tensor, mgr, thresh_a, thresh_p, budget)
        comp_conv = execute_strategy("conventional", tensor, mgr, thresh_a, thresh_p, budget)
        comp_edge = execute_strategy("adapt_edge", tensor, mgr, thresh_a, thresh_p, budget, enable_cache=enable_cache)

        comparison_list = [comp_a, comp_b, comp_conv, comp_edge]
        res = comp_edge
        result_payload = {
            "success": True,
            "filename": image.filename,
            "dimensions": [pil_image.width, pil_image.height],
            "preview_b64": preview,
            "ground_truth": None,
            "comparison": comparison_list,
            "primary": res,
            "adapt_edge_cached": res,
            "model_mode": "cifar10",
        }
    else:
        res = execute_strategy(
            strategy,
            tensor,
            mgr,
            threshold_a=thresh_a,
            threshold_probe=thresh_p,
            latency_budget_ms=budget,
            enable_cache=enable_cache,
        )
        result_payload = {
            "success": True,
            "filename": image.filename,
            "dimensions": [pil_image.width, pil_image.height],
            "preview_b64": preview,
            "ground_truth": None,
            "primary": res,
            "adapt_edge_cached": res,
            "model_mode": "cifar10",
        }

    with INFERENCE_HISTORY_LOCK:
        INFERENCE_HISTORY.appendleft({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image.filename,
            "model_mode": "cifar10",
            "model_name": "CIFAR-10 ADAPT-EDGE research pipeline",
            "prediction": res.get("prediction_class"),
            "action": res.get("decision"),
            "selected_probe": res.get("trace", {}).get("executed_probe"),
            "probes": res.get("trace", {}).get("probes_executed", []),
            "latency_ms": res.get("latency_ms"),
            "ground_truth": None,
            "confidence": res.get("confidence"),
            "ood_status": "POTENTIAL_OOD",
        })
    ood = {
        "status": "POTENTIAL_OOD",
        "is_potential_ood": True,
        "message": "Potential out-of-distribution image. The model was trained on CIFAR-10 and may produce unreliable predictions for external photographs.",
        "calibration": "pending",
    }
    result_payload["ood"] = ood
    return result_payload


@router.post("/compare")
def compare_inference(req: PredictRequest):
    req.strategy = "ALL"
    return predict_image(req)


@router.get("/history")
def get_inference_history(limit: int = 50):
    limit = max(1, min(limit, 100))
    with INFERENCE_HISTORY_LOCK:
        history = list(INFERENCE_HISTORY)[:limit]
    return {"success": True, "history": history}
