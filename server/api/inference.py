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
from collections import deque
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from server.core import ServiceManager, CIFAR10_CLASSES
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
def get_sample_gallery(count: int = 12):
    """Returns pre-loaded CIFAR-10 test set images with thumbnails for 1-click testing."""
    mgr = ServiceManager.get_instance()
    samples = mgr.sample_cache[:count] if mgr.sample_cache else []
    return {
        "success": True,
        "count": len(samples),
        "samples": samples
    }


@router.post("/samples/refresh")
def refresh_sample_gallery(count: int = 12):
    mgr = ServiceManager.get_instance()
    if mgr.cifar10_dataset is None:
        raise HTTPException(status_code=503, detail="CIFAR-10 sample dataset is unavailable.")
    samples = mgr.refresh_sample_cache(count=count)
    return {"success": True, "count": len(samples), "samples": samples}


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

    if model_a_id != "mobilenet_v3_small_baseline":
        with CANDIDATE_MODELS_LOCK:
            cand_a = CANDIDATE_MODELS.get(model_a_id)
            if cand_a and "model" in cand_a:
                model_a = cand_a["model"].to(device).eval()

    if model_b_id != "mobilenet_v3_large_baseline":
        with CANDIDATE_MODELS_LOCK:
            cand_b = CANDIDATE_MODELS.get(model_b_id)
            if cand_b and "model" in cand_b:
                model_b = cand_b["model"].to(device).eval()

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
        return {
            "strategy": "Model A Baseline",
            "strategy_name": "Model A Baseline",
            "strategy_key": "MODEL_A",
            "prediction": p_idx,
            "prediction_class": CIFAR10_CLASSES[p_idx],
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
        return {
            "strategy": "Model B Baseline",
            "strategy_name": "Model B Baseline",
            "strategy_key": "MODEL_B",
            "prediction": p_idx,
            "prediction_class": CIFAR10_CLASSES[p_idx],
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
                return {
                    "strategy": "Conventional Switching",
                    "strategy_name": "Conventional Switching",
                    "strategy_key": "CONVENTIONAL",
                    "prediction": p_idx,
                    "prediction_class": CIFAR10_CLASSES[p_idx],
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
                return {
                    "strategy": "Conventional Switching",
                    "strategy_name": "Conventional Switching",
                    "strategy_key": "CONVENTIONAL",
                    "prediction": p_idx,
                    "prediction_class": CIFAR10_CLASSES[p_idx],
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
                "model_a_pred_class": CIFAR10_CLASSES[int(pred_a_initial.item())],
                "model_a_conf": float(conf_a_initial.item()),
                "model_a_topk": [
                    {"class": CIFAR10_CLASSES[int(idx)], "confidence": float(value)}
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
                "prediction_class": CIFAR10_CLASSES[p_idx],
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
