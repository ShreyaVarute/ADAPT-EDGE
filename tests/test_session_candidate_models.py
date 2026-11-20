import io
from types import SimpleNamespace

import torch
from PIL import Image
from starlette.datastructures import UploadFile

from server.api import inference, training


def test_custom_model_inference_uses_candidate_kept_in_memory(monkeypatch):
    candidate_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 2),
    )
    candidate = {
        "model": candidate_model,
        "metadata": {
            "run_id": "session-candidate",
            "num_classes": 2,
            "class_names": ["cat", "dog"],
            "preprocessing": {
                "resize": [4, 4],
                "normalize_mean": [0.0, 0.0, 0.0],
                "normalize_std": [1.0, 1.0, 1.0],
            },
        },
        "metrics": {"best_val_accuracy": 75.0},
        "history": [{"epoch": 1, "val_loss": 0.5}],
    }
    monkeypatch.setitem(training.CANDIDATE_MODELS, "session-candidate", candidate)
    monkeypatch.setattr(
        inference.ServiceManager,
        "get_instance",
        lambda: SimpleNamespace(device=torch.device("cpu")),
    )
    image = Image.new("RGB", (8, 8), color=(100, 120, 140))
    image_data = io.BytesIO()
    image.save(image_data, format="PNG")
    image_data.seek(0)

    response = inference.predict_uploaded_image(
        UploadFile(filename="sample.png", file=image_data),
        model_mode="custom",
        custom_run_id="session-candidate",
    )

    assert response["success"] is True
    assert response["model_mode"] == "custom"
    assert response["primary"]["prediction_class"] in {"cat", "dog"}
    assert response["primary"]["metadata"]["run_id"] == "session-candidate"


def test_custom_model_is_unavailable_without_session_registry_entry(monkeypatch):
    monkeypatch.setattr(
        inference.ServiceManager,
        "get_instance",
        lambda: SimpleNamespace(device=torch.device("cpu")),
    )
    image = Image.new("RGB", (8, 8), color=(100, 120, 140))
    image_data = io.BytesIO()
    image.save(image_data, format="PNG")
    image_data.seek(0)
    try:
        inference.predict_uploaded_image(
            UploadFile(filename="sample.png", file=image_data),
            model_mode="custom",
            custom_run_id="not-in-this-session",
        )
    except inference.HTTPException as exc:
        assert exc.status_code == 404
        assert "backend session" in exc.detail
    else:
        raise AssertionError("A candidate not present in RAM must not be loaded from disk.")


def test_custom_model_listing_returns_session_candidates_only(monkeypatch):
    candidate = {
        "metadata": {
            "architecture": "mobilenet_v3_small",
            "class_names": ["cat", "dog"],
            "status": "COMPLETED",
            "dataset_id": "test-dataset",
        },
        "metrics": {"best_val_accuracy": 75.0},
        "history": [{"epoch": 1, "val_loss": 0.5}],
    }
    monkeypatch.setitem(training.CANDIDATE_MODELS, "session-candidate", candidate)

    response = training.list_custom_models()

    assert [model["run_id"] for model in response["models"]] == ["session-candidate"]
    assert response["models"][0]["history"] == candidate["history"]
    assert "checkpoint" not in response["models"][0]
