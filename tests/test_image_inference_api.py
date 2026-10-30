import base64
import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.image_inference import (
    ImageClassifierInferenceService,
    ImageInferenceNotReadyError,
)
from backend.app.main import create_app
from ml.models.image_classifier import StaticSignCNN


API_KEY = "test-image-api-key-that-is-at-least-32-chars"


def create_image_checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = StaticSignCNN(class_count=2, dropout=0.2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias[1] = 5.0
    checkpoint = {
        "checkpoint_format_version": 1,
        "model_config": {
            "architecture": "StaticSignCNN",
            "class_count": 2,
            "dropout": 0.2,
        },
        "preprocessing": {
            "schema_version": 1,
            "color_space": "RGB",
            "image_size": 64,
            "resize": "PIL bilinear",
            "tensor_layout": "CHW",
            "pixel_range": [0.0, 1.0],
            "normalization": "uint8 pixel values divided by 255",
        },
        "class_to_idx": {"A": 0, "B": 1},
        "dataset_version": "sha256:" + ("b" * 64),
        "dataset_schema_version": 1,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    return checkpoint


def create_settings(checkpoint_path=None, *, api_key=API_KEY):
    return Settings(
        checkpoint_path=None,
        api_key=api_key,
        image_checkpoint_path=checkpoint_path,
    )


def png_image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (40, 32), color=(90, 130, 170)).save(stream, format="PNG")
    return stream.getvalue()


def image_request(image_bytes=None, *, schema_version=1):
    if image_bytes is None:
        image_bytes = png_image_bytes()
    return {
        "preprocessing_schema_version": schema_version,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
    }


def create_client(tmp_path, *, api_key=API_KEY):
    checkpoint_path = tmp_path / "image-checkpoint.pt"
    create_image_checkpoint(checkpoint_path)
    service = ImageClassifierInferenceService(checkpoint_path)
    app = create_app(
        create_settings(checkpoint_path, api_key=api_key),
        image_inference_service=service,
    )
    return TestClient(app), service


def test_image_checkpoint_loads_safely_and_predicts_saved_label(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = create_image_checkpoint(checkpoint_path)
    service = ImageClassifierInferenceService(checkpoint_path)

    assert service.load() is True
    label, class_index = service.predict(png_image_bytes())

    assert label == "B"
    assert class_index == checkpoint["class_to_idx"]["B"]
    assert service.checkpoint_id.startswith("sha256:")


def test_wide_camera_image_is_center_cropped_without_stretching(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_image_checkpoint(checkpoint_path)
    service = ImageClassifierInferenceService(checkpoint_path)
    assert service.load() is True

    image = Image.new("RGB", (200, 100), color=(255, 0, 0))
    for x in range(50, 150):
        for y in range(100):
            image.putpixel((x, y), (0, 255, 0))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    tensor = service._preprocess_image(stream.getvalue())

    assert tuple(tensor.shape) == (1, 3, 64, 64)
    center = tensor[:, :, 8:-8, 8:-8]
    assert torch.all(center[:, 0] < 0.01)
    assert torch.all(center[:, 1] > 0.99)
    assert torch.all(center[:, 2] < 0.01)


def test_missing_image_checkpoint_reports_not_ready(tmp_path):
    service = ImageClassifierInferenceService(tmp_path / "missing.pt")

    assert service.load() is False
    assert service.ready is False
    with pytest.raises(ImageInferenceNotReadyError):
        service.predict(png_image_bytes())


def test_image_endpoint_returns_saved_class_without_confidence(tmp_path):
    client, service = create_client(tmp_path)
    with client:
        response = client.post(
            "/predict/image",
            json=image_request(),
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "label": "B",
        "class_index": 1,
        "model_version": service.checkpoint_id,
        "preprocessing_schema_version": 1,
        "status": "prototype_image_classification",
    }
    assert "confidence" not in response.json()
    assert health.json()["image_classifier_status"] == "ready"
    assert health.json()["image_classifier_prediction_enabled"] == "true"


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"preprocessing_schema_version": 1, "image_base64": ""}, 422),
        ({"preprocessing_schema_version": 1, "image_base64": "not-base64!"}, 422),
        ({"preprocessing_schema_version": 2, "image_base64": "YWJj"}, 422),
        ({"preprocessing_schema_version": 1, "image_base64": "YWJj"}, 422),
    ],
)
def test_image_endpoint_rejects_malformed_requests(
    tmp_path, payload, expected_status
):
    client, _ = create_client(tmp_path)
    with client:
        response = client.post(
            "/predict/image",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    assert response.status_code == expected_status
    assert "checkpoint" not in response.text


def test_image_endpoint_returns_not_ready_without_checkpoint(tmp_path):
    app = create_app(create_settings())
    with TestClient(app) as client:
        response = client.post(
            "/predict/image",
            json=image_request(),
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        health = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "No valid image-classifier model is configured."
    )
    assert health.json()["image_classifier_status"] == "model_not_ready"


def test_image_endpoint_requires_api_key(tmp_path):
    client, _ = create_client(tmp_path)
    with client:
        response = client.post("/predict/image", json=image_request())

    assert response.status_code == 401


def test_image_endpoint_obeys_request_body_limit(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_image_checkpoint(checkpoint_path)
    service = ImageClassifierInferenceService(checkpoint_path)
    settings = Settings(
        checkpoint_path=None,
        api_key=API_KEY,
        max_request_bytes=128,
        image_checkpoint_path=checkpoint_path,
    )
    client = TestClient(
        create_app(settings, image_inference_service=service)
    )
    with client:
        response = client.post(
            "/predict/image",
            content=json.dumps(image_request()),
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 413


def test_invalid_image_checkpoint_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "invalid-checkpoint.pt"
    checkpoint = create_image_checkpoint(checkpoint_path)
    checkpoint["preprocessing"]["normalization"] = "different preprocessing"
    torch.save(checkpoint, checkpoint_path)

    service = ImageClassifierInferenceService(checkpoint_path)

    assert service.load() is False
    assert service.ready is False
