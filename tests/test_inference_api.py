import math
import json

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.inference import InferenceNotReadyError, InferenceService
from backend.app.main import ClipPredictionRequest, create_app
from ml.models.sequence_model import HandLandmarkBiLSTM
from ml.models.vocabulary import GlossVocabulary


API_KEY = "test-inference-api-key-that-is-at-least-32-chars"


def create_checkpoint(path, *, description="Test checkpoint"):
    path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary = GlossVocabulary.from_sequences([["HELLO"]])
    model = HandLandmarkBiLSTM(
        class_count=vocabulary.class_count,
        hidden_size=4,
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[1] = 5.0
    checkpoint = {
        "checkpoint_format_version": 1,
        "dataset_version": "sha256:" + ("a" * 64),
        "preprocessing_schema_version": 1,
        "model_config": {
            "architecture": "bidirectional_lstm_ctc",
            "num_layers": 1,
            "bidirectional": True,
            "class_count": vocabulary.class_count,
            "blank_id": 0,
            "hidden_size": 4,
            "dropout": 0.0,
            "input_size": 128,
            "feature_shape": [2, 21, 3],
            "missing_hand_mask_channels": 2,
        },
        "vocabulary": vocabulary.to_dict(),
        "training_settings": {"seed": 1},
        "data_description": description,
        "best_epoch": 1,
        "best_validation_loss": 1.0,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    return checkpoint


def create_settings(
    checkpoint_path=None,
    *,
    max_frames=300,
    max_request_bytes=2 * 1024 * 1024,
    allow_synthetic_checkpoint=False,
):
    return Settings(
        checkpoint_path=checkpoint_path,
        api_key=API_KEY,
        max_frames=max_frames,
        max_request_bytes=max_request_bytes,
        allow_synthetic_checkpoint=allow_synthetic_checkpoint,
    )


def observed_hand(offset=0):
    return [[float(index + offset), float(index + 1), -0.5] for index in range(21)]


def valid_request(frame_count=2):
    return {
        "preprocessing_schema_version": 1,
        "frames": [
            {
                "timestamp_ms": index * 40,
                "left_hand": observed_hand(),
                "right_hand": None,
            }
            for index in range(frame_count)
        ],
    }


def api_client(tmp_path, *, max_frames=300, max_request_bytes=2 * 1024 * 1024):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_checkpoint(checkpoint_path)
    service = InferenceService(checkpoint_path)
    app = create_app(
        create_settings(
            checkpoint_path,
            max_frames=max_frames,
            max_request_bytes=max_request_bytes,
        ),
        inference_service=service,
    )
    return TestClient(app), service


def test_valid_landmark_request_validation_and_explicit_missing_hand():
    request = ClipPredictionRequest.model_validate(valid_request())

    assert len(request.frames) == 2
    assert request.frames[0].left_hand == observed_hand()
    assert request.frames[0].right_hand is None


def test_missing_hand_fields_and_clips_with_no_observed_hands_are_rejected():
    payload = valid_request(frame_count=1)
    del payload["frames"][0]["right_hand"]
    with pytest.raises(ValueError):
        ClipPredictionRequest.model_validate(payload)

    payload = valid_request(frame_count=1)
    payload["frames"][0]["left_hand"] = None
    with pytest.raises(ValueError, match="at least one observed hand"):
        ClipPredictionRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"preprocessing_schema_version": 1, "frames": []},
            "at least 1 item",
        ),
        (
            {
                "preprocessing_schema_version": 1,
                "frames": [
                    {
                        "timestamp_ms": 0,
                        "left_hand": [[0.0, 0.0, 0.0]] * 20,
                        "right_hand": None,
                    }
                ],
            },
            "exactly 21 landmarks",
        ),
    ],
)
def test_invalid_clip_and_landmark_count_are_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        ClipPredictionRequest.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_are_rejected(value):
    payload = valid_request(frame_count=1)
    payload["frames"][0]["left_hand"][0][0] = value

    with pytest.raises(ValueError, match="finite"):
        ClipPredictionRequest.model_validate(payload)


def test_unsupported_schema_and_invalid_timestamps_are_rejected():
    payload = valid_request(frame_count=1)
    payload["preprocessing_schema_version"] = 2
    request = ClipPredictionRequest.model_validate(payload)
    assert request.preprocessing_schema_version == 2

    payload = valid_request(frame_count=2)
    payload["frames"][1]["timestamp_ms"] = 0
    with pytest.raises(ValueError, match="strictly increasing"):
        ClipPredictionRequest.model_validate(payload)


def test_request_with_non_numeric_landmarks_is_rejected():
    payload = valid_request(frame_count=1)
    payload["frames"][0]["left_hand"][0][0] = True

    with pytest.raises(ValueError):
        ClipPredictionRequest.model_validate(payload)


def test_checkpoint_missing_returns_service_not_ready(tmp_path):
    service = InferenceService(tmp_path / "missing.pt")

    assert service.load() is False
    assert service.ready is False
    with pytest.raises(InferenceNotReadyError):
        service.predict([(observed_hand(), None)])


def test_no_checkpoint_api_returns_useful_503():
    app = create_app(create_settings())
    with TestClient(app) as client:
        response = client.post(
            "/predict/clip",
            json=valid_request(),
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "No valid inference model is configured."


def test_synthetic_checkpoint_is_rejected_unless_explicitly_enabled(tmp_path):
    checkpoint_path = tmp_path / "synthetic.pt"
    create_checkpoint(
        checkpoint_path,
        description="SYNTHETIC SMOKE TEST ONLY. Not real data.",
    )

    disabled_service = InferenceService(checkpoint_path)
    enabled_service = InferenceService(checkpoint_path, allow_synthetic_checkpoint=True)

    assert disabled_service.load() is False
    assert enabled_service.load() is True


def test_ctc_decoder_collapses_repeats_and_removes_blank():
    vocabulary = GlossVocabulary.from_sequences([["HELLO", "THANKS"]])

    assert vocabulary.decode_ctc([1, 1, 0, 1, 2, 2, 0]) == [
        "HELLO",
        "HELLO",
        "THANKS",
    ]


def test_model_output_decodes_using_saved_checkpoint_vocabulary(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = create_checkpoint(checkpoint_path)
    service = InferenceService(checkpoint_path)

    assert service.load() is True
    glosses, sequence = service.predict(
        [(observed_hand(), None), (observed_hand(offset=2), None)]
    )

    assert checkpoint["vocabulary"]["token_to_id"] == {"HELLO": 1}
    assert glosses == ["HELLO"]
    assert sequence == "HELLO"
    assert service.checkpoint_id.startswith("sha256:")


def test_api_response_and_model_not_ready_status(tmp_path):
    client, service = api_client(tmp_path)
    with client:
        response = client.post(
            "/predict/clip",
            json=valid_request(),
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        health = client.get("/health")

    assert service.ready is True
    assert health.json() == {
        "status": "ready",
        "service": "clip_inference",
        "image_classifier_status": "model_not_ready",
        "image_classifier_prediction_enabled": "false",
    }
    assert response.status_code == 200
    assert response.json() == {
        "glosses": ["HELLO"],
        "sequence": "HELLO",
        "model_version": service.checkpoint_id,
        "preprocessing_schema_version": 1,
        "status": "prototype_prediction",
    }
    assert "confidence" not in response.json()


def test_api_rejects_bad_landmarks_without_echoing_input(tmp_path):
    client, _ = api_client(tmp_path)
    payload = valid_request(frame_count=1)
    payload["frames"][0]["left_hand"] = [[0.0, 0.0, 0.0]] * 20
    with client:
        response = client.post(
            "/predict/clip",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    assert response.status_code == 422
    assert "Invalid inference request." == response.json()["detail"]
    assert "left_hand" not in response.text


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_api_rejects_non_finite_json_values(tmp_path, value):
    client, _ = api_client(tmp_path)
    payload = valid_request(frame_count=1)
    content = json.dumps(payload).replace(
        "[0.0, 1.0, -0.5]", f"[{value}, 1.0, -0.5]", 1
    )
    with client:
        response = client.post(
            "/predict/clip",
            content=content,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid inference request."


def test_api_rejects_unsupported_schema_and_frame_limit(tmp_path):
    client, _ = api_client(tmp_path, max_frames=2)
    unsupported = valid_request(frame_count=1)
    unsupported["preprocessing_schema_version"] = 2
    too_many_frames = valid_request(frame_count=3)
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with client:
        unsupported_response = client.post(
            "/predict/clip", json=unsupported, headers=headers
        )
        frame_response = client.post(
            "/predict/clip", json=too_many_frames, headers=headers
        )

    assert unsupported_response.status_code == 422
    assert "Unsupported preprocessing schema" in unsupported_response.json()["detail"]
    assert frame_response.status_code == 413
    assert "frame limit" in frame_response.json()["detail"]


def test_api_limits_body_size_and_requires_api_key(tmp_path):
    limited_client, _ = api_client(tmp_path, max_request_bytes=128)
    auth_client, _ = api_client(tmp_path / "auth")
    with limited_client, auth_client:
        oversized = limited_client.post(
            "/predict/clip",
            content=b"x" * 129,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        unauthorized = auth_client.post("/predict/clip", json=valid_request())

    assert oversized.status_code == 413
    assert unauthorized.status_code == 401


def test_api_hides_checkpoint_path_and_rejects_invalid_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "invalid-checkpoint.pt"
    checkpoint = create_checkpoint(checkpoint_path)
    checkpoint["model_config"]["architecture"] = "other"
    torch.save(checkpoint, checkpoint_path)
    service = InferenceService(checkpoint_path)
    app = create_app(create_settings(checkpoint_path), inference_service=service)
    with TestClient(app) as client:
        response = client.post(
            "/predict/clip",
            json=valid_request(),
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    assert response.status_code == 503
    assert str(checkpoint_path) not in response.text
