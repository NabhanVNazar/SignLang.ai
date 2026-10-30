"""Safe inference service for the isolated-image character classifier."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import pickle
import re
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor

from ml.image_labels import IMAGE_CLASS_TO_IDX
from ml.models.image_classifier import StaticSignCNN

logger = logging.getLogger(__name__)
_DATASET_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_IMAGE_BYTES = 1_000_000
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 16_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageInferenceNotReadyError(RuntimeError):
    """Raised when an image prediction is requested without a valid model."""


class ImageCheckpointValidationError(ValueError):
    """Raised when image checkpoint metadata does not match its contract."""


class ImageClassifierInferenceService:
    """Loads a trusted local image checkpoint once and serializes predictions."""

    def __init__(self, checkpoint_path: Path | None) -> None:
        self._checkpoint_path = checkpoint_path
        self._model: StaticSignCNN | None = None
        self._class_to_idx: dict[str, int] | None = None
        self._image_size: int | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint_id: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return (
            self._model is not None
            and self._class_to_idx is not None
            and self._image_size is not None
        )

    @property
    def checkpoint_id(self) -> str | None:
        return self._checkpoint_id

    def load(self) -> bool:
        if self.ready:
            return True
        if self._checkpoint_path is None:
            return False
        if not self._checkpoint_path.is_file():
            logger.warning("configured image-classifier checkpoint is unavailable")
            return False

        try:
            checkpoint_bytes = self._checkpoint_path.read_bytes()
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise ImageCheckpointValidationError(
                    "checkpoint root must be a mapping"
                )
            self._validate_checkpoint(checkpoint)
            model_config = checkpoint["model_config"]
            preprocessing = checkpoint["preprocessing"]
            class_to_idx = checkpoint["class_to_idx"]
            state_dict = checkpoint.get("model_state_dict")
            if not isinstance(model_config, dict):
                raise ImageCheckpointValidationError(
                    "checkpoint model configuration is missing"
                )
            if not isinstance(preprocessing, dict):
                raise ImageCheckpointValidationError(
                    "checkpoint preprocessing configuration is missing"
                )
            if not isinstance(class_to_idx, dict) or not isinstance(state_dict, dict):
                raise ImageCheckpointValidationError(
                    "checkpoint classes or model weights are missing"
                )

            model = StaticSignCNN(
                class_count=model_config["class_count"],
                dropout=float(model_config["dropout"]),
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(self._device)
            model.eval()
            checkpoint_id = f"sha256:{hashlib.sha256(checkpoint_bytes).hexdigest()}"
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            EOFError,
            pickle.UnpicklingError,
        ):
            logger.warning(
                "configured image-classifier checkpoint failed validation; "
                "image predictions disabled"
            )
            return False

        self._model = model
        self._class_to_idx = dict(class_to_idx)
        self._image_size = preprocessing["image_size"]
        self._checkpoint_id = checkpoint_id
        return True

    @staticmethod
    def _validate_checkpoint(checkpoint: dict[str, object]) -> None:
        checkpoint_format_version = checkpoint.get("checkpoint_format_version")
        if (
            isinstance(checkpoint_format_version, bool)
            or checkpoint_format_version != 1
        ):
            raise ImageCheckpointValidationError("unsupported checkpoint format")
        dataset_schema_version = checkpoint.get("dataset_schema_version")
        if (
            isinstance(dataset_schema_version, bool)
            or dataset_schema_version != 1
        ):
            raise ImageCheckpointValidationError("unsupported dataset schema")
        dataset_version = checkpoint.get("dataset_version")
        if (
            not isinstance(dataset_version, str)
            or not _DATASET_FINGERPRINT.fullmatch(dataset_version)
        ):
            raise ImageCheckpointValidationError(
                "checkpoint dataset fingerprint is invalid"
            )

        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ImageCheckpointValidationError(
                "checkpoint model configuration is missing"
            )
        if model_config.get("architecture") != "StaticSignCNN":
            raise ImageCheckpointValidationError(
                "checkpoint architecture is unsupported"
            )
        class_count = model_config.get("class_count")
        if (
            isinstance(class_count, bool)
            or not isinstance(class_count, int)
            or not 2 <= class_count <= 1000
        ):
            raise ImageCheckpointValidationError("checkpoint class count is invalid")
        dropout = model_config.get("dropout")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ImageCheckpointValidationError("checkpoint dropout is invalid")

        preprocessing = checkpoint.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ImageCheckpointValidationError(
                "checkpoint preprocessing configuration is missing"
            )
        preprocessing_schema_version = preprocessing.get("schema_version")
        if (
            isinstance(preprocessing_schema_version, bool)
            or preprocessing_schema_version != 1
        ):
            raise ImageCheckpointValidationError(
                "unsupported preprocessing schema"
            )
        expected_preprocessing = {
            "color_space": "RGB",
            "resize": "PIL bilinear",
            "tensor_layout": "CHW",
            "pixel_range": [0.0, 1.0],
            "normalization": "uint8 pixel values divided by 255",
        }
        for key, expected in expected_preprocessing.items():
            if preprocessing.get(key) != expected:
                raise ImageCheckpointValidationError(
                    f"checkpoint preprocessing {key} is unsupported"
                )
        image_size = preprocessing.get("image_size")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or not 16 <= image_size <= 512
        ):
            raise ImageCheckpointValidationError(
                "checkpoint image size is invalid"
            )

        class_to_idx = checkpoint.get("class_to_idx")
        if (
            class_count != len(IMAGE_CLASS_TO_IDX)
            or not isinstance(class_to_idx, dict)
            or class_to_idx != IMAGE_CLASS_TO_IDX
        ):
            raise ImageCheckpointValidationError(
                "checkpoint must use the canonical 36-class mapping"
            )
        indices = []
        for label, index in class_to_idx.items():
            if (
                not isinstance(label, str)
                or not label
                or len(label) > 64
                or isinstance(index, bool)
                or not isinstance(index, int)
            ):
                raise ImageCheckpointValidationError(
                    "checkpoint class mapping is invalid"
                )
            indices.append(index)
        if sorted(indices) != list(range(class_count)):
            raise ImageCheckpointValidationError(
                "checkpoint class indices must be unique and contiguous"
            )

    def _preprocess_image(self, image_bytes: bytes) -> Tensor:
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("image size is outside the accepted range")

        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                if source.format not in SUPPORTED_IMAGE_FORMATS:
                    raise ValueError("unsupported image format")
                if (
                    source.width > MAX_IMAGE_DIMENSION
                    or source.height > MAX_IMAGE_DIMENSION
                    or source.width * source.height > MAX_IMAGE_PIXELS
                ):
                    raise ValueError("image dimensions exceed the accepted limit")
                source.verify()
            with Image.open(io.BytesIO(image_bytes)) as source:
                image_size = self._image_size
                if image_size is None:
                    raise ImageInferenceNotReadyError
                resized = ImageOps.fit(
                    source.convert("RGB"),
                    (image_size, image_size),
                    method=Image.Resampling.BILINEAR,
                    centering=(0.5, 0.5),
                )
                pixels = np.asarray(resized, dtype=np.uint8).copy()
        except (
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as exc:
            raise ValueError("image could not be decoded") from exc

        return (
            torch.from_numpy(pixels)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )

    def predict(self, image_bytes: bytes) -> tuple[str, int]:
        if not self.ready:
            raise ImageInferenceNotReadyError
        tensor = self._preprocess_image(image_bytes).to(self._device)
        model = self._model
        class_to_idx = self._class_to_idx
        if model is None or class_to_idx is None:
            raise ImageInferenceNotReadyError
        with self._lock, torch.inference_mode():
            predicted_index = int(model(tensor).argmax(dim=1).item())
        label = next(
            (name for name, index in class_to_idx.items() if index == predicted_index),
            None,
        )
        if label is None:
            raise ImageInferenceNotReadyError
        return label, predicted_index
