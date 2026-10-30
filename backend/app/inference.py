"""Safe, process-local inference service for configured CTC checkpoints."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import pickle
import re
import threading
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from ml.features import build_hand_features
from ml.models.sequence_model import HandLandmarkBiLSTM
from ml.models.vocabulary import GlossVocabulary
from ml.training.train import CHECKPOINT_FORMAT_VERSION

logger = logging.getLogger(__name__)
_SYNTHETIC_MARKER = "SYNTHETIC SMOKE TEST ONLY."
_DATASET_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


class InferenceNotReadyError(RuntimeError):
    """Raised when a prediction is requested without a valid model."""


class CheckpointValidationError(ValueError):
    """Raised when checkpoint metadata does not match the inference contract."""


class InferenceService:
    """Loads a trusted local checkpoint once and serializes model inference."""

    def __init__(
        self,
        checkpoint_path: Path | None,
        allow_synthetic_checkpoint: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._allow_synthetic_checkpoint = allow_synthetic_checkpoint
        self._model: HandLandmarkBiLSTM | None = None
        self._vocabulary: GlossVocabulary | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint_id: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None and self._vocabulary is not None

    def load(self) -> bool:
        if self._model is not None and self._vocabulary is not None:
            return True
        if self._checkpoint_path is None:
            return False
        if not self._checkpoint_path.is_file():
            logger.warning("configured inference checkpoint is unavailable")
            return False

        try:
            checkpoint_bytes = self._checkpoint_path.read_bytes()
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise CheckpointValidationError("checkpoint root must be a mapping")
            vocabulary = GlossVocabulary.from_dict(checkpoint.get("vocabulary"))
            self._validate_checkpoint(checkpoint, vocabulary)
            data_description = checkpoint.get("data_description")
            if (
                isinstance(data_description, str)
                and data_description.startswith(_SYNTHETIC_MARKER)
                and not self._allow_synthetic_checkpoint
            ):
                raise CheckpointValidationError(
                    "synthetic checkpoints are disabled for inference"
                )
            model_config = checkpoint["model_config"]
            state_dict = checkpoint.get("model_state_dict")
            if not isinstance(state_dict, dict):
                raise CheckpointValidationError("checkpoint model weights are missing")
            model = HandLandmarkBiLSTM(
                class_count=vocabulary.class_count,
                hidden_size=model_config["hidden_size"],
                dropout=float(model_config["dropout"]),
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(self._device)
            model.eval()
            checkpoint_digest = hashlib.sha256(checkpoint_bytes)
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
                "configured inference checkpoint failed validation; predictions disabled"
            )
            return False

        self._model = model
        self._vocabulary = vocabulary
        self._checkpoint_id = f"sha256:{checkpoint_digest.hexdigest()}"
        return True

    @staticmethod
    def _validate_checkpoint(
        checkpoint: dict[str, object], vocabulary: GlossVocabulary
    ) -> None:
        checkpoint_version = checkpoint.get("checkpoint_format_version")
        if (
            isinstance(checkpoint_version, bool)
            or checkpoint_version != CHECKPOINT_FORMAT_VERSION
        ):
            raise CheckpointValidationError("unsupported checkpoint format")
        if (
            isinstance(checkpoint.get("preprocessing_schema_version"), bool)
            or checkpoint.get("preprocessing_schema_version") != 1
        ):
            raise CheckpointValidationError("unsupported preprocessing schema")
        if not isinstance(checkpoint.get("data_description"), str):
            raise CheckpointValidationError("checkpoint data description is missing")
        dataset_version = checkpoint.get("dataset_version")
        if (
            not isinstance(dataset_version, str)
            or not _DATASET_FINGERPRINT.fullmatch(dataset_version)
        ):
            raise CheckpointValidationError("checkpoint dataset fingerprint is invalid")

        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise CheckpointValidationError("checkpoint model configuration is missing")
        required_config = {
            "architecture": "bidirectional_lstm_ctc",
            "num_layers": 1,
            "bidirectional": True,
            "blank_id": 0,
            "input_size": 128,
            "feature_shape": [2, 21, 3],
            "missing_hand_mask_channels": 2,
        }
        for key, expected in required_config.items():
            actual = model_config.get(key)
            if (
                actual != expected
                or type(expected) is int
                and (isinstance(actual, bool) or not isinstance(actual, int))
                or type(expected) is bool
                and not isinstance(actual, bool)
            ):
                raise CheckpointValidationError(
                    f"checkpoint model configuration has an unsupported {key}"
                )
        class_count = model_config.get("class_count")
        if (
            isinstance(class_count, bool)
            or not isinstance(class_count, int)
            or class_count > 10000
            or class_count != vocabulary.class_count
        ):
            raise CheckpointValidationError(
                "checkpoint model class count does not match its vocabulary"
            )
        hidden_size = model_config.get("hidden_size")
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or not 1 <= hidden_size <= 1024
        ):
            raise CheckpointValidationError("checkpoint hidden size is invalid")
        dropout = model_config.get("dropout")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise CheckpointValidationError("checkpoint dropout is invalid")

    def predict(
        self,
        frames: Sequence[
            tuple[
                list[list[float | int]] | None,
                list[list[float | int]] | None,
            ]
        ],
    ) -> tuple[list[str], str]:
        if not self.ready:
            raise InferenceNotReadyError
        model = self._model
        vocabulary = self._vocabulary
        if model is None or vocabulary is None:
            raise InferenceNotReadyError

        features, observed = build_hand_features(frames)
        model_input = torch.from_numpy(features).unsqueeze(0).to(self._device)
        observed_input = torch.from_numpy(observed).unsqueeze(0).to(self._device)
        lengths: Tensor = torch.tensor([len(features)], dtype=torch.int64)
        with self._lock, torch.inference_mode():
            logits = model(model_input, observed_input, lengths)
            if logits.shape != (1, len(frames), vocabulary.class_count):
                raise RuntimeError("model output does not match the checkpoint vocabulary")
            if not torch.isfinite(logits).all():
                raise RuntimeError("model produced non-finite logits")
            frame_class_ids = logits[0].argmax(dim=-1).tolist()
            glosses = vocabulary.decode_ctc(frame_class_ids)
        return glosses, " ".join(glosses)

    @property
    def checkpoint_id(self) -> str | None:
        return self._checkpoint_id
