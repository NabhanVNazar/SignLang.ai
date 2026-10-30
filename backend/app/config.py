"""Server-side configuration for the clip inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MAX_ALLOWED_FRAMES = 3000
MAX_ALLOWED_REQUEST_BYTES = 16 * 1024 * 1024


def _read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be greater than zero")
    maximum = (
        MAX_ALLOWED_FRAMES if name == "SIGNLEARN_MAX_FRAMES" else MAX_ALLOWED_REQUEST_BYTES
    )
    if parsed > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return parsed


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path | None
    api_key: str | None
    max_frames: int = 300
    max_request_bytes: int = 2 * 1024 * 1024
    allow_synthetic_checkpoint: bool = False
    image_checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_frames <= MAX_ALLOWED_FRAMES:
            raise ValueError(
                f"max_frames must be between 1 and {MAX_ALLOWED_FRAMES}"
            )
        if not 1 <= self.max_request_bytes <= MAX_ALLOWED_REQUEST_BYTES:
            raise ValueError(
                "max_request_bytes must be between 1 and "
                f"{MAX_ALLOWED_REQUEST_BYTES}"
            )
        if self.api_key is not None and len(self.api_key) < 32:
            raise ValueError("api_key must contain at least 32 characters")

    @classmethod
    def from_env(cls) -> Settings:
        checkpoint_value = os.getenv("SIGNLEARN_CHECKPOINT_PATH")
        image_checkpoint_value = os.getenv("SIGNLEARN_IMAGE_CHECKPOINT_PATH")
        key_value = os.getenv("SIGNLEARN_API_KEY")
        if key_value and len(key_value) < 32:
            raise ValueError("SIGNLEARN_API_KEY must contain at least 32 characters")
        return cls(
            checkpoint_path=Path(checkpoint_value) if checkpoint_value else None,
            api_key=key_value if key_value else None,
            max_frames=_read_positive_int("SIGNLEARN_MAX_FRAMES", 300),
            max_request_bytes=_read_positive_int(
                "SIGNLEARN_MAX_REQUEST_BYTES", 2 * 1024 * 1024
            ),
            allow_synthetic_checkpoint=_read_bool(
                "SIGNLEARN_ALLOW_SYNTHETIC_CHECKPOINT"
            ),
            image_checkpoint_path=(
                Path(image_checkpoint_value) if image_checkpoint_value else None
            ),
        )
