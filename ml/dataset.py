"""Versioned JSONL schema and validation for sign-language samples."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1
MAX_MANIFEST_LINE_BYTES = 32 * 1024 * 1024
MAX_SAMPLE_FRAMES = 1800
HAND_LANDMARK_COUNT = 21
MAX_POSE_LANDMARK_COUNT = 100
MAX_FACE_LANDMARK_COUNT = 500
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_ANNOTATION_STATUSES = {"pending", "approved", "rejected"}
_FRAME_FIELDS = {
    "timestamp_ms",
    "left_hand",
    "right_hand",
    "body_pose",
    "face",
}
_SAMPLE_FIELDS = {
    "schema_version",
    "sample_id",
    "language_code",
    "signer_id",
    "consent",
    "annotation_status",
    "created_at",
    "glosses",
    "frames",
}

Point3D = tuple[float, float, float]
Landmarks = tuple[Point3D, ...] | None


class DatasetValidationError(ValueError):
    """Raised when a manifest record does not match the dataset schema."""


@dataclass(frozen=True)
class Frame:
    timestamp_ms: int
    left_hand: Landmarks
    right_hand: Landmarks
    body_pose: Landmarks
    face: Landmarks


@dataclass(frozen=True)
class Sample:
    schema_version: int
    sample_id: str
    language_code: str
    signer_id: str
    consent: bool
    annotation_status: str
    created_at: str
    glosses: tuple[str, ...]
    frames: tuple[Frame, ...]
    source_video_key: str | None = None


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{field} must be an object")
    return value


def _require_fields(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    field: str,
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise DatasetValidationError(
            f"{field} is missing required fields: {', '.join(sorted(missing))}"
        )
    if extra:
        raise DatasetValidationError(
            f"{field} has unknown fields: {', '.join(sorted(extra))}"
        )


def _require_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field} must be a UUID string")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise DatasetValidationError(f"{field} must be a UUID string") from exc


def _parse_landmarks(
    value: Any, field: str, maximum_count: int, exact_count: int | None = None
) -> Landmarks:
    if value is None:
        return None
    if not isinstance(value, list):
        raise DatasetValidationError(f"{field} must be an array or null")
    if exact_count is not None and len(value) != exact_count:
        raise DatasetValidationError(
            f"{field} must contain exactly {exact_count} landmarks"
        )
    if not value or len(value) > maximum_count:
        raise DatasetValidationError(
            f"{field} must contain between 1 and {maximum_count} landmarks"
        )

    points: list[Point3D] = []
    for point_index, point in enumerate(value):
        point_field = f"{field}[{point_index}]"
        if not isinstance(point, list) or len(point) != 3:
            raise DatasetValidationError(f"{point_field} must contain x, y, and z")
        coordinates: list[float] = []
        for coordinate_index, coordinate in enumerate(point):
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
            ):
                axis = ("x", "y", "z")[coordinate_index]
                raise DatasetValidationError(
                    f"{point_field}.{axis} must be a finite number"
                )
            coordinates.append(float(coordinate))
        points.append((coordinates[0], coordinates[1], coordinates[2]))
    return tuple(points)


def _parse_frame(value: Any, frame_index: int) -> Frame:
    field = f"frames[{frame_index}]"
    frame = _require_mapping(value, field)
    _require_fields(frame, _FRAME_FIELDS, set(), field)

    timestamp = frame["timestamp_ms"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise DatasetValidationError(f"{field}.timestamp_ms must be a non-negative integer")

    return Frame(
        timestamp_ms=timestamp,
        left_hand=_parse_landmarks(
            frame["left_hand"],
            f"{field}.left_hand",
            HAND_LANDMARK_COUNT,
            HAND_LANDMARK_COUNT,
        ),
        right_hand=_parse_landmarks(
            frame["right_hand"],
            f"{field}.right_hand",
            HAND_LANDMARK_COUNT,
            HAND_LANDMARK_COUNT,
        ),
        body_pose=_parse_landmarks(
            frame["body_pose"], f"{field}.body_pose", MAX_POSE_LANDMARK_COUNT
        ),
        face=_parse_landmarks(frame["face"], f"{field}.face", MAX_FACE_LANDMARK_COUNT),
    )


def validate_sample(value: Any) -> Sample:
    """Validate a decoded JSON object and return its typed representation."""
    data = _require_mapping(value, "sample")
    _require_fields(data, _SAMPLE_FIELDS, {"source_video_key"}, "sample")

    version = data["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise DatasetValidationError(
            f"schema_version must be {SCHEMA_VERSION}; received {version!r}"
        )

    language_code = data["language_code"]
    if not isinstance(language_code, str) or not _LANGUAGE_CODE.fullmatch(
        language_code
    ):
        raise DatasetValidationError(
            "language_code must use lowercase language and optional uppercase region"
        )

    consent = data["consent"]
    if not isinstance(consent, bool):
        raise DatasetValidationError("consent must be a boolean")

    annotation_status = data["annotation_status"]
    if (
        not isinstance(annotation_status, str)
        or annotation_status not in _ANNOTATION_STATUSES
    ):
        raise DatasetValidationError(
            "annotation_status must be pending, approved, or rejected"
        )

    created_at = data["created_at"]
    if not isinstance(created_at, str):
        raise DatasetValidationError("created_at must be an ISO 8601 timestamp")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetValidationError(
            "created_at must be an ISO 8601 timestamp"
        ) from exc
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() is None:
        raise DatasetValidationError("created_at must include a timezone")

    glosses = data["glosses"]
    if (
        not isinstance(glosses, list)
        or not glosses
        or any(
            not isinstance(gloss, str)
            or not gloss.strip()
            or gloss != gloss.strip()
            or len(gloss) > 64
            or any(ord(char) < 32 for char in gloss)
            for gloss in glosses
        )
    ):
        raise DatasetValidationError(
            "glosses must be a non-empty array of non-empty strings up to 64 characters"
        )

    raw_frames = data["frames"]
    if (
        not isinstance(raw_frames, list)
        or not raw_frames
        or len(raw_frames) > MAX_SAMPLE_FRAMES
    ):
        raise DatasetValidationError(
            f"frames must contain between 1 and {MAX_SAMPLE_FRAMES} frames"
        )
    frames = tuple(_parse_frame(frame, index) for index, frame in enumerate(raw_frames))
    if any(
        current.timestamp_ms <= previous.timestamp_ms
        for previous, current in zip(frames, frames[1:])
    ):
        raise DatasetValidationError("frame timestamps must be strictly increasing")

    source_video_key = data.get("source_video_key")
    if source_video_key is not None:
        if (
            not isinstance(source_video_key, str)
            or not source_video_key
            or len(source_video_key) > 1024
            or "\\" in source_video_key
            or "://" in source_video_key
            or "?" in source_video_key
            or "#" in source_video_key
            or source_video_key.startswith("/")
            or any(part in {"", ".", ".."} for part in source_video_key.split("/"))
        ):
            raise DatasetValidationError(
                "source_video_key must be a relative private-storage object key"
            )

    return Sample(
        schema_version=SCHEMA_VERSION,
        sample_id=_require_uuid(data["sample_id"], "sample_id"),
        language_code=language_code,
        signer_id=_require_uuid(data["signer_id"], "signer_id"),
        consent=consent,
        annotation_status=annotation_status,
        created_at=created_at,
        glosses=tuple(glosses),
        frames=frames,
        source_video_key=source_video_key,
    )


def load_manifest(path: str | Path) -> Iterator[Sample]:
    """Yield validated samples from UTF-8 JSON Lines, reporting invalid line numbers."""
    manifest_path = Path(path)
    with manifest_path.open("rb") as stream:
        line_number = 0
        while True:
            raw_line = stream.readline(MAX_MANIFEST_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            if len(raw_line) > MAX_MANIFEST_LINE_BYTES:
                raise DatasetValidationError(
                    f"{manifest_path}:{line_number}: line exceeds "
                    f"{MAX_MANIFEST_LINE_BYTES} bytes"
                )
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                yield validate_sample(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, DatasetValidationError) as exc:
                raise DatasetValidationError(
                    f"{manifest_path}:{line_number}: {exc}"
                ) from exc


def write_manifest(samples: Iterable[Sample], path: str | Path) -> None:
    """Write validated samples as UTF-8 JSON Lines."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            encoded = json.dumps(asdict(sample), ensure_ascii=False, allow_nan=False)
            validate_sample(json.loads(encoded))
            stream.write(encoded)
            stream.write("\n")
