"""Safe loading and validation of the hand-landmark NPZ data contract."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.dataset import HAND_LANDMARK_COUNT, SCHEMA_VERSION


ARCHIVE_KEYS = {
    "schema_version",
    "features",
    "hand_observed",
    "offsets",
    "sample_ids",
    "signer_ids",
    "language_codes",
    "gloss_sequences",
}


@dataclass(frozen=True)
class SequenceRecord:
    sample_id: str
    signer_id: str
    language_code: str
    glosses: tuple[str, ...]
    features: np.ndarray
    hand_observed: np.ndarray


@dataclass(frozen=True)
class LandmarkDataset:
    schema_version: int
    dataset_version: str
    records: tuple[SequenceRecord, ...]

    def by_sample_id(self) -> dict[str, SequenceRecord]:
        return {record.sample_id: record for record in self.records}


def _read_unicode_vector(array: np.ndarray, name: str, expected_size: int) -> list[str]:
    if array.ndim != 1 or len(array) != expected_size:
        raise ValueError(f"{name} must be a one-dimensional array of length {expected_size}")
    if array.dtype.kind != "U":
        raise ValueError(f"{name} must use a Unicode string dtype")
    values = [str(value) for value in array.tolist()]
    if any(not value for value in values):
        raise ValueError(f"{name} values must not be empty")
    return values


def load_npz_dataset(path: str | Path) -> LandmarkDataset:
    """Load only the documented NPZ arrays; never unpickle object arrays."""
    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        names = set(archive.files)
        if names != ARCHIVE_KEYS:
            missing = sorted(ARCHIVE_KEYS - names)
            extra = sorted(names - ARCHIVE_KEYS)
            raise ValueError(
                f"NPZ keys do not match preprocessing contract; "
                f"missing={missing}, extra={extra}"
            )
        version = archive["schema_version"]
        if version.shape != (1,) or version.dtype != np.dtype("int16"):
            raise ValueError("schema_version must be int16 with shape (1,)")
        if int(version[0]) != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataset version {int(version[0])}; expected {SCHEMA_VERSION}"
            )
        features = archive["features"]
        observed = archive["hand_observed"]
        offsets = archive["offsets"]
        if features.ndim != 4 or tuple(features.shape[1:]) != (
            2,
            HAND_LANDMARK_COUNT,
            3,
        ):
            raise ValueError("features must have shape (frames, 2, 21, 3)")
        if features.dtype != np.dtype("float32"):
            raise ValueError("features must have dtype float32")
        if len(features) == 0 or not np.isfinite(features).all():
            raise ValueError("features must contain finite values and at least one frame")
        if observed.shape != (len(features), 2) or observed.dtype != np.dtype("bool"):
            raise ValueError("hand_observed must have shape (frames, 2) and dtype bool")
        if np.any(features[~observed] != 0):
            raise ValueError("missing-hand feature slots must be zero-filled")
        if offsets.ndim != 1 or offsets.dtype != np.dtype("int64") or len(offsets) < 2:
            raise ValueError("offsets must be int64 with one start and end per sample")
        if offsets[0] != 0 or offsets[-1] != len(features):
            raise ValueError("offsets must start at zero and end at the frame count")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("offsets must increase strictly for non-empty samples")

        sample_count = len(offsets) - 1
        sample_ids = _read_unicode_vector(archive["sample_ids"], "sample_ids", sample_count)
        signer_ids = _read_unicode_vector(archive["signer_ids"], "signer_ids", sample_count)
        language_codes = _read_unicode_vector(
            archive["language_codes"], "language_codes", sample_count
        )
        gloss_sequences = _read_unicode_vector(
            archive["gloss_sequences"], "gloss_sequences", sample_count
        )
        if len(set(sample_ids)) != sample_count:
            raise ValueError("sample_ids must be unique")

        records: list[SequenceRecord] = []
        for index in range(sample_count):
            try:
                glosses: Any = json.loads(gloss_sequences[index])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"gloss_sequences[{index}] must contain a JSON array"
                ) from exc
            if (
                not isinstance(glosses, list)
                or not glosses
                or any(
                    not isinstance(gloss, str)
                    or not gloss
                    or gloss != gloss.strip()
                    or len(gloss) > 64
                    or any(ord(char) < 32 for char in gloss)
                    for gloss in glosses
                )
            ):
                raise ValueError(
                    f"gloss_sequences[{index}] must be a non-empty gloss array"
                )
            start, end = int(offsets[index]), int(offsets[index + 1])
            records.append(
                SequenceRecord(
                    sample_id=sample_ids[index],
                    signer_id=signer_ids[index],
                    language_code=language_codes[index],
                    glosses=tuple(glosses),
                    features=features[start:end],
                    hand_observed=observed[start:end],
                )
            )
    digest = hashlib.sha256()
    with dataset_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return LandmarkDataset(
        schema_version=SCHEMA_VERSION,
        dataset_version=f"sha256:{digest.hexdigest()}",
        records=tuple(records),
    )
