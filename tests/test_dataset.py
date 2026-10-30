import json
import uuid

import numpy as np
import pytest

from ml.dataset import (
    DatasetValidationError,
    load_manifest,
    validate_sample,
    write_manifest as write_dataset_manifest,
)
from ml.preprocess import export_hand_dataset


def make_sample(**overrides):
    hand = [[index / 21, index / 42, -index / 100] for index in range(21)]
    frame = {
        "timestamp_ms": 0,
        "left_hand": hand,
        "right_hand": None,
        "body_pose": None,
        "face": None,
    }
    sample = {
        "schema_version": 1,
        "sample_id": str(uuid.uuid4()),
        "language_code": "isl",
        "signer_id": str(uuid.uuid4()),
        "consent": True,
        "annotation_status": "approved",
        "created_at": "2026-10-29T04:30:00Z",
        "glosses": ["HELLO"],
        "frames": [frame],
    }
    sample.update(overrides)
    return sample


def write_manifest(path, samples):
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )


def test_validate_sample_accepts_valid_hand_landmarks():
    sample = validate_sample(make_sample())

    assert sample.language_code == "isl"
    assert len(sample.frames[0].left_hand) == 21
    assert sample.frames[0].right_hand is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["frames"][0].update(left_hand=[[0, 0, 0]]),
            "exactly 21",
        ),
        (
            lambda data: data["frames"][0].update(
                left_hand=[[float("nan"), 0, 0]] * 21
            ),
            "finite number",
        ),
        (
            lambda data: data["frames"][0].update(timestamp_ms=True),
            "non-negative integer",
        ),
        (lambda data: data.update(consent="yes"), "consent must be a boolean"),
        (lambda data: data.update(annotation_status=[]), "annotation_status"),
        (lambda data: data.update(unexpected=True), "unknown fields"),
    ],
)
def test_validate_sample_rejects_invalid_data(mutate, message):
    sample = make_sample()
    mutate(sample)

    with pytest.raises(DatasetValidationError, match=message):
        validate_sample(sample)


def test_validate_sample_requires_increasing_timestamps():
    sample = make_sample(
        frames=[
            {
                "timestamp_ms": timestamp,
                "left_hand": None,
                "right_hand": None,
                "body_pose": None,
                "face": None,
            }
            for timestamp in (10, 10)
        ]
    )

    with pytest.raises(DatasetValidationError, match="strictly increasing"):
        validate_sample(sample)


def test_load_manifest_reports_invalid_line_number(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=r"manifest\.jsonl:1"):
        list(load_manifest(manifest_path))


def test_write_manifest_round_trips_validated_samples(tmp_path):
    output_path = tmp_path / "round_trip.jsonl"
    sample = validate_sample(make_sample())

    write_dataset_manifest([sample], output_path)

    loaded = list(load_manifest(output_path))
    assert loaded == [sample]


def test_export_skips_unconsented_or_unapproved_and_preserves_sequence(tmp_path):
    approved = make_sample(
        frames=[
            {
                "timestamp_ms": 0,
                "left_hand": [[1, 2, 3]] * 21,
                "right_hand": None,
                "body_pose": None,
                "face": None,
            },
            {
                "timestamp_ms": 40,
                "left_hand": None,
                "right_hand": [[4, 5, 6]] * 21,
                "body_pose": None,
                "face": None,
            },
        ]
    )
    no_consent = make_sample(consent=False)
    pending = make_sample(annotation_status="pending")
    manifest_path = tmp_path / "manifest.jsonl"
    output_path = tmp_path / "hand_dataset.npz"
    write_manifest(manifest_path, [approved, no_consent, pending])

    summary = export_hand_dataset(manifest_path, output_path)

    assert summary.exported_samples == 1
    assert summary.skipped_samples == 2
    assert summary.frame_count == 2
    with np.load(output_path, allow_pickle=False) as dataset:
        assert dataset["features"].shape == (2, 2, 21, 3)
        assert dataset["features"].dtype == np.float32
        assert dataset["hand_observed"].tolist() == [[True, False], [False, True]]
        assert dataset["offsets"].tolist() == [0, 2]
        assert dataset["sample_ids"].tolist() == [approved["sample_id"]]
        assert dataset["gloss_sequences"].tolist() == ['["HELLO"]']
        assert "source_video_key" not in dataset.files


def test_export_requires_eligible_samples_and_npz_output(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    output_path = tmp_path / "hand_dataset.npz"
    write_manifest(manifest_path, [make_sample(consent=False)])

    with pytest.raises(ValueError, match="no consented, approved"):
        export_hand_dataset(manifest_path, output_path)

    with pytest.raises(ValueError, match="end in .npz"):
        export_hand_dataset(manifest_path, tmp_path / "dataset.bin")


def test_export_rejects_duplicate_sample_ids(tmp_path):
    sample = make_sample()
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, [sample, sample])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        export_hand_dataset(manifest_path, tmp_path / "hand_dataset.npz")
