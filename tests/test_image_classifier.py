from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.image_data import load_image_manifest
from ml.models.image_classifier import StaticSignCNN
from ml.training.train_image_classifier import (
    ImageTrainingConfig,
    train_image_classifier,
)


def _write_class_images(root: Path, images_per_class: int = 20) -> None:
    for class_index, label in enumerate(("0", "A", "B")):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for index in range(images_per_class):
            pixels = torch.zeros((32, 32, 3), dtype=torch.uint8)
            pixels[:, :, class_index] = 180 + index % 70
            Image.fromarray(pixels.numpy(), mode="RGB").save(
                class_dir / f"{label}_{index}.png"
            )


def test_image_manifest_builds_deterministic_class_and_splits(tmp_path: Path) -> None:
    _write_class_images(tmp_path)

    manifest = load_image_manifest(tmp_path)

    assert manifest.classes == ("0", "A", "B")
    assert manifest.split_counts == {"train": 42, "validation": 9, "test": 9}
    assert all(
        {record.split for record in manifest.records if record.label == label}
        == {"train", "validation", "test"}
        for label in manifest.classes
    )
    assert manifest.dataset_version == load_image_manifest(tmp_path).dataset_version


def test_exact_duplicates_do_not_cross_splits(tmp_path: Path) -> None:
    _write_class_images(tmp_path)
    repeated = tmp_path / "0" / "0_13.png"
    (tmp_path / "0" / "0_14.png").write_bytes(repeated.read_bytes())

    manifest = load_image_manifest(tmp_path)
    duplicate_records = [
        record for record in manifest.records if record.path.name in {"0_13.png", "0_14.png"}
    ]

    assert len(duplicate_records) == 2
    assert len({record.split for record in duplicate_records}) == 1


def test_model_outputs_one_logit_vector_per_image() -> None:
    model = StaticSignCNN(class_count=36)
    logits = model(torch.zeros((2, 3, 64, 64)))

    assert logits.shape == (2, 36)


def test_cpu_training_saves_safely_loadable_checkpoint(tmp_path: Path) -> None:
    dataset_root = tmp_path / "images"
    _write_class_images(dataset_root, images_per_class=12)
    output = tmp_path / "artifacts"
    manifest = load_image_manifest(dataset_root)

    result = train_image_classifier(
        manifest,
        ImageTrainingConfig(
            epochs=1,
            batch_size=8,
            image_size=32,
            output_directory=output,
        ),
    )

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    evaluation = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    assert checkpoint["class_to_idx"] == {"0": 0, "A": 1, "B": 2}
    assert checkpoint["dataset_version"] == manifest.dataset_version
    assert checkpoint["preprocessing"]["schema_version"] == 1
    assert isinstance(checkpoint["model_state_dict"], dict)
    assert evaluation["split_counts"] == manifest.split_counts
    assert "signer-independent" in evaluation["interpretation_warning"]
    assert result.device == "cpu"


def test_manifest_rejects_bad_image(tmp_path: Path) -> None:
    _write_class_images(tmp_path)
    (tmp_path / "A" / "A_20.jpg").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="invalid image file"):
        load_image_manifest(tmp_path)
