"""Train and evaluate an isolated-sign image classifier from class folders."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.image_data import (
    ImageDatasetManifest,
    ImageRecord,
    StaticSignImageDataset,
    load_image_manifest,
    write_image_manifest,
)
from ml.image_labels import IMAGE_CLASS_LABELS, class_mapping_for
from ml.models.image_classifier import StaticSignCNN


@dataclass(frozen=True)
class ImageTrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    image_size: int = 64
    seed: int = 42
    output_directory: Path = Path("ml/artifacts/static_classifier")
    num_workers: int = 0
    allow_subset_classes: bool = False

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not isinstance(self.allow_subset_classes, bool):
            raise ValueError("allow_subset_classes must be a boolean")


@dataclass(frozen=True)
class ImageTrainingResult:
    checkpoint_path: Path
    history_path: Path
    evaluation_path: Path
    device: str
    best_validation_loss: float
    test_accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(value, stream)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _loader(
    records: list[ImageRecord],
    class_to_idx: dict[str, int],
    config: ImageTrainingConfig,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        StaticSignImageDataset(records, class_to_idx, config.image_size),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: StaticSignCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("classification loss became non-finite")
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch_size
    if total == 0:
        raise ValueError("cannot train or evaluate on an empty split")
    return {"loss": total_loss / total, "accuracy": correct / total}


def _per_class_accuracy(
    model: StaticSignCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    classes: tuple[str, ...],
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    totals = [0] * len(classes)
    correct = [0] * len(classes)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                totals[label] += 1
                correct[label] += int(label == prediction)
    return {
        class_name: {
            "samples": totals[index],
            "accuracy": correct[index] / totals[index] if totals[index] else 0.0,
        }
        for index, class_name in enumerate(classes)
    }


def train_image_classifier(
    manifest: ImageDatasetManifest,
    config: ImageTrainingConfig,
) -> ImageTrainingResult:
    config.validate()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = class_mapping_for(
        manifest.classes, allow_subset=config.allow_subset_classes
    )
    split_records = {
        split: [record for record in manifest.records if record.split == split]
        for split in ("train", "validation", "test")
    }
    data_generator = torch.Generator()
    data_generator.manual_seed(config.seed)
    train_loader = _loader(
        split_records["train"], class_to_idx, config, True, data_generator
    )
    validation_loader = _loader(
        split_records["validation"], class_to_idx, config, False, data_generator
    )
    test_loader = _loader(
        split_records["test"], class_to_idx, config, False, data_generator
    )

    model = StaticSignCNN(class_count=len(manifest.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    output_directory = config.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "best_checkpoint.pt"
    history_path = output_directory / "history.json"
    evaluation_path = output_directory / "evaluation.json"
    split_manifest_path = output_directory / "split_manifest.json"
    write_image_manifest(manifest, split_manifest_path)

    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    model_configuration = {
        "architecture": "StaticSignCNN",
        "class_count": len(manifest.classes),
        "dropout": 0.2,
    }
    preprocessing = {
        "schema_version": 1,
        "color_space": "RGB",
        "image_size": config.image_size,
        "resize": "PIL bilinear",
        "tensor_layout": "CHW",
        "pixel_range": [0.0, 1.0],
        "normalization": "uint8 pixel values divided by 255",
    }
    training_settings = asdict(config)
    training_settings["output_directory"] = str(config.output_directory)

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, criterion, device, None
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": train_metrics["loss"],
                "training_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            _atomic_torch_save(
                {
                    "checkpoint_format_version": 1,
                    "model_config": model_configuration,
                    "preprocessing": preprocessing,
                    "class_to_idx": class_to_idx,
                    "dataset_version": manifest.dataset_version,
                    "dataset_schema_version": manifest.schema_version,
                    "dataset_split_method": manifest.split_method,
                    "dataset_language_claim": "unverified_user_supplied_image_archive",
                    "dataset_source_and_license": "not_provided",
                    "training_settings": training_settings,
                    "best_epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        print(
            f"epoch {epoch}/{config.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f}"
        )

    safe_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(safe_checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, test_loader, criterion, device, None)
    per_class = _per_class_accuracy(model, test_loader, manifest.classes, device)
    evaluation = {
        "dataset_version": manifest.dataset_version,
        "split_counts": manifest.split_counts,
        "best_epoch_by_validation_loss": best_epoch,
        "validation_loss_at_best_epoch": best_validation_loss,
        "test_image_top1_accuracy": test_metrics["accuracy"],
        "test_image_loss": test_metrics["loss"],
        "per_class_test_accuracy": per_class,
        "interpretation_warning": (
            "These are image-level metrics on an ordered filename-suffix holdout. "
            "The dataset has no verified signer or recording identifiers, so this "
            "does not establish signer-independent or real-world ISL recognition."
        ),
    }
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Final test image top-1 accuracy: "
        f"{test_metrics['accuracy']:.4f} "
        "(not signer-independent; see evaluation.json)"
    )
    return ImageTrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        evaluation_path=evaluation_path,
        device=str(device),
        best_validation_loss=best_validation_loss,
        test_accuracy=test_metrics["accuracy"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an isolated-sign image classifier from class folders."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("archive/isl_dataset"),
        help="Folder containing one image subfolder per class.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/artifacts/static_classifier"),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-subset-classes",
        action="store_true",
        help="Permit non-36-class training for unit tests only; never use for deployment.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ImageTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        seed=args.seed,
        output_directory=args.output,
        num_workers=args.num_workers,
        allow_subset_classes=args.allow_subset_classes,
    )
    manifest = load_image_manifest(args.dataset)
    print(
        f"Scanned {sum(manifest.split_counts.values())} images across "
        f"{len(manifest.classes)} classes; split counts: {manifest.split_counts}"
    )
    print(
        "Warning: signer and recording metadata are absent; evaluation is not "
        "signer-independent."
    )
    result = train_image_classifier(manifest, config)
    print(f"Device: {result.device}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Evaluation: {result.evaluation_path}")


if __name__ == "__main__":
    main()
