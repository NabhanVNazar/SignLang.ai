"""Train the baseline CTC model using signer-independent split manifests."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import CTCLoss

from ml.data import LandmarkDataset, SequenceRecord, load_npz_dataset
from ml.models.sequence_model import HandLandmarkBiLSTM
from ml.models.vocabulary import GlossVocabulary
from ml.splitting import (
    SplitManifest,
    load_split_directory,
    validate_split_manifests,
)


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.1
    seed: int = 42
    output_directory: Path = Path("ml/artifacts/baseline")
    data_description: str = "User-provided dataset; provenance not inferred."

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint: Path
    vocabulary_path: Path
    history_path: Path
    best_validation_loss: float
    device: str


def minimum_ctc_frames(target: Sequence[int]) -> int:
    """Minimum frame count: one frame per token plus repeats needing a blank."""
    if not target:
        raise ValueError("CTC targets must contain at least one gloss")
    repeated_neighbors = sum(left == right for left, right in zip(target, target[1:]))
    return len(target) + repeated_neighbors


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _records_for_manifest(
    manifest: SplitManifest, dataset: LandmarkDataset
) -> list[SequenceRecord]:
    records_by_id = dataset.by_sample_id()
    return [records_by_id[sample_id] for sample_id in manifest.sample_ids]


def _collate_records(
    records: Sequence[SequenceRecord],
    vocabulary: GlossVocabulary,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    max_length = max(len(record.features) for record in records)
    batch_features = torch.zeros(
        (len(records), max_length, 2, 21, 3), dtype=torch.float32
    )
    batch_observed = torch.zeros((len(records), max_length, 2), dtype=torch.bool)
    lengths_values = [len(record.features) for record in records]
    targets_values: list[int] = []
    target_lengths_values: list[int] = []
    for batch_index, record in enumerate(records):
        labels = vocabulary.encode(record.glosses)
        minimum_frames = minimum_ctc_frames(labels)
        if len(record.features) < minimum_frames:
            raise ValueError(
                f"sample {record.sample_id} has {len(record.features)} input frames "
                f"but its CTC target requires at least {minimum_frames}; "
                f"target glosses: {', '.join(record.glosses)}"
            )
        batch_features[batch_index, : len(record.features)] = torch.from_numpy(
            record.features
        )
        batch_observed[batch_index, : len(record.features)] = torch.from_numpy(
            record.hand_observed
        )
        targets_values.extend(labels)
        target_lengths_values.append(len(labels))

    features = batch_features.to(device)
    observed = batch_observed.to(device)
    # PyTorch's packed sequence and CTCLoss APIs accept CPU length tensors.
    input_lengths = torch.tensor(lengths_values, dtype=torch.int64)
    target_lengths = torch.tensor(target_lengths_values, dtype=torch.int64)
    targets = torch.tensor(targets_values, dtype=torch.int64, device=device)
    return features, observed, input_lengths, targets, target_lengths


def _run_epoch(
    model: HandLandmarkBiLSTM,
    criterion: CTCLoss,
    records: Sequence[SequenceRecord],
    vocabulary: GlossVocabulary,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    generator: random.Random,
) -> float:
    training = optimizer is not None
    model.train(training)
    order = list(range(len(records)))
    if training:
        generator.shuffle(order)
    weighted_loss = 0.0
    seen = 0

    for start in range(0, len(order), batch_size):
        batch_records = [records[index] for index in order[start : start + batch_size]]
        features, observed, input_lengths, targets, target_lengths = _collate_records(
            batch_records, vocabulary, device
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, observed, input_lengths)
            log_probabilities = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(
                log_probabilities,
                targets,
                input_lengths,
                target_lengths,
            )
            if not torch.isfinite(loss):
                raise ValueError(
                    "CTC loss became non-finite; check landmark values and CTC "
                    "alignment lengths."
                )
            if training:
                loss.backward()
                optimizer.step()
        weighted_loss += float(loss.detach().item()) * len(batch_records)
        seen += len(batch_records)

    if not seen:
        raise ValueError("cannot train or validate on an empty split")
    return weighted_loss / seen


def _atomic_torch_save(value: object, path: Path) -> None:
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


def train_model(
    dataset: LandmarkDataset,
    manifests: dict[str, SplitManifest],
    config: TrainingConfig,
) -> TrainingResult:
    config.validate()
    validate_split_manifests(manifests, dataset)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = _records_for_manifest(manifests["train"], dataset)
    validation_records = _records_for_manifest(manifests["validation"], dataset)
    vocabulary = GlossVocabulary.from_sequences(
        record.glosses for record in train_records
    )

    # Validate labels and CTC feasibility before starting optimization.
    for split_name, records in (
        ("train", train_records),
        ("validation", validation_records),
    ):
        for record in records:
            labels = vocabulary.encode(record.glosses)
            minimum_frames = minimum_ctc_frames(labels)
            if len(record.features) < minimum_frames:
                raise ValueError(
                    f"{split_name} sample {record.sample_id} has "
                    f"{len(record.features)} frames; CTC requires at least "
                    f"{minimum_frames} for its target sequence"
                )

    model_config = {
        "architecture": "bidirectional_lstm_ctc",
        "num_layers": 1,
        "bidirectional": True,
        "class_count": vocabulary.class_count,
        "blank_id": 0,
        "hidden_size": config.hidden_size,
        "dropout": config.dropout,
        "input_size": 128,
        "feature_shape": [2, 21, 3],
        "missing_hand_mask_channels": 2,
    }
    model = HandLandmarkBiLSTM(
        class_count=vocabulary.class_count,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = CTCLoss(blank=GlossVocabulary.BLANK_ID, zero_infinity=False)
    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    vocabulary_path = output_directory / "vocabulary.json"
    checkpoint_path = output_directory / "best_model.pt"
    history_path = output_directory / "training_history.json"
    vocabulary.save(vocabulary_path)

    settings = asdict(config)
    settings["output_directory"] = str(config.output_directory)
    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    epoch_rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        training_loss = _run_epoch(
            model,
            criterion,
            train_records,
            vocabulary,
            config.batch_size,
            device,
            optimizer,
            epoch_rng,
        )
        validation_loss = _run_epoch(
            model,
            criterion,
            validation_records,
            vocabulary,
            config.batch_size,
            device,
            None,
            epoch_rng,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            checkpoint = {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "dataset_version": dataset.dataset_version,
                "preprocessing_schema_version": dataset.schema_version,
                "split_metadata": {
                    "seed": manifests["train"].seed,
                    "fractions": manifests["train"].fractions,
                    "train_signer_count": len(manifests["train"].signer_ids),
                    "validation_signer_count": len(
                        manifests["validation"].signer_ids
                    ),
                    "test_signer_count": len(manifests["test"].signer_ids),
                },
                "model_config": model_config,
                "vocabulary": vocabulary.to_dict(),
                "training_settings": settings,
                "data_description": config.data_description,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            }
            _atomic_torch_save(checkpoint, checkpoint_path)

    history_payload = {
        "dataset_version": dataset.dataset_version,
        "preprocessing_schema_version": dataset.schema_version,
        "split_seed": manifests["train"].seed,
        "split_fractions": manifests["train"].fractions,
        "seed": config.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }
    history_path.write_text(
        json.dumps(history_payload, indent=2) + "\n", encoding="utf-8"
    )
    return TrainingResult(
        best_checkpoint=checkpoint_path,
        vocabulary_path=vocabulary_path,
        history_path=history_path,
        best_validation_loss=best_validation_loss,
        device=str(device),
    )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[HandLandmarkBiLSTM, GlossVocabulary, dict[str, object]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
        EOFError,
    ) as exc:
        raise ValueError(f"cannot safely load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if (
        not isinstance(checkpoint.get("dataset_version"), str)
        or not checkpoint["dataset_version"].startswith("sha256:")
        or checkpoint.get("preprocessing_schema_version") != 1
    ):
        raise ValueError("checkpoint dataset/preprocessing version is unsupported")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing model configuration or weights")
    vocabulary = GlossVocabulary.from_dict(checkpoint.get("vocabulary"))
    if model_config.get("class_count") != vocabulary.class_count:
        raise ValueError("checkpoint model class count does not match vocabulary")
    model = HandLandmarkBiLSTM(
        class_count=vocabulary.class_count,
        hidden_size=int(model_config["hidden_size"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, vocabulary, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the BiLSTM/CTC baseline from an NPZ and signer splits."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    dataset = load_npz_dataset(arguments.dataset)
    manifests = load_split_directory(arguments.splits, dataset)
    config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        hidden_size=arguments.hidden_size,
        dropout=arguments.dropout,
        seed=arguments.seed,
        output_directory=arguments.output_dir,
    )
    result = train_model(dataset, manifests, config)
    print(
        f"Training complete on {result.device}. Best validation loss: "
        f"{result.best_validation_loss:.6f}. Checkpoint: {result.best_checkpoint}"
    )


if __name__ == "__main__":
    main()
