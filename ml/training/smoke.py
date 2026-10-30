"""Run a tiny explicitly synthetic smoke training job; not a recognition test."""

from __future__ import annotations

import argparse
import tempfile
import uuid
from pathlib import Path

import numpy as np

from ml.data import load_npz_dataset
from ml.splitting import split_by_signer
from ml.training.train import TrainingConfig, train_model


def _create_synthetic_archive(path: Path) -> None:
    random = np.random.default_rng(42)
    signer_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"smoke-signer-{i}")) for i in range(3)]
    sample_ids: list[str] = []
    samples_signers: list[str] = []
    glosses: list[str] = []
    features: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    offsets = [0]
    for signer_index, signer_id in enumerate(signer_ids):
        for gloss_index, gloss in enumerate(("HELLO", "THANKS")):
            sample_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"smoke-sample-{signer_index}-{gloss_index}",
                )
            )
            sample_ids.append(sample_id)
            samples_signers.append(signer_id)
            glosses.append(f'["{gloss}"]')
            sample_features = random.normal(0, 0.1, size=(4, 2, 21, 3)).astype(
                np.float32
            )
            sample_observed = np.ones((4, 2), dtype=np.bool_)
            sample_features[:, gloss_index, :, :] += float(gloss_index + 1)
            features.append(sample_features)
            observed.append(sample_observed)
            offsets.append(offsets[-1] + len(sample_features))
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int16),
        features=np.concatenate(features),
        hand_observed=np.concatenate(observed),
        offsets=np.asarray(offsets, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        signer_ids=np.asarray(samples_signers, dtype=np.str_),
        language_codes=np.asarray(["isl"] * len(sample_ids), dtype=np.str_),
        gloss_sequences=np.asarray(glosses, dtype=np.str_),
    )


def run_smoke_training(output_directory: Path, epochs: int = 1) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="signlearn-synthetic-smoke-") as temporary:
        temporary_directory = Path(temporary)
        dataset_path = temporary_directory / "synthetic_smoke_only.npz"
        _create_synthetic_archive(dataset_path)
        dataset = load_npz_dataset(dataset_path)
        manifests = split_by_signer(
            dataset,
            temporary_directory / "splits",
            seed=42,
            fractions=(1 / 3, 1 / 3, 1 / 3),
        )
        result = train_model(
            dataset,
            manifests,
            TrainingConfig(
                epochs=epochs,
                batch_size=2,
                hidden_size=8,
                dropout=0.0,
                seed=42,
                output_directory=output_directory,
                data_description=(
                    "SYNTHETIC SMOKE TEST ONLY. Not real sign data and not an "
                    "ISL recognition result."
                ),
            ),
        )
    print(
        "Synthetic smoke training completed; this does not demonstrate learned "
        f"sign recognition. Device: {result.device}; checkpoint: "
        f"{result.best_checkpoint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one small synthetic CPU/GPU training smoke test only."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/synthetic-smoke"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    arguments = parser.parse_args()
    run_smoke_training(arguments.output_dir, arguments.epochs)


if __name__ == "__main__":
    main()
