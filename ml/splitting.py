"""Deterministic signer-independent dataset splitting."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.data import LandmarkDataset, SequenceRecord, load_npz_dataset


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_FRACTIONS = (0.7, 0.15, 0.15)


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    split: str
    dataset_version: str
    seed: int
    fractions: dict[str, float]
    sample_ids: tuple[str, ...]
    signer_ids: tuple[str, ...]


def _split_counts(signer_count: int, fractions: tuple[float, float, float]) -> list[int]:
    if signer_count < 3:
        raise ValueError(
            f"at least 3 distinct signers are required for train/validation/test; "
            f"found {signer_count}"
        )
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("train, validation, and test fractions must all be positive")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to 1")

    remaining = signer_count - 3
    raw_additional = [remaining * fraction for fraction in fractions]
    additional = [int(value) for value in raw_additional]
    left = remaining - sum(additional)
    order = sorted(
        range(3),
        key=lambda index: (raw_additional[index] - additional[index], -index),
        reverse=True,
    )
    for index in order[:left]:
        additional[index] += 1
    return [count + 1 for count in additional]


def split_by_signer(
    dataset: LandmarkDataset,
    output_directory: str | Path,
    seed: int = 42,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> dict[str, SplitManifest]:
    signers: dict[str, list[SequenceRecord]] = {}
    for record in dataset.records:
        signers.setdefault(record.signer_id, []).append(record)
    sorted_signers = sorted(signers)
    counts = _split_counts(len(sorted_signers), fractions)

    shuffled_signers = sorted_signers.copy()
    random.Random(seed).shuffle(shuffled_signers)
    signer_splits: dict[str, list[str]] = {}
    cursor = 0
    for split_name, count in zip(SPLIT_NAMES, counts):
        signer_splits[split_name] = shuffled_signers[cursor : cursor + count]
        cursor += count

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    fraction_map = dict(zip(SPLIT_NAMES, fractions))
    result: dict[str, SplitManifest] = {}
    for split_name in SPLIT_NAMES:
        selected_signers = set(signer_splits[split_name])
        selected_records = [
            record for record in dataset.records if record.signer_id in selected_signers
        ]
        manifest_path = directory / f"{split_name}.json"
        payload = {
            "format_version": 1,
            "dataset_version": dataset.dataset_version,
            "preprocessing_schema_version": dataset.schema_version,
            "seed": seed,
            "fractions": fraction_map,
            "split": split_name,
            "signer_ids": sorted(selected_signers),
            "samples": [
                {"sample_id": record.sample_id, "signer_id": record.signer_id}
                for record in selected_records
            ],
        }
        _atomic_write_json(manifest_path, payload)
        result[split_name] = SplitManifest(
            path=manifest_path,
            split=split_name,
            dataset_version=dataset.dataset_version,
            seed=seed,
            fractions=fraction_map,
            sample_ids=tuple(item["sample_id"] for item in payload["samples"]),
            signer_ids=tuple(payload["signer_ids"]),
        )
    validate_split_manifests(result, dataset)
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_split_manifest(path: str | Path, dataset: LandmarkDataset) -> SplitManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read split manifest {manifest_path}: {exc}") from exc
    expected_fields = {
        "format_version",
        "dataset_version",
        "preprocessing_schema_version",
        "seed",
        "fractions",
        "split",
        "signer_ids",
        "samples",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(f"{manifest_path} has missing or unknown manifest fields")
    if (
        isinstance(payload.get("format_version"), bool)
        or payload.get("format_version") != 1
    ):
        raise ValueError(f"{manifest_path} has an unsupported split manifest format")
    if payload.get("dataset_version") != dataset.dataset_version:
        raise ValueError(f"{manifest_path} dataset version does not match the NPZ")
    schema_version = payload.get("preprocessing_schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != dataset.schema_version
    ):
        raise ValueError(
            f"{manifest_path} preprocessing schema version does not match the NPZ"
        )
    split_name = payload.get("split")
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"{manifest_path} has an invalid split name")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{manifest_path} seed must be an integer")
    fractions = payload.get("fractions")
    if not isinstance(fractions, dict) or set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"{manifest_path} must record all split fractions")
    if any(
        isinstance(fractions[name], bool)
        or not isinstance(fractions[name], (int, float))
        or not math.isfinite(fractions[name])
        or fractions[name] <= 0
        for name in SPLIT_NAMES
    ):
        raise ValueError(
            f"{manifest_path} split fractions must be finite positive numbers"
        )
    if abs(sum(float(fractions[name]) for name in SPLIT_NAMES) - 1.0) > 1e-8:
        raise ValueError(f"{manifest_path} split fractions must sum to 1")
    signer_ids = payload.get("signer_ids")
    samples = payload.get("samples")
    if (
        not isinstance(signer_ids, list)
        or any(not isinstance(signer_id, str) for signer_id in signer_ids)
        or len(set(signer_ids)) != len(signer_ids)
        or not isinstance(samples, list)
        or any(
            not isinstance(sample, dict)
            or not isinstance(sample.get("sample_id"), str)
            or not isinstance(sample.get("signer_id"), str)
            for sample in samples
        )
    ):
        raise ValueError(f"{manifest_path} has malformed signer or sample entries")
    record_map = dataset.by_sample_id()
    sample_ids = tuple(item["sample_id"] for item in samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{manifest_path} contains duplicate sample IDs")
    if not sample_ids:
        raise ValueError(f"{manifest_path} contains no samples")
    for item in samples:
        record = record_map.get(item["sample_id"])
        if record is None:
            raise ValueError(
                f"{manifest_path} references unknown sample {item['sample_id']}"
            )
        if record.signer_id != item["signer_id"]:
            raise ValueError(
                f"{manifest_path} signer does not match sample {item['sample_id']}"
            )
        if record.signer_id not in signer_ids:
            raise ValueError(
                f"{manifest_path} sample signer is missing from its signer list"
            )
    actual_signers = {item["signer_id"] for item in samples}
    if actual_signers != set(signer_ids):
        raise ValueError(f"{manifest_path} signer list does not match its samples")
    return SplitManifest(
        path=manifest_path,
        split=split_name,
        dataset_version=dataset.dataset_version,
        seed=seed,
        fractions={name: float(fractions[name]) for name in SPLIT_NAMES},
        sample_ids=sample_ids,
        signer_ids=tuple(signer_ids),
    )


def validate_split_manifests(
    manifests: dict[str, SplitManifest], dataset: LandmarkDataset
) -> None:
    if set(manifests) != set(SPLIT_NAMES):
        raise ValueError("exactly train, validation, and test manifests are required")
    seen_signers: set[str] = set()
    seen_samples: set[str] = set()
    reference_seed: int | None = None
    reference_fractions: dict[str, float] | None = None
    records_by_id = dataset.by_sample_id()
    for split_name in SPLIT_NAMES:
        manifest = manifests[split_name]
        if manifest.split != split_name:
            raise ValueError(f"expected {split_name} manifest, got {manifest.split}")
        if manifest.dataset_version != dataset.dataset_version:
            raise ValueError(f"{split_name} manifest dataset version does not match")
        if reference_seed is None:
            reference_seed = manifest.seed
            reference_fractions = manifest.fractions
        elif manifest.seed != reference_seed or manifest.fractions != reference_fractions:
            raise ValueError("split manifests must share the same seed and fractions")
        if len(set(manifest.sample_ids)) != len(manifest.sample_ids):
            raise ValueError(f"{split_name} manifest contains duplicate sample IDs")
        if len(set(manifest.signer_ids)) != len(manifest.signer_ids):
            raise ValueError(f"{split_name} manifest contains duplicate signer IDs")
        expected_signers: set[str] = set()
        for sample_id in manifest.sample_ids:
            record = records_by_id.get(sample_id)
            if record is None:
                raise ValueError(
                    f"{split_name} manifest references unknown sample {sample_id}"
                )
            expected_signers.add(record.signer_id)
        if expected_signers != set(manifest.signer_ids):
            raise ValueError(
                f"{split_name} manifest signer list does not match its samples"
            )
        overlap = seen_signers.intersection(manifest.signer_ids)
        if overlap:
            raise ValueError(
                f"signer leakage detected in {split_name}: {', '.join(sorted(overlap))}"
            )
        seen_signers.update(manifest.signer_ids)
        sample_overlap = seen_samples.intersection(manifest.sample_ids)
        if sample_overlap:
            raise ValueError(
                f"sample appears in multiple splits: {', '.join(sorted(sample_overlap))}"
            )
        seen_samples.update(manifest.sample_ids)
    if seen_samples != {record.sample_id for record in dataset.records}:
        raise ValueError("split manifests do not partition every dataset sample")


def load_split_directory(
    directory: str | Path, dataset: LandmarkDataset
) -> dict[str, SplitManifest]:
    split_directory = Path(directory)
    manifests = {
        split_name: load_split_manifest(
            split_directory / f"{split_name}.json", dataset
        )
        for split_name in SPLIT_NAMES
    }
    validate_split_manifests(manifests, dataset)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic signer-independent train/validation/test splits."
    )
    parser.add_argument("dataset", type=Path, help="Preprocessed hand-landmark NPZ")
    parser.add_argument("output_directory", type=Path, help="Split manifest directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    arguments = parser.parse_args()

    dataset = load_npz_dataset(arguments.dataset)
    manifests = split_by_signer(
        dataset,
        arguments.output_directory,
        seed=arguments.seed,
        fractions=(
            arguments.train_fraction,
            arguments.validation_fraction,
            arguments.test_fraction,
        ),
    )
    for split_name, manifest in manifests.items():
        print(
            f"{split_name}: {len(manifest.sample_ids)} samples, "
            f"{len(manifest.signer_ids)} signers -> {manifest.path}"
        )
    if len({signer for item in manifests.values() for signer in item.signer_ids}) < 10:
        print(
            "WARNING: fewer than 10 distinct signers are available; "
            "signer-independent estimates may be unreliable."
        )


if __name__ == "__main__":
    main()
