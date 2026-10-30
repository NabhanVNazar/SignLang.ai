"""Consent-gated conversion of JSONL samples to hand-landmark NumPy datasets."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.dataset import Sample, load_manifest
from ml.features import build_hand_features


@dataclass(frozen=True)
class ExportSummary:
    exported_samples: int
    skipped_samples: int
    frame_count: int
    output_path: Path


def _hand_features(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    return build_hand_features(
        (frame.left_hand, frame.right_hand) for frame in sample.frames
    )


def export_hand_dataset(
    manifest_path: str | Path, output_path: str | Path
) -> ExportSummary:
    """Export approved, consented hand sequences to a non-pickled NPZ archive.

    Per-sample boundaries are stored in ``offsets``. The archive contains landmarks
    and labels only; source video objects are not copied into the training dataset.
    """
    manifest = Path(manifest_path)
    destination = Path(output_path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"output directory does not exist: {destination.parent}"
        )
    if manifest.resolve() == destination.resolve():
        raise ValueError("manifest and output paths must be different")

    features_by_sample: list[np.ndarray] = []
    masks_by_sample: list[np.ndarray] = []
    sample_ids: list[str] = []
    signer_ids: list[str] = []
    language_codes: list[str] = []
    gloss_sequences: list[str] = []
    offsets = [0]
    skipped_samples = 0
    seen_sample_ids: set[str] = set()

    for sample in load_manifest(manifest):
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id in manifest: {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)
        if not sample.consent or sample.annotation_status != "approved":
            skipped_samples += 1
            continue
        features, observed = _hand_features(sample)
        features_by_sample.append(features)
        masks_by_sample.append(observed)
        offsets.append(offsets[-1] + len(features))
        sample_ids.append(sample.sample_id)
        signer_ids.append(sample.signer_id)
        language_codes.append(sample.language_code)
        gloss_sequences.append(
            json.dumps(sample.glosses, ensure_ascii=False, separators=(",", ":"))
        )

    if not features_by_sample:
        raise ValueError("manifest contains no consented, approved samples")

    arrays = {
        "schema_version": np.asarray([1], dtype=np.int16),
        "features": np.concatenate(features_by_sample, axis=0),
        "hand_observed": np.concatenate(masks_by_sample, axis=0),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "signer_ids": np.asarray(signer_ids, dtype=np.str_),
        "language_codes": np.asarray(language_codes, dtype=np.str_),
        "gloss_sequences": np.asarray(gloss_sequences, dtype=np.str_),
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **arrays)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return ExportSummary(
        exported_samples=len(sample_ids),
        skipped_samples=skipped_samples,
        frame_count=int(offsets[-1]),
        output_path=destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export consented, approved hand-landmark sequences to NPZ."
    )
    parser.add_argument("manifest", type=Path, help="Input JSONL manifest")
    parser.add_argument("output", type=Path, help="Output .npz file")
    arguments = parser.parse_args()

    summary = export_hand_dataset(arguments.manifest, arguments.output)
    print(
        f"Exported {summary.exported_samples} samples "
        f"({summary.frame_count} frames); skipped {summary.skipped_samples}. "
        f"Output: {summary.output_path}"
    )


if __name__ == "__main__":
    main()
