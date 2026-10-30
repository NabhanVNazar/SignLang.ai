"""Evaluate a checkpoint on the held-out signer-independent test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.data import LandmarkDataset, SequenceRecord, load_npz_dataset
from ml.metrics import sequence_metrics
from ml.splitting import SplitManifest, load_split_directory
from ml.training.train import load_model_checkpoint


def _select_records(
    manifest: SplitManifest, dataset: LandmarkDataset
) -> list[SequenceRecord]:
    by_id = dataset.by_sample_id()
    return [by_id[sample_id] for sample_id in manifest.sample_ids]


def _missing_hand_statistics(records: list[SequenceRecord]) -> dict[str, object]:
    frames = sum(len(record.features) for record in records)
    left_missing = sum(int((~record.hand_observed[:, 0]).sum()) for record in records)
    right_missing = sum(int((~record.hand_observed[:, 1]).sum()) for record in records)
    both_missing = sum(
        int((~record.hand_observed).all(axis=1).sum()) for record in records
    )
    return {
        "frame_count": frames,
        "left_hand_missing_frames": left_missing,
        "right_hand_missing_frames": right_missing,
        "both_hands_missing_frames": both_missing,
        "left_hand_missing_rate": left_missing / frames if frames else 0.0,
        "right_hand_missing_rate": right_missing / frames if frames else 0.0,
        "both_hands_missing_rate": both_missing / frames if frames else 0.0,
    }


def evaluate_checkpoint(
    dataset_path: str | Path,
    split_directory: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    dataset = load_npz_dataset(dataset_path)
    manifests = load_split_directory(split_directory, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary, checkpoint = load_model_checkpoint(checkpoint_path, device)
    if checkpoint.get("dataset_version") != dataset.dataset_version:
        raise ValueError("checkpoint dataset version does not match the evaluation data")
    test_records = _select_records(manifests["test"], dataset)
    references: list[tuple[str, ...]] = []
    hypotheses: list[tuple[str, ...]] = []
    with torch.inference_mode():
        for record in test_records:
            features = torch.from_numpy(record.features).unsqueeze(0).to(device)
            observed = torch.from_numpy(record.hand_observed).unsqueeze(0).to(device)
            lengths = torch.tensor([len(record.features)], dtype=torch.int64)
            logits = model(features, observed, lengths)[0, : len(record.features)]
            frame_ids = logits.argmax(dim=-1).tolist()
            references.append(record.glosses)
            hypotheses.append(tuple(vocabulary.decode_ctc(frame_ids)))

    metrics = sequence_metrics(references, hypotheses)
    test_signers = len(manifests["test"].signer_ids)
    warnings: list[str] = []
    if len(test_records) < 10:
        warnings.append(
            "Fewer than 10 test samples: evaluation metrics are highly uncertain."
        )
    if test_signers < 3:
        warnings.append(
            "Fewer than 3 held-out test signers: signer-diverse evaluation is limited."
        )
    known_glosses = vocabulary.token_to_id.keys()
    unknown_reference_glosses = sorted(
        {gloss for sequence in references for gloss in sequence if gloss not in known_glosses}
    )
    if unknown_reference_glosses:
        warnings.append(
            "Some test reference glosses were absent from the training vocabulary; "
            "the model cannot emit them: " + ", ".join(unknown_reference_glosses)
        )

    split_counts = {
        name: {
            "sample_count": len(manifest.sample_ids),
            "signer_count": len(manifest.signer_ids),
        }
        for name, manifest in manifests.items()
    }
    return {
        "dataset_version": dataset.dataset_version,
        "preprocessing_schema_version": dataset.schema_version,
        "checkpoint": str(checkpoint_path),
        "evaluation_split": "test",
        "split_counts": split_counts,
        "test_missing_hand_statistics": _missing_hand_statistics(test_records),
        "metrics": metrics,
        "unknown_reference_glosses": unknown_reference_glosses,
        "warnings": warnings,
        "note": (
            "Gloss token error rate treats each gloss string as one token. Character "
            "error rate uses characters in space-joined gloss sequences. Neither is "
            "frame accuracy; results are not a claim of general ISL translation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the signer-independent test split."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate_checkpoint(
        arguments.dataset, arguments.splits, arguments.checkpoint
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
