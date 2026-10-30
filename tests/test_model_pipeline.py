import json
import uuid

import numpy as np
import pytest
import torch

from ml.data import load_npz_dataset
from ml.evaluation import evaluate_checkpoint
from ml.metrics import edit_distance, sequence_metrics
from ml.models.sequence_model import HandLandmarkBiLSTM
from ml.models.vocabulary import GlossVocabulary
from ml.preprocess import export_hand_dataset
from ml.splitting import load_split_directory, split_by_signer
from ml.training.smoke import run_smoke_training
from ml.training.train import (
    TrainingConfig,
    load_model_checkpoint,
    minimum_ctc_frames,
    train_model,
)


def create_synthetic_npz(path, signer_count=3):
    features = []
    hand_masks = []
    offsets = [0]
    sample_ids = []
    signer_ids = []
    gloss_sequences = []
    language_codes = []
    for signer_index in range(signer_count):
        signer_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"test-signer-{signer_index}"))
        for gloss_index, gloss in enumerate(("HELLO", "THANKS")):
            sample_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"test-sample-{signer_index}-{gloss_index}",
                )
            )
            frames = np.full(
                (4, 2, 21, 3),
                fill_value=float(gloss_index + 1),
                dtype=np.float32,
            )
            observed = np.ones((4, 2), dtype=np.bool_)
            if gloss_index:
                frames[:, 0] = 0
                observed[:, 0] = False
            features.append(frames)
            hand_masks.append(observed)
            offsets.append(offsets[-1] + len(frames))
            sample_ids.append(sample_id)
            signer_ids.append(signer_id)
            gloss_sequences.append(json.dumps([gloss]))
            language_codes.append("isl")
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int16),
        features=np.concatenate(features),
        hand_observed=np.concatenate(hand_masks),
        offsets=np.asarray(offsets, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        signer_ids=np.asarray(signer_ids, dtype=np.str_),
        language_codes=np.asarray(language_codes, dtype=np.str_),
        gloss_sequences=np.asarray(gloss_sequences, dtype=np.str_),
    )
    return sample_ids, signer_ids


def test_npz_contract_loads_exporter_output_without_format_changes(tmp_path):
    samples = []
    for signer_index in range(3):
        samples.append(
            {
                "schema_version": 1,
                "sample_id": str(uuid.uuid4()),
                "language_code": "isl",
                "signer_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"exporter-{signer_index}")),
                "consent": True,
                "annotation_status": "approved",
                "created_at": "2026-10-29T04:30:00Z",
                "glosses": ["HELLO"],
                "frames": [
                    {
                        "timestamp_ms": 0,
                        "left_hand": [[0.1, 0.2, 0.3]] * 21,
                        "right_hand": None,
                        "body_pose": None,
                        "face": None,
                    },
                    {
                        "timestamp_ms": 40,
                        "left_hand": None,
                        "right_hand": [[0.4, 0.5, 0.6]] * 21,
                        "body_pose": None,
                        "face": None,
                    },
                ],
            }
        )
    manifest_path = tmp_path / "source.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    npz_path = tmp_path / "dataset.npz"
    export_hand_dataset(manifest_path, npz_path)

    dataset = load_npz_dataset(npz_path)

    assert dataset.dataset_version.startswith("sha256:")
    assert len(dataset.records) == 3
    first = dataset.records[0]
    assert first.features.shape == (2, 2, 21, 3)
    assert first.features.dtype == np.float32
    assert first.hand_observed.dtype == np.bool_
    assert first.hand_observed.tolist() == [[True, False], [False, True]]
    assert first.glosses == ("HELLO",)


def test_vocabulary_is_deterministic_and_ctc_decodes():
    first = GlossVocabulary.from_sequences([["THANKS"], ["HELLO", "THANKS"]])
    second = GlossVocabulary.from_sequences([["HELLO", "THANKS"], ["THANKS"]])
    assert first.token_to_id == second.token_to_id == {"HELLO": 1, "THANKS": 2}
    assert first.BLANK_ID == 0
    assert first.decode_ctc([1, 1, 0, 1, 2, 2]) == ["HELLO", "HELLO", "THANKS"]


def test_vocabulary_save_load(tmp_path):
    vocabulary = GlossVocabulary.from_sequences([["HELLO"], ["THANKS"]])
    path = tmp_path / "vocabulary.json"
    vocabulary.save(path)
    loaded = GlossVocabulary.load(path)

    assert json.loads(path.read_text(encoding="utf-8"))["blank_id"] == 0
    assert loaded.to_dict() == vocabulary.to_dict()
    with pytest.raises(ValueError, match="absent from the training vocabulary"):
        loaded.encode(["UNKNOWN"])


def test_model_outputs_frame_logits_for_variable_lengths_and_ctc_loss():
    model = HandLandmarkBiLSTM(class_count=4, hidden_size=8, dropout=0.0)
    features = torch.randn(2, 5, 2, 21, 3)
    observed = torch.ones(2, 5, 2, dtype=torch.bool)
    observed[1, 3:] = False
    features[1, 3:] = 0
    lengths = torch.tensor([5, 3], dtype=torch.int64)
    logits = model(features, observed, lengths)
    assert logits.shape == (2, 5, 4)
    assert torch.isfinite(logits).all()

    ctc_model = HandLandmarkBiLSTM(class_count=3, hidden_size=4, dropout=0.0)
    ctc_logits = ctc_model(
        torch.randn(1, 3, 2, 21, 3),
        torch.ones(1, 3, 2, dtype=torch.bool),
        torch.tensor([3], dtype=torch.int64),
    )
    targets = torch.tensor([1, 1], dtype=torch.int64)
    loss = torch.nn.CTCLoss(blank=0)(
        ctc_logits.log_softmax(-1).transpose(0, 1),
        targets,
        torch.tensor([3], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert minimum_ctc_frames([1, 1]) == 3


def test_signer_split_is_deterministic_complete_and_disjoint(tmp_path):
    archive_path = tmp_path / "synthetic.npz"
    create_synthetic_npz(archive_path, signer_count=9)
    dataset = load_npz_dataset(archive_path)
    first = split_by_signer(dataset, tmp_path / "splits-one", seed=9)
    second = split_by_signer(dataset, tmp_path / "splits-two", seed=9)
    loaded = load_split_directory(tmp_path / "splits-one", dataset)

    assert {
        key: value.sample_ids for key, value in first.items()
    } == {key: value.sample_ids for key, value in second.items()}
    assert set(loaded) == {"train", "validation", "test"}
    assert set().union(*(set(item.sample_ids) for item in loaded.values())) == {
        record.sample_id for record in dataset.records
    }
    signer_sets = [set(item.signer_ids) for item in loaded.values()]
    assert signer_sets[0].isdisjoint(signer_sets[1])
    assert signer_sets[0].isdisjoint(signer_sets[2])
    assert signer_sets[1].isdisjoint(signer_sets[2])
    for path in (tmp_path / "splits-one").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["seed"] == 9
        assert payload["dataset_version"] == dataset.dataset_version


def test_split_fails_with_too_few_signers(tmp_path):
    archive_path = tmp_path / "two-signers.npz"
    create_synthetic_npz(archive_path, signer_count=2)
    dataset = load_npz_dataset(archive_path)

    with pytest.raises(ValueError, match="at least 3 distinct signers"):
        split_by_signer(dataset, tmp_path / "splits")


def test_split_manifest_rejects_different_dataset_build(tmp_path):
    original_path = tmp_path / "original.npz"
    changed_path = tmp_path / "changed.npz"
    create_synthetic_npz(original_path)
    create_synthetic_npz(changed_path)
    with np.load(changed_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["features"][0, 0, 0, 0] += 0.25
    np.savez_compressed(changed_path, **arrays)
    original_dataset = load_npz_dataset(original_path)
    changed_dataset = load_npz_dataset(changed_path)
    split_by_signer(original_dataset, tmp_path / "splits")

    with pytest.raises(ValueError, match="dataset version does not match"):
        load_split_directory(tmp_path / "splits", changed_dataset)


def test_training_checkpoint_safe_load_and_test_evaluation(tmp_path):
    archive_path = tmp_path / "synthetic.npz"
    create_synthetic_npz(archive_path, signer_count=3)
    dataset = load_npz_dataset(archive_path)
    manifests = split_by_signer(
        dataset,
        tmp_path / "splits",
        fractions=(1 / 3, 1 / 3, 1 / 3),
    )
    result = train_model(
        dataset,
        manifests,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            hidden_size=4,
            dropout=0.0,
            seed=13,
            output_directory=tmp_path / "model",
        ),
    )
    loaded_model, vocabulary, checkpoint = load_model_checkpoint(
        result.best_checkpoint, "cpu"
    )
    record = dataset.records[0]
    with torch.inference_mode():
        logits = loaded_model(
            torch.from_numpy(record.features).unsqueeze(0),
            torch.from_numpy(record.hand_observed).unsqueeze(0),
            torch.tensor([len(record.features)]),
        )
    report = evaluate_checkpoint(archive_path, tmp_path / "splits", result.best_checkpoint)

    assert result.device == "cpu"
    assert result.best_checkpoint.is_file()
    assert result.vocabulary_path.is_file()
    assert result.history_path.is_file()
    assert checkpoint["dataset_version"] == dataset.dataset_version
    assert checkpoint["preprocessing_schema_version"] == 1
    assert checkpoint["model_config"]["architecture"] == "bidirectional_lstm_ctc"
    assert checkpoint["best_epoch"] == 1
    assert vocabulary.class_count == 3
    assert logits.shape == (1, 4, 3)
    assert report["evaluation_split"] == "test"
    assert report["split_counts"]["test"]["signer_count"] == 1
    assert "metrics" in report
    assert report["test_missing_hand_statistics"]["left_hand_missing_frames"] == 4
    assert any("Fewer than 10 test samples" in warning for warning in report["warnings"])


def test_training_reports_impossible_ctc_alignment(tmp_path):
    archive_path = tmp_path / "synthetic.npz"
    create_synthetic_npz(archive_path, signer_count=3)
    initial_dataset = load_npz_dataset(archive_path)
    initial_manifests = split_by_signer(
        initial_dataset,
        tmp_path / "initial-splits",
        fractions=(1 / 3, 1 / 3, 1 / 3),
    )
    train_sample_id = initial_manifests["train"].sample_ids[0]
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    sample_index = arrays["sample_ids"].tolist().index(train_sample_id)
    labels = arrays["gloss_sequences"].tolist()
    labels[sample_index] = json.dumps(["HELLO"] * 5)
    arrays["gloss_sequences"] = np.asarray(labels, dtype=np.str_)
    np.savez_compressed(archive_path, **arrays)
    altered_dataset = load_npz_dataset(archive_path)
    manifests = split_by_signer(
        altered_dataset,
        tmp_path / "splits",
        seed=42,
        fractions=(1 / 3, 1 / 3, 1 / 3),
    )

    with pytest.raises(ValueError, match="CTC requires at least"):
        train_model(
            altered_dataset,
            manifests,
            TrainingConfig(
                epochs=1,
                hidden_size=4,
                output_directory=tmp_path / "model",
            ),
        )


def test_edit_distance_and_sequence_metrics():
    assert edit_distance(["HELLO", "WORLD"], ["HELLO", "THERE"]) == 1
    assert edit_distance(list("kitten"), list("sitting")) == 3
    result = sequence_metrics(
        [["HELLO", "WORLD"], ["HELLO"]],
        [["HELLO", "THERE"], ["HELLO"]],
        minimum_per_gloss_occurrences=1,
    )
    assert result["gloss_token_error_rate"] == pytest.approx(1 / 3)
    assert result["exact_match_accuracy"] == pytest.approx(0.5)
    assert result["character_error_rate"] > 0
    assert result["per_gloss"]["HELLO"]["f1"] == 1.0
    assert result["per_gloss"]["WORLD"]["recall"] == 0.0


def test_synthetic_cpu_smoke_training_is_explicitly_labeled(tmp_path):
    run_smoke_training(tmp_path / "smoke-output", epochs=1)
    _, _, checkpoint = load_model_checkpoint(
        tmp_path / "smoke-output" / "best_model.pt", "cpu"
    )

    assert "SYNTHETIC SMOKE TEST ONLY" in checkpoint["data_description"]
