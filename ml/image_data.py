"""Image-folder dataset support for isolated sign/character classification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from ml.image_labels import IMAGE_CLASS_LABELS

ImageSplit = Literal["train", "validation", "test"]
IMAGE_SCHEMA_VERSION = 1
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_INDEX_PATTERN = re.compile(r"^.+_(\d+)$")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    label: str
    numeric_index: int
    sha256: str
    split: ImageSplit


@dataclass(frozen=True)
class ImageDatasetManifest:
    schema_version: int
    dataset_version: str
    classes: tuple[str, ...]
    split_counts: dict[str, int]
    split_method: str
    exact_duplicates_kept_together: int
    records: tuple[ImageRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "classes": list(self.classes),
            "split_counts": self.split_counts,
            "split_method": self.split_method,
            "exact_duplicates_kept_together": self.exact_duplicates_kept_together,
            "splits": {
                name: [
                    record.relative_path
                    for record in self.records
                    if record.split == name
                ]
                for name in ("train", "validation", "test")
            },
        }


def load_image_manifest(dataset_root: Path) -> ImageDatasetManifest:
    """Scan class folders and create ordered, duplicate-safe dataset splits.

    Filename suffixes are treated as capture order. Each class is partitioned
    into contiguous 70/15/15 ranges so neighboring frames are not randomly
    scattered across splits. Exact duplicate images that cross a boundary are
    moved into the earliest split containing that image.
    """
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"image dataset directory does not exist: {dataset_root}")

    class_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(class_dirs) < 2:
        raise ValueError("image dataset must contain at least two class directories")

    provisional: list[ImageRecord] = []
    dataset_hasher = hashlib.sha256()
    class_names = {path.name for path in class_dirs}
    unknown_classes = class_names.difference(IMAGE_CLASS_LABELS)
    if unknown_classes:
        raise ValueError(f"unsupported image class labels: {sorted(unknown_classes)}")
    classes = tuple(label for label in IMAGE_CLASS_LABELS if label in class_names)

    for class_dir in class_dirs:
        image_paths = sorted(
            (
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            ),
            key=lambda path: path.name,
        )
        indexed_paths: list[tuple[int, Path]] = []
        for path in image_paths:
            match = _INDEX_PATTERN.fullmatch(path.stem)
            if match is None:
                raise ValueError(
                    f"image filename must end with an underscore and numeric index: "
                    f"{path.relative_to(root)}"
                )
            indexed_paths.append((int(match.group(1)), path))
        indexed_paths.sort(key=lambda item: (item[0], item[1].name))
        if len(indexed_paths) < 3:
            raise ValueError(
                f"class {class_dir.name!r} needs at least 3 supported images"
            )
        indices = [index for index, _ in indexed_paths]
        if len(indices) != len(set(indices)):
            raise ValueError(f"class {class_dir.name!r} has duplicate numeric indices")

        count = len(indexed_paths)
        train_end = max(1, int(count * 0.70))
        validation_end = max(train_end + 1, int(count * 0.85))
        validation_end = min(validation_end, count - 1)

        for position, (numeric_index, path) in enumerate(indexed_paths):
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"invalid image file: {path.relative_to(root)}") from exc

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative_path = path.relative_to(root).as_posix()
            if position < train_end:
                split: ImageSplit = "train"
            elif position < validation_end:
                split = "validation"
            else:
                split = "test"
            record = ImageRecord(
                path=path,
                relative_path=relative_path,
                label=class_dir.name,
                numeric_index=numeric_index,
                sha256=digest,
                split=split,
            )
            provisional.append(record)
            dataset_hasher.update(relative_path.encode("utf-8"))
            dataset_hasher.update(b"\0")
            dataset_hasher.update(digest.encode("ascii"))
            dataset_hasher.update(b"\n")

    split_order = {"train": 0, "validation": 1, "test": 2}
    first_split_by_digest: dict[str, ImageSplit] = {}
    duplicate_groups: dict[str, set[ImageSplit]] = {}
    for record in provisional:
        duplicate_groups.setdefault(record.sha256, set()).add(record.split)
        current = first_split_by_digest.get(record.sha256)
        if current is None or split_order[record.split] < split_order[current]:
            first_split_by_digest[record.sha256] = record.split

    records: list[ImageRecord] = []
    for record in provisional:
        target_split = first_split_by_digest[record.sha256]
        records.append(
            record
            if record.split == target_split
            else ImageRecord(
                path=record.path,
                relative_path=record.relative_path,
                label=record.label,
                numeric_index=record.numeric_index,
                sha256=record.sha256,
                split=target_split,
            )
        )

    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test")
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError("dataset split produced an empty train, validation, or test set")

    return ImageDatasetManifest(
        schema_version=IMAGE_SCHEMA_VERSION,
        dataset_version=f"sha256:{dataset_hasher.hexdigest()}",
        classes=classes,
        split_counts=counts,
        split_method="ordered_per_class_numeric_suffix_70_15_15_exact_duplicates_grouped",
        exact_duplicates_kept_together=sum(
            len(splits) > 1 for splits in duplicate_groups.values()
        ),
        records=tuple(records),
    )


class StaticSignImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Lazily decode RGB images into normalized CHW tensors."""

    def __init__(
        self,
        records: tuple[ImageRecord, ...] | list[ImageRecord],
        class_to_idx: dict[str, int],
        image_size: int = 64,
    ) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16 pixels")
        self.records = tuple(records)
        self.class_to_idx = dict(class_to_idx)
        self.image_size = image_size
        if not self.records:
            raise ValueError("image dataset split cannot be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            pixels = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
        return tensor, self.class_to_idx[record.label]


def write_image_manifest(manifest: ImageDatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
