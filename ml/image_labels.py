"""Canonical class vocabulary for isolated letter/digit image models."""

from __future__ import annotations

IMAGE_CLASS_LABELS = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_CLASS_TO_IDX = {
    label: index for index, label in enumerate(IMAGE_CLASS_LABELS)
}


def class_mapping_for(
    labels: tuple[str, ...] | list[str], *, allow_subset: bool = False
) -> dict[str, int]:
    """Build the canonical mapping, optionally for explicitly marked test runs."""
    label_set = set(labels)
    if len(label_set) != len(labels):
        raise ValueError("class labels must be unique")
    unknown = label_set.difference(IMAGE_CLASS_TO_IDX)
    if unknown:
        raise ValueError(f"unsupported image class labels: {sorted(unknown)}")
    if not allow_subset and label_set != set(IMAGE_CLASS_LABELS):
        missing = sorted(set(IMAGE_CLASS_LABELS).difference(label_set))
        raise ValueError(f"dataset must contain all 36 labels; missing: {missing}")
    return {
        label: IMAGE_CLASS_TO_IDX[label]
        for label in IMAGE_CLASS_LABELS
        if label in label_set
    }
