"""Shared hand-landmark feature construction for training and inference."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from ml.dataset import HAND_LANDMARK_COUNT

HandLandmarks: TypeAlias = Iterable[Iterable[float]] | None


def build_hand_features(
    frames: Iterable[tuple[HandLandmarks, HandLandmarks]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build left/right float32 landmarks and explicit per-hand observation masks."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("at least one frame is required")
    features = np.zeros((len(frame_list), 2, HAND_LANDMARK_COUNT, 3), dtype=np.float32)
    observed = np.zeros((len(frame_list), 2), dtype=np.bool_)
    for frame_index, hands in enumerate(frame_list):
        if len(hands) != 2:
            raise ValueError("each frame must contain left and right hand values")
        for hand_index, hand in enumerate(hands):
            if hand is None:
                continue
            points = np.asarray(list(hand), dtype=np.float32)
            if points.shape != (HAND_LANDMARK_COUNT, 3):
                raise ValueError(
                    f"hand landmarks must have shape ({HAND_LANDMARK_COUNT}, 3)"
                )
            if not np.isfinite(points).all():
                raise ValueError("hand landmarks must contain finite values")
            features[frame_index, hand_index] = points
            observed[frame_index, hand_index] = True
    return features, observed
