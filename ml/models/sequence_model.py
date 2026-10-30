"""CPU-friendly bidirectional LSTM for frame-level CTC gloss recognition."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


HAND_COUNT = 2
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
INPUT_SIZE = HAND_COUNT * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK + HAND_COUNT


class HandLandmarkBiLSTM(nn.Module):
    """Predict per-frame gloss logits from explicit landmarks and hand masks."""

    def __init__(
        self,
        class_count: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must include blank and at least one gloss")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.class_count = class_count
        self.hidden_size = hidden_size
        self.dropout_probability = dropout
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, class_count)

    def forward(
        self,
        features: Tensor,
        hand_observed: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        if features.ndim != 5 or tuple(features.shape[2:]) != (2, 21, 3):
            raise ValueError(
                "features must have shape (batch, time, 2, 21, 3)"
            )
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must have a non-empty batch and time dimension")
        if hand_observed.shape != features.shape[:2] + (2,):
            raise ValueError("hand_observed must have shape (batch, time, 2)")
        if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
            raise ValueError("lengths must have one value per batch item")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must use an integer dtype")
        if bool(torch.any(lengths <= 0)) or bool(torch.any(lengths > features.shape[1])):
            raise ValueError("each sequence length must be between 1 and padded time")

        # The observed bits distinguish an absent hand from real landmarks at zero.
        flattened = features.flatten(start_dim=2)
        inputs = torch.cat((flattened, hand_observed.to(features.dtype)), dim=-1)
        packed = pack_padded_sequence(
            inputs,
            lengths.detach().to(device="cpu", dtype=torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=features.shape[1],
        )
        return self.classifier(self.dropout(outputs))
