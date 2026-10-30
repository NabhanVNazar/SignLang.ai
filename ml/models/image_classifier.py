"""Small CNN baseline for isolated RGB sign-character images."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StaticSignCNN(nn.Module):
    """Classify a single RGB image; this model does not recognize motion."""

    def __init__(self, class_count: int, dropout: float = 0.2) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.class_count = class_count
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, class_count),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        if images.shape[0] == 0 or min(images.shape[2:]) < 8:
            raise ValueError("images must have a non-empty batch and dimensions >= 8")
        return self.classifier(self.features(images))
