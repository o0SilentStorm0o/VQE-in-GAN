"""Frozen MNIST classifier used for semantic and feature-space evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class MnistFeatureClassifier(nn.Module):
    """Architecture matching the repository's independently trained classifier checkpoint."""

    def __init__(self, *, num_classes: int = 10) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        logits, _ = self.forward_with_features(images)
        return logits

    def forward_with_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return class logits and a stable 64-dimensional convolutional feature vector."""

        features = self.conv_block1(images)
        features = self.conv_block2(features)
        pooled_features = features.mean(dim=(-2, -1))
        flat_features = features.flatten(start_dim=1)
        logits = self.fc_block(flat_features)
        return logits, pooled_features


def load_mnist_classifier(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> MnistFeatureClassifier:
    classifier = MnistFeatureClassifier().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier
