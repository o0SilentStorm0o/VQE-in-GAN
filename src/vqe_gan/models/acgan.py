"""ACGAN modules with an explicit shared generator representation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SharedQuantumGenerator(nn.Module):
    """MNIST ACGAN generator whose circuit head consumes the generated image.

    The classical image path retains the historical label embedding, input projection, and
    convolutional decoder. The angle head reads only the completed generated image. Consequently,
    regularizer gradients traverse the full image path and cannot read labels through a separate
    shortcut.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 100,
        num_classes: int = 10,
        image_channels: int = 1,
        num_angles: int = 16,
        feature_channels: int = 256,
    ) -> None:
        super().__init__()
        if min(latent_dim, num_classes, image_channels, num_angles, feature_channels) <= 0:
            raise ValueError("all generator dimensions must be positive")

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_channels = image_channels
        self.num_angles = num_angles
        self.feature_channels = feature_channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.input_projection = nn.Linear(
            latent_dim + num_classes,
            feature_channels * 7 * 7,
        )
        self.image_decoder = nn.Sequential(
            nn.BatchNorm2d(feature_channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        angle_output = nn.Linear(128, num_angles)
        angle_output.is_angle_output = True
        self.angle_head = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            angle_output,
            nn.Tanh(),
        )

    def shared_features(self, noise: Tensor, class_labels: Tensor) -> Tensor:
        """Return the representation consumed by both output heads."""

        self._validate_inputs(noise, class_labels)
        embedded_labels = self.label_embedding(class_labels)
        conditioned_input = torch.cat((noise, embedded_labels), dim=1)
        flat_features = self.input_projection(conditioned_input)
        return flat_features.view(noise.shape[0], self.feature_channels, 7, 7)

    def images_from_features(self, features: Tensor) -> Tensor:
        return self.image_decoder(features)

    def angles_from_images(self, images: Tensor) -> Tensor:
        return self.angle_head(images) * torch.pi

    def forward(self, noise: Tensor, class_labels: Tensor) -> Tensor:
        """Generate images without evaluating the angle head or a quantum backend."""

        return self.images_from_features(self.shared_features(noise, class_labels))

    def forward_with_angles(
        self,
        noise: Tensor,
        class_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Generate images and circuit angles from one shared-feature computation."""

        features = self.shared_features(noise, class_labels)
        images = self.images_from_features(features)
        return images, self.angles_from_images(images)

    def quantum_angles(self, noise: Tensor, class_labels: Tensor) -> Tensor:
        """Produce circuit angles without running the image decoder."""

        return self.angles_from_images(self(noise, class_labels))

    def _validate_inputs(self, noise: Tensor, class_labels: Tensor) -> None:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(
                f"noise must have shape (batch, {self.latent_dim}), got {tuple(noise.shape)}"
            )
        if class_labels.ndim != 1 or class_labels.shape[0] != noise.shape[0]:
            raise ValueError("class_labels must have shape (batch,)")
        if class_labels.dtype != torch.long:
            raise TypeError("class_labels must have dtype torch.long")
        if torch.any(class_labels < 0) or torch.any(class_labels >= self.num_classes):
            raise ValueError("class_labels contains an out-of-range class index")


class ACGANDiscriminator(nn.Module):
    """MNIST discriminator with adversarial and auxiliary-class logits."""

    def __init__(self, *, num_classes: int = 10, image_channels: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_channels = image_channels
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adversarial_head = nn.Linear(256 * 4 * 4, 1)
        self.class_head = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(images)
        flat_features = features.flatten(start_dim=1)
        return self.adversarial_head(flat_features), self.class_head(flat_features)


def initialize_weights(module: nn.Module) -> None:
    """Apply the initialization used by the historical ACGAN implementation."""

    if isinstance(module, nn.Conv2d | nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            if getattr(module, "is_angle_output", False):
                with torch.no_grad():
                    module.bias.copy_(torch.linspace(-0.75, 0.75, module.bias.numel()))
            else:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
