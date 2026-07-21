"""Isolated optimizer steps with an explicit quantum-computation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor

from vqe_gan.models.acgan import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.losses import contrastive_energy_loss


class AllClassEnergyBackend(Protocol):
    def all_energies(self, angles: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class DiscriminatorStepMetrics:
    total: float
    real_adversarial: float
    fake_adversarial: float
    real_auxiliary: float
    fake_auxiliary: float


@dataclass(frozen=True)
class GeneratorStepMetrics:
    total: float
    adversarial: float
    auxiliary: float
    quantum: float
    mean_target_energy: float


def discriminator_step(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    optimizer: torch.optim.Optimizer,
    real_images: Tensor,
    real_labels: Tensor,
    noise: Tensor,
    generated_labels: Tensor,
    *,
    real_label_smoothing: float = 0.9,
) -> DiscriminatorStepMetrics:
    """Update only the discriminator; no circuit angles or energies are computed."""

    if not 0 < real_label_smoothing <= 1:
        raise ValueError("real_label_smoothing must be in (0, 1]")
    optimizer.zero_grad(set_to_none=True)
    generator_was_training = generator.training
    generator.eval()
    try:
        with torch.no_grad():
            fake_images = generator(noise, generated_labels)
    finally:
        generator.train(generator_was_training)

    real_logits, real_class_logits = discriminator(real_images)
    fake_logits, fake_class_logits = discriminator(fake_images)
    real_adversarial = functional.binary_cross_entropy_with_logits(
        real_logits,
        torch.full_like(real_logits, real_label_smoothing),
    )
    fake_adversarial = functional.binary_cross_entropy_with_logits(
        fake_logits,
        torch.zeros_like(fake_logits),
    )
    real_auxiliary = functional.cross_entropy(real_class_logits, real_labels)
    fake_auxiliary = functional.cross_entropy(fake_class_logits, generated_labels)
    total = 0.5 * (
        real_adversarial + fake_adversarial + real_auxiliary + fake_auxiliary
    )
    total.backward()
    optimizer.step()

    return DiscriminatorStepMetrics(
        total=total.detach().item(),
        real_adversarial=real_adversarial.detach().item(),
        fake_adversarial=fake_adversarial.detach().item(),
        real_auxiliary=real_auxiliary.detach().item(),
        fake_auxiliary=fake_auxiliary.detach().item(),
    )


def generator_step(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    energy_backend: AllClassEnergyBackend | None,
    optimizer: torch.optim.Optimizer,
    noise: Tensor,
    class_labels: Tensor,
    *,
    quantum_weight: float,
    quantum_temperature: float = 1.0,
) -> GeneratorStepMetrics:
    """Update the generator while keeping discriminator parameters gradient-free."""

    if quantum_weight < 0:
        raise ValueError("quantum_weight must be non-negative")
    if quantum_weight > 0 and energy_backend is None:
        raise ValueError("energy_backend is required when quantum_weight is positive")

    optimizer.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    previous_requires_grad = [parameter.requires_grad for parameter in discriminator.parameters()]
    discriminator_was_training = discriminator.training
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    try:
        if quantum_weight > 0:
            images, angles = generator.forward_with_angles(noise, class_labels)
            assert energy_backend is not None
            all_energies = energy_backend.all_energies(angles)
            quantum = contrastive_energy_loss(
                all_energies,
                class_labels,
                temperature=quantum_temperature,
            )
            target_energies = all_energies.gather(1, class_labels.unsqueeze(1)).squeeze(1)
            mean_target_energy = target_energies.mean()
        else:
            images = generator(noise, class_labels)
            quantum = images.new_zeros(())
            mean_target_energy = images.new_zeros(())

        adversarial_logits, class_logits = discriminator(images)
        adversarial = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary = functional.cross_entropy(class_logits, class_labels)
        total = adversarial + auxiliary + quantum_weight * quantum
        total.backward()
        optimizer.step()
    finally:
        discriminator.train(discriminator_was_training)
        for parameter, requires_grad in zip(
            discriminator.parameters(),
            previous_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

    return GeneratorStepMetrics(
        total=total.detach().item(),
        adversarial=adversarial.detach().item(),
        auxiliary=auxiliary.detach().item(),
        quantum=quantum.detach().item(),
        mean_target_energy=mean_target_energy.detach().item(),
    )
