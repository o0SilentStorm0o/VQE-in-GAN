"""Isolated optimizer steps with an explicit quantum-computation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor

from vqe_gan.models.acgan import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.losses import contrastive_energy_loss


class AllClassEnergyModel(Protocol):
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
    regularizer: float
    effective_regularizer_weight: float
    regularizer_to_gan_gradient_ratio: float | None
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
    energy_model: AllClassEnergyModel | None,
    optimizer: torch.optim.Optimizer,
    noise: Tensor,
    class_labels: Tensor,
    *,
    regularizer_weight: float,
    regularizer_temperature: float = 1.0,
    regularizer_gradient_ratio: float | None = None,
) -> GeneratorStepMetrics:
    """Update the generator while keeping discriminator parameters gradient-free."""

    if regularizer_weight < 0:
        raise ValueError("regularizer_weight must be non-negative")
    if regularizer_weight > 0 and energy_model is None:
        raise ValueError("energy_model is required when regularizer_weight is positive")
    if regularizer_gradient_ratio is not None and regularizer_gradient_ratio <= 0:
        raise ValueError("regularizer_gradient_ratio must be positive when set")

    optimizer.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    previous_requires_grad = [parameter.requires_grad for parameter in discriminator.parameters()]
    discriminator_was_training = discriminator.training
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    try:
        if regularizer_weight > 0:
            images, angles = generator.forward_with_angles(noise, class_labels)
            assert energy_model is not None
            all_energies = energy_model.all_energies(angles)
            regularizer = contrastive_energy_loss(
                all_energies,
                class_labels,
                temperature=regularizer_temperature,
            )
            target_energies = all_energies.gather(1, class_labels.unsqueeze(1)).squeeze(1)
            mean_target_energy = target_energies.mean()
        else:
            images = generator(noise, class_labels)
            regularizer = images.new_zeros(())
            mean_target_energy = images.new_zeros(())

        adversarial_logits, class_logits = discriminator(images)
        adversarial = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary = functional.cross_entropy(class_logits, class_labels)
        gan_objective = adversarial + auxiliary
        effective_regularizer_weight = regularizer_weight
        achieved_gradient_ratio = None
        if regularizer_gradient_ratio is not None and regularizer_weight > 0:
            shared_parameters = (
                *generator.label_embedding.parameters(),
                *generator.input_projection.parameters(),
                *generator.image_decoder.parameters(),
            )
            gan_gradients = torch.autograd.grad(
                gan_objective,
                shared_parameters,
                retain_graph=True,
            )
            regularizer_gradients = torch.autograd.grad(
                regularizer,
                shared_parameters,
                retain_graph=True,
            )
            gan_norm = _tensor_gradient_norm(gan_gradients)
            regularizer_norm = _tensor_gradient_norm(regularizer_gradients)
            if regularizer_norm.item() <= torch.finfo(regularizer_norm.dtype).eps:
                raise RuntimeError("regularizer gradient is too small for dynamic balancing")
            effective_weight_tensor = (
                regularizer_gradient_ratio * gan_norm / regularizer_norm
            ).detach()
            effective_regularizer_weight = effective_weight_tensor.item()
            achieved_gradient_ratio = regularizer_gradient_ratio
        total = gan_objective + effective_regularizer_weight * regularizer
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
        regularizer=regularizer.detach().item(),
        effective_regularizer_weight=effective_regularizer_weight,
        regularizer_to_gan_gradient_ratio=achieved_gradient_ratio,
        mean_target_energy=mean_target_energy.detach().item(),
    )


def _tensor_gradient_norm(gradients: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
