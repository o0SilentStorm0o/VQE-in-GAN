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


class ImageDistributionRegularizer(Protocol):
    def __call__(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor: ...


class CoherenceGuidanceRegularizer(Protocol):
    def loss_components(
        self,
        generated: Tensor,
        real: Tensor,
        class_labels: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


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
    classical_regularizer: float | None = None
    quantum_regularizer: float | None = None
    effective_quantum_weight: float | None = None
    quantum_to_gan_gradient_ratio: float | None = None
    quantum_classical_gradient_cosine: float | None = None
    quantum_gan_gradient_cosine: float | None = None
    retained_quantum_gradient_fraction: float | None = None


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


def distribution_regularized_generator_step(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer_model: ImageDistributionRegularizer,
    optimizer: torch.optim.Optimizer,
    real_images: Tensor,
    noise: Tensor,
    class_labels: Tensor,
    *,
    regularizer_weight: float,
    regularizer_gradient_ratio: float | None = None,
) -> GeneratorStepMetrics:
    """Update the generator with a real-data-anchored distribution loss.

    Generated labels intentionally equal the labels of ``real_images``. This gives every class
    present in a minibatch identical real and generated counts, making class-conditional mixed
    states and MMD estimates well-defined without a separate replay buffer.
    """

    if regularizer_weight <= 0:
        raise ValueError("regularizer_weight must be positive")
    if real_images.shape[0] != noise.shape[0] or class_labels.shape != (noise.shape[0],):
        raise ValueError("real images, noise, and class labels must share the batch dimension")
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
        images = generator(noise, class_labels)
        regularizer = regularizer_model(images, real_images, class_labels)
        if regularizer.ndim != 0 or not torch.isfinite(regularizer):
            raise RuntimeError("distribution regularizer must return one finite scalar")

        adversarial_logits, class_logits = discriminator(images)
        adversarial = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary = functional.cross_entropy(class_logits, class_labels)
        gan_objective = adversarial + auxiliary
        effective_regularizer_weight = regularizer_weight
        achieved_gradient_ratio = None
        if regularizer_gradient_ratio is not None:
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
        mean_target_energy=0.0,
    )


def coherence_guided_generator_step(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer_model: CoherenceGuidanceRegularizer,
    optimizer: torch.optim.Optimizer,
    real_images: Tensor,
    noise: Tensor,
    class_labels: Tensor,
    *,
    regularizer_weight: float,
    regularizer_gradient_ratio: float | None,
    coherence_gradient_ratio: float,
) -> GeneratorStepMetrics:
    """Add only the quantum gradient not explained by a strong classical control.

    The classical RBF loss is balanced exactly like the standalone RBF variant. The quantum
    dephasing-residual gradient is then projected out of the classical-gradient direction. If its
    remaining component conflicts with the GAN gradient in that orthogonal subspace, the
    conflicting component is removed as well. The retained direction receives its own fixed
    gradient budget. This makes any improvement incremental to the classical control and prevents
    the quantum term from worsening the current GAN objective to first order.
    """

    if regularizer_weight <= 0:
        raise ValueError("regularizer_weight must be positive")
    if regularizer_gradient_ratio is not None and regularizer_gradient_ratio <= 0:
        raise ValueError("regularizer_gradient_ratio must be positive when set")
    if coherence_gradient_ratio <= 0:
        raise ValueError("coherence_gradient_ratio must be positive")
    if real_images.shape[0] != noise.shape[0] or class_labels.shape != (noise.shape[0],):
        raise ValueError("real images, noise, and class labels must share the batch dimension")

    optimizer.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    previous_requires_grad = [parameter.requires_grad for parameter in discriminator.parameters()]
    discriminator_was_training = discriminator.training
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    try:
        images = generator(noise, class_labels)
        classical_loss, quantum_loss = regularizer_model.loss_components(
            images,
            real_images,
            class_labels,
        )
        for name, loss in (("classical", classical_loss), ("quantum", quantum_loss)):
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(f"{name} regularizer must return one finite scalar")

        adversarial_logits, class_logits = discriminator(images)
        adversarial = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary = functional.cross_entropy(class_logits, class_labels)
        gan_objective = adversarial + auxiliary
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
        classical_gradients = torch.autograd.grad(
            classical_loss,
            shared_parameters,
            retain_graph=True,
        )
        quantum_gradients = torch.autograd.grad(
            quantum_loss,
            shared_parameters,
            retain_graph=False,
        )
        gan_norm = _tensor_gradient_norm(gan_gradients)
        classical_norm = _tensor_gradient_norm(classical_gradients)
        quantum_norm = _tensor_gradient_norm(quantum_gradients)
        epsilon = torch.finfo(gan_norm.dtype).eps
        if classical_norm.item() <= epsilon:
            raise RuntimeError("classical regularizer gradient is too small for balancing")

        if regularizer_gradient_ratio is None:
            classical_weight = gan_norm.new_tensor(regularizer_weight)
        else:
            classical_weight = (
                regularizer_gradient_ratio * gan_norm / classical_norm
            ).detach()
        weighted_classical = _scale_gradients(classical_gradients, classical_weight)

        quantum_classical_cosine = _gradient_cosine(
            quantum_gradients,
            classical_gradients,
        )
        quantum_gan_cosine = _gradient_cosine(quantum_gradients, gan_gradients)
        quantum_unique = _remove_gradient_projection(
            quantum_gradients,
            classical_gradients,
        )
        gan_unique = _remove_gradient_projection(gan_gradients, classical_gradients)
        if _gradient_dot(quantum_unique, gan_unique).item() < 0:
            quantum_safe = _remove_gradient_projection(quantum_unique, gan_unique)
        else:
            quantum_safe = quantum_unique
        quantum_safe_norm = _tensor_gradient_norm(quantum_safe)
        retained_fraction = (
            (quantum_safe_norm / quantum_norm).item() if quantum_norm.item() > epsilon else 0.0
        )
        if quantum_safe_norm.item() <= epsilon:
            quantum_weight = gan_norm.new_zeros(())
            weighted_quantum = tuple(torch.zeros_like(value) for value in quantum_safe)
            achieved_quantum_ratio = 0.0
        else:
            quantum_weight = (
                coherence_gradient_ratio * gan_norm / quantum_safe_norm
            ).detach()
            weighted_quantum = _scale_gradients(quantum_safe, quantum_weight)
            achieved_quantum_ratio = coherence_gradient_ratio

        final_gradients = tuple(
            gan + classical + quantum
            for gan, classical, quantum in zip(
                gan_gradients,
                weighted_classical,
                weighted_quantum,
                strict=True,
            )
        )
        for parameter, gradient in zip(shared_parameters, final_gradients, strict=True):
            parameter.grad = gradient.detach()
        optimizer.step()

        combined_regularizer_gradients = tuple(
            classical + quantum
            for classical, quantum in zip(
                weighted_classical,
                weighted_quantum,
                strict=True,
            )
        )
        combined_ratio = (
            _tensor_gradient_norm(combined_regularizer_gradients) / gan_norm
        ).item()
        total = (
            gan_objective.detach()
            + classical_weight * classical_loss.detach()
            + quantum_weight * quantum_loss.detach()
        )
    finally:
        discriminator.train(discriminator_was_training)
        for parameter, requires_grad in zip(
            discriminator.parameters(),
            previous_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

    return GeneratorStepMetrics(
        total=total.item(),
        adversarial=adversarial.detach().item(),
        auxiliary=auxiliary.detach().item(),
        regularizer=(classical_loss + quantum_loss).detach().item(),
        effective_regularizer_weight=classical_weight.item(),
        regularizer_to_gan_gradient_ratio=combined_ratio,
        mean_target_energy=0.0,
        classical_regularizer=classical_loss.detach().item(),
        quantum_regularizer=quantum_loss.detach().item(),
        effective_quantum_weight=quantum_weight.item(),
        quantum_to_gan_gradient_ratio=achieved_quantum_ratio,
        quantum_classical_gradient_cosine=quantum_classical_cosine,
        quantum_gan_gradient_cosine=quantum_gan_cosine,
        retained_quantum_gradient_fraction=retained_fraction,
    )


def _tensor_gradient_norm(gradients: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()


def _gradient_dot(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> Tensor:
    return torch.stack(
        [(left * right).sum() for left, right in zip(first, second, strict=True)]
    ).sum()


def _gradient_cosine(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> float | None:
    denominator = _tensor_gradient_norm(first) * _tensor_gradient_norm(second)
    if denominator.item() <= torch.finfo(denominator.dtype).eps:
        return None
    return (_gradient_dot(first, second) / denominator).item()


def _scale_gradients(
    gradients: tuple[Tensor, ...],
    scale: Tensor,
) -> tuple[Tensor, ...]:
    return tuple(scale * gradient for gradient in gradients)


def _remove_gradient_projection(
    source: tuple[Tensor, ...],
    reference: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    reference_squared_norm = _gradient_dot(reference, reference)
    if reference_squared_norm.item() <= torch.finfo(reference_squared_norm.dtype).eps:
        return source
    coefficient = _gradient_dot(source, reference) / reference_squared_norm
    return tuple(
        value - coefficient * direction
        for value, direction in zip(source, reference, strict=True)
    )
