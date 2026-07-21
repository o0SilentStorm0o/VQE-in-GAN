"""Read-only diagnostics for gradient interaction and label shortcuts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.losses import contrastive_energy_loss
from vqe_gan.training.steps import (
    AllClassEnergyModel,
    ImageDistributionRegularizer,
    RelationalCoverageRegularizer,
)


@dataclass(frozen=True)
class GradientDiagnostics:
    adversarial_norm: float
    auxiliary_norm: float
    gan_objective_norm: float
    regularizer_norm: float
    regularizer_to_gan_norm_ratio: float
    effective_regularizer_weight: float
    regularizer_label_embedding_norm: float
    regularizer_input_projection_norm: float
    regularizer_image_decoder_norm: float
    adversarial_auxiliary_cosine: float | None
    regularizer_gan_objective_cosine: float | None
    regularizer_adversarial_cosine: float | None
    regularizer_auxiliary_cosine: float | None
    regularizer_accuracy: float | None
    zero_noise_regularizer_accuracy: float | None
    angle_noise_sensitivity: float | None


@dataclass(frozen=True)
class RelationalGradientDiagnostics:
    gan_objective_norm: float
    kde_norm: float
    coverage_shared_norm: float
    coverage_angle_head_norm: float
    direct_coverage_shared_norm: float
    indirect_coverage_shared_norm: float
    direct_to_full_coverage_norm_ratio: float
    weighted_kde_to_gan_norm_ratio: float
    weighted_coverage_to_gan_norm_ratio: float
    kde_gan_cosine: float | None
    coverage_gan_cosine: float | None
    coverage_kde_cosine: float | None
    direct_full_coverage_cosine: float | None
    kde_loss: float
    coverage_loss: float
    angle_residual_rms: float


def measure_gradient_diagnostics(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    energy_model: AllClassEnergyModel | None,
    noise: Tensor,
    class_labels: Tensor,
    *,
    regularizer_weight: float,
    regularizer_temperature: float,
    regularizer_gradient_ratio: float | None = None,
) -> GradientDiagnostics:
    """Measure loss-gradient relationships without changing model state or parameter gradients."""

    if regularizer_weight < 0:
        raise ValueError("regularizer_weight must be non-negative")
    if regularizer_weight > 0 and energy_model is None:
        raise ValueError("energy_model is required when regularizer_weight is positive")
    if regularizer_gradient_ratio is not None and regularizer_gradient_ratio <= 0:
        raise ValueError("regularizer_gradient_ratio must be positive when set")

    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    discriminator_requires_grad = [
        parameter.requires_grad for parameter in discriminator.parameters()
    ]
    generator.eval()
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    label_parameters = tuple(generator.label_embedding.parameters())
    projection_parameters = tuple(generator.input_projection.parameters())
    decoder_parameters = tuple(generator.image_decoder.parameters())
    shared_parameters = (*label_parameters, *projection_parameters, *decoder_parameters)
    try:
        if energy_model is not None and regularizer_weight > 0:
            images, angles = generator.forward_with_angles(noise, class_labels)
            all_energies = energy_model.all_energies(angles)
            regularizer_loss = contrastive_energy_loss(
                all_energies,
                class_labels,
                temperature=regularizer_temperature,
            )
            regularizer_accuracy = _energy_accuracy(all_energies, class_labels)
            with torch.no_grad():
                zero_angles = generator.quantum_angles(torch.zeros_like(noise), class_labels)
                zero_energies = energy_model.all_energies(zero_angles)
                zero_noise_accuracy = _energy_accuracy(zero_energies, class_labels)
                angle_noise_sensitivity = (
                    (angles.detach() - zero_angles).square().mean().sqrt().item()
                )
        else:
            images = generator(noise, class_labels)
            regularizer_loss = None
            regularizer_accuracy = None
            zero_noise_accuracy = None
            angle_noise_sensitivity = None

        adversarial_logits, class_logits = discriminator(images)
        adversarial_loss = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary_loss = functional.cross_entropy(class_logits, class_labels)

        adversarial_gradients = _gradients(
            adversarial_loss,
            shared_parameters,
            retain_graph=True,
        )
        auxiliary_gradients = _gradients(
            auxiliary_loss,
            shared_parameters,
            retain_graph=regularizer_loss is not None,
        )
        regularizer_gradients = (
            _gradients(regularizer_loss, shared_parameters, retain_graph=False)
            if regularizer_loss is not None
            else tuple(torch.zeros_like(parameter) for parameter in shared_parameters)
        )
    finally:
        generator.train(generator_was_training)
        discriminator.train(discriminator_was_training)
        for parameter, requires_grad in zip(
            discriminator.parameters(),
            discriminator_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

    label_count = len(label_parameters)
    projection_end = label_count + len(projection_parameters)
    gan_gradients = tuple(
        adversarial + auxiliary
        for adversarial, auxiliary in zip(
            adversarial_gradients,
            auxiliary_gradients,
            strict=True,
        )
    )
    gan_norm = _norm(gan_gradients)
    unweighted_regularizer_norm = _norm(regularizer_gradients)
    effective_regularizer_weight = regularizer_weight
    if regularizer_loss is not None and regularizer_gradient_ratio is not None:
        if unweighted_regularizer_norm <= torch.finfo(noise.dtype).eps:
            raise RuntimeError("regularizer gradient is too small for dynamic balancing")
        effective_regularizer_weight = (
            regularizer_gradient_ratio * gan_norm / unweighted_regularizer_norm
        )
    regularizer_gradients = tuple(
        effective_regularizer_weight * gradient for gradient in regularizer_gradients
    )
    regularizer_norm = _norm(regularizer_gradients)
    return GradientDiagnostics(
        adversarial_norm=_norm(adversarial_gradients),
        auxiliary_norm=_norm(auxiliary_gradients),
        gan_objective_norm=gan_norm,
        regularizer_norm=regularizer_norm,
        regularizer_to_gan_norm_ratio=(regularizer_norm / gan_norm if gan_norm > 0 else 0.0),
        effective_regularizer_weight=effective_regularizer_weight,
        regularizer_label_embedding_norm=_norm(regularizer_gradients[:label_count]),
        regularizer_input_projection_norm=_norm(regularizer_gradients[label_count:projection_end]),
        regularizer_image_decoder_norm=_norm(regularizer_gradients[projection_end:]),
        adversarial_auxiliary_cosine=_cosine(adversarial_gradients, auxiliary_gradients),
        regularizer_gan_objective_cosine=_cosine(
            regularizer_gradients,
            gan_gradients,
        ),
        regularizer_adversarial_cosine=_cosine(
            regularizer_gradients,
            adversarial_gradients,
        ),
        regularizer_auxiliary_cosine=_cosine(
            regularizer_gradients,
            auxiliary_gradients,
        ),
        regularizer_accuracy=regularizer_accuracy,
        zero_noise_regularizer_accuracy=zero_noise_accuracy,
        angle_noise_sensitivity=angle_noise_sensitivity,
    )


def measure_distribution_gradient_diagnostics(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer_model: ImageDistributionRegularizer,
    real_images: Tensor,
    noise: Tensor,
    class_labels: Tensor,
    *,
    regularizer_weight: float,
    regularizer_gradient_ratio: float | None = None,
) -> GradientDiagnostics:
    """Measure a real-data distribution loss against both ACGAN generator objectives."""

    if regularizer_weight <= 0:
        raise ValueError("regularizer_weight must be positive")
    if regularizer_gradient_ratio is not None and regularizer_gradient_ratio <= 0:
        raise ValueError("regularizer_gradient_ratio must be positive when set")

    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    discriminator_requires_grad = [
        parameter.requires_grad for parameter in discriminator.parameters()
    ]
    generator.eval()
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    label_parameters = tuple(generator.label_embedding.parameters())
    projection_parameters = tuple(generator.input_projection.parameters())
    decoder_parameters = tuple(generator.image_decoder.parameters())
    shared_parameters = (*label_parameters, *projection_parameters, *decoder_parameters)
    try:
        images = generator(noise, class_labels)
        regularizer_loss = regularizer_model(images, real_images, class_labels)
        adversarial_logits, class_logits = discriminator(images)
        adversarial_loss = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary_loss = functional.cross_entropy(class_logits, class_labels)
        adversarial_gradients = _gradients(
            adversarial_loss,
            shared_parameters,
            retain_graph=True,
        )
        auxiliary_gradients = _gradients(
            auxiliary_loss,
            shared_parameters,
            retain_graph=True,
        )
        regularizer_gradients = _gradients(
            regularizer_loss,
            shared_parameters,
            retain_graph=False,
        )
    finally:
        generator.train(generator_was_training)
        discriminator.train(discriminator_was_training)
        for parameter, requires_grad in zip(
            discriminator.parameters(),
            discriminator_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

    gan_gradients = tuple(
        adversarial + auxiliary
        for adversarial, auxiliary in zip(
            adversarial_gradients,
            auxiliary_gradients,
            strict=True,
        )
    )
    gan_norm = _norm(gan_gradients)
    unweighted_regularizer_norm = _norm(regularizer_gradients)
    effective_regularizer_weight = regularizer_weight
    if regularizer_gradient_ratio is not None:
        if unweighted_regularizer_norm <= torch.finfo(noise.dtype).eps:
            raise RuntimeError("regularizer gradient is too small for dynamic balancing")
        effective_regularizer_weight = (
            regularizer_gradient_ratio * gan_norm / unweighted_regularizer_norm
        )
    regularizer_gradients = tuple(
        effective_regularizer_weight * gradient for gradient in regularizer_gradients
    )
    label_count = len(label_parameters)
    projection_end = label_count + len(projection_parameters)
    regularizer_norm = _norm(regularizer_gradients)
    return GradientDiagnostics(
        adversarial_norm=_norm(adversarial_gradients),
        auxiliary_norm=_norm(auxiliary_gradients),
        gan_objective_norm=gan_norm,
        regularizer_norm=regularizer_norm,
        regularizer_to_gan_norm_ratio=(regularizer_norm / gan_norm if gan_norm > 0 else 0.0),
        effective_regularizer_weight=effective_regularizer_weight,
        regularizer_label_embedding_norm=_norm(regularizer_gradients[:label_count]),
        regularizer_input_projection_norm=_norm(regularizer_gradients[label_count:projection_end]),
        regularizer_image_decoder_norm=_norm(regularizer_gradients[projection_end:]),
        adversarial_auxiliary_cosine=_cosine(adversarial_gradients, auxiliary_gradients),
        regularizer_gan_objective_cosine=_cosine(regularizer_gradients, gan_gradients),
        regularizer_adversarial_cosine=_cosine(
            regularizer_gradients,
            adversarial_gradients,
        ),
        regularizer_auxiliary_cosine=_cosine(
            regularizer_gradients,
            auxiliary_gradients,
        ),
        regularizer_accuracy=None,
        zero_noise_regularizer_accuracy=None,
        angle_noise_sensitivity=None,
    )


def measure_relational_gradient_diagnostics(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer_model: RelationalCoverageRegularizer,
    real_images: Tensor,
    noise: Tensor,
    class_labels: Tensor,
    *,
    kde_weight: float,
    coverage_weight: float,
) -> RelationalGradientDiagnostics:
    """Measure the additive KDE and coverage paths without changing model state."""

    if kde_weight <= 0 or coverage_weight <= 0:
        raise ValueError("KDE and coverage weights must be positive")
    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    discriminator_requires_grad = [
        parameter.requires_grad for parameter in discriminator.parameters()
    ]
    generator.eval()
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    shared_parameters = (
        *generator.label_embedding.parameters(),
        *generator.input_projection.parameters(),
        *generator.image_decoder.parameters(),
    )
    angle_parameters = tuple(generator.angle_head.parameters())
    try:
        images, angle_residuals = generator.forward_with_angles(noise, class_labels)
        kde_loss, coverage_loss = regularizer_model.loss_components(
            images,
            angle_residuals,
            real_images,
            class_labels,
        )
        _, direct_coverage_loss = regularizer_model.loss_components(
            images,
            angle_residuals.detach(),
            real_images,
            class_labels,
        )
        adversarial_logits, class_logits = discriminator(images)
        gan_objective = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        ) + functional.cross_entropy(class_logits, class_labels)

        gan_gradients = _gradients(gan_objective, shared_parameters, retain_graph=True)
        kde_gradients = _gradients(kde_loss, shared_parameters, retain_graph=True)
        coverage_gradients = _gradients(
            coverage_loss,
            (*shared_parameters, *angle_parameters),
            retain_graph=True,
        )
        direct_gradients = _gradients(
            direct_coverage_loss,
            shared_parameters,
            retain_graph=False,
        )
    finally:
        generator.train(generator_was_training)
        discriminator.train(discriminator_was_training)
        for parameter, requires_grad in zip(
            discriminator.parameters(),
            discriminator_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

    coverage_shared = coverage_gradients[: len(shared_parameters)]
    coverage_angle = coverage_gradients[len(shared_parameters) :]
    indirect_gradients = tuple(
        full - direct for full, direct in zip(coverage_shared, direct_gradients, strict=True)
    )
    gan_norm = _norm(gan_gradients)
    kde_norm = _norm(kde_gradients)
    coverage_norm = _norm(coverage_shared)
    direct_norm = _norm(direct_gradients)
    return RelationalGradientDiagnostics(
        gan_objective_norm=gan_norm,
        kde_norm=kde_norm,
        coverage_shared_norm=coverage_norm,
        coverage_angle_head_norm=_norm(coverage_angle),
        direct_coverage_shared_norm=direct_norm,
        indirect_coverage_shared_norm=_norm(indirect_gradients),
        direct_to_full_coverage_norm_ratio=(
            direct_norm / coverage_norm if coverage_norm > 0 else 0.0
        ),
        weighted_kde_to_gan_norm_ratio=(kde_weight * kde_norm / gan_norm if gan_norm > 0 else 0.0),
        weighted_coverage_to_gan_norm_ratio=(
            coverage_weight * coverage_norm / gan_norm if gan_norm > 0 else 0.0
        ),
        kde_gan_cosine=_cosine(kde_gradients, gan_gradients),
        coverage_gan_cosine=_cosine(coverage_shared, gan_gradients),
        coverage_kde_cosine=_cosine(coverage_shared, kde_gradients),
        direct_full_coverage_cosine=_cosine(direct_gradients, coverage_shared),
        kde_loss=kde_loss.detach().item(),
        coverage_loss=coverage_loss.detach().item(),
        angle_residual_rms=angle_residuals.detach().square().mean().sqrt().item(),
    )


def _gradients(
    loss: Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> tuple[Tensor, ...]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        gradient if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(gradients, parameters, strict=True)
    )


def _norm(gradients: tuple[Tensor, ...]) -> float:
    squared_norm = sum((gradient.square().sum() for gradient in gradients), start=0.0)
    if not isinstance(squared_norm, Tensor):
        return 0.0
    return squared_norm.sqrt().item()


def _cosine(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> float | None:
    first_squared = sum((gradient.square().sum() for gradient in first), start=0.0)
    second_squared = sum((gradient.square().sum() for gradient in second), start=0.0)
    dot_product = sum(
        ((left * right).sum() for left, right in zip(first, second, strict=True)),
        start=0.0,
    )
    if not all(isinstance(value, Tensor) for value in (first_squared, second_squared, dot_product)):
        return None
    denominator = first_squared.sqrt() * second_squared.sqrt()
    if denominator.item() <= torch.finfo(denominator.dtype).eps:
        return None
    return (dot_product / denominator).item()


def _energy_accuracy(energies: Tensor, class_labels: Tensor) -> float:
    predictions = energies.argmin(dim=1)
    return (predictions == class_labels).float().mean().item()
