"""Isolated optimizer steps with an explicit quantum-computation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor

from vqe_gan.models.acgan import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.losses import contrastive_energy_loss
from vqe_gan.training.budget import CoverageBudgetTarget


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


class RelationalCoverageRegularizer(Protocol):
    def loss_components(
        self,
        generated: Tensor,
        angle_residuals: Tensor,
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
    coverage_regularizer: float | None = None
    effective_coverage_weight: float | None = None
    coverage_shared_target_ratio: float | None = None
    coverage_shared_achieved_ratio: float | None = None
    coverage_shared_ratio_relative_error: float | None = None
    coverage_shared_anchor_gradient_norm: float | None = None
    coverage_shared_gradient_norm: float | None = None
    coverage_shared_weighted_gradient_norm: float | None = None
    coverage_shared_gradient_cosine: float | None = None
    coverage_shared_adam_update_norm: float | None = None
    coverage_shared_adam_auxiliary_ratio: float | None = None
    coverage_shared_adam_target_ratio: float | None = None
    coverage_shared_adam_ratio_relative_error: float | None = None
    coverage_shared_adam_uncorrected_ratio: float | None = None
    coverage_shared_adam_uncorrected_relative_error: float | None = None
    coverage_shared_adam_update_multiplier: float | None = None
    coverage_shared_adam_proposal_relative_error: float | None = None
    coverage_shared_adam_correction_iterations: int | None = None
    coverage_angle_gradient_target_norm: float | None = None
    coverage_angle_gradient_achieved_norm: float | None = None
    coverage_angle_gradient_relative_error: float | None = None
    coverage_angle_update_target_norm: float | None = None
    coverage_angle_update_achieved_norm: float | None = None
    coverage_angle_update_relative_error: float | None = None
    coverage_angle_update_multiplier: float | None = None
    coverage_angle_update_correction_iterations: int | None = None


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
    total = 0.5 * (real_adversarial + fake_adversarial + real_auxiliary + fake_auxiliary)
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
            classical_weight = (regularizer_gradient_ratio * gan_norm / classical_norm).detach()
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
            quantum_weight = (coherence_gradient_ratio * gan_norm / quantum_safe_norm).detach()
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
        combined_ratio = (_tensor_gradient_norm(combined_regularizer_gradients) / gan_norm).item()
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


def relational_coverage_generator_step(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer_model: RelationalCoverageRegularizer,
    optimizer: torch.optim.Optimizer,
    real_images: Tensor,
    noise: Tensor,
    class_labels: Tensor,
    *,
    kde_weight: float,
    coverage_weight: float,
    budget_target: CoverageBudgetTarget | None = None,
    record_budget: bool = False,
    match_angle_head_budget: bool = False,
) -> GeneratorStepMetrics:
    """Apply a relational update with an optional full-trajectory reference budget."""

    if kde_weight <= 0 or coverage_weight <= 0:
        raise ValueError("KDE and coverage weights must be positive")
    if record_budget and budget_target is not None:
        raise ValueError("a step cannot record and replay a coverage budget simultaneously")
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
        images, angle_residuals = generator.forward_with_angles(noise, class_labels)
        kde_loss, coverage_loss = regularizer_model.loss_components(
            images,
            angle_residuals,
            real_images,
            class_labels,
        )
        for name, loss in (("KDE", kde_loss), ("coverage", coverage_loss)):
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(f"{name} regularizer must return one finite scalar")

        adversarial_logits, class_logits = discriminator(images)
        adversarial = functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        )
        auxiliary = functional.cross_entropy(class_logits, class_labels)
        gan_objective = adversarial + auxiliary
        anchor_objective = gan_objective + kde_weight * kde_loss
        if not record_budget and budget_target is None and not match_angle_head_budget:
            total = anchor_objective + coverage_weight * coverage_loss
            total.backward()
            optimizer.step()
            return GeneratorStepMetrics(
                total=total.detach().item(),
                adversarial=adversarial.detach().item(),
                auxiliary=auxiliary.detach().item(),
                regularizer=(kde_loss + coverage_loss).detach().item(),
                effective_regularizer_weight=kde_weight,
                regularizer_to_gan_gradient_ratio=None,
                mean_target_energy=0.0,
                classical_regularizer=kde_loss.detach().item(),
                coverage_regularizer=coverage_loss.detach().item(),
                effective_coverage_weight=coverage_weight,
            )
        shared_parameters = (
            *generator.label_embedding.parameters(),
            *generator.input_projection.parameters(),
            *generator.image_decoder.parameters(),
        )
        angle_parameters = tuple(
            parameter for parameter in generator.angle_head.parameters() if parameter.requires_grad
        )
        if match_angle_head_budget and not angle_parameters:
            raise ValueError("angle-head budget matching requires trainable angle parameters")

        anchor_gradients = torch.autograd.grad(
            anchor_objective,
            shared_parameters,
            retain_graph=True,
        )
        coverage_shared_gradients = torch.autograd.grad(
            coverage_loss,
            shared_parameters,
            retain_graph=bool(angle_parameters) or budget_target is None,
        )
        coverage_angle_gradients = (
            torch.autograd.grad(
                coverage_loss,
                angle_parameters,
                retain_graph=budget_target is None,
            )
            if angle_parameters
            else ()
        )
        anchor_norm = _tensor_gradient_norm(anchor_gradients)
        coverage_shared_norm = _tensor_gradient_norm(coverage_shared_gradients)
        epsilon = torch.finfo(anchor_norm.dtype).eps
        if not torch.isfinite(anchor_norm) or anchor_norm.item() <= epsilon:
            raise RuntimeError("anchor gradient is non-finite or too small for budget matching")
        if not torch.isfinite(coverage_shared_norm) or coverage_shared_norm.item() <= epsilon:
            raise RuntimeError("coverage gradient is non-finite or too small for budget matching")

        if budget_target is None:
            target_shared_ratio = (coverage_weight * coverage_shared_norm / anchor_norm).detach()
            effective_shared_weight = anchor_norm.new_tensor(coverage_weight)
        else:
            target_shared_ratio = anchor_norm.new_tensor(budget_target.shared_ratio)
            effective_shared_weight = (
                target_shared_ratio * anchor_norm / coverage_shared_norm
            ).detach()
        weighted_coverage_shared = _scale_gradients(
            coverage_shared_gradients,
            effective_shared_weight,
        )
        final_shared_gradients = tuple(
            anchor + coverage
            for anchor, coverage in zip(
                anchor_gradients,
                weighted_coverage_shared,
                strict=True,
            )
        )
        achieved_shared_ratio = _tensor_gradient_norm(weighted_coverage_shared) / anchor_norm
        shared_ratio_relative_error = _relative_error(
            achieved_shared_ratio,
            target_shared_ratio,
        )

        angle_gradient_target_norm = None
        angle_gradient_achieved_norm = None
        angle_gradient_relative_error = None
        if angle_parameters:
            coverage_angle_norm = _tensor_gradient_norm(coverage_angle_gradients)
            if not torch.isfinite(coverage_angle_norm) or coverage_angle_norm.item() <= epsilon:
                raise RuntimeError(
                    "angle-head coverage gradient is non-finite or too small for budget matching"
                )
            if match_angle_head_budget and budget_target is not None:
                if budget_target.angle_gradient_norm is None:
                    raise ValueError("Phase B replay requires an angle-gradient target")
                angle_gradient_target = coverage_angle_norm.new_tensor(
                    budget_target.angle_gradient_norm
                )
                effective_angle_weight = (angle_gradient_target / coverage_angle_norm).detach()
            else:
                effective_angle_weight = coverage_angle_norm.new_tensor(coverage_weight)
                angle_gradient_target = (effective_angle_weight * coverage_angle_norm).detach()
            weighted_coverage_angle = _scale_gradients(
                coverage_angle_gradients,
                effective_angle_weight,
            )
            if match_angle_head_budget:
                achieved_angle_gradient = _tensor_gradient_norm(weighted_coverage_angle)
                angle_gradient_target_norm = angle_gradient_target.item()
                angle_gradient_achieved_norm = achieved_angle_gradient.item()
                angle_gradient_relative_error = _relative_error(
                    achieved_angle_gradient,
                    angle_gradient_target,
                )
        else:
            weighted_coverage_angle = ()

        if budget_target is None:
            backward_total = anchor_objective + coverage_weight * coverage_loss
            backward_total.backward()
            optimizer_shared_gradients = tuple(
                parameter.grad.detach() for parameter in shared_parameters
            )
        else:
            optimizer_shared_gradients = final_shared_gradients
            for parameter, gradient in zip(
                shared_parameters,
                optimizer_shared_gradients,
                strict=True,
            ):
                parameter.grad = gradient.detach()
            for parameter, gradient in zip(
                angle_parameters,
                weighted_coverage_angle,
                strict=True,
            ):
                parameter.grad = gradient.detach()

        adam_anchor_updates = _adam_proposed_updates(
            optimizer,
            shared_parameters,
            anchor_gradients,
        )
        adam_total_updates = _adam_proposed_updates(
            optimizer,
            shared_parameters,
            optimizer_shared_gradients,
        )
        adam_anchor_update_norm = _tensor_gradient_norm(adam_anchor_updates)
        if adam_anchor_update_norm.item() <= epsilon:
            raise RuntimeError("counterfactual anchor Adam update is too small")

        shared_before = tuple(parameter.detach().clone() for parameter in shared_parameters)
        angle_before = (
            tuple(parameter.detach().clone() for parameter in angle_parameters)
            if match_angle_head_budget
            else ()
        )
        optimizer.step()

        ordinary_shared_updates = tuple(
            parameter.detach() - before
            for parameter, before in zip(shared_parameters, shared_before, strict=True)
        )
        ordinary_shared_update_norm = _tensor_gradient_norm(ordinary_shared_updates)
        if ordinary_shared_update_norm.item() <= epsilon:
            raise RuntimeError("ordinary shared Adam update is too small")
        adam_proposal_relative_error = (
            _tensor_gradient_norm(
                tuple(
                    actual - proposed
                    for actual, proposed in zip(
                        ordinary_shared_updates,
                        adam_total_updates,
                        strict=True,
                    )
                )
            )
            / ordinary_shared_update_norm
        )
        uncorrected_adam_auxiliary_updates = tuple(
            total_update - anchor_update
            for total_update, anchor_update in zip(
                ordinary_shared_updates,
                adam_anchor_updates,
                strict=True,
            )
        )
        uncorrected_adam_auxiliary_norm = _tensor_gradient_norm(uncorrected_adam_auxiliary_updates)
        if (
            not torch.isfinite(uncorrected_adam_auxiliary_norm)
            or uncorrected_adam_auxiliary_norm.item() <= epsilon
        ):
            raise RuntimeError("shared auxiliary Adam update is non-finite or too small")
        uncorrected_adam_auxiliary_ratio = uncorrected_adam_auxiliary_norm / adam_anchor_update_norm
        adam_target_ratio = (
            uncorrected_adam_auxiliary_ratio.detach()
            if budget_target is None
            else uncorrected_adam_auxiliary_ratio.new_tensor(
                budget_target.shared_adam_auxiliary_ratio
            )
        )
        adam_uncorrected_relative_error = _relative_error(
            uncorrected_adam_auxiliary_ratio,
            adam_target_ratio,
        )
        adam_update_multiplier = (
            (adam_target_ratio * adam_anchor_update_norm / uncorrected_adam_auxiliary_norm)
            .detach()
            .to(torch.float64)
        )
        adam_correction_iterations = 0
        if budget_target is not None:
            for iteration in range(1, 9):
                _write_scaled_parameter_displacement(
                    shared_parameters,
                    shared_before,
                    adam_anchor_updates,
                    uncorrected_adam_auxiliary_updates,
                    adam_update_multiplier,
                )
                achieved_shared_updates = _parameter_displacements(
                    shared_parameters,
                    shared_before,
                )
                achieved_adam_auxiliary_updates = tuple(
                    total_update - anchor_update
                    for total_update, anchor_update in zip(
                        achieved_shared_updates,
                        adam_anchor_updates,
                        strict=True,
                    )
                )
                adam_auxiliary_ratio = (
                    _tensor_gradient_norm(achieved_adam_auxiliary_updates) / adam_anchor_update_norm
                )
                adam_ratio_relative_error = _relative_error(
                    adam_auxiliary_ratio,
                    adam_target_ratio,
                )
                adam_correction_iterations = iteration
                if adam_ratio_relative_error <= 1e-5:
                    break
                if adam_auxiliary_ratio.item() <= epsilon:
                    raise RuntimeError("realized shared Adam correction is too small")
                adam_update_multiplier = (
                    adam_update_multiplier
                    * adam_target_ratio.to(torch.float64)
                    / adam_auxiliary_ratio.to(torch.float64)
                )
        else:
            achieved_shared_updates = ordinary_shared_updates
            achieved_adam_auxiliary_updates = uncorrected_adam_auxiliary_updates
            adam_auxiliary_ratio = uncorrected_adam_auxiliary_ratio
            adam_ratio_relative_error = _relative_error(
                adam_auxiliary_ratio,
                adam_target_ratio,
            )
        shared_update_norm = _parameter_displacement_norm(shared_parameters, shared_before)
        angle_update_target_norm = None
        angle_update_achieved_norm = None
        angle_update_relative_error = None
        angle_update_multiplier = None
        angle_update_correction_iterations = None
        if match_angle_head_budget:
            ordinary_angle_updates = _parameter_displacements(
                angle_parameters,
                angle_before,
            )
            proposed_angle_update_norm = _tensor_gradient_norm(ordinary_angle_updates)
            if (
                not torch.isfinite(proposed_angle_update_norm)
                or proposed_angle_update_norm.item() <= epsilon
            ):
                raise RuntimeError("angle-head Adam update is non-finite or too small")
            if budget_target is None:
                target_angle_update = proposed_angle_update_norm.detach()
            else:
                if budget_target.angle_update_norm is None:
                    raise ValueError("Phase B replay requires an angle-update target")
                target_angle_update = proposed_angle_update_norm.new_tensor(
                    budget_target.angle_update_norm
                )
            multiplier = (
                (target_angle_update / proposed_angle_update_norm).detach().to(torch.float64)
            )
            angle_update_correction_iterations = 0
            if budget_target is not None:
                zero_updates = tuple(torch.zeros_like(update) for update in ordinary_angle_updates)
                for iteration in range(1, 9):
                    _write_scaled_parameter_displacement(
                        angle_parameters,
                        angle_before,
                        zero_updates,
                        ordinary_angle_updates,
                        multiplier,
                    )
                    achieved_angle_update = _parameter_displacement_norm(
                        angle_parameters,
                        angle_before,
                    )
                    angle_update_relative_error = _relative_error(
                        achieved_angle_update,
                        target_angle_update,
                    )
                    angle_update_correction_iterations = iteration
                    if angle_update_relative_error <= 1e-5:
                        break
                    if achieved_angle_update.item() <= epsilon:
                        raise RuntimeError("realized angle-head correction is too small")
                    multiplier = (
                        multiplier
                        * target_angle_update.to(torch.float64)
                        / achieved_angle_update.to(torch.float64)
                    )
            else:
                achieved_angle_update = proposed_angle_update_norm
            angle_update_target_norm = target_angle_update.item()
            angle_update_achieved_norm = achieved_angle_update.item()
            angle_update_relative_error = _relative_error(
                achieved_angle_update,
                target_angle_update,
            )
            angle_update_multiplier = multiplier.item()

        total = anchor_objective.detach() + effective_shared_weight * coverage_loss.detach()
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
        regularizer=(kde_loss + coverage_loss).detach().item(),
        effective_regularizer_weight=kde_weight,
        regularizer_to_gan_gradient_ratio=None,
        mean_target_energy=0.0,
        classical_regularizer=kde_loss.detach().item(),
        coverage_regularizer=coverage_loss.detach().item(),
        effective_coverage_weight=(
            coverage_weight if budget_target is None else effective_shared_weight.item()
        ),
        coverage_shared_target_ratio=target_shared_ratio.item(),
        coverage_shared_achieved_ratio=achieved_shared_ratio.item(),
        coverage_shared_ratio_relative_error=shared_ratio_relative_error,
        coverage_shared_anchor_gradient_norm=anchor_norm.item(),
        coverage_shared_gradient_norm=coverage_shared_norm.item(),
        coverage_shared_weighted_gradient_norm=(
            _tensor_gradient_norm(weighted_coverage_shared).item()
        ),
        coverage_shared_gradient_cosine=_gradient_cosine(
            coverage_shared_gradients,
            anchor_gradients,
        ),
        coverage_shared_adam_update_norm=shared_update_norm.item(),
        coverage_shared_adam_auxiliary_ratio=adam_auxiliary_ratio.item(),
        coverage_shared_adam_target_ratio=adam_target_ratio.item(),
        coverage_shared_adam_ratio_relative_error=adam_ratio_relative_error,
        coverage_shared_adam_uncorrected_ratio=(uncorrected_adam_auxiliary_ratio.item()),
        coverage_shared_adam_uncorrected_relative_error=(adam_uncorrected_relative_error),
        coverage_shared_adam_update_multiplier=adam_update_multiplier.item(),
        coverage_shared_adam_proposal_relative_error=(adam_proposal_relative_error.item()),
        coverage_shared_adam_correction_iterations=adam_correction_iterations,
        coverage_angle_gradient_target_norm=angle_gradient_target_norm,
        coverage_angle_gradient_achieved_norm=angle_gradient_achieved_norm,
        coverage_angle_gradient_relative_error=angle_gradient_relative_error,
        coverage_angle_update_target_norm=angle_update_target_norm,
        coverage_angle_update_achieved_norm=angle_update_achieved_norm,
        coverage_angle_update_relative_error=angle_update_relative_error,
        coverage_angle_update_multiplier=angle_update_multiplier,
        coverage_angle_update_correction_iterations=(angle_update_correction_iterations),
    )


def _tensor_gradient_norm(gradients: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()


def _relative_error(actual: Tensor, target: Tensor) -> float:
    denominator = torch.maximum(target.abs(), target.new_tensor(torch.finfo(target.dtype).eps))
    return ((actual - target).abs() / denominator).item()


def _parameter_displacement_norm(
    parameters: tuple[torch.nn.Parameter, ...],
    before: tuple[Tensor, ...],
) -> Tensor:
    return _tensor_gradient_norm(_parameter_displacements(parameters, before))


def _parameter_displacements(
    parameters: tuple[torch.nn.Parameter, ...],
    before: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    return tuple(
        parameter.detach() - value for parameter, value in zip(parameters, before, strict=True)
    )


def _write_scaled_parameter_displacement(
    parameters: tuple[torch.nn.Parameter, ...],
    before: tuple[Tensor, ...],
    base_updates: tuple[Tensor, ...],
    auxiliary_updates: tuple[Tensor, ...],
    multiplier: Tensor,
) -> None:
    """Apply one scalar displacement in float64 before the final parameter cast."""

    if not torch.isfinite(multiplier):
        raise RuntimeError("parameter displacement multiplier is non-finite")
    with torch.no_grad():
        for parameter, value, base, auxiliary in zip(
            parameters,
            before,
            base_updates,
            auxiliary_updates,
            strict=True,
        ):
            corrected = (
                value.to(torch.float64)
                + base.to(torch.float64)
                + multiplier * auxiliary.to(torch.float64)
            )
            parameter.copy_(corrected.to(parameter.dtype))


def _adam_proposed_updates(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Return the next Adam displacements without mutating optimizer state."""

    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("relational budget diagnostics require torch.optim.Adam")
    groups_by_parameter = {
        id(parameter): group for group in optimizer.param_groups for parameter in group["params"]
    }
    updates = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        group = groups_by_parameter.get(id(parameter))
        if group is None:
            raise ValueError("budgeted parameter is missing from the optimizer")
        if (
            group.get("amsgrad", False)
            or group.get("maximize", False)
            or group.get("weight_decay", 0) != 0
        ):
            raise ValueError("budget diagnostics require plain Adam without optional transforms")
        beta1, beta2 = group["betas"]
        state = optimizer.state.get(parameter, {})
        prior_step = state.get("step", 0)
        if isinstance(prior_step, Tensor):
            prior_step = int(prior_step.item())
        step = int(prior_step) + 1
        prior_mean = state.get("exp_avg")
        prior_squared_mean = state.get("exp_avg_sq")
        if prior_mean is None:
            prior_mean = torch.zeros_like(parameter)
        if prior_squared_mean is None:
            prior_squared_mean = torch.zeros_like(parameter)
        mean = beta1 * prior_mean + (1 - beta1) * gradient
        squared_mean = beta2 * prior_squared_mean + (1 - beta2) * gradient.square()
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denominator = squared_mean.sqrt() / (bias_correction2**0.5)
        denominator = denominator + group["eps"]
        update = -(group["lr"] / bias_correction1) * mean / denominator
        updates.append(update)
    return tuple(updates)


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
        value - coefficient * direction for value, direction in zip(source, reference, strict=True)
    )
