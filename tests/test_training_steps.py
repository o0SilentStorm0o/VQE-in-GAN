from __future__ import annotations

import torch
from torch import Tensor, nn

from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.spec import HamiltonianFamily, IsingHamiltonianSpec
from vqe_gan.quantum.torch_backend import TorchStatevectorEnergy
from vqe_gan.training import (
    coherence_guided_generator_step,
    discriminator_step,
    distribution_regularized_generator_step,
    generator_step,
)


class CountingEnergyBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backend = TorchStatevectorEnergy(
            hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
        )
        self.calls = 0

    def all_energies(self, angles: Tensor) -> Tensor:
        self.calls += 1
        return self.backend.all_energies(angles)


def _models_and_batch() -> tuple[
    SharedQuantumGenerator,
    ACGANDiscriminator,
    Tensor,
    Tensor,
    Tensor,
]:
    torch.manual_seed(37)
    generator = SharedQuantumGenerator()
    discriminator = ACGANDiscriminator()
    noise = torch.randn(3, generator.latent_dim)
    labels = torch.tensor([1, 4, 8], dtype=torch.long)
    real_images = torch.randn(3, 1, 28, 28).clamp(-1, 1)
    return generator, discriminator, noise, labels, real_images


def test_quantum_backend_is_called_once_only_in_quantum_generator_step() -> None:
    generator, discriminator, noise, labels, real_images = _models_and_batch()
    backend = CountingEnergyBackend()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

    discriminator_step(
        generator,
        discriminator,
        discriminator_optimizer,
        real_images,
        labels,
        noise,
        labels,
    )
    generator.eval()
    with torch.no_grad():
        generator(noise, labels)
    assert backend.calls == 0

    metrics = generator_step(
        generator,
        discriminator,
        backend,
        generator_optimizer,
        noise,
        labels,
        regularizer_weight=0.1,
        regularizer_gradient_ratio=0.1,
    )
    assert backend.calls == 1
    assert metrics.regularizer_to_gan_gradient_ratio == 0.1
    assert metrics.effective_regularizer_weight > 0

    generator_step(
        generator,
        discriminator,
        None,
        generator_optimizer,
        noise,
        labels,
        regularizer_weight=0.0,
    )
    assert backend.calls == 1


def test_generator_step_does_not_change_or_populate_discriminator_gradients() -> None:
    generator, discriminator, noise, labels, _ = _models_and_batch()
    backend = CountingEnergyBackend()
    optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    state_before = {
        name: value.detach().clone() for name, value in discriminator.state_dict().items()
    }
    discriminator.train()

    generator_step(
        generator,
        discriminator,
        backend,
        optimizer,
        noise,
        labels,
        regularizer_weight=0.1,
    )

    assert all(parameter.grad is None for parameter in discriminator.parameters())
    for name, after in discriminator.state_dict().items():
        torch.testing.assert_close(state_before[name], after)
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert discriminator.training


def test_discriminator_step_does_not_populate_generator_gradients() -> None:
    generator, discriminator, noise, labels, real_images = _models_and_batch()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    state_before = {name: value.detach().clone() for name, value in generator.state_dict().items()}
    generator.train()

    discriminator_step(
        generator,
        discriminator,
        optimizer,
        real_images,
        labels,
        noise,
        labels,
    )

    assert all(parameter.grad is None for parameter in generator.parameters())
    for name, after in generator.state_dict().items():
        torch.testing.assert_close(state_before[name], after)
    assert generator.training


class MeanMatchingRegularizer(nn.Module):
    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        del class_labels
        return (generated.mean(dim=0) - real.mean(dim=0)).square().mean()


class DecomposedMeanMatchingRegularizer(nn.Module):
    def loss_components(
        self,
        generated: Tensor,
        real: Tensor,
        class_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del class_labels
        classical = (generated.mean(dim=0) - real.mean(dim=0)).square().mean()
        quantum = (
            generated[:, :, ::2, ::2].mean(dim=0)
            - real[:, :, ::2, ::2].mean(dim=0)
        ).square().mean()
        return classical, quantum


def test_distribution_step_updates_only_the_generator_with_balanced_gradient() -> None:
    generator, discriminator, noise, labels, real_images = _models_and_batch()
    optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    discriminator_state = {
        name: value.detach().clone() for name, value in discriminator.state_dict().items()
    }
    generator_state = {
        name: value.detach().clone() for name, value in generator.state_dict().items()
    }

    metrics = distribution_regularized_generator_step(
        generator,
        discriminator,
        MeanMatchingRegularizer(),
        optimizer,
        real_images,
        noise,
        labels,
        regularizer_weight=1.0,
        regularizer_gradient_ratio=0.1,
    )

    assert metrics.regularizer_to_gan_gradient_ratio == 0.1
    assert metrics.effective_regularizer_weight > 0
    assert any(
        not torch.equal(generator_state[name], value)
        for name, value in generator.state_dict().items()
    )
    for name, value in discriminator.state_dict().items():
        torch.testing.assert_close(value, discriminator_state[name])
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_coherence_guidance_has_separate_bounded_gradient_budget() -> None:
    generator, discriminator, noise, labels, real_images = _models_and_batch()
    optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    discriminator_state = {
        name: value.detach().clone() for name, value in discriminator.state_dict().items()
    }

    metrics = coherence_guided_generator_step(
        generator,
        discriminator,
        DecomposedMeanMatchingRegularizer(),
        optimizer,
        real_images,
        noise,
        labels,
        regularizer_weight=1.0,
        regularizer_gradient_ratio=0.1,
        coherence_gradient_ratio=0.05,
    )

    assert metrics.classical_regularizer is not None
    assert metrics.quantum_regularizer is not None
    assert metrics.effective_quantum_weight is not None
    assert metrics.quantum_to_gan_gradient_ratio == 0.05
    assert metrics.regularizer_to_gan_gradient_ratio is not None
    assert 0 <= metrics.retained_quantum_gradient_fraction <= 1.00001
    for name, value in discriminator.state_dict().items():
        torch.testing.assert_close(value, discriminator_state[name])
    assert all(parameter.grad is None for parameter in discriminator.parameters())
