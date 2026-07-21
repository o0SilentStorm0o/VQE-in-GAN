from __future__ import annotations

import math

import torch

from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.spec import HamiltonianFamily, IsingHamiltonianSpec
from vqe_gan.quantum.torch_backend import TorchStatevectorEnergy
from vqe_gan.training import measure_gradient_diagnostics


def test_diagnostics_are_finite_read_only_and_restore_modes() -> None:
    torch.manual_seed(61)
    generator = SharedQuantumGenerator()
    discriminator = ACGANDiscriminator()
    energy_model = TorchStatevectorEnergy(
        hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
    ).to(dtype=torch.float32)
    generator.train()
    discriminator.train()
    generator_state = {
        name: value.detach().clone() for name, value in generator.state_dict().items()
    }
    discriminator_state = {
        name: value.detach().clone() for name, value in discriminator.state_dict().items()
    }
    noise = torch.randn(5, generator.latent_dim)
    labels = torch.tensor([0, 2, 4, 7, 9], dtype=torch.long)

    diagnostics = measure_gradient_diagnostics(
        generator,
        discriminator,
        energy_model,
        noise,
        labels,
        regularizer_weight=0.1,
        regularizer_temperature=1.0,
        regularizer_gradient_ratio=0.1,
    )

    for norm in (
        diagnostics.adversarial_norm,
        diagnostics.auxiliary_norm,
        diagnostics.gan_objective_norm,
        diagnostics.regularizer_norm,
        diagnostics.regularizer_label_embedding_norm,
        diagnostics.regularizer_input_projection_norm,
        diagnostics.regularizer_image_decoder_norm,
    ):
        assert math.isfinite(norm)
        assert norm > 0
    for cosine in (
        diagnostics.adversarial_auxiliary_cosine,
        diagnostics.regularizer_gan_objective_cosine,
        diagnostics.regularizer_adversarial_cosine,
        diagnostics.regularizer_auxiliary_cosine,
    ):
        assert cosine is not None
        assert -1 <= cosine <= 1
    assert diagnostics.regularizer_accuracy is not None
    assert diagnostics.zero_noise_regularizer_accuracy is not None
    assert diagnostics.angle_noise_sensitivity is not None
    assert diagnostics.angle_noise_sensitivity > 0
    assert math.isclose(diagnostics.regularizer_to_gan_norm_ratio, 0.1, rel_tol=1e-5)
    assert diagnostics.effective_regularizer_weight > 0

    assert generator.training
    assert discriminator.training
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    for name, value in generator.state_dict().items():
        torch.testing.assert_close(value, generator_state[name])
    for name, value in discriminator.state_dict().items():
        torch.testing.assert_close(value, discriminator_state[name])


def test_no_regularizer_reports_zero_regularizer_gradient() -> None:
    generator = SharedQuantumGenerator()
    discriminator = ACGANDiscriminator()
    diagnostics = measure_gradient_diagnostics(
        generator,
        discriminator,
        None,
        torch.randn(3, generator.latent_dim),
        torch.tensor([1, 3, 5], dtype=torch.long),
        regularizer_weight=0.0,
        regularizer_temperature=1.0,
    )

    assert diagnostics.regularizer_norm == 0
    assert diagnostics.regularizer_adversarial_cosine is None
    assert diagnostics.regularizer_accuracy is None
