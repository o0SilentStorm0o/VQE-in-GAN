from __future__ import annotations

import torch

from vqe_gan.models import SharedQuantumGenerator, initialize_weights
from vqe_gan.quantum.losses import contrastive_energy_loss
from vqe_gan.quantum.spec import HamiltonianFamily, IsingHamiltonianSpec
from vqe_gan.quantum.torch_backend import TorchStatevectorEnergy


def test_generator_shapes_and_image_only_forward() -> None:
    generator = SharedQuantumGenerator()
    noise = torch.randn(3, generator.latent_dim)
    labels = torch.tensor([0, 4, 9], dtype=torch.long)

    images = generator(noise, labels)
    images_with_angles, angles = generator.forward_with_angles(noise, labels)

    assert images.shape == (3, 1, 28, 28)
    assert angles.shape == (3, 16)
    torch.testing.assert_close(images, images_with_angles)
    assert torch.all(angles >= -torch.pi)
    assert torch.all(angles <= torch.pi)


def test_quantum_loss_updates_shared_image_path() -> None:
    torch.manual_seed(31)
    generator = SharedQuantumGenerator()
    generator.eval()
    backend = TorchStatevectorEnergy(
        hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
    )
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.02)
    noise = torch.randn(4, generator.latent_dim)
    labels = torch.tensor([0, 2, 5, 9], dtype=torch.long)

    before = generator(noise, labels).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    angles = generator.quantum_angles(noise, labels)
    loss = contrastive_energy_loss(backend.all_energies(angles), labels)
    loss.backward()

    shared_gradient = generator.input_projection.weight.grad
    assert shared_gradient is not None
    assert torch.isfinite(shared_gradient).all()
    assert torch.count_nonzero(shared_gradient) > 0
    decoder_gradient = generator.image_decoder[2].weight.grad
    assert decoder_gradient is not None
    assert torch.count_nonzero(decoder_gradient) > 0

    optimizer.step()
    after = generator(noise, labels).detach()
    assert not torch.equal(before, after)


def test_experiment_initialization_avoids_zero_angle_stationary_point() -> None:
    generator = SharedQuantumGenerator()
    generator.apply(initialize_weights)
    output_layer = generator.angle_head[-2]

    assert torch.count_nonzero(output_layer.bias) == generator.num_angles
    assert output_layer.bias.min() < 0
    assert output_layer.bias.max() > 0
