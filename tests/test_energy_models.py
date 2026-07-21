from __future__ import annotations

import torch

from vqe_gan.regularizers import ClassicalPrototypeEnergy, PermutedClassEnergy


class FixedEnergy(torch.nn.Module):
    def all_energies(self, angles: torch.Tensor) -> torch.Tensor:
        return angles[:, :3]


def test_classical_prototypes_are_unique_parameter_free_minima() -> None:
    energy_model = ClassicalPrototypeEnergy(num_classes=10, num_angles=16)

    assert list(energy_model.parameters()) == []
    prototype_energies = energy_model.all_energies(energy_model.prototypes)
    torch.testing.assert_close(prototype_energies.argmin(dim=1), torch.arange(10))
    torch.testing.assert_close(
        prototype_energies.diagonal(),
        torch.zeros(10, dtype=prototype_energies.dtype),
        atol=1e-12,
        rtol=0,
    )


def test_classical_energy_has_finite_angle_gradients() -> None:
    energy_model = ClassicalPrototypeEnergy(num_classes=10, num_angles=16)
    angles = torch.randn(4, 16, requires_grad=True)

    energy_model.all_energies(angles).square().mean().backward()

    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
    assert torch.count_nonzero(angles.grad) > 0


def test_permuted_energy_applies_fixed_cyclic_class_mapping() -> None:
    energy_model = PermutedClassEnergy(FixedEnergy(), num_classes=3)
    angles = torch.tensor([[10.0, 20.0, 30.0]])

    energies = energy_model.all_energies(angles)

    torch.testing.assert_close(energies, torch.tensor([[30.0, 10.0, 20.0]]))
    assert list(energy_model.parameters()) == []
