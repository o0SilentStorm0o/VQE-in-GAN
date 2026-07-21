from __future__ import annotations

import torch

from vqe_gan.quantum.losses import contrastive_energy_loss
from vqe_gan.quantum.qiskit_backend import QiskitReferenceEnergy
from vqe_gan.quantum.spec import HamiltonianFamily, IsingHamiltonianSpec
from vqe_gan.quantum.torch_backend import TorchStatevectorEnergy


def test_qiskit_and_torch_energies_and_gradients_match() -> None:
    torch.manual_seed(13)
    fast_backend = TorchStatevectorEnergy()
    reference_backend = QiskitReferenceEnergy(reverse_gradient=True)
    labels = torch.tensor([0, 7, 3], dtype=torch.long)
    base_angles = torch.randn(3, fast_backend.num_parameters, dtype=torch.float64) * 0.7
    weights = torch.tensor([0.2, -0.5, 1.1], dtype=torch.float64)

    fast_angles = base_angles.clone().requires_grad_(True)
    reference_angles = base_angles.clone().requires_grad_(True)
    fast_energies = fast_backend(fast_angles, labels)
    reference_energies = reference_backend(reference_angles, labels)

    torch.testing.assert_close(fast_energies, reference_energies, atol=1e-7, rtol=1e-7)

    (fast_energies * weights).sum().backward()
    (reference_energies * weights).sum().backward()
    assert fast_angles.grad is not None
    assert reference_angles.grad is not None
    torch.testing.assert_close(
        fast_angles.grad,
        reference_angles.grad,
        atol=2e-6,
        rtol=2e-6,
    )


def test_class_encoded_all_energies_and_contrastive_gradients_match() -> None:
    torch.manual_seed(23)
    hamiltonian_spec = IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
    fast_backend = TorchStatevectorEnergy(hamiltonian_spec=hamiltonian_spec)
    reference_backend = QiskitReferenceEnergy(
        hamiltonian_spec=hamiltonian_spec,
        reverse_gradient=True,
    )
    labels = torch.tensor([1, 8], dtype=torch.long)
    base_angles = torch.randn(2, fast_backend.num_parameters, dtype=torch.float64) * 0.4

    fast_angles = base_angles.clone().requires_grad_(True)
    reference_angles = base_angles.clone().requires_grad_(True)
    fast_energies = fast_backend.all_energies(fast_angles)
    reference_energies = reference_backend.all_energies(reference_angles)

    torch.testing.assert_close(fast_energies, reference_energies, atol=1e-7, rtol=1e-7)

    fast_loss = contrastive_energy_loss(fast_energies, labels, temperature=0.8)
    reference_loss = contrastive_energy_loss(reference_energies, labels, temperature=0.8)
    fast_loss.backward()
    reference_loss.backward()
    assert fast_angles.grad is not None
    assert reference_angles.grad is not None
    torch.testing.assert_close(
        fast_angles.grad,
        reference_angles.grad,
        atol=2e-6,
        rtol=2e-6,
    )
