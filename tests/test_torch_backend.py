from __future__ import annotations

import torch

from vqe_gan.quantum.losses import contrastive_energy_loss
from vqe_gan.quantum.spec import (
    HamiltonianFamily,
    IsingHamiltonianSpec,
    QuantumCircuitSpec,
)
from vqe_gan.quantum.torch_backend import DeviceBridgedEnergy, TorchStatevectorEnergy


def test_zero_angles_match_all_zero_state_energy() -> None:
    circuit_spec = QuantumCircuitSpec()
    hamiltonian_spec = IsingHamiltonianSpec()
    backend = TorchStatevectorEnergy(circuit_spec, hamiltonian_spec)
    angles = torch.zeros(2, circuit_spec.num_parameters, dtype=torch.float64)
    labels = torch.tensor([0, 9], dtype=torch.long)

    energies = backend(angles, labels)

    expected = torch.tensor(
        [
            -3 * hamiltonian_spec.coupling - 4 * hamiltonian_spec.field_for_class(0),
            -3 * hamiltonian_spec.coupling - 4 * hamiltonian_spec.field_for_class(9),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(energies, expected)


def test_statevectors_remain_normalized_and_gradients_are_finite() -> None:
    torch.manual_seed(7)
    backend = TorchStatevectorEnergy()
    angles = torch.randn(8, backend.num_parameters, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(8, dtype=torch.long) % 10

    states = backend.statevector(angles)
    norms = states.abs().square().sum(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-12, rtol=1e-12)

    loss = backend(angles, labels).square().mean()
    loss.backward()

    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
    assert torch.count_nonzero(angles.grad) > 0


def test_class_encoded_hamiltonians_have_unique_class_ground_states() -> None:
    backend = TorchStatevectorEnergy(
        hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
    )
    diagonals = backend.energy_diagonals

    ground_states = diagonals.argmin(dim=1)
    torch.testing.assert_close(ground_states, torch.arange(10))
    assert all(torch.count_nonzero(row == row.min()) == 1 for row in diagonals)

    reference_spectrum = diagonals[0].sort().values
    for spectrum in diagonals[1:]:
        torch.testing.assert_close(spectrum.sort().values, reference_spectrum)


def test_all_energies_select_target_and_support_contrastive_gradient() -> None:
    torch.manual_seed(17)
    backend = TorchStatevectorEnergy(
        hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
    )
    angles = torch.randn(4, backend.num_parameters, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 3, 7, 9], dtype=torch.long)

    all_energies = backend.all_energies(angles)
    target_energies = backend(angles, labels)
    torch.testing.assert_close(
        target_energies,
        all_energies.gather(1, labels.unsqueeze(1)).squeeze(1),
    )

    loss = contrastive_energy_loss(all_energies, labels, temperature=0.7)
    loss.backward()
    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
    assert torch.count_nonzero(angles.grad) > 0


def test_rejects_invalid_shapes_and_labels() -> None:
    backend = TorchStatevectorEnergy()
    angles = torch.zeros(2, backend.num_parameters)

    try:
        backend(angles[:, :-1], torch.tensor([0, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid angle shape was accepted")

    try:
        backend(angles, torch.tensor([0, 10]))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range label was accepted")


def test_device_bridge_preserves_energies_and_gradients() -> None:
    torch.manual_seed(29)
    direct = TorchStatevectorEnergy(
        hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
    ).to(dtype=torch.float32)
    bridged = DeviceBridgedEnergy(
        TorchStatevectorEnergy(
            hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
        ),
        torch.device("cpu"),
    )
    base_angles = torch.randn(3, direct.num_parameters)
    direct_angles = base_angles.clone().requires_grad_(True)
    bridged_angles = base_angles.clone().requires_grad_(True)

    direct_energies = direct.all_energies(direct_angles)
    bridged_energies = bridged.all_energies(bridged_angles)
    torch.testing.assert_close(direct_energies, bridged_energies)
    direct_energies.square().mean().backward()
    bridged_energies.square().mean().backward()
    torch.testing.assert_close(direct_angles.grad, bridged_angles.grad)
