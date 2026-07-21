"""Batched differentiable statevector backend implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .spec import HamiltonianFamily, IsingHamiltonianSpec, QuantumCircuitSpec


class TorchStatevectorEnergy(nn.Module):
    """Evaluate the original EfficientSU2/Ising energy for a full batch.

    This module simulates the same four-qubit circuit as the Qiskit reference backend. It keeps
    every operation in PyTorch so gradients propagate through the circuit in one autograd pass.
    """

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        hamiltonian_spec: IsingHamiltonianSpec | None = None,
    ) -> None:
        super().__init__()
        self.circuit_spec = circuit_spec or QuantumCircuitSpec()
        self.hamiltonian_spec = hamiltonian_spec or IsingHamiltonianSpec()

        diagonals = self._build_energy_diagonals()
        self.register_buffer("energy_diagonals", diagonals, persistent=True)

        permutations = [
            self._cnot_permutation(control, target)
            for control, target in self._entanglement_pairs()
        ]
        self.register_buffer(
            "cnot_permutations",
            torch.stack(permutations),
            persistent=False,
        )

    @property
    def num_parameters(self) -> int:
        return self.circuit_spec.num_parameters

    def forward(self, angles: Tensor, class_labels: Tensor) -> Tensor:
        """Return one class-conditioned energy per input sample."""

        self._validate_inputs(angles, class_labels)
        energies = self.all_energies(angles)
        return energies.gather(1, class_labels.unsqueeze(1)).squeeze(1)

    def all_energies(self, angles: Tensor) -> Tensor:
        """Return the energy of every class Hamiltonian for every input sample."""

        state = self.statevector(angles)
        probabilities = state.abs().square().real
        diagonals = self.energy_diagonals.to(device=angles.device, dtype=angles.dtype)
        return probabilities @ diagonals.transpose(0, 1)

    def statevector(self, angles: Tensor, *, apply_entanglement: bool = True) -> Tensor:
        """Return batched final statevectors in Qiskit's little-endian basis order.

        ``apply_entanglement=False`` retains every rotation and parameter while removing only the
        CNOT ring. It is used as a controlled product-state ablation of distributional losses; the
        default remains parity-tested against Qiskit.
        """

        if angles.ndim != 2 or angles.shape[1] != self.num_parameters:
            raise ValueError(
                f"angles must have shape (batch, {self.num_parameters}), got {tuple(angles.shape)}"
            )
        if not angles.is_floating_point():
            raise TypeError("angles must use a floating-point dtype")

        complex_dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
        batch_size = angles.shape[0]
        state = torch.zeros(
            batch_size,
            2**self.circuit_spec.num_qubits,
            dtype=complex_dtype,
            device=angles.device,
        )
        state[:, 0] = 1

        offset = 0
        for layer in range(self.circuit_spec.reps + 1):
            for qubit in range(self.circuit_spec.num_qubits):
                state = self._apply_ry(state, angles[:, offset + qubit], qubit)
            offset += self.circuit_spec.num_qubits

            for qubit in range(self.circuit_spec.num_qubits):
                state = self._apply_rz(state, angles[:, offset + qubit], qubit)
            offset += self.circuit_spec.num_qubits

            if apply_entanglement and layer < self.circuit_spec.reps:
                for permutation in self.cnot_permutations:
                    state = state.index_select(1, permutation)

        return state

    def _validate_inputs(self, angles: Tensor, class_labels: Tensor) -> None:
        if angles.ndim != 2 or angles.shape[1] != self.num_parameters:
            raise ValueError(
                f"angles must have shape (batch, {self.num_parameters}), got {tuple(angles.shape)}"
            )
        if class_labels.ndim != 1 or class_labels.shape[0] != angles.shape[0]:
            raise ValueError(
                "class_labels must have shape (batch,) and match the angles batch dimension"
            )
        if class_labels.dtype != torch.long:
            raise TypeError("class_labels must have dtype torch.long")
        if torch.any(class_labels < 0) or torch.any(
            class_labels >= self.hamiltonian_spec.num_classes
        ):
            raise ValueError("class_labels contains an out-of-range class index")

    def _apply_ry(self, state: Tensor, angles: Tensor, qubit: int) -> Tensor:
        cosine = torch.cos(angles / 2)
        sine = torch.sin(angles / 2)
        gate = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        ).to(dtype=state.dtype)
        return self._apply_single_qubit_gate(state, gate, qubit)

    def _apply_rz(self, state: Tensor, angles: Tensor, qubit: int) -> Tensor:
        half_angles = angles.to(dtype=state.dtype) / 2
        phase_zero = torch.exp(-1j * half_angles)
        phase_one = torch.exp(1j * half_angles)
        zeros = torch.zeros_like(phase_zero)
        gate = torch.stack(
            (
                torch.stack((phase_zero, zeros), dim=-1),
                torch.stack((zeros, phase_one), dim=-1),
            ),
            dim=-2,
        )
        return self._apply_single_qubit_gate(state, gate, qubit)

    def _apply_single_qubit_gate(self, state: Tensor, gate: Tensor, qubit: int) -> Tensor:
        num_qubits = self.circuit_spec.num_qubits
        qubit_axis = 1 + (num_qubits - 1 - qubit)
        permutation = [0, *[axis for axis in range(1, num_qubits + 1) if axis != qubit_axis]]
        permutation.append(qubit_axis)
        inverse_permutation = [permutation.index(axis) for axis in range(num_qubits + 1)]

        tensor_state = state.reshape(state.shape[0], *([2] * num_qubits))
        transposed = tensor_state.permute(permutation)
        paired = transposed.reshape(state.shape[0], -1, 2)
        updated = torch.einsum("bij,bpj->bpi", gate, paired)
        restored = updated.reshape(transposed.shape).permute(inverse_permutation)
        return restored.reshape_as(state)

    def _entanglement_pairs(self) -> list[tuple[int, int]]:
        # Qiskit's circular EfficientSU2 ordering is last->first, then the linear chain.
        last = self.circuit_spec.num_qubits - 1
        return [(last, 0), *[(qubit, qubit + 1) for qubit in range(last)]]

    def _cnot_permutation(self, control: int, target: int) -> Tensor:
        dimension = 2**self.circuit_spec.num_qubits
        basis = torch.arange(dimension, dtype=torch.long)
        control_is_one = ((basis >> control) & 1).bool()
        return torch.where(control_is_one, basis ^ (1 << target), basis)

    def _build_energy_diagonals(self) -> Tensor:
        num_qubits = self.circuit_spec.num_qubits
        dimension = 2**num_qubits
        basis = torch.arange(dimension, dtype=torch.long)
        z_values = torch.stack(
            [1.0 - 2.0 * ((basis >> qubit) & 1).to(torch.float64) for qubit in range(num_qubits)],
            dim=-1,
        )
        diagonals = []
        for class_index in range(self.hamiltonian_spec.num_classes):
            field = self.hamiltonian_spec.field_for_class(class_index)
            target_spins = torch.tensor(
                self.hamiltonian_spec.target_spins(class_index, num_qubits)
                if self.hamiltonian_spec.family is HamiltonianFamily.CLASS_ENCODED
                else (1,) * num_qubits,
                dtype=torch.float64,
            )
            coupling_signs = target_spins[:-1] * target_spins[1:]
            coupling_energy = -self.hamiltonian_spec.coupling * (
                z_values[:, :-1] * z_values[:, 1:] * coupling_signs
            ).sum(dim=-1)
            field_energy = -field * (z_values * target_spins).sum(dim=-1)
            diagonals.append(coupling_energy + field_energy)
        return torch.stack(diagonals)


class DeviceBridgedEnergy(nn.Module):
    """Evaluate a backend on another device while preserving autograd connectivity."""

    def __init__(self, backend: TorchStatevectorEnergy, execution_device: torch.device) -> None:
        super().__init__()
        self.backend = backend.to(device=execution_device, dtype=torch.float32)
        self.execution_device = execution_device

    def all_energies(self, angles: Tensor) -> Tensor:
        execution_angles = angles.to(self.execution_device)
        energies = self.backend.all_energies(execution_angles)
        return energies.to(angles.device)
