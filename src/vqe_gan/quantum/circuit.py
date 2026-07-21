"""Qiskit construction helpers for the reference implementation."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

from .spec import HamiltonianFamily, IsingHamiltonianSpec, QuantumCircuitSpec


def build_efficient_su2(spec: QuantumCircuitSpec) -> QuantumCircuit:
    """Build the exact ansatz used by the original experiment, without barriers."""

    return efficient_su2(
        num_qubits=spec.num_qubits,
        su2_gates=["ry", "rz"],
        entanglement=spec.entanglement,
        reps=spec.reps,
        insert_barriers=False,
    )


def build_class_hamiltonian(
    circuit_spec: QuantumCircuitSpec,
    hamiltonian_spec: IsingHamiltonianSpec,
    class_index: int,
) -> SparsePauliOp:
    """Build a class-conditioned linear-chain Ising Hamiltonian."""

    num_qubits = circuit_spec.num_qubits
    field = hamiltonian_spec.field_for_class(class_index)
    terms: list[tuple[str, float]] = []
    target_spins = (
        hamiltonian_spec.target_spins(class_index, num_qubits)
        if hamiltonian_spec.family is HamiltonianFamily.CLASS_ENCODED
        else (1,) * num_qubits
    )

    for qubit in range(num_qubits - 1):
        pauli = ["I"] * num_qubits
        # Qiskit Pauli labels are big-endian strings, while logical qubits are little-endian.
        pauli[num_qubits - 1 - qubit] = "Z"
        pauli[num_qubits - 2 - qubit] = "Z"
        coefficient = (
            -hamiltonian_spec.coupling
            * target_spins[qubit]
            * target_spins[qubit + 1]
        )
        terms.append(("".join(pauli), coefficient))

    for qubit in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[num_qubits - 1 - qubit] = "Z"
        terms.append(("".join(pauli), -field * target_spins[qubit]))

    return SparsePauliOp.from_list(terms)
