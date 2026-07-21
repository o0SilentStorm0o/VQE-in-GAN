"""Shared, backend-independent definitions of the quantum experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HamiltonianFamily(str, Enum):
    """Supported class-conditioning schemes for the Ising Hamiltonian."""

    LEGACY_UNIFORM_FIELD = "legacy_uniform_field"
    CLASS_ENCODED = "class_encoded"


@dataclass(frozen=True)
class QuantumCircuitSpec:
    """Specification matching the original four-qubit EfficientSU2 ansatz."""

    num_qubits: int = 4
    reps: int = 1
    entanglement: str = "circular"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.reps < 0:
            raise ValueError("reps must be non-negative")
        if self.entanglement != "circular":
            raise ValueError("Only circular entanglement is supported in the parity-tested backend")

    @property
    def num_parameters(self) -> int:
        # EfficientSU2 defaults to one RY and one RZ per qubit in each rotation layer.
        return 2 * self.num_qubits * (self.reps + 1)


@dataclass(frozen=True)
class IsingHamiltonianSpec:
    """Specification of either the historical or corrected Hamiltonian family."""

    coupling: float = 1.0
    global_field: float = 0.1
    class_field_step: float = 0.01
    num_classes: int = 10
    family: HamiltonianFamily = HamiltonianFamily.LEGACY_UNIFORM_FIELD

    def __post_init__(self) -> None:
        if isinstance(self.family, str):
            object.__setattr__(self, "family", HamiltonianFamily(self.family))
        if self.coupling <= 0:
            raise ValueError("coupling must be positive")
        if self.global_field <= 0:
            raise ValueError("global_field must be positive")
        if self.class_field_step < 0:
            raise ValueError("class_field_step must be non-negative")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")

    def field_for_class(self, class_index: int) -> float:
        """Return the field magnitude used for a class.

        Only the historical family varies the magnitude by class. The corrected family varies
        the signs of otherwise identical terms so every class has the same spectrum.
        """

        self.validate_class_index(class_index)
        if self.family is HamiltonianFamily.CLASS_ENCODED:
            return self.global_field
        return self.global_field + class_index * self.class_field_step

    def validate_class_index(self, class_index: int) -> None:
        if not 0 <= class_index < self.num_classes:
            raise ValueError(
                f"class_index must be in [0, {self.num_classes}), got {class_index}"
            )

    def target_spins(self, class_index: int, num_qubits: int) -> tuple[int, ...]:
        """Encode a class as target Z eigenvalues in little-endian qubit order."""

        self.validate_class_index(class_index)
        if self.num_classes > 2**num_qubits:
            raise ValueError(
                f"{self.num_classes} classes cannot be uniquely encoded by {num_qubits} qubits"
            )
        return tuple(
            1 if ((class_index >> qubit) & 1) == 0 else -1
            for qubit in range(num_qubits)
        )
