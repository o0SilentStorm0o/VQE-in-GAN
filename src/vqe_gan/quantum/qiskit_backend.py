"""Qiskit reference backend with class-grouped batched evaluation."""

from __future__ import annotations

import torch
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, ReverseEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import Tensor, nn

from .circuit import build_class_hamiltonian, build_efficient_su2
from .spec import IsingHamiltonianSpec, QuantumCircuitSpec


class QiskitReferenceEnergy(nn.Module):
    """Evaluate energies with Qiskit, batching samples that share a Hamiltonian."""

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        hamiltonian_spec: IsingHamiltonianSpec | None = None,
        *,
        reverse_gradient: bool = True,
    ) -> None:
        super().__init__()
        self.circuit_spec = circuit_spec or QuantumCircuitSpec()
        self.hamiltonian_spec = hamiltonian_spec or IsingHamiltonianSpec()
        circuit = build_efficient_su2(self.circuit_spec)
        input_parameters = list(circuit.parameters)

        modules = []
        for class_index in range(self.hamiltonian_spec.num_classes):
            estimator = StatevectorEstimator()
            gradient = (
                ReverseEstimatorGradient()
                if reverse_gradient
                else ParamShiftEstimatorGradient(estimator)
            )
            qnn = EstimatorQNN(
                circuit=circuit,
                observables=build_class_hamiltonian(
                    self.circuit_spec,
                    self.hamiltonian_spec,
                    class_index,
                ),
                input_params=input_parameters,
                weight_params=[],
                input_gradients=True,
                estimator=estimator,
                gradient=gradient,
                default_precision=0.0,
            )
            modules.append(TorchConnector(qnn))
        self.class_modules = nn.ModuleList(modules)

    @property
    def num_parameters(self) -> int:
        return self.circuit_spec.num_parameters

    def forward(self, angles: Tensor, class_labels: Tensor) -> Tensor:
        self._validate_inputs(angles, class_labels)
        energies = torch.zeros(angles.shape[0], dtype=angles.dtype, device=angles.device)

        for class_index, module in enumerate(self.class_modules):
            indices = torch.nonzero(class_labels == class_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            class_angles = angles.index_select(0, indices)
            class_energies = module(class_angles).reshape(-1).to(dtype=angles.dtype)
            energies = energies.index_copy(0, indices, class_energies)

        return energies

    def all_energies(self, angles: Tensor) -> Tensor:
        """Return the energy of every class Hamiltonian for every input sample."""

        self._validate_angles(angles)
        class_energies = [
            module(angles).reshape(-1).to(dtype=angles.dtype)
            for module in self.class_modules
        ]
        return torch.stack(class_energies, dim=1)

    def _validate_inputs(self, angles: Tensor, class_labels: Tensor) -> None:
        self._validate_angles(angles)
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

    def _validate_angles(self, angles: Tensor) -> None:
        if angles.ndim != 2 or angles.shape[1] != self.num_parameters:
            raise ValueError(
                f"angles must have shape (batch, {self.num_parameters}), got {tuple(angles.shape)}"
            )
        if not angles.is_floating_point():
            raise TypeError("angles must use a floating-point dtype")
