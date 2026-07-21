"""Real-data-anchored distribution regularizers for generated MNIST images."""

from __future__ import annotations

import math
from enum import Enum

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from vqe_gan.quantum.spec import QuantumCircuitSpec
from vqe_gan.quantum.torch_backend import TorchStatevectorEnergy


class ModularAblation(str, Enum):
    """Controlled removals from the modular free-energy construction."""

    FULL = "full"
    DEPHASED = "dephased"
    PRODUCT = "product"
    ENERGY_ONLY = "energy_only"


class QuantumModularFreeEnergy(nn.Module):
    r"""Match class-conditional generated and real mixed quantum states.

    Each image is deterministically pooled to the 16 angles of the parity-tested four-qubit
    circuit. For every class in a batch, pure image states are averaged into mixed states
    :math:`\rho_G` and :math:`\rho_R`. The full loss is

    .. math::
       D(\rho_G\Vert\rho_R)
       = \operatorname{Tr}[\rho_G(-\log\rho_R)] - S(\rho_G).

    Thus the real-data density defines a batchwise modular Hamiltonian while the entropy term
    prevents minimizing its energy by collapsing every sample onto one eigenstate. A small
    depolarizing mixture makes the matrix logarithm finite. Real states and Hamiltonians are
    detached: only generated images receive gradients.
    """

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.125,
        depolarization: float = 0.01,
        ablation: ModularAblation | str = ModularAblation.FULL,
    ) -> None:
        super().__init__()
        self.circuit_spec = circuit_spec or QuantumCircuitSpec()
        if self.circuit_spec.num_parameters != 16:
            raise ValueError("modular image encoding requires exactly 16 circuit parameters")
        if angle_scale <= 0:
            raise ValueError("angle_scale must be positive")
        if not 0 < depolarization < 1:
            raise ValueError("depolarization must be in (0, 1)")
        self.execution_device = torch.device(execution_device)
        if self.execution_device.type == "mps":
            raise ValueError("modular matrix operations are not supported on the MPS backend")
        self.angle_scale = angle_scale
        self.depolarization = depolarization
        self.ablation = ModularAblation(ablation)
        self.backend = TorchStatevectorEnergy(self.circuit_spec).to(
            device=self.execution_device,
            dtype=torch.float32,
        )

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        states_generated = self._statevectors(generated)
        with torch.no_grad():
            states_real = self._statevectors(real)
        labels = class_labels.to(self.execution_device)

        if self.ablation is ModularAblation.DEPHASED:
            loss = self._dephased_relative_entropy(
                states_generated.abs().square().real,
                states_real.abs().square().real,
                labels,
            )
        else:
            generated_density = _pure_state_densities(states_generated)
            real_density = _pure_state_densities(states_real)
            loss = self._matrix_free_energy(generated_density, real_density, labels)
        return loss.to(generated.device)

    def _statevectors(self, images: Tensor) -> Tensor:
        angles = _pooled_angles(
            images,
            angle_scale=self.angle_scale,
            execution_device=self.execution_device,
        )
        return self.backend.statevector(
            angles,
            apply_entanglement=self.ablation is not ModularAblation.PRODUCT,
        )

    def _matrix_free_energy(
        self,
        generated_density: Tensor,
        real_density: Tensor,
        labels: Tensor,
    ) -> Tensor:
        losses = []
        dimension = generated_density.shape[-1]
        identity = torch.eye(
            dimension,
            device=generated_density.device,
            dtype=generated_density.dtype,
        )
        for class_index in labels.unique(sorted=True):
            mask = labels == class_index
            generated_class = _depolarize(
                generated_density[mask].mean(dim=0),
                identity,
                self.depolarization,
            )
            real_class = _depolarize(
                real_density[mask].mean(dim=0),
                identity,
                self.depolarization,
            )
            modular_hamiltonian = -_hermitian_log(real_class)
            cross_energy = torch.trace(generated_class @ modular_hamiltonian).real
            if self.ablation is ModularAblation.ENERGY_ONLY:
                losses.append(cross_energy)
            else:
                losses.append(cross_energy - _von_neumann_entropy(generated_class))
        return torch.stack(losses).mean()

    def _dephased_relative_entropy(
        self,
        generated_probabilities: Tensor,
        real_probabilities: Tensor,
        labels: Tensor,
    ) -> Tensor:
        losses = []
        dimension = generated_probabilities.shape[-1]
        for class_index in labels.unique(sorted=True):
            mask = labels == class_index
            generated_class = (
                (1 - self.depolarization) * generated_probabilities[mask].mean(dim=0)
                + self.depolarization / dimension
            )
            real_class = (
                (1 - self.depolarization) * real_probabilities[mask].mean(dim=0)
                + self.depolarization / dimension
            )
            losses.append(
                (generated_class * (generated_class.log() - real_class.log())).sum()
            )
        return torch.stack(losses).mean()


class QuantumDensityMMD(nn.Module):
    """Known quantum-kernel control using class-conditional density-mean distance."""

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.circuit_spec = circuit_spec or QuantumCircuitSpec()
        if self.circuit_spec.num_parameters != 16:
            raise ValueError("density MMD image encoding requires exactly 16 circuit parameters")
        if angle_scale <= 0:
            raise ValueError("angle_scale must be positive")
        self.execution_device = torch.device(execution_device)
        if self.execution_device.type == "mps":
            raise ValueError("complex density gradients are not supported on the MPS backend")
        self.angle_scale = angle_scale
        self.backend = TorchStatevectorEnergy(self.circuit_spec).to(
            device=self.execution_device,
            dtype=torch.float32,
        )

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        generated_states = self.backend.statevector(
            _pooled_angles(
                generated,
                angle_scale=self.angle_scale,
                execution_device=self.execution_device,
            )
        )
        with torch.no_grad():
            real_states = self.backend.statevector(
                _pooled_angles(
                    real,
                    angle_scale=self.angle_scale,
                    execution_device=self.execution_device,
                )
            )
        generated_density = _pure_state_densities(generated_states)
        real_density = _pure_state_densities(real_states)
        labels = class_labels.to(self.execution_device)
        losses = []
        for class_index in labels.unique(sorted=True):
            mask = labels == class_index
            difference = (
                generated_density[mask].mean(dim=0) - real_density[mask].mean(dim=0)
            )
            losses.append(difference.abs().square().real.sum())
        return torch.stack(losses).mean().to(generated.device)


class QuantumCoherenceResidual(QuantumModularFreeEnergy):
    r"""Return only the information discarded by computational-basis dephasing.

    The residual

    .. math::
       R_Q = D(\rho_G\Vert\rho_R)
       - D(\Delta\rho_G\Vert\Delta\rho_R)

    is non-negative by data processing for the dephasing channel :math:`\Delta` (up to
    floating-point error). Unlike the full modular loss, it removes the population-matching
    term that a classical probability model can already express. This is a diagnostic quantum
    contribution, not by itself evidence of a computational advantage.
    """

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.125,
        depolarization: float = 0.01,
    ) -> None:
        super().__init__(
            circuit_spec,
            execution_device=execution_device,
            angle_scale=angle_scale,
            depolarization=depolarization,
            ablation=ModularAblation.FULL,
        )

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        states_generated = self._statevectors(generated)
        with torch.no_grad():
            states_real = self._statevectors(real)
        labels = class_labels.to(self.execution_device)
        generated_density = _pure_state_densities(states_generated)
        real_density = _pure_state_densities(states_real)
        full_relative_entropy = self._matrix_free_energy(
            generated_density,
            real_density,
            labels,
        )
        dephased_relative_entropy = self._dephased_relative_entropy(
            states_generated.abs().square().real,
            states_real.abs().square().real,
            labels,
        )
        return (full_relative_entropy - dephased_relative_entropy).to(generated.device)


class ClassConditionalRBFMMD(nn.Module):
    """Strong classical control on exactly the same pooled image input."""

    def __init__(
        self,
        *,
        execution_device: torch.device | str = "cpu",
        sigma_squared: float = 1.213,
    ) -> None:
        super().__init__()
        if sigma_squared <= 0:
            raise ValueError("sigma_squared must be positive")
        self.execution_device = torch.device(execution_device)
        self.sigma_squared = sigma_squared

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        generated_features = functional.adaptive_avg_pool2d(generated, (4, 4)).flatten(1)
        generated_features = generated_features.to(self.execution_device)
        real_features = functional.adaptive_avg_pool2d(real, (4, 4)).flatten(1)
        real_features = real_features.to(self.execution_device).detach()
        labels = class_labels.to(self.execution_device)
        losses = []
        for class_index in labels.unique(sorted=True):
            mask = labels == class_index
            generated_class = generated_features[mask]
            real_class = real_features[mask]
            losses.append(
                _rbf_kernel(generated_class, generated_class, self.sigma_squared).mean()
                + _rbf_kernel(real_class, real_class, self.sigma_squared).mean()
                - 2 * _rbf_kernel(generated_class, real_class, self.sigma_squared).mean()
            )
        return torch.stack(losses).mean().to(generated.device)


class RBFQuantumCoherenceGuidance(nn.Module):
    """Expose a strong classical loss and a separate quantum-coherence residual.

    The training step keeps the components separate so that the quantum gradient can be
    stripped of directions already supplied by the classical RBF control before it reaches the
    generator. Returning their sum from ``forward`` keeps ordinary read-only diagnostics useful;
    optimization uses ``loss_components`` instead.
    """

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.125,
        depolarization: float = 0.01,
        sigma_squared: float = 1.213,
    ) -> None:
        super().__init__()
        self.classical = ClassConditionalRBFMMD(
            execution_device=execution_device,
            sigma_squared=sigma_squared,
        )
        self.quantum = QuantumCoherenceResidual(
            circuit_spec,
            execution_device=execution_device,
            angle_scale=angle_scale,
            depolarization=depolarization,
        )

    def loss_components(
        self,
        generated: Tensor,
        real: Tensor,
        class_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(classical_rbf, quantum_dephasing_residual)`` losses."""

        return (
            self.classical(generated, real, class_labels),
            self.quantum(generated, real, class_labels),
        )

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        classical, quantum = self.loss_components(generated, real, class_labels)
        return classical + quantum


class QuantumModularReference(QuantumModularFreeEnergy):
    """Match generated class states to stable real-data modular Hamiltonians.

    A balanced real reference bank is encoded once before GAN training. Its class density
    matrices define fixed modular Hamiltonians, eliminating the high-variance matrix logarithm
    of a handful of real examples in every minibatch. Generated mixed states retain the entropy
    term that discourages collapse.
    """

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        num_classes: int = 10,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.125,
        depolarization: float = 0.01,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        super().__init__(
            circuit_spec,
            execution_device=execution_device,
            angle_scale=angle_scale,
            depolarization=depolarization,
            ablation=ModularAblation.FULL,
        )
        self.num_classes = num_classes
        dimension = 2**self.circuit_spec.num_qubits
        self.register_buffer(
            "reference_hamiltonians",
            torch.empty(
                0,
                dimension,
                dimension,
                device=self.execution_device,
                dtype=torch.complex64,
            ),
        )

    @property
    def reference_is_fitted(self) -> bool:
        return self.reference_hamiltonians.shape[0] == self.num_classes

    def fit_reference(self, images: Tensor, class_labels: Tensor) -> None:
        """Build one fixed modular Hamiltonian per class from balanced real images."""

        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("reference images must have shape (batch, 1, height, width)")
        if class_labels.shape != (images.shape[0],) or class_labels.dtype != torch.long:
            raise ValueError("reference labels must be one-dimensional torch.long values")
        with torch.no_grad():
            states = self._statevectors(images)
            densities = _pure_state_densities(states)
            labels = class_labels.to(self.execution_device)
            dimension = densities.shape[-1]
            identity = torch.eye(
                dimension,
                device=self.execution_device,
                dtype=densities.dtype,
            )
            hamiltonians = []
            for class_index in range(self.num_classes):
                mask = labels == class_index
                if not torch.any(mask):
                    raise ValueError(f"reference bank has no examples for class {class_index}")
                density = _depolarize(
                    densities[mask].mean(dim=0),
                    identity,
                    self.depolarization,
                )
                hamiltonians.append(-_hermitian_log(density))
            self.reference_hamiltonians = torch.stack(hamiltonians).detach()

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        if not self.reference_is_fitted:
            raise RuntimeError("fit_reference must be called before optimization")
        states = self._statevectors(generated)
        densities = _pure_state_densities(states)
        labels = class_labels.to(self.execution_device)
        dimension = densities.shape[-1]
        identity = torch.eye(
            dimension,
            device=self.execution_device,
            dtype=densities.dtype,
        )
        losses = []
        for class_index in labels.unique(sorted=True):
            mask = labels == class_index
            generated_class = _depolarize(
                densities[mask].mean(dim=0),
                identity,
                self.depolarization,
            )
            cross_energy = torch.trace(
                generated_class @ self.reference_hamiltonians[class_index]
            ).real
            losses.append(cross_energy - _von_neumann_entropy(generated_class))
        return torch.stack(losses).mean().to(generated.device)


class ClassConditionalRBFReferenceMMD(nn.Module):
    """Classical reference-bank control matched to ``QuantumModularReference``."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        execution_device: torch.device | str = "cpu",
        sigma_squared: float = 1.213,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if sigma_squared <= 0:
            raise ValueError("sigma_squared must be positive")
        self.num_classes = num_classes
        self.execution_device = torch.device(execution_device)
        self.sigma_squared = sigma_squared
        self.register_buffer(
            "reference_features",
            torch.empty(0, 0, 16, device=self.execution_device),
        )
        self.register_buffer(
            "reference_self_kernel_means",
            torch.empty(0, device=self.execution_device),
        )

    @property
    def reference_is_fitted(self) -> bool:
        return self.reference_features.shape[0] == self.num_classes

    def fit_reference(self, images: Tensor, class_labels: Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("reference images must have shape (batch, 1, height, width)")
        if class_labels.shape != (images.shape[0],) or class_labels.dtype != torch.long:
            raise ValueError("reference labels must be one-dimensional torch.long values")
        features = functional.adaptive_avg_pool2d(images, (4, 4)).flatten(1)
        features = features.to(self.execution_device).detach()
        labels = class_labels.to(self.execution_device)
        counts = [
            int((labels == class_index).sum().item())
            for class_index in range(self.num_classes)
        ]
        if min(counts) <= 0 or len(set(counts)) != 1:
            raise ValueError("reference bank must contain the same positive count for every class")
        self.reference_features = torch.stack(
            [features[labels == class_index] for class_index in range(self.num_classes)]
        )
        self.reference_self_kernel_means = torch.stack(
            [
                _rbf_kernel(reference, reference, self.sigma_squared).mean()
                for reference in self.reference_features
            ]
        )

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        if not self.reference_is_fitted:
            raise RuntimeError("fit_reference must be called before optimization")
        generated_features = functional.adaptive_avg_pool2d(generated, (4, 4)).flatten(1)
        generated_features = generated_features.to(self.execution_device)
        labels = class_labels.to(self.execution_device)
        losses = []
        for class_index in labels.unique(sorted=True):
            generated_class = generated_features[labels == class_index]
            reference_class = self.reference_features[class_index]
            losses.append(
                _rbf_kernel(generated_class, generated_class, self.sigma_squared).mean()
                + self.reference_self_kernel_means[class_index]
                - 2
                * _rbf_kernel(
                    generated_class,
                    reference_class,
                    self.sigma_squared,
                ).mean()
            )
        return torch.stack(losses).mean().to(generated.device)


class QuantumModularContrastiveReference(nn.Module):
    r"""Classify generated images with normalized real-data modular energies.

    The fixed image encoder assigns one spatial quadrant to each qubit in CNOT-ring order.
    Within every quadrant, the calibrated pixel order is bottom-right, top-left, top-right,
    bottom-left for the four rotation layers. Real class densities define modular Hamiltonians
    ``-log(rho_c)``. Their identity components are removed and their Frobenius norms are matched
    before using all ten energies as contrastive logits.

    Unlike :class:`QuantumModularReference`, this objective compares the requested class against
    every competing class for each image. This removes class-dependent energy offsets and the
    target-only failure mode found in the reference audit.
    """

    _QUADRANT_RING_PERMUTATION = (
        5,
        7,
        15,
        13,
        0,
        2,
        10,
        8,
        1,
        3,
        11,
        9,
        4,
        6,
        14,
        12,
    )

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        num_classes: int = 10,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.75,
        depolarization: float = 0.01,
        temperature: float = 0.04,
        ablation: ModularAblation | str = ModularAblation.FULL,
    ) -> None:
        super().__init__()
        self.circuit_spec = circuit_spec or QuantumCircuitSpec()
        if self.circuit_spec.num_parameters != 16:
            raise ValueError("contrastive modular encoding requires exactly 16 parameters")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if angle_scale <= 0 or temperature <= 0:
            raise ValueError("angle_scale and temperature must be positive")
        if not 0 < depolarization < 1:
            raise ValueError("depolarization must be in (0, 1)")
        self.ablation = ModularAblation(ablation)
        if self.ablation is ModularAblation.ENERGY_ONLY:
            raise ValueError("energy_only is not a contrastive circuit ablation")
        self.num_classes = num_classes
        self.execution_device = torch.device(execution_device)
        if self.execution_device.type == "mps":
            raise ValueError("complex modular operations are not supported on MPS")
        self.angle_scale = angle_scale
        self.depolarization = depolarization
        self.temperature = temperature
        self.backend = TorchStatevectorEnergy(self.circuit_spec).to(
            device=self.execution_device,
            dtype=torch.float32,
        )
        dimension = 2**self.circuit_spec.num_qubits
        self.register_buffer(
            "pixel_permutation",
            torch.tensor(
                self._QUADRANT_RING_PERMUTATION,
                device=self.execution_device,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "reference_observables",
            torch.empty(0, dimension, dimension, device=self.execution_device),
        )

    @property
    def reference_is_fitted(self) -> bool:
        return self.reference_observables.shape[0] == self.num_classes

    def fit_reference(self, images: Tensor, class_labels: Tensor) -> None:
        """Fit one centered, unit-norm observable for every real-data class."""

        _validate_reference_inputs(images, class_labels)
        with torch.no_grad():
            states = self._statevectors(images)
            labels = class_labels.to(self.execution_device)
            if self.ablation is ModularAblation.DEPHASED:
                probabilities = states.abs().square().real
                observables = []
                dimension = probabilities.shape[-1]
                for class_index in range(self.num_classes):
                    mask = labels == class_index
                    if not torch.any(mask):
                        raise ValueError(
                            f"reference bank has no examples for class {class_index}"
                        )
                    reference = (
                        (1 - self.depolarization) * probabilities[mask].mean(dim=0)
                        + self.depolarization / dimension
                    )
                    energy = -reference.log()
                    centered = energy - energy.mean()
                    observables.append(centered / centered.square().sum().sqrt())
                diagonal = torch.stack(observables).to(states.dtype)
                self.reference_observables = torch.diag_embed(diagonal).detach()
                return

            densities = _pure_state_densities(states)
            dimension = densities.shape[-1]
            identity = torch.eye(
                dimension,
                device=self.execution_device,
                dtype=densities.dtype,
            )
            observables = []
            for class_index in range(self.num_classes):
                mask = labels == class_index
                if not torch.any(mask):
                    raise ValueError(
                        f"reference bank has no examples for class {class_index}"
                    )
                density = _depolarize(
                    densities[mask].mean(dim=0),
                    identity,
                    self.depolarization,
                )
                hamiltonian = -_hermitian_log(density)
                centered = hamiltonian - torch.trace(hamiltonian) * identity / dimension
                norm = centered.abs().square().sum().sqrt()
                observables.append(centered / norm)
            self.reference_observables = torch.stack(observables).detach()

    def class_logits(self, generated: Tensor) -> Tensor:
        """Return calibrated negative-energy logits for every generated image and class."""

        if not self.reference_is_fitted:
            raise RuntimeError("fit_reference must be called before optimization")
        states = self._statevectors(generated)
        energies = torch.einsum(
            "bi,cij,bj->bc",
            states.conj(),
            self.reference_observables,
            states,
        ).real
        return (-energies / self.temperature).to(generated.device)

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        return functional.cross_entropy(
            self.class_logits(generated),
            class_labels.to(generated.device),
        )

    def _statevectors(self, images: Tensor) -> Tensor:
        pooled = functional.adaptive_avg_pool2d(images, (4, 4)).flatten(start_dim=1)
        angles = (
            pooled.to(self.execution_device)[:, self.pixel_permutation]
            * (torch.pi * self.angle_scale)
        )
        return self.backend.statevector(
            angles,
            apply_entanglement=self.ablation is not ModularAblation.PRODUCT,
        )


class ClassConditionalLogKDEReference(nn.Module):
    """Strong classical control using class-conditional log kernel densities."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        execution_device: torch.device | str = "cpu",
        sigma_squared: float = 0.03125,
        temperature: float = 0.75,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if sigma_squared <= 0 or temperature <= 0:
            raise ValueError("sigma_squared and temperature must be positive")
        self.num_classes = num_classes
        self.execution_device = torch.device(execution_device)
        self.sigma_squared = sigma_squared
        self.temperature = temperature
        self.register_buffer(
            "reference_features",
            torch.empty(0, 0, 16, device=self.execution_device),
        )

    @property
    def reference_is_fitted(self) -> bool:
        return self.reference_features.shape[0] == self.num_classes

    def fit_reference(self, images: Tensor, class_labels: Tensor) -> None:
        _validate_reference_inputs(images, class_labels)
        features = functional.adaptive_avg_pool2d(images, (4, 4)).flatten(1)
        features = features.to(self.execution_device).detach()
        labels = class_labels.to(self.execution_device)
        counts = [
            int((labels == class_index).sum().item())
            for class_index in range(self.num_classes)
        ]
        if min(counts) <= 0 or len(set(counts)) != 1:
            raise ValueError("reference bank must contain equal positive class counts")
        self.reference_features = torch.stack(
            [features[labels == class_index] for class_index in range(self.num_classes)]
        )

    def class_logits(self, generated: Tensor) -> Tensor:
        if not self.reference_is_fitted:
            raise RuntimeError("fit_reference must be called before optimization")
        features = functional.adaptive_avg_pool2d(generated, (4, 4)).flatten(1)
        features = features.to(self.execution_device)
        squared_distances = (
            features[:, None, None, :]
            - self.reference_features[None, :, :, :]
        ).square().sum(dim=-1)
        log_density = torch.logsumexp(
            -squared_distances / (2 * self.sigma_squared),
            dim=-1,
        ) - math.log(self.reference_features.shape[1])
        return (log_density / self.temperature).to(generated.device)

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        return functional.cross_entropy(
            self.class_logits(generated),
            class_labels.to(generated.device),
        )


class HybridQuantumKDEContrastiveReference(nn.Module):
    """Fuse calibrated log-KDE logits with a small modular quantum contribution."""

    def __init__(
        self,
        circuit_spec: QuantumCircuitSpec | None = None,
        *,
        num_classes: int = 10,
        execution_device: torch.device | str = "cpu",
        angle_scale: float = 0.75,
        depolarization: float = 0.01,
        quantum_temperature: float = 0.04,
        kde_sigma_squared: float = 0.03125,
        kde_temperature: float = 0.75,
        quantum_mixture_weight: float = 0.05,
        ablation: ModularAblation | str = ModularAblation.FULL,
    ) -> None:
        super().__init__()
        if not 0 <= quantum_mixture_weight <= 1:
            raise ValueError("quantum_mixture_weight must be in [0, 1]")
        self.quantum_mixture_weight = quantum_mixture_weight
        self.quantum = QuantumModularContrastiveReference(
            circuit_spec,
            num_classes=num_classes,
            execution_device=execution_device,
            angle_scale=angle_scale,
            depolarization=depolarization,
            temperature=quantum_temperature,
            ablation=ablation,
        )
        self.classical = ClassConditionalLogKDEReference(
            num_classes=num_classes,
            execution_device=execution_device,
            sigma_squared=kde_sigma_squared,
            temperature=kde_temperature,
        )

    @property
    def reference_is_fitted(self) -> bool:
        return self.quantum.reference_is_fitted and self.classical.reference_is_fitted

    def fit_reference(self, images: Tensor, class_labels: Tensor) -> None:
        self.quantum.fit_reference(images, class_labels)
        self.classical.fit_reference(images, class_labels)

    def class_logits(self, generated: Tensor) -> Tensor:
        quantum_logits = self.quantum.class_logits(generated)
        classical_logits = self.classical.class_logits(generated)
        return (
            self.quantum_mixture_weight * quantum_logits
            + (1 - self.quantum_mixture_weight) * classical_logits
        )

    def forward(self, generated: Tensor, real: Tensor, class_labels: Tensor) -> Tensor:
        _validate_distribution_inputs(generated, real, class_labels)
        return functional.cross_entropy(
            self.class_logits(generated),
            class_labels.to(generated.device),
        )


def _pooled_angles(
    images: Tensor,
    *,
    angle_scale: float,
    execution_device: torch.device,
) -> Tensor:
    pooled = functional.adaptive_avg_pool2d(images, (4, 4)).flatten(start_dim=1)
    return pooled.to(execution_device) * (torch.pi * angle_scale)


def _pure_state_densities(states: Tensor) -> Tensor:
    return states.unsqueeze(-1) * states.conj().unsqueeze(-2)


def _depolarize(state: Tensor, identity: Tensor, amount: float) -> Tensor:
    return (1 - amount) * state + amount * identity / state.shape[-1]


def _hermitian_log(matrix: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    log_eigenvalues = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps).log()
    return (eigenvectors * log_eigenvalues.unsqueeze(0)) @ eigenvectors.conj().transpose(-2, -1)


def _von_neumann_entropy(matrix: Tensor) -> Tensor:
    eigenvalues = torch.linalg.eigvalsh(matrix)
    eigenvalues = eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).eps)
    return -(eigenvalues * eigenvalues.log()).sum()


def _rbf_kernel(first: Tensor, second: Tensor, sigma_squared: float) -> Tensor:
    squared_distances = (
        first.square().sum(dim=1, keepdim=True)
        + second.square().sum(dim=1).unsqueeze(0)
        - 2 * first @ second.transpose(0, 1)
    ).clamp_min(0)
    return torch.exp(-squared_distances / (2 * sigma_squared))


def _validate_distribution_inputs(
    generated: Tensor,
    real: Tensor,
    class_labels: Tensor,
) -> None:
    if generated.ndim != 4 or generated.shape[1:] != real.shape[1:]:
        raise ValueError("generated and real images must have matching BCHW shapes")
    if generated.shape != real.shape:
        raise ValueError("generated and real batches must have identical shapes")
    if generated.shape[1] != 1:
        raise ValueError("the current deterministic encoder requires one image channel")
    if class_labels.shape != (generated.shape[0],):
        raise ValueError("class_labels must match the image batch dimension")
    if class_labels.dtype != torch.long:
        raise TypeError("class_labels must use dtype torch.long")
    if not generated.is_floating_point() or not real.is_floating_point():
        raise TypeError("images must use a floating-point dtype")


def _validate_reference_inputs(images: Tensor, class_labels: Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("reference images must have shape (batch, 1, height, width)")
    if class_labels.shape != (images.shape[0],) or class_labels.dtype != torch.long:
        raise ValueError("reference labels must be one-dimensional torch.long values")
