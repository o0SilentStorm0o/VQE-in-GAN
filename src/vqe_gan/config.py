"""Validated experiment configuration with stable JSON serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentVariant(str, Enum):
    NO_REGULARIZER = "no_regularizer"
    QUANTUM_CONTRASTIVE = "quantum_contrastive"
    CLASSICAL_PROTOTYPE = "classical_prototype"
    QUANTUM_PERMUTED = "quantum_permuted"
    QUANTUM_MODULAR_FREE_ENERGY = "quantum_modular_free_energy"
    QUANTUM_MODULAR_DEPHASED = "quantum_modular_dephased"
    QUANTUM_MODULAR_PRODUCT = "quantum_modular_product"
    QUANTUM_MODULAR_ENERGY_ONLY = "quantum_modular_energy_only"
    QUANTUM_DENSITY_MMD = "quantum_density_mmd"
    CLASSICAL_RBF_MMD = "classical_rbf_mmd"
    QUANTUM_COHERENCE_GUIDANCE = "quantum_coherence_guidance"
    QUANTUM_MODULAR_REFERENCE = "quantum_modular_reference"
    CLASSICAL_RBF_REFERENCE = "classical_rbf_reference"
    CLASSICAL_LOG_KDE_CONTRASTIVE = "classical_log_kde_contrastive"
    CLASSICAL_LOG_KDE_SCALE_CONTROL = "classical_log_kde_scale_control"
    CLASSICAL_KDE_RELATIONAL_COVERAGE = "classical_kde_relational_coverage"
    QUANTUM_KDE_RELATIONAL_COVERAGE = "quantum_kde_relational_coverage"
    QUANTUM_KDE_RELATIONAL_PRODUCT = "quantum_kde_relational_product"
    QUANTUM_KDE_RELATIONAL_DEPHASED = "quantum_kde_relational_dephased"
    HYBRID_MODULAR_KDE_CONTRASTIVE = "hybrid_modular_kde_contrastive"
    HYBRID_MODULAR_KDE_PRODUCT = "hybrid_modular_kde_product"
    HYBRID_MODULAR_KDE_DEPHASED = "hybrid_modular_kde_dephased"

    @property
    def uses_distribution_regularizer(self) -> bool:
        return self in {
            ExperimentVariant.QUANTUM_MODULAR_FREE_ENERGY,
            ExperimentVariant.QUANTUM_MODULAR_DEPHASED,
            ExperimentVariant.QUANTUM_MODULAR_PRODUCT,
            ExperimentVariant.QUANTUM_MODULAR_ENERGY_ONLY,
            ExperimentVariant.QUANTUM_DENSITY_MMD,
            ExperimentVariant.CLASSICAL_RBF_MMD,
            ExperimentVariant.QUANTUM_COHERENCE_GUIDANCE,
            ExperimentVariant.QUANTUM_MODULAR_REFERENCE,
            ExperimentVariant.CLASSICAL_RBF_REFERENCE,
            ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE,
            ExperimentVariant.CLASSICAL_LOG_KDE_SCALE_CONTROL,
            ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
            ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE,
            ExperimentVariant.HYBRID_MODULAR_KDE_PRODUCT,
            ExperimentVariant.HYBRID_MODULAR_KDE_DEPHASED,
        }

    @property
    def uses_quantum_backend(self) -> bool:
        return self in {
            ExperimentVariant.QUANTUM_CONTRASTIVE,
            ExperimentVariant.QUANTUM_PERMUTED,
            ExperimentVariant.QUANTUM_MODULAR_FREE_ENERGY,
            ExperimentVariant.QUANTUM_MODULAR_DEPHASED,
            ExperimentVariant.QUANTUM_MODULAR_PRODUCT,
            ExperimentVariant.QUANTUM_MODULAR_ENERGY_ONLY,
            ExperimentVariant.QUANTUM_DENSITY_MMD,
            ExperimentVariant.QUANTUM_COHERENCE_GUIDANCE,
            ExperimentVariant.QUANTUM_MODULAR_REFERENCE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
            ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE,
            ExperimentVariant.HYBRID_MODULAR_KDE_PRODUCT,
            ExperimentVariant.HYBRID_MODULAR_KDE_DEPHASED,
        }

    @property
    def uses_coherence_guidance(self) -> bool:
        return self is ExperimentVariant.QUANTUM_COHERENCE_GUIDANCE

    @property
    def uses_relational_coverage(self) -> bool:
        return self in {
            ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
        }

    @property
    def uses_contrastive_reference(self) -> bool:
        return self in {
            ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE,
            ExperimentVariant.CLASSICAL_LOG_KDE_SCALE_CONTROL,
            ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
            ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE,
            ExperimentVariant.HYBRID_MODULAR_KDE_PRODUCT,
            ExperimentVariant.HYBRID_MODULAR_KDE_DEPHASED,
        }

    @property
    def default_regularizer_weight(self) -> float:
        if self is ExperimentVariant.NO_REGULARIZER:
            return 0.0
        if self is ExperimentVariant.CLASSICAL_LOG_KDE_SCALE_CONTROL:
            return 1.9e-5
        if self.uses_contrastive_reference:
            return 2e-5
        return 1.0

    @property
    def default_coverage_weight(self) -> float:
        """Return the fixed median-gradient calibration for relational variants."""

        if self is ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE:
            return 0.0033809364895852783
        if self in {
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
        }:
            return 0.0005662128371831583
        return 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration required to reproduce one training run."""

    run_name: str
    output_root: str = "runs"
    dataset_root: str = "data"
    variant: ExperimentVariant = ExperimentVariant.QUANTUM_CONTRASTIVE
    seed: int = 42
    device: str = "auto"
    quantum_device: str = "auto"
    epochs: int = 1
    max_steps: int | None = None
    dataset_limit: int | None = None
    batch_size: int = 64
    num_workers: int = 0
    latent_dim: int = 100
    num_classes: int = 10
    image_channels: int = 1
    learning_rate_generator: float = 2e-4
    learning_rate_discriminator: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    real_label_smoothing: float = 0.9
    num_qubits: int = 4
    ansatz_reps: int = 1
    ising_coupling: float = 1.0
    ising_field: float = 0.1
    regularizer_weight: float = 1.0
    regularizer_temperature: float = 1.0
    regularizer_gradient_ratio: float | None = None
    modular_angle_scale: float = 0.125
    modular_depolarization: float = 0.01
    density_mmd_angle_scale: float = 0.25
    rbf_sigma_squared: float = 1.213
    coherence_gradient_ratio: float = 0.05
    reference_samples_per_class: int = 256
    contrastive_reference_samples_per_class: int = 1024
    contrastive_angle_scale: float = 0.75
    contrastive_quantum_temperature: float = 0.04
    kde_sigma_squared: float = 0.03125
    kde_temperature: float = 0.75
    quantum_mixture_weight: float = 0.05
    coverage_weight: float = 0.0
    coverage_temperature: float = 0.10
    angle_residual_fraction: float = 0.10
    gradient_diagnostics_every_steps: int = 0
    log_every_steps: int = 1
    checkpoint_every_steps: int = 0
    download_dataset: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.variant, str):
            object.__setattr__(self, "variant", ExperimentVariant(self.variant))
        if not self.run_name or Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must be one non-empty path component")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.quantum_device not in {"auto", "same", "cpu"}:
            raise ValueError("quantum_device must be one of: auto, same, cpu")
        for name in (
            "epochs",
            "batch_size",
            "latent_dim",
            "num_classes",
            "image_channels",
            "num_qubits",
            "log_every_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_steps", "dataset_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.regularizer_gradient_ratio is not None and self.regularizer_gradient_ratio <= 0:
            raise ValueError("regularizer_gradient_ratio must be positive when set")
        if (
            self.num_workers < 0
            or self.checkpoint_every_steps < 0
            or self.gradient_diagnostics_every_steps < 0
        ):
            raise ValueError("worker, checkpoint, and diagnostic intervals must be non-negative")
        if min(self.learning_rate_generator, self.learning_rate_discriminator) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if not 0 < self.real_label_smoothing <= 1:
            raise ValueError("real_label_smoothing must be in (0, 1]")
        if self.num_classes > 2**self.num_qubits:
            raise ValueError("num_qubits is insufficient for unique class encoding")
        if self.regularizer_weight < 0 or self.regularizer_temperature <= 0:
            raise ValueError("regularizer weight must be non-negative and temperature positive")
        if self.modular_angle_scale <= 0:
            raise ValueError("modular_angle_scale must be positive")
        if self.density_mmd_angle_scale <= 0:
            raise ValueError("density_mmd_angle_scale must be positive")
        if not 0 < self.modular_depolarization < 1:
            raise ValueError("modular_depolarization must be in (0, 1)")
        if self.rbf_sigma_squared <= 0:
            raise ValueError("rbf_sigma_squared must be positive")
        if self.coherence_gradient_ratio <= 0:
            raise ValueError("coherence_gradient_ratio must be positive")
        if self.reference_samples_per_class <= 0:
            raise ValueError("reference_samples_per_class must be positive")
        if self.contrastive_reference_samples_per_class <= 0:
            raise ValueError("contrastive_reference_samples_per_class must be positive")
        if self.contrastive_angle_scale <= 0 or self.contrastive_quantum_temperature <= 0:
            raise ValueError("contrastive angle scale and temperature must be positive")
        if self.kde_sigma_squared <= 0 or self.kde_temperature <= 0:
            raise ValueError("KDE sigma and temperature must be positive")
        if not 0 <= self.quantum_mixture_weight <= 1:
            raise ValueError("quantum_mixture_weight must be in [0, 1]")
        if self.coverage_weight < 0 or self.coverage_temperature <= 0:
            raise ValueError("coverage weight must be non-negative and temperature positive")
        if not 0 < self.angle_residual_fraction <= 1:
            raise ValueError("angle_residual_fraction must be in (0, 1]")
        if self.variant is ExperimentVariant.NO_REGULARIZER and self.regularizer_weight != 0:
            raise ValueError("no_regularizer requires regularizer_weight=0")
        if (
            self.variant is ExperimentVariant.NO_REGULARIZER
            and self.regularizer_gradient_ratio is not None
        ):
            raise ValueError("no_regularizer cannot request dynamic gradient balancing")
        if self.variant.uses_contrastive_reference and self.regularizer_gradient_ratio is not None:
            raise ValueError("contrastive reference variants require a fixed calibrated weight")
        if self.variant is not ExperimentVariant.NO_REGULARIZER and self.regularizer_weight <= 0:
            raise ValueError("regularized variants require a positive regularizer_weight")
        if self.variant.uses_relational_coverage and self.coverage_weight <= 0:
            raise ValueError("relational coverage variants require a calibrated coverage_weight")
        if not self.variant.uses_relational_coverage and self.coverage_weight != 0:
            raise ValueError("coverage_weight is only valid for relational coverage variants")

    @property
    def output_directory(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant.value
        return payload
