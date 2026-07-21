"""Validated experiment configuration with stable JSON serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentVariant(str, Enum):
    NO_REGULARIZER = "no_regularizer"
    QUANTUM_CONTRASTIVE = "quantum_contrastive"


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
    quantum_weight: float = 0.1
    quantum_temperature: float = 1.0
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
        if self.num_workers < 0 or self.checkpoint_every_steps < 0:
            raise ValueError("worker and checkpoint intervals must be non-negative")
        if min(self.learning_rate_generator, self.learning_rate_discriminator) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if not 0 < self.real_label_smoothing <= 1:
            raise ValueError("real_label_smoothing must be in (0, 1]")
        if self.num_classes > 2**self.num_qubits:
            raise ValueError("num_qubits is insufficient for unique class encoding")
        if self.quantum_weight < 0 or self.quantum_temperature <= 0:
            raise ValueError("quantum weight must be non-negative and temperature positive")
        if self.variant is ExperimentVariant.NO_REGULARIZER and self.quantum_weight != 0:
            raise ValueError("no_regularizer requires quantum_weight=0")
        if self.variant is ExperimentVariant.QUANTUM_CONTRASTIVE and self.quantum_weight <= 0:
            raise ValueError("quantum_contrastive requires a positive quantum_weight")

    @property
    def output_directory(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant.value
        return payload
