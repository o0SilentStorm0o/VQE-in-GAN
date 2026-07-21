"""Validated full-reference schedules for relational gradient budgets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.reproducibility import write_json

SCHEDULE_SCHEMA_VERSION = 1
REFERENCE_VARIANT = ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE


@dataclass(frozen=True)
class CoverageBudgetTarget:
    """One full-branch optimization budget consumed by a paired control step."""

    step: int
    shared_ratio: float
    shared_anchor_gradient_norm: float
    shared_coverage_gradient_norm: float
    shared_weighted_coverage_gradient_norm: float
    shared_adam_update_norm: float
    shared_adam_auxiliary_ratio: float
    angle_gradient_norm: float | None = None
    angle_update_norm: float | None = None

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("budget step must be positive")
        positive = {
            "shared_ratio": self.shared_ratio,
            "shared_anchor_gradient_norm": self.shared_anchor_gradient_norm,
            "shared_coverage_gradient_norm": self.shared_coverage_gradient_norm,
            "shared_weighted_coverage_gradient_norm": (self.shared_weighted_coverage_gradient_norm),
            "shared_adam_update_norm": self.shared_adam_update_norm,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.shared_adam_auxiliary_ratio)
            or self.shared_adam_auxiliary_ratio < 0
        ):
            raise ValueError("shared_adam_auxiliary_ratio must be finite and non-negative")
        if (self.angle_gradient_norm is None) != (self.angle_update_norm is None):
            raise ValueError("angle gradient and update targets must be present together")
        for name, value in (
            ("angle_gradient_norm", self.angle_gradient_norm),
            ("angle_update_norm", self.angle_update_norm),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoverageBudgetTarget:
        return cls(**payload)


@dataclass(frozen=True)
class CoverageBudgetSchedule:
    """A sealed ordered schedule recorded by the coherent-entangled reference branch."""

    metadata: dict[str, Any]
    records: tuple[CoverageBudgetTarget, ...]

    def __post_init__(self) -> None:
        expected_steps = tuple(range(1, len(self.records) + 1))
        actual_steps = tuple(record.step for record in self.records)
        if actual_steps != expected_steps:
            raise ValueError("budget records must be contiguous, ordered, and one-indexed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "metadata": self.metadata,
            "records": [asdict(record) for record in self.records],
        }

    def write(self, path: str | Path) -> None:
        write_json(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: str | Path) -> CoverageBudgetSchedule:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
            raise ValueError("unsupported coverage budget schedule schema")
        metadata = payload.get("metadata")
        records = payload.get("records")
        if not isinstance(metadata, dict) or not isinstance(records, list):
            raise ValueError("coverage budget schedule is malformed")
        return cls(
            metadata=metadata,
            records=tuple(CoverageBudgetTarget.from_dict(record) for record in records),
        )

    def validate_for(self, config: ExperimentConfig) -> None:
        expected = coverage_budget_metadata(config)
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"budget schedule metadata mismatch for {key}: "
                    f"expected {value!r}, got {self.metadata.get(key)!r}"
                )
        if len(self.records) != config.max_steps:
            raise ValueError("budget schedule length does not match max_steps")
        expects_angles = config.match_angle_head_budget
        has_mismatched_angle_target = any(
            (record.angle_gradient_norm is not None) != expects_angles for record in self.records
        )
        if has_mismatched_angle_target:
            raise ValueError("budget schedule angle targets do not match the requested phase")


def coverage_budget_phase(config: ExperimentConfig) -> str:
    if config.freeze_angle_head:
        return "a"
    if config.match_angle_head_budget:
        return "b"
    raise ValueError("reference-budget runs must select Phase A or Phase B")


def coverage_budget_metadata(config: ExperimentConfig) -> dict[str, Any]:
    """Return the cross-variant settings that must match the reference schedule."""

    if config.max_steps is None:
        raise ValueError("reference-budget runs require max_steps")
    return {
        "seed": config.seed,
        "phase": coverage_budget_phase(config),
        "reference_variant": REFERENCE_VARIANT.value,
        "max_steps": config.max_steps,
        "reference_coverage_weight": REFERENCE_VARIANT.default_coverage_weight,
        "kde_weight": config.regularizer_weight,
        "batch_size": config.batch_size,
        "reference_samples_per_class": config.contrastive_reference_samples_per_class,
        "coverage_temperature": config.coverage_temperature,
        "angle_residual_fraction": config.angle_residual_fraction,
        "learning_rate_generator": config.learning_rate_generator,
        "beta1": config.beta1,
        "beta2": config.beta2,
    }
