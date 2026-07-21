from __future__ import annotations

import pytest

from vqe_gan.config import CoverageBudgetMode, ExperimentConfig, ExperimentVariant
from vqe_gan.training import (
    CoverageBudgetSchedule,
    CoverageBudgetTarget,
    coverage_budget_metadata,
)


def _config(
    variant: ExperimentVariant,
    mode: CoverageBudgetMode,
    *,
    schedule: str | None = None,
    phase: str = "a",
) -> ExperimentConfig:
    return ExperimentConfig(
        run_name=f"budget-{variant.value}",
        variant=variant,
        seed=42,
        device="cpu",
        quantum_device="cpu",
        max_steps=2,
        regularizer_weight=2e-5,
        coverage_weight=(ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE.default_coverage_weight),
        coverage_budget_mode=mode,
        coverage_budget_schedule=schedule,
        freeze_angle_head=phase == "a",
        match_angle_head_budget=phase == "b",
    )


def _target(step: int, *, phase: str = "a") -> CoverageBudgetTarget:
    return CoverageBudgetTarget(
        step=step,
        shared_ratio=0.01,
        shared_anchor_gradient_norm=2.0,
        shared_coverage_gradient_norm=3.0,
        shared_weighted_coverage_gradient_norm=0.02,
        shared_adam_update_norm=0.03,
        shared_adam_auxiliary_ratio=0.004,
        angle_gradient_norm=0.05 if phase == "b" else None,
        angle_update_norm=0.006 if phase == "b" else None,
    )


@pytest.mark.parametrize("phase", ["a", "b"])
def test_budget_schedule_round_trip_and_control_validation(tmp_path, phase: str) -> None:
    reference = _config(
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
        CoverageBudgetMode.RECORD,
        phase=phase,
    )
    path = tmp_path / "coverage-budget.json"
    CoverageBudgetSchedule(
        metadata=coverage_budget_metadata(reference),
        records=(_target(1, phase=phase), _target(2, phase=phase)),
    ).write(path)

    control = _config(
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
        CoverageBudgetMode.REPLAY,
        schedule=str(path),
        phase=phase,
    )
    loaded = CoverageBudgetSchedule.read(path)
    loaded.validate_for(control)

    assert loaded.records[0].shared_ratio == 0.01
    assert loaded.metadata["phase"] == phase


def test_budget_schedule_rejects_wrong_seed(tmp_path) -> None:
    reference = _config(
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
        CoverageBudgetMode.RECORD,
    )
    path = tmp_path / "coverage-budget.json"
    CoverageBudgetSchedule(
        metadata=coverage_budget_metadata(reference),
        records=(_target(1), _target(2)),
    ).write(path)
    mismatched = ExperimentConfig(
        **{
            **_config(
                ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
                CoverageBudgetMode.REPLAY,
                schedule=str(path),
            ).to_dict(),
            "run_name": "wrong-seed",
            "seed": 43,
        }
    )

    with pytest.raises(ValueError, match="metadata mismatch for seed"):
        CoverageBudgetSchedule.read(path).validate_for(mismatched)


def test_budget_configuration_rejects_ambiguous_angle_control() -> None:
    with pytest.raises(ValueError, match="cannot request"):
        _config(
            ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            CoverageBudgetMode.RECORD,
            phase="a",
        ).__class__(
            **{
                **_config(
                    ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
                    CoverageBudgetMode.RECORD,
                    phase="a",
                ).to_dict(),
                "freeze_angle_head": True,
                "match_angle_head_budget": True,
            }
        )
