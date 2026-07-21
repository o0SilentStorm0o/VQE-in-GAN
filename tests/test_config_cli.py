from __future__ import annotations

import pytest

from vqe_gan.cli import build_parser, config_from_arguments
from vqe_gan.config import ExperimentConfig, ExperimentVariant


def test_no_regularizer_cli_sets_zero_quantum_weight() -> None:
    arguments = build_parser().parse_args(
        ["--run-name", "control", "--variant", "no_regularizer"]
    )
    config = config_from_arguments(arguments)

    assert config.variant is ExperimentVariant.NO_REGULARIZER
    assert config.quantum_weight == 0
    assert config.to_dict()["variant"] == "no_regularizer"


def test_quantum_cli_keeps_explicit_weight_and_smoke_limits() -> None:
    arguments = build_parser().parse_args(
        [
            "--run-name",
            "quantum-smoke",
            "--max-steps",
            "2",
            "--dataset-limit",
            "128",
            "--quantum-weight",
            "0.25",
        ]
    )
    config = config_from_arguments(arguments)

    assert config.variant is ExperimentVariant.QUANTUM_CONTRASTIVE
    assert config.max_steps == 2
    assert config.dataset_limit == 128
    assert config.quantum_weight == 0.25


def test_configuration_rejects_inconsistent_variant_and_unsafe_name() -> None:
    with pytest.raises(ValueError, match="no_regularizer"):
        ExperimentConfig(
            run_name="control",
            variant=ExperimentVariant.NO_REGULARIZER,
            quantum_weight=0.1,
        )
    with pytest.raises(ValueError, match="path component"):
        ExperimentConfig(run_name="../outside")
