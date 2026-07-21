from __future__ import annotations

import pytest

from vqe_gan.cli import build_parser, config_from_arguments
from vqe_gan.config import ExperimentConfig, ExperimentVariant


def test_no_regularizer_cli_sets_zero_regularizer_weight() -> None:
    arguments = build_parser().parse_args(
        ["--run-name", "control", "--variant", "no_regularizer"]
    )
    config = config_from_arguments(arguments)

    assert config.variant is ExperimentVariant.NO_REGULARIZER
    assert config.regularizer_weight == 0
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
            "--regularizer-weight",
            "0.25",
        ]
    )
    config = config_from_arguments(arguments)

    assert config.variant is ExperimentVariant.QUANTUM_CONTRASTIVE
    assert config.max_steps == 2
    assert config.dataset_limit == 128
    assert config.regularizer_weight == 0.25


def test_configuration_rejects_inconsistent_variant_and_unsafe_name() -> None:
    with pytest.raises(ValueError, match="no_regularizer"):
        ExperimentConfig(
            run_name="control",
            variant=ExperimentVariant.NO_REGULARIZER,
            regularizer_weight=0.1,
        )
    with pytest.raises(ValueError, match="path component"):
        ExperimentConfig(run_name="../outside")


def test_modular_cli_exposes_stabilization_parameters() -> None:
    arguments = build_parser().parse_args(
        [
            "--run-name",
            "modular-smoke",
            "--variant",
            "quantum_modular_free_energy",
            "--modular-angle-scale",
            "0.2",
            "--modular-depolarization",
            "0.02",
        ]
    )
    config = config_from_arguments(arguments)

    assert config.variant.uses_distribution_regularizer
    assert config.variant.uses_quantum_backend
    assert config.modular_angle_scale == 0.2
    assert config.modular_depolarization == 0.02


def test_coherence_guidance_cli_exposes_its_independent_gradient_budget() -> None:
    arguments = build_parser().parse_args(
        [
            "--run-name",
            "coherence-smoke",
            "--variant",
            "quantum_coherence_guidance",
            "--coherence-gradient-ratio",
            "0.03",
        ]
    )
    config = config_from_arguments(arguments)

    assert config.variant.uses_coherence_guidance
    assert config.variant.uses_quantum_backend
    assert config.coherence_gradient_ratio == 0.03


def test_reference_cli_exposes_balanced_bank_size() -> None:
    arguments = build_parser().parse_args(
        [
            "--run-name",
            "reference-smoke",
            "--variant",
            "quantum_modular_reference",
            "--reference-samples-per-class",
            "32",
        ]
    )
    config = config_from_arguments(arguments)

    assert config.variant.uses_distribution_regularizer
    assert config.variant.uses_quantum_backend
    assert config.reference_samples_per_class == 32


def test_hybrid_contrastive_cli_exposes_frozen_calibration() -> None:
    arguments = build_parser().parse_args(
        [
            "--run-name",
            "hybrid-contrastive-smoke",
            "--variant",
            "hybrid_modular_kde_contrastive",
            "--contrastive-reference-samples-per-class",
            "64",
            "--contrastive-angle-scale",
            "0.7",
            "--contrastive-quantum-temperature",
            "0.05",
            "--kde-sigma-squared",
            "0.04",
            "--kde-temperature",
            "0.8",
            "--quantum-mixture-weight",
            "0.1",
        ]
    )
    config = config_from_arguments(arguments)

    assert config.variant.uses_distribution_regularizer
    assert config.variant.uses_quantum_backend
    assert config.variant.uses_contrastive_reference
    assert config.contrastive_reference_samples_per_class == 64
    assert config.contrastive_angle_scale == 0.7
    assert config.contrastive_quantum_temperature == 0.05
    assert config.kde_sigma_squared == 0.04
    assert config.kde_temperature == 0.8
    assert config.quantum_mixture_weight == 0.1
    assert config.regularizer_weight == 2e-5


def test_contrastive_reference_rejects_online_gradient_balancing() -> None:
    with pytest.raises(ValueError, match="fixed calibrated weight"):
        ExperimentConfig(
            run_name="invalid-online-balance",
            variant=ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE,
            regularizer_weight=2e-5,
            regularizer_gradient_ratio=0.1,
        )


def test_relational_coverage_requires_an_explicit_precalibrated_weight() -> None:
    with pytest.raises(ValueError, match="calibrated coverage_weight"):
        build_parser().parse_args(
            [
                "--run-name",
                "uncalibrated-coverage",
                "--variant",
                "quantum_kde_relational_coverage",
            ]
        )
        config_from_arguments(
            build_parser().parse_args(
                [
                    "--run-name",
                    "uncalibrated-coverage",
                    "--variant",
                    "quantum_kde_relational_coverage",
                ]
            )
        )

    config = config_from_arguments(
        build_parser().parse_args(
            [
                "--run-name",
                "calibrated-coverage",
                "--variant",
                "quantum_kde_relational_coverage",
                "--coverage-weight",
                "0.03",
            ]
        )
    )
    assert config.variant.uses_relational_coverage
    assert config.coverage_weight == 0.03
    assert config.regularizer_weight == 2e-5


def test_kde_scale_control_changes_only_the_outer_default_weight() -> None:
    config = config_from_arguments(
        build_parser().parse_args(
            [
                "--run-name",
                "kde-scale-control",
                "--variant",
                "classical_log_kde_scale_control",
            ]
        )
    )

    assert config.regularizer_weight == 1.9e-5
    assert config.coverage_weight == 0
