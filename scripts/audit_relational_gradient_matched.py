#!/usr/bin/env python3
"""Audit the post-hoc gradient-matched relational circuit diagnostic."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.quantum.spec import QuantumCircuitSpec
from vqe_gan.regularizers import (
    KDERelationalCoverageReference,
    RelationalKernel,
)
from vqe_gan.reproducibility import (
    collect_provenance,
    file_sha256,
    seed_everything,
    write_json,
)
from vqe_gan.runner import _build_data_loader, _build_models, _fit_reference_regularizer
from vqe_gan.training import measure_relational_gradient_diagnostics

_FULL = ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE
_PRODUCT = ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT
_DEPHASED = ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED
_VARIANTS = (_FULL, _PRODUCT, _DEPHASED)
_CONTROLS = (_PRODUCT, _DEPHASED)
_KERNELS = {
    _FULL: RelationalKernel.QUANTUM_FULL,
    _PRODUCT: RelationalKernel.QUANTUM_PRODUCT,
    _DEPHASED: RelationalKernel.QUANTUM_DEPHASED,
}
_CALIBRATION_KEYS = {
    _FULL: "quantum_kde_relational_coverage",
    _PRODUCT: "quantum_kde_relational_product",
    _DEPHASED: "quantum_kde_relational_dephased",
}
_METRICS = (
    "conditional_accuracy",
    "feature_fid_64",
    "class_conditional_feature_fid_64",
    "feature_precision",
    "feature_recall",
    "diversity_ratio",
)
_TRAJECTORY_METRICS = (
    "adversarial",
    "auxiliary",
    "classical_regularizer",
    "coverage_regularizer",
    "total",
)
_DIAGNOSTIC_NOISE_BASE_SEED = 94_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-roots", nargs=2, required=True)
    parser.add_argument("--product-roots", nargs=2, required=True)
    parser.add_argument("--dephased-roots", nargs=2, required=True)
    parser.add_argument(
        "--calibration",
        default="docs/relational_gradient_matched_calibration.json",
    )
    parser.add_argument(
        "--same-weight-results",
        default="docs/relational_coverage_ablation_results.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"gradient-matched audit output already exists: {output}")
    repository_root = Path(__file__).resolve().parents[1]
    calibration_path = Path(arguments.calibration)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_weights = {
        variant: float(calibration["suggested_weights"][_CALIBRATION_KEYS[variant]])
        for variant in _VARIANTS
    }
    roots = {
        _FULL: _roots_by_seed(arguments.full_roots),
        _PRODUCT: _roots_by_seed(arguments.product_roots),
        _DEPHASED: _roots_by_seed(arguments.dephased_roots),
    }

    runs = []
    for seed in (42, 43):
        variants = {
            variant.value: _read_run(
                roots[variant][seed],
                seed,
                variant,
                expected_weights[variant],
            )
            for variant in _VARIANTS
        }
        variants[_FULL.value]["common_checkpoint_kernel_geometry"] = (
            _common_checkpoint_kernel_geometry(
                roots[_FULL][seed],
                seed,
                expected_weights,
            )
        )
        runs.append({"seed": seed, "variants": variants})

    mean_metrics = {
        variant.value: {
            metric: statistics.mean(
                float(run["variants"][variant.value]["metrics"][metric]) for run in runs
            )
            for metric in _METRICS
        }
        for variant in _VARIANTS
    }
    paired_deltas = {control.value: _paired_deltas(runs, _FULL, control) for control in _CONTROLS}
    checks_by_control = {
        control.value: _margin_checks(mean_metrics, control) for control in _CONTROLS
    }
    common_geometry = _common_geometry_summary(runs)
    same_weight_path = Path(arguments.same_weight_results)
    same_weight = json.loads(same_weight_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "status": "post_hoc_development_only",
        "protocol": "docs/relational_gradient_matched_diagnostic_protocol.md",
        "heldout_authorized": False,
        "calibration": {
            "path": str(calibration_path),
            "sha256": file_sha256(calibration_path),
            "target_initial_shared_gradient_ratio": calibration[
                "target_initial_shared_gradient_ratio"
            ],
            "weights": {variant.value: expected_weights[variant] for variant in _VARIANTS},
            "realized_ratio_summaries": _calibration_ratio_summaries(
                calibration,
                expected_weights,
            ),
            "initial_common_geometry": calibration["structural_checks"],
        },
        "runs": runs,
        "mean_metrics": mean_metrics,
        "full_minus_controls": paired_deltas,
        "checks_by_control": checks_by_control,
        "all_descriptive_margins_passed": all(
            all(checks.values()) for checks in checks_by_control.values()
        ),
        "seed_consistency": {
            control.value: _seed_consistency(paired_deltas[control.value]) for control in _CONTROLS
        },
        "common_full_checkpoint_geometry": common_geometry,
        "same_coefficient_comparison": {
            "path": str(same_weight_path),
            "sha256": file_sha256(same_weight_path),
            "control_mean_shift_after_gradient_matching": (
                _same_weight_control_shifts(mean_metrics, same_weight)
            ),
        },
        "interpretation": {
            "mean_distributional_advantage_over_product": (
                checks_by_control[_PRODUCT.value]["full_class_fid_better_by_at_least_0_005"]
                and checks_by_control[_PRODUCT.value]["full_diversity_not_lower_by_more_than_0_005"]
            ),
            "mean_distributional_advantage_over_dephased": (
                checks_by_control[_DEPHASED.value]["full_class_fid_better_by_at_least_0_005"]
                and checks_by_control[_DEPHASED.value][
                    "full_diversity_not_lower_by_more_than_0_005"
                ]
            ),
            "accuracy_tradeoff_within_frozen_margin": all(
                checks_by_control[control.value]["full_accuracy_not_lower_by_more_than_0_005"]
                for control in _CONTROLS
            ),
            "phase_coherent_contribution_isolated": all(
                checks_by_control[_DEPHASED.value].values()
            ),
            "entanglement_contribution_isolated": all(checks_by_control[_PRODUCT.value].values()),
            "control_shared_gradient_direction_close_to_full": all(
                summary["minimum_shared_gradient_cosine"] > 0.9
                for summary in common_geometry.values()
            ),
            "initial_gradient_match_persisted_at_full_checkpoint": all(
                0.5 <= summary["minimum_weighted_shared_norm_control_over_full"]
                and summary["maximum_weighted_shared_norm_control_over_full"] <= 2.0
                for summary in common_geometry.values()
            ),
            "confirmatory_claim_permitted": False,
        },
        "audit_provenance": collect_provenance(repository_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _roots_by_seed(values: list[str]) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for value in values:
        root = Path(value)
        summaries = list(root.glob("matrix-seed-*.json"))
        if len(summaries) != 1:
            raise ValueError(f"expected one matrix summary under {root}")
        matrix = json.loads(summaries[0].read_text(encoding="utf-8"))
        roots[int(matrix["seed"])] = root
    if set(roots) != {42, 43}:
        raise ValueError("gradient-matched audit requires exactly seeds 42 and 43")
    return roots


def _read_run(
    root: Path,
    seed: int,
    variant: ExperimentVariant,
    expected_weight: float,
) -> dict[str, Any]:
    matrix_path = root / f"matrix-seed-{seed}.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    run_directory = root / f"{variant.value}-seed-{seed}"
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_directory / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_directory / "evaluation.json").read_text(encoding="utf-8"))
    if config["variant"] != variant.value or config["seed"] != seed:
        raise RuntimeError("run directory and serialized configuration disagree")
    if config["regularizer_weight"] != 2e-5:
        raise RuntimeError("diagnostic changed the exact KDE weight")
    if config["coverage_weight"] != expected_weight:
        raise RuntimeError("diagnostic run does not use the committed calibrated weight")
    if summary["completed_steps"] != 200 or evaluation["samples"] != 5_000:
        raise RuntimeError("diagnostic run does not match the frozen horizon")
    if evaluation["seed"] != 91_001:
        raise RuntimeError("diagnostic run changed the frozen evaluation seed")
    checkpoint_path = run_directory / "checkpoint-final.pt"
    return {
        "run_directory": str(run_directory),
        "config": config,
        "source_revision": provenance["source_revision"],
        "source_dirty": provenance["source_dirty"],
        "matrix_sha256": file_sha256(matrix_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_seconds": matrix["training_seconds"][variant.value],
        "evaluation_seconds": matrix["evaluation_seconds"][variant.value],
        "metrics": {metric: evaluation["metrics"][metric] for metric in _METRICS},
        "trajectory": _trajectory_summary(run_directory / "metrics.jsonl", expected_weight),
        "post_run_mechanism": _audit_checkpoint(checkpoint_path),
    }


def _trajectory_summary(path: Path, coverage_weight: float) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 200 or [record["step"] for record in records] != list(range(1, 201)):
        raise RuntimeError("training trajectory is incomplete or out of order")

    def summarize(window: list[dict[str, Any]]) -> dict[str, float]:
        values = {
            metric: statistics.mean(float(record["generator"][metric]) for record in window)
            for metric in _TRAJECTORY_METRICS
        }
        values["weighted_coverage_contribution"] = statistics.mean(
            coverage_weight * float(record["generator"]["coverage_regularizer"])
            for record in window
        )
        values["weighted_kde_contribution"] = statistics.mean(
            2e-5 * float(record["generator"]["classical_regularizer"]) for record in window
        )
        return values

    return {
        "first_20_steps": summarize(records[:20]),
        "last_20_steps": summarize(records[-20:]),
        "all_steps": summarize(records),
    }


def _audit_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    seed_everything(config.seed)
    device = torch.device("cpu")
    data_loader = _build_data_loader(config, device)
    generator, discriminator, regularizer = _build_models(config, device, device)
    assert isinstance(regularizer, KDERelationalCoverageReference)
    initial_angle_state = {
        name: value.detach().clone() for name, value in generator.angle_head.state_dict().items()
    }
    _fit_reference_regularizer(
        regularizer,
        data_loader.dataset,
        num_classes=config.num_classes,
        samples_per_class=config.contrastive_reference_samples_per_class,
    )
    generator.load_state_dict(checkpoint["generator"], strict=True)
    discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
    real_images, class_labels = next(iter(data_loader))
    noise = torch.randn(
        real_images.shape[0],
        config.latent_dim,
        generator=torch.Generator().manual_seed(_DIAGNOSTIC_NOISE_BASE_SEED + config.seed),
    )
    diagnostics = measure_relational_gradient_diagnostics(
        generator,
        discriminator,
        regularizer,
        real_images,
        noise,
        class_labels,
        kde_weight=config.regularizer_weight,
        coverage_weight=config.coverage_weight,
    )
    angle_differences = [
        value.detach() - initial_angle_state[name]
        for name, value in generator.angle_head.state_dict().items()
    ]
    angle_parameter_changes = {
        name: {
            "changed_values": torch.count_nonzero(
                value.detach() - initial_angle_state[name]
            ).item(),
            "change_norm": (value.detach() - initial_angle_state[name]).norm().item(),
            "max_abs_change": (value.detach() - initial_angle_state[name]).abs().max().item(),
        }
        for name, value in generator.angle_head.state_dict().items()
    }
    return {
        "diagnostic_noise_seed": _DIAGNOSTIC_NOISE_BASE_SEED + config.seed,
        "angle_head_changed_values": sum(
            torch.count_nonzero(difference).item() for difference in angle_differences
        ),
        "angle_head_change_norm": sum(difference.square().sum() for difference in angle_differences)
        .sqrt()
        .item(),
        "angle_head_max_abs_change": max(
            difference.abs().max().item() for difference in angle_differences
        ),
        "angle_parameter_changes": angle_parameter_changes,
        "applied_angle_correction_rms": (
            config.angle_residual_fraction * diagnostics.angle_residual_rms
        ),
        "gradient_diagnostics": asdict(diagnostics),
    }


def _common_checkpoint_kernel_geometry(
    root: Path,
    seed: int,
    weights: dict[ExperimentVariant, float],
) -> dict[str, Any]:
    checkpoint_path = root / f"{_FULL.value}-seed-{seed}" / "checkpoint-final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    seed_everything(config.seed)
    device = torch.device("cpu")
    data_loader = _build_data_loader(config, device)
    generator, _, _ = _build_models(config, device, device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()
    real_images, class_labels = next(iter(data_loader))
    noise = torch.randn(
        real_images.shape[0],
        config.latent_dim,
        generator=torch.Generator().manual_seed(_DIAGNOSTIC_NOISE_BASE_SEED + 100 + config.seed),
    )
    circuit_spec = QuantumCircuitSpec(
        num_qubits=config.num_qubits,
        reps=config.ansatz_reps,
    )
    measurements: dict[ExperimentVariant, dict[str, Any]] = {}
    reference_angles: Tensor | None = None
    for variant in _VARIANTS:
        regularizer = KDERelationalCoverageReference(
            circuit_spec,
            num_classes=config.num_classes,
            execution_device=device,
            angle_scale=config.contrastive_angle_scale,
            angle_residual_fraction=config.angle_residual_fraction,
            coverage_temperature=config.coverage_temperature,
            kde_sigma_squared=config.kde_sigma_squared,
            kde_temperature=config.kde_temperature,
            kernel=_KERNELS[variant],
        )
        _fit_reference_regularizer(
            regularizer,
            data_loader.dataset,
            num_classes=config.num_classes,
            samples_per_class=config.contrastive_reference_samples_per_class,
        )
        if reference_angles is None:
            reference_angles = regularizer.coverage.reference_angles.detach().clone()
        elif not torch.equal(reference_angles, regularizer.coverage.reference_angles):
            raise RuntimeError("kernel controls did not receive identical reference angles")
        measurements[variant] = _coverage_gradient_measurement(
            generator,
            regularizer,
            noise,
            class_labels,
            weights[variant],
        )

    full_measurement = measurements[_FULL]
    comparisons = {}
    for control in _CONTROLS:
        control_measurement = measurements[control]
        comparisons[control.value] = {
            "shared_gradient_cosine": _cosine(
                full_measurement["shared_gradients"],
                control_measurement["shared_gradients"],
            ),
            "angle_gradient_cosine": _cosine(
                full_measurement["angle_gradients"],
                control_measurement["angle_gradients"],
            ),
            "weighted_shared_norm_control_over_full": (
                control_measurement["weighted_shared_gradient_norm"]
                / full_measurement["weighted_shared_gradient_norm"]
            ),
            "weighted_angle_norm_control_over_full": (
                control_measurement["weighted_angle_gradient_norm"]
                / full_measurement["weighted_angle_gradient_norm"]
            ),
            "coverage_loss_control_minus_full": (
                control_measurement["coverage_loss"] - full_measurement["coverage_loss"]
            ),
        }
    for measurement in measurements.values():
        measurement.pop("shared_gradients", None)
        measurement.pop("angle_gradients", None)
    return {
        "diagnostic_noise_seed": _DIAGNOSTIC_NOISE_BASE_SEED + 100 + config.seed,
        "reference_angles_exactly_shared": True,
        "measurements": {variant.value: measurements[variant] for variant in _VARIANTS},
        "full_vs_controls": comparisons,
    }


def _coverage_gradient_measurement(
    generator: torch.nn.Module,
    regularizer: KDERelationalCoverageReference,
    noise: Tensor,
    class_labels: Tensor,
    weight: float,
) -> dict[str, Any]:
    shared_parameters = (
        *generator.label_embedding.parameters(),
        *generator.input_projection.parameters(),
        *generator.image_decoder.parameters(),
    )
    angle_parameters = tuple(generator.angle_head.parameters())
    images, angle_residuals = generator.forward_with_angles(noise, class_labels)
    coverage_loss = regularizer.coverage(
        images,
        angle_residuals,
        class_labels,
    )
    gradients = torch.autograd.grad(
        coverage_loss,
        (*shared_parameters, *angle_parameters),
        allow_unused=True,
    )
    gradients = tuple(
        gradient if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(
            gradients,
            (*shared_parameters, *angle_parameters),
            strict=True,
        )
    )
    shared_gradients = gradients[: len(shared_parameters)]
    angle_gradients = gradients[len(shared_parameters) :]
    shared_norm = _norm(shared_gradients)
    angle_norm = _norm(angle_gradients)
    return {
        "coverage_loss": coverage_loss.detach().item(),
        "coverage_weight": weight,
        "shared_gradient_norm": shared_norm,
        "angle_gradient_norm": angle_norm,
        "weighted_shared_gradient_norm": weight * shared_norm,
        "weighted_angle_gradient_norm": weight * angle_norm,
        "angle_residual_rms": angle_residuals.detach().square().mean().sqrt().item(),
        "shared_gradients": shared_gradients,
        "angle_gradients": angle_gradients,
    }


def _paired_deltas(
    runs: list[dict[str, Any]],
    first: ExperimentVariant,
    second: ExperimentVariant,
) -> list[dict[str, Any]]:
    deltas = []
    for run in runs:
        first_metrics = run["variants"][first.value]["metrics"]
        second_metrics = run["variants"][second.value]["metrics"]
        deltas.append(
            {
                "seed": run["seed"],
                **{metric: first_metrics[metric] - second_metrics[metric] for metric in _METRICS},
            }
        )
    deltas.append(
        {
            "seed": "mean",
            **{
                metric: statistics.mean(float(item[metric]) for item in deltas)
                for metric in _METRICS
            },
        }
    )
    return deltas


def _margin_checks(
    mean_metrics: dict[str, dict[str, float]],
    control: ExperimentVariant,
) -> dict[str, bool]:
    full = mean_metrics[_FULL.value]
    comparison = mean_metrics[control.value]
    return {
        "full_class_fid_better_by_at_least_0_005": (
            comparison["class_conditional_feature_fid_64"]
            - full["class_conditional_feature_fid_64"]
            >= 0.005
        ),
        "full_diversity_not_lower_by_more_than_0_005": (
            full["diversity_ratio"] >= comparison["diversity_ratio"] - 0.005
        ),
        "full_accuracy_not_lower_by_more_than_0_005": (
            full["conditional_accuracy"] >= comparison["conditional_accuracy"] - 0.005
        ),
    }


def _seed_consistency(deltas: list[dict[str, Any]]) -> dict[str, bool]:
    seeds = [item for item in deltas if item["seed"] != "mean"]
    return {
        "full_class_fid_lower_on_both_seeds": all(
            item["class_conditional_feature_fid_64"] < 0 for item in seeds
        ),
        "full_diversity_higher_on_both_seeds": all(item["diversity_ratio"] > 0 for item in seeds),
        "full_accuracy_not_lower_on_both_seeds": all(
            item["conditional_accuracy"] >= 0 for item in seeds
        ),
    }


def _common_geometry_summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for control in _CONTROLS:
        records = [
            run["variants"][_FULL.value]["common_checkpoint_kernel_geometry"]["full_vs_controls"][
                control.value
            ]
            for run in runs
        ]
        shared_cosines = [float(record["shared_gradient_cosine"]) for record in records]
        angle_cosines = [float(record["angle_gradient_cosine"]) for record in records]
        norm_ratios = [
            float(record["weighted_shared_norm_control_over_full"]) for record in records
        ]
        result[control.value] = {
            "mean_shared_gradient_cosine": statistics.mean(shared_cosines),
            "minimum_shared_gradient_cosine": min(shared_cosines),
            "maximum_shared_gradient_cosine": max(shared_cosines),
            "mean_angle_gradient_cosine": statistics.mean(angle_cosines),
            "minimum_angle_gradient_cosine": min(angle_cosines),
            "maximum_angle_gradient_cosine": max(angle_cosines),
            "mean_weighted_shared_norm_control_over_full": statistics.mean(norm_ratios),
            "minimum_weighted_shared_norm_control_over_full": min(norm_ratios),
            "maximum_weighted_shared_norm_control_over_full": max(norm_ratios),
        }
    return result


def _calibration_ratio_summaries(
    calibration: dict[str, Any],
    weights: dict[ExperimentVariant, float],
) -> dict[str, dict[str, float]]:
    measurement_keys = {
        _FULL: "quantum_full",
        _PRODUCT: "quantum_product",
        _DEPHASED: "quantum_dephased",
    }
    result = {}
    for variant, key in measurement_keys.items():
        ratios = sorted(
            weights[variant]
            * float(measurement["coverage_shared_norm"])
            / float(measurement["gan_objective_norm"])
            for measurement in calibration["measurements"][key]
        )
        result[variant.value] = {
            "minimum": min(ratios),
            "median": statistics.median(ratios),
            "maximum": max(ratios),
        }
    return result


def _same_weight_control_shifts(
    matched_means: dict[str, dict[str, float]],
    same_weight: dict[str, Any],
) -> dict[str, dict[str, float]]:
    return {
        control.value: {
            metric: (
                matched_means[control.value][metric]
                - float(same_weight["mean_metrics"][control.value][metric])
            )
            for metric in _METRICS
        }
        for control in _CONTROLS
    }


def _norm(gradients: tuple[Tensor, ...]) -> float:
    return sum((gradient.square().sum() for gradient in gradients), start=0.0).sqrt().item()


def _cosine(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> float | None:
    first_norm = _norm(first)
    second_norm = _norm(second)
    if min(first_norm, second_norm) <= torch.finfo(torch.float32).eps:
        return None
    dot = sum(
        ((left * right).sum() for left, right in zip(first, second, strict=True)),
        start=0.0,
    )
    assert isinstance(dot, Tensor)
    return (dot / (first_norm * second_norm)).item()


if __name__ == "__main__":
    main()
