#!/usr/bin/env python3
"""Audit frozen relational development runs and evaluate their preregistered gate."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.regularizers import KDERelationalCoverageReference
from vqe_gan.reproducibility import collect_provenance, file_sha256, seed_everything, write_json
from vqe_gan.runner import _build_data_loader, _build_models, _fit_reference_regularizer
from vqe_gan.training import measure_relational_gradient_diagnostics

_VARIANTS = (
    ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE,
    ExperimentVariant.CLASSICAL_LOG_KDE_SCALE_CONTROL,
    ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE,
    ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
)
_METRICS = (
    "conditional_accuracy",
    "feature_fid_64",
    "class_conditional_feature_fid_64",
    "feature_precision",
    "feature_recall",
    "diversity_ratio",
)
_DIAGNOSTIC_NOISE_BASE_SEED = 92_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    roots = [Path(value) for value in arguments.run_roots]
    repository_root = Path(__file__).resolve().parents[1]

    runs = [_read_seed(root) for root in roots]
    runs.sort(key=lambda item: int(item["seed"]))
    if [item["seed"] for item in runs] != [42, 43]:
        raise ValueError("development audit requires exactly seeds 42 and 43")
    mean_metrics = {
        variant.value: {
            metric: statistics.mean(
                float(run["variants"][variant.value]["metrics"][metric])
                for run in runs
            )
            for metric in _METRICS
        }
        for variant in _VARIANTS
    }
    quantum_deltas = _paired_deltas(
        runs,
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
        ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE,
    )
    quantum_vs_matched = _paired_deltas(
        runs,
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
        ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE,
    )
    gate_checks = {
        "class_fid_lower_than_kde_on_both_seeds": all(
            delta["class_conditional_feature_fid_64"] < 0 for delta in quantum_deltas
        ),
        "mean_diversity_not_lower_than_kde": (
            mean_metrics[ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE.value][
                "diversity_ratio"
            ]
            >= mean_metrics[ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE.value][
                "diversity_ratio"
            ]
        ),
        "mean_accuracy_within_0_005_of_kde": (
            mean_metrics[ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE.value][
                "conditional_accuracy"
            ]
            >= mean_metrics[ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE.value][
                "conditional_accuracy"
            ]
            - 0.005
        ),
        "mean_class_fid_lower_than_matched_classical": (
            mean_metrics[ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE.value][
                "class_conditional_feature_fid_64"
            ]
            < mean_metrics[ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE.value][
                "class_conditional_feature_fid_64"
            ]
        ),
        "angle_head_changed_and_shared_gradient_remains": all(
            _mechanism_gate(run) for run in runs
        ),
        "runs_complete_and_finite": all(_run_is_complete(run) for run in runs),
    }
    result = {
        "schema_version": 1,
        "protocol": "docs/trainable_relational_coverage_protocol.md",
        "source_revision_under_test": _single_revision(runs),
        "evaluation_samples": 5_000,
        "evaluation_seed": 91_001,
        "runs": runs,
        "mean_metrics": mean_metrics,
        "quantum_minus_kde": quantum_deltas,
        "quantum_minus_matched_classical": quantum_vs_matched,
        "gate_checks": gate_checks,
        "development_gate_passed": all(gate_checks.values()),
        "audit_provenance": collect_provenance(repository_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _read_seed(root: Path) -> dict[str, Any]:
    matrix_paths = list(root.glob("matrix-seed-*.json"))
    if len(matrix_paths) != 1:
        raise ValueError(f"expected one matrix summary under {root}")
    matrix = json.loads(matrix_paths[0].read_text(encoding="utf-8"))
    seed = int(matrix["seed"])
    variants = {}
    for variant in _VARIANTS:
        run_directory = root / f"{variant.value}-seed-{seed}"
        config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
        provenance = json.loads(
            (run_directory / "provenance.json").read_text(encoding="utf-8")
        )
        summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
        evaluation = json.loads(
            (run_directory / "evaluation.json").read_text(encoding="utf-8")
        )
        if config["seed"] != seed or config["variant"] != variant.value:
            raise RuntimeError("run directory and serialized configuration disagree")
        if summary["completed_steps"] != 200 or evaluation["samples"] != 5_000:
            raise RuntimeError("development run does not match the frozen horizon")
        record: dict[str, Any] = {
            "config": config,
            "source_revision": provenance["source_revision"],
            "source_dirty": provenance["source_dirty"],
            "training_seconds": matrix["training_seconds"][variant.value],
            "evaluation_seconds": matrix["evaluation_seconds"][variant.value],
            "checkpoint_sha256": file_sha256(run_directory / "checkpoint-final.pt"),
            "metrics": {metric: evaluation["metrics"][metric] for metric in _METRICS},
            "last_generator_metrics": summary["last_metrics"]["generator"],
        }
        if variant.uses_relational_coverage:
            record["post_run_mechanism"] = _audit_checkpoint(
                run_directory / "checkpoint-final.pt"
            )
        variants[variant.value] = record
    return {"seed": seed, "matrix_total_seconds": matrix["total_seconds"], "variants": variants}


def _audit_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    if not config.variant.uses_relational_coverage:
        raise ValueError("mechanism audit requires a relational coverage checkpoint")

    seed_everything(config.seed)
    device = torch.device("cpu")
    data_loader = _build_data_loader(config, device)
    quantum_device = device if config.variant.uses_quantum_backend else None
    generator, discriminator, regularizer = _build_models(config, device, quantum_device)
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
    noise_generator = torch.Generator().manual_seed(
        _DIAGNOSTIC_NOISE_BASE_SEED + config.seed
    )
    noise = torch.randn(
        real_images.shape[0],
        config.latent_dim,
        generator=noise_generator,
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
    changed_values = sum(
        torch.count_nonzero(difference).item() for difference in angle_differences
    )
    return {
        "diagnostic_noise_seed": _DIAGNOSTIC_NOISE_BASE_SEED + config.seed,
        "angle_head_changed_values": changed_values,
        "angle_head_change_norm": sum(
            difference.square().sum() for difference in angle_differences
        ).sqrt().item(),
        "angle_head_max_abs_change": max(
            difference.abs().max().item() for difference in angle_differences
        ),
        "gradient_diagnostics": asdict(diagnostics),
    }


def _paired_deltas(
    runs: list[dict[str, Any]],
    first: ExperimentVariant,
    second: ExperimentVariant,
) -> list[dict[str, Any]]:
    result = []
    for run in runs:
        first_metrics = run["variants"][first.value]["metrics"]
        second_metrics = run["variants"][second.value]["metrics"]
        result.append(
            {
                "seed": run["seed"],
                **{
                    metric: first_metrics[metric] - second_metrics[metric]
                    for metric in _METRICS
                },
            }
        )
    result.append(
        {
            "seed": "mean",
            **{
                metric: statistics.mean(float(item[metric]) for item in result)
                for metric in _METRICS
            },
        }
    )
    return result


def _mechanism_gate(run: dict[str, Any]) -> bool:
    mechanism = run["variants"][
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE.value
    ]["post_run_mechanism"]
    gradients = mechanism["gradient_diagnostics"]
    return bool(
        mechanism["angle_head_changed_values"] > 0
        and mechanism["angle_head_change_norm"] > 0
        and gradients["coverage_shared_norm"] > 0
        and gradients["direct_coverage_shared_norm"] > 0
        and gradients["coverage_angle_head_norm"] > 0
    )


def _run_is_complete(run: dict[str, Any]) -> bool:
    return all(
        all(torch.isfinite(torch.tensor(value)) for value in record["metrics"].values())
        for record in run["variants"].values()
    )


def _single_revision(runs: list[dict[str, Any]]) -> str:
    revisions = {
        record["source_revision"]
        for run in runs
        for record in run["variants"].values()
    }
    if len(revisions) != 1:
        raise RuntimeError("development variants were not run from one source revision")
    return revisions.pop()


if __name__ == "__main__":
    main()
