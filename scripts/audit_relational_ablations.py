#!/usr/bin/env python3
"""Audit relational circuit ablations and apply the frozen circuit-specific gate."""

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

_FULL = ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE
_PRODUCT = ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT
_DEPHASED = ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED
_VARIANTS = (_FULL, _PRODUCT, _DEPHASED)
_METRICS = (
    "conditional_accuracy",
    "feature_fid_64",
    "class_conditional_feature_fid_64",
    "feature_precision",
    "feature_recall",
    "diversity_ratio",
)
_DIAGNOSTIC_NOISE_BASE_SEED = 93_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-roots", nargs=2, required=True)
    parser.add_argument("--development-roots", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"ablation audit output already exists: {output}")
    repository_root = Path(__file__).resolve().parents[1]

    development_roots = _roots_by_seed(arguments.development_roots)
    runs = [
        _read_seed(Path(root), development_roots[seed])
        for seed, root in sorted(_roots_by_seed(arguments.ablation_roots).items())
    ]
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
    full_minus_controls = {
        control.value: {
            metric: (
                mean_metrics[_FULL.value][metric]
                - mean_metrics[control.value][metric]
            )
            for metric in _METRICS
        }
        for control in (_PRODUCT, _DEPHASED)
    }
    checks_by_control = {
        control.value: {
            "full_class_fid_better_by_at_least_0_005": (
                mean_metrics[control.value]["class_conditional_feature_fid_64"]
                - mean_metrics[_FULL.value]["class_conditional_feature_fid_64"]
                >= 0.005
            ),
            "full_diversity_not_lower_by_more_than_0_005": (
                mean_metrics[_FULL.value]["diversity_ratio"]
                >= mean_metrics[control.value]["diversity_ratio"] - 0.005
            ),
            "full_accuracy_not_lower_by_more_than_0_005": (
                mean_metrics[_FULL.value]["conditional_accuracy"]
                >= mean_metrics[control.value]["conditional_accuracy"] - 0.005
            ),
        }
        for control in (_PRODUCT, _DEPHASED)
    }
    replay_checks = {
        str(run["seed"]): run["full_replay"] for run in runs
    }
    result = {
        "schema_version": 1,
        "protocol": "docs/trainable_relational_coverage_protocol.md",
        "source_revision_under_test": _single_revision(runs),
        "evaluation_samples": 5_000,
        "evaluation_seed": 91_001,
        "runs": runs,
        "mean_metrics": mean_metrics,
        "full_minus_controls": full_minus_controls,
        "checks_by_control": checks_by_control,
        "full_replay_checks": replay_checks,
        "circuit_specific_gate_passed": (
            all(all(checks.values()) for checks in checks_by_control.values())
            and all(check["all_state_sections_equal"] for check in replay_checks.values())
        ),
        "audit_provenance": collect_provenance(repository_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _roots_by_seed(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        root = Path(value)
        matrix_paths = list(root.glob("matrix-seed-*.json"))
        if len(matrix_paths) != 1:
            raise ValueError(f"expected one matrix summary under {root}")
        matrix = json.loads(matrix_paths[0].read_text(encoding="utf-8"))
        result[int(matrix["seed"])] = root
    if set(result) != {42, 43}:
        raise ValueError("ablation audit requires exactly seeds 42 and 43")
    return result


def _read_seed(root: Path, development_root: Path) -> dict[str, Any]:
    matrix_path = next(root.glob("matrix-seed-*.json"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
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
        if config["variant"] != variant.value or config["seed"] != seed:
            raise RuntimeError("ablation directory and serialized configuration disagree")
        if config["regularizer_weight"] != 2e-5:
            raise RuntimeError("ablation changed the frozen KDE weight")
        if config["coverage_weight"] != 0.0005662128371831583:
            raise RuntimeError("ablation changed the frozen quantum coverage weight")
        if summary["completed_steps"] != 200 or evaluation["samples"] != 5_000:
            raise RuntimeError("ablation run does not match the frozen horizon")
        variants[variant.value] = {
            "config": config,
            "source_revision": provenance["source_revision"],
            "source_dirty": provenance["source_dirty"],
            "training_seconds": matrix["training_seconds"][variant.value],
            "evaluation_seconds": matrix["evaluation_seconds"][variant.value],
            "checkpoint_sha256": file_sha256(run_directory / "checkpoint-final.pt"),
            "metrics": {metric: evaluation["metrics"][metric] for metric in _METRICS},
            "last_generator_metrics": summary["last_metrics"]["generator"],
            "initial_gradient_diagnostics": _initial_gradient_audit(
                run_directory / "checkpoint-final.pt"
            ),
        }

    development_checkpoint = (
        development_root / f"{_FULL.value}-seed-{seed}" / "checkpoint-final.pt"
    )
    replay_checkpoint = root / f"{_FULL.value}-seed-{seed}" / "checkpoint-final.pt"
    return {
        "seed": seed,
        "matrix_total_seconds": matrix["total_seconds"],
        "variants": variants,
        "full_replay": _compare_replay(development_checkpoint, replay_checkpoint),
    }


def _initial_gradient_audit(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    seed_everything(config.seed)
    device = torch.device("cpu")
    data_loader = _build_data_loader(config, device)
    generator, discriminator, regularizer = _build_models(config, device, device)
    assert isinstance(regularizer, KDERelationalCoverageReference)
    _fit_reference_regularizer(
        regularizer,
        data_loader.dataset,
        num_classes=config.num_classes,
        samples_per_class=config.contrastive_reference_samples_per_class,
    )
    real_images, class_labels = next(iter(data_loader))
    noise_generator = torch.Generator().manual_seed(
        _DIAGNOSTIC_NOISE_BASE_SEED + config.seed
    )
    noise = torch.randn(
        real_images.shape[0],
        config.latent_dim,
        generator=noise_generator,
    )
    return asdict(
        measure_relational_gradient_diagnostics(
            generator,
            discriminator,
            regularizer,
            real_images,
            noise,
            class_labels,
            kde_weight=config.regularizer_weight,
            coverage_weight=config.coverage_weight,
        )
    )


def _compare_replay(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    second = torch.load(second_path, map_location="cpu", weights_only=True)
    sections = {
        name: _nested_equal(first[name], second[name])
        for name in (
            "global_step",
            "epoch",
            "generator",
            "discriminator",
            "generator_optimizer",
            "discriminator_optimizer",
            "torch_rng_state",
        )
    }
    return {
        "development_checkpoint_sha256": file_sha256(first_path),
        "replay_checkpoint_sha256": file_sha256(second_path),
        "whole_file_hash_expected_to_differ_due_to_run_metadata": True,
        "state_sections_equal": sections,
        "all_state_sections_equal": all(sections.values()),
    }


def _nested_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return bool(torch.equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _nested_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, list | tuple) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            _nested_equal(left, right)
            for left, right in zip(first, second, strict=True)
        )
    return bool(first == second)


def _single_revision(runs: list[dict[str, Any]]) -> str:
    revisions = {
        record["source_revision"]
        for run in runs
        for record in run["variants"].values()
    }
    if len(revisions) != 1:
        raise RuntimeError("ablation variants were not run from one source revision")
    return revisions.pop()


if __name__ == "__main__":
    main()
