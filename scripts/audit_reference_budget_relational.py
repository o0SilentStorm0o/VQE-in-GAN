#!/usr/bin/env python3
"""Audit one frozen full-trajectory reference-budget development phase."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from vqe_gan.config import ExperimentVariant
from vqe_gan.reproducibility import collect_provenance, file_sha256, write_json
from vqe_gan.training import CoverageBudgetSchedule

SEEDS = (42, 43)
FULL = ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE
CONTROLS = (
    ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
    ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
)
VARIANTS = (FULL, *CONTROLS)
METRICS = (
    "conditional_accuracy",
    "class_conditional_feature_fid_64",
    "diversity_ratio",
    "feature_fid_64",
    "feature_precision",
    "feature_recall",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def run_name(phase: str, variant: ExperimentVariant, seed: int) -> str:
    return f"reference-budget-{phase}-{variant.value}-seed-{seed}"


def main() -> None:
    arguments = parse_arguments()
    repository_root = Path(__file__).resolve().parents[1]
    root = Path(arguments.output_root).resolve()
    matrix_path = root / f"reference-budget-phase-{arguments.phase}.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix["phase"] != arguments.phase or matrix["seeds"] != list(SEEDS):
        raise RuntimeError("phase matrix does not match the frozen audit")
    output = (
        Path(arguments.output)
        if arguments.output
        else root / f"reference-budget-phase-{arguments.phase}-audit.json"
    )
    if output.exists():
        raise FileExistsError(f"phase audit already exists: {output}")

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        full_directory = root / run_name(arguments.phase, FULL, seed)
        schedule_path = full_directory / "coverage-budget.json"
        schedule = CoverageBudgetSchedule.read(schedule_path)
        schedule_hash = file_sha256(schedule_path)
        for variant in VARIANTS:
            directory = root / run_name(arguments.phase, variant, seed)
            runs.append(
                _read_run(
                    directory,
                    seed=seed,
                    phase=arguments.phase,
                    variant=variant,
                    schedule=schedule,
                    schedule_hash=schedule_hash,
                )
            )

    by_key = {(run["seed"], run["variant"]): run for run in runs}
    means = {
        variant.value: {
            metric: statistics.mean(
                by_key[(seed, variant.value)]["metrics"][metric] for seed in SEEDS
            )
            for metric in METRICS
        }
        for variant in VARIANTS
    }
    checks = {control.value: _circuit_checks(by_key, means, control) for control in CONTROLS}
    source_revision_consistent = len({run["source_revision"] for run in runs}) == 1
    technical_passed = source_revision_consistent and all(
        run["technical_checks"]["all_passed"] for run in runs
    )
    circuit_gate_passed = technical_passed and all(
        all(values.values()) for values in checks.values()
    )
    incremental_angle_head = (
        _phase_b_incremental_check(root, means[FULL.value]) if arguments.phase == "b" else None
    )
    result = {
        "schema_version": 1,
        "status": "development_audit",
        "protocol": "docs/reference_budget_relational_protocol.md",
        "protocol_sha256": file_sha256(
            repository_root / "docs/reference_budget_relational_protocol.md"
        ),
        "phase": arguments.phase,
        "matrix": str(matrix_path),
        "matrix_sha256": file_sha256(matrix_path),
        "runs": runs,
        "mean_metrics": means,
        "circuit_checks": checks,
        "source_revision_consistent": source_revision_consistent,
        "technical_checks_passed": technical_passed,
        "circuit_gate_passed": circuit_gate_passed,
        "incremental_angle_head_check": incremental_angle_head,
        "heldout_authorized": False,
        "audit_provenance": collect_provenance(repository_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _read_run(
    directory: Path,
    *,
    seed: int,
    phase: str,
    variant: ExperimentVariant,
    schedule: CoverageBudgetSchedule,
    schedule_hash: str | None,
) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_mode = "record" if variant is FULL else "replay"
    frozen_config = (
        config["seed"] == seed
        and config["variant"] == variant.value
        and config["coverage_budget_mode"] == expected_mode
        and config["freeze_angle_head"] == (phase == "a")
        and config["match_angle_head_budget"] == (phase == "b")
        and config["max_steps"] == 200
        and config["batch_size"] == 64
        and config["regularizer_weight"] == 2e-5
        and config["coverage_weight"] == FULL.default_coverage_weight
        and config["contrastive_reference_samples_per_class"] == 1_024
        and summary["completed_steps"] == 200
        and evaluation["samples"] == 5_000
        and evaluation["seed"] == 91_001
    )
    tracked_source_clean = not provenance["source_tracked_dirty"]
    configured_schedule = (
        Path(config["coverage_budget_schedule"])
        if variant is not FULL
        else directory / "coverage-budget.json"
    )
    schedule_consistent = (
        len(schedule.records) == 200
        and provenance["coverage_budget_schedule_sha256"] == schedule_hash
        and file_sha256(configured_schedule) == schedule_hash
    )
    shared_errors = [
        float(record["generator"]["coverage_shared_ratio_relative_error"]) for record in records
    ]
    adam_errors = [
        float(record["generator"]["coverage_shared_adam_ratio_relative_error"])
        for record in records
    ]
    uncorrected_adam_errors = [
        float(record["generator"]["coverage_shared_adam_uncorrected_relative_error"])
        for record in records
    ]
    adam_proposal_errors = [
        float(record["generator"]["coverage_shared_adam_proposal_relative_error"])
        for record in records
    ]
    adam_multipliers = [
        float(record["generator"]["coverage_shared_adam_update_multiplier"]) for record in records
    ]
    adam_correction_iterations = [
        int(record["generator"]["coverage_shared_adam_correction_iterations"]) for record in records
    ]
    angle_gradient_errors = [
        float(record["generator"]["coverage_angle_gradient_relative_error"])
        for record in records
        if record["generator"]["coverage_angle_gradient_relative_error"] is not None
    ]
    angle_update_errors = [
        float(record["generator"]["coverage_angle_update_relative_error"])
        for record in records
        if record["generator"]["coverage_angle_update_relative_error"] is not None
    ]
    angle_correction_iterations = [
        int(record["generator"]["coverage_angle_update_correction_iterations"])
        for record in records
        if record["generator"]["coverage_angle_update_correction_iterations"] is not None
    ]
    technical = {
        "frozen_config": frozen_config,
        "tracked_source_clean": tracked_source_clean,
        "exactly_200_logged_steps": len(records) == 200,
        "schedule_consistent": schedule_consistent,
        "maximum_shared_ratio_relative_error": max(shared_errors),
        "shared_ratio_within_1e_5": max(shared_errors) <= 1e-5,
        "maximum_shared_adam_ratio_relative_error": max(adam_errors),
        "shared_adam_ratio_within_1e_4": max(adam_errors) <= 1e-4,
        "maximum_adam_proposal_relative_error": max(adam_proposal_errors),
        "adam_proposal_within_1e_5": max(adam_proposal_errors) <= 1e-5,
        "maximum_angle_gradient_relative_error": (
            max(angle_gradient_errors) if angle_gradient_errors else None
        ),
        "angle_gradient_within_1e_5": (
            max(angle_gradient_errors) <= 1e-5 if phase == "b" else not angle_gradient_errors
        ),
        "maximum_angle_update_relative_error": (
            max(angle_update_errors) if angle_update_errors else None
        ),
        "angle_update_within_1e_4": (
            max(angle_update_errors) <= 1e-4 if phase == "b" else not angle_update_errors
        ),
        "uncorrected_adam_error_median": statistics.median(uncorrected_adam_errors),
        "uncorrected_adam_error_maximum": max(uncorrected_adam_errors),
        "adam_update_multiplier_minimum": min(adam_multipliers),
        "adam_update_multiplier_maximum": max(adam_multipliers),
        "adam_correction_iterations_maximum": max(adam_correction_iterations),
        "adam_steps_requiring_refinement": sum(
            iterations > 1 for iterations in adam_correction_iterations
        ),
        "angle_correction_iterations_maximum": (
            max(angle_correction_iterations) if angle_correction_iterations else None
        ),
        "angle_steps_requiring_refinement": (
            sum(iterations > 1 for iterations in angle_correction_iterations)
            if angle_correction_iterations
            else None
        ),
    }
    technical["all_passed"] = all(
        (
            technical["frozen_config"],
            technical["tracked_source_clean"],
            technical["exactly_200_logged_steps"],
            technical["schedule_consistent"],
            technical["shared_ratio_within_1e_5"],
            technical["shared_adam_ratio_within_1e_4"],
            technical["adam_proposal_within_1e_5"],
            technical["angle_gradient_within_1e_5"],
            technical["angle_update_within_1e_4"],
        )
    )
    return {
        "seed": seed,
        "variant": variant.value,
        "run_directory": str(directory),
        "checkpoint_sha256": file_sha256(directory / "checkpoint-final.pt"),
        "evaluation_sha256": file_sha256(directory / "evaluation.json"),
        "schedule_sha256": schedule_hash,
        "source_revision": provenance["source_revision"],
        "source_dirty": provenance["source_dirty"],
        "source_tracked_dirty": provenance["source_tracked_dirty"],
        "source_dirty_reason": provenance["source_dirty_reason"],
        "metrics": {metric: float(evaluation["metrics"][metric]) for metric in METRICS},
        "technical_checks": technical,
    }


def _circuit_checks(
    by_key: dict[tuple[int, str], dict[str, Any]],
    means: dict[str, dict[str, float]],
    control: ExperimentVariant,
) -> dict[str, bool]:
    fid = "class_conditional_feature_fid_64"
    return {
        "full_class_fid_lower_on_each_seed": all(
            by_key[(seed, FULL.value)]["metrics"][fid]
            < by_key[(seed, control.value)]["metrics"][fid]
            for seed in SEEDS
        ),
        "full_mean_class_fid_better_by_at_least_0_005": (
            means[control.value][fid] - means[FULL.value][fid] >= 0.005
        ),
        "full_mean_diversity_not_lower_by_more_than_0_005": (
            means[FULL.value]["diversity_ratio"] >= means[control.value]["diversity_ratio"] - 0.005
        ),
        "full_mean_accuracy_not_lower_by_more_than_0_005": (
            means[FULL.value]["conditional_accuracy"]
            >= means[control.value]["conditional_accuracy"] - 0.005
        ),
    }


def _phase_b_incremental_check(
    root: Path,
    phase_b_full: dict[str, float],
) -> dict[str, Any]:
    phase_a_evaluations = [root / run_name("a", FULL, seed) / "evaluation.json" for seed in SEEDS]
    if not all(path.is_file() for path in phase_a_evaluations):
        return {"available": False, "passed": False}
    phase_a = {
        metric: statistics.mean(
            json.loads(path.read_text(encoding="utf-8"))["metrics"][metric]
            for path in phase_a_evaluations
        )
        for metric in METRICS
    }
    checks = {
        "mean_class_fid_better_by_at_least_0_005": (
            phase_a["class_conditional_feature_fid_64"]
            - phase_b_full["class_conditional_feature_fid_64"]
            >= 0.005
        ),
        "mean_diversity_not_lower_by_more_than_0_005": (
            phase_b_full["diversity_ratio"] >= phase_a["diversity_ratio"] - 0.005
        ),
        "mean_accuracy_not_lower_by_more_than_0_005": (
            phase_b_full["conditional_accuracy"] >= phase_a["conditional_accuracy"] - 0.005
        ),
    }
    return {
        "available": True,
        "phase_a_full_mean_metrics": phase_a,
        "checks": checks,
        "passed": all(checks.values()),
    }


if __name__ == "__main__":
    main()
