"""End-to-end experiment runner for the corrected MNIST ACGAN."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from vqe_gan.config import CoverageBudgetMode, ExperimentConfig, ExperimentVariant
from vqe_gan.data import create_seeded_data_loader
from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator, initialize_weights
from vqe_gan.quantum.spec import (
    HamiltonianFamily,
    IsingHamiltonianSpec,
    QuantumCircuitSpec,
)
from vqe_gan.quantum.torch_backend import DeviceBridgedEnergy, TorchStatevectorEnergy
from vqe_gan.regularizers import (
    ClassConditionalLogKDEReference,
    ClassConditionalRBFMMD,
    ClassConditionalRBFReferenceMMD,
    ClassicalPrototypeEnergy,
    HybridQuantumKDEContrastiveReference,
    KDERelationalCoverageReference,
    ModularAblation,
    PermutedClassEnergy,
    QuantumDensityMMD,
    QuantumModularFreeEnergy,
    QuantumModularReference,
    RBFQuantumCoherenceGuidance,
    RelationalKernel,
)
from vqe_gan.reproducibility import (
    collect_provenance,
    file_sha256,
    resolve_device,
    seed_everything,
    write_json,
)
from vqe_gan.training import (
    CoverageBudgetSchedule,
    CoverageBudgetTarget,
    coherence_guided_generator_step,
    discriminator_step,
    distribution_regularized_generator_step,
    generator_step,
    measure_distribution_gradient_diagnostics,
    measure_gradient_diagnostics,
    relational_coverage_generator_step,
)
from vqe_gan.training.budget import coverage_budget_metadata, coverage_budget_phase


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train one configured run and return its final summary."""

    repository_root = Path(__file__).resolve().parents[2]
    output_directory = config.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    write_json(output_directory / "config.json", config.to_dict())
    provenance = collect_provenance(repository_root)
    provenance["regularizer_model"] = config.variant.value
    provenance["generator_label_source"] = "paired_real_batch"

    seed_everything(config.seed)
    device = resolve_device(config.device)
    quantum_device = (
        _resolve_quantum_device(config, device) if config.variant.uses_quantum_backend else None
    )
    provenance["training_device"] = str(device)
    provenance["quantum_execution_device"] = (
        str(quantum_device) if quantum_device is not None else None
    )
    provenance["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    provenance["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
    provenance["backend_reproducibility"] = (
        "exploratory_only"
        if device.type == "mps" and config.variant.uses_distribution_regularizer
        else "confirmation_eligible_after_paired_replay"
    )
    data_loader = _build_data_loader(config, device)
    generator, discriminator, regularizer_model = _build_models(
        config,
        device,
        quantum_device,
    )
    budget_schedule = None
    budget_schedule_path = None
    recorded_budget: list[CoverageBudgetTarget] = []
    if config.coverage_budget_mode is CoverageBudgetMode.REPLAY:
        assert config.coverage_budget_schedule is not None
        budget_schedule_path = Path(config.coverage_budget_schedule).resolve()
        budget_schedule = CoverageBudgetSchedule.read(budget_schedule_path)
        budget_schedule.validate_for(config)
        provenance["coverage_budget_schedule"] = str(budget_schedule_path)
        provenance["coverage_budget_schedule_sha256"] = file_sha256(budget_schedule_path)
    elif config.coverage_budget_mode is CoverageBudgetMode.RECORD:
        budget_schedule_path = output_directory / "coverage-budget.json"
    if config.coverage_budget_mode is not CoverageBudgetMode.FIXED:
        provenance["coverage_budget_mode"] = config.coverage_budget_mode.value
        provenance["coverage_budget_phase"] = coverage_budget_phase(config)
    if isinstance(
        regularizer_model,
        QuantumModularReference
        | ClassConditionalRBFReferenceMMD
        | ClassConditionalLogKDEReference
        | HybridQuantumKDEContrastiveReference
        | KDERelationalCoverageReference,
    ):
        samples_per_class = (
            config.contrastive_reference_samples_per_class
            if config.variant.uses_contrastive_reference
            else config.reference_samples_per_class
        )
        reference_count = _fit_reference_regularizer(
            regularizer_model,
            data_loader.dataset,
            num_classes=config.num_classes,
            samples_per_class=samples_per_class,
        )
        provenance["reference_selection"] = "first_balanced_training_examples"
        provenance["reference_samples_per_class"] = reference_count
    provenance["regularizer_weight"] = config.regularizer_weight
    provenance["regularizer_gradient_ratio"] = config.regularizer_gradient_ratio
    provenance["coverage_weight"] = config.coverage_weight
    if config.variant.uses_relational_coverage:
        provenance["regularizer_weight_calibration"] = (
            "unchanged_kde_plus_fixed_coverage_from_development_seeds_42_43"
        )
        provenance["coverage_calibration_target_gradient_ratio"] = 0.01
    elif config.variant.uses_contrastive_reference:
        provenance["regularizer_weight_calibration"] = (
            "fixed_from_initial_gradient_norms_on_development_seeds_42_43"
        )
        provenance["calibration_target_gradient_ratio"] = 0.1
    write_json(output_directory / "provenance.json", provenance)
    generator_optimizer = _build_generator_optimizer(config, generator)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate_discriminator,
        betas=(config.beta1, config.beta2),
    )

    log_path = output_directory / "metrics.jsonl"
    global_step = 0
    last_record: dict[str, Any] = {}
    for epoch in range(config.epochs):
        generator.train()
        discriminator.train()
        for real_images, real_labels in data_loader:
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.shape[0]

            discriminator_noise = torch.randn(batch_size, config.latent_dim, device=device)
            discriminator_labels = torch.randint(
                config.num_classes,
                (batch_size,),
                device=device,
            )
            discriminator_metrics = discriminator_step(
                generator,
                discriminator,
                discriminator_optimizer,
                real_images,
                real_labels,
                discriminator_noise,
                discriminator_labels,
                real_label_smoothing=config.real_label_smoothing,
            )

            generator_noise = torch.randn(batch_size, config.latent_dim, device=device)
            # Reuse the real-batch labels for every variant. Besides giving the
            # distribution losses equal real/generated class counts, this keeps
            # the generator's labels and RNG stream exactly paired across an
            # ablation matrix.
            generator_labels = real_labels
            next_step = global_step + 1
            gradient_diagnostics = None
            if (
                config.gradient_diagnostics_every_steps > 0
                and next_step % config.gradient_diagnostics_every_steps == 0
            ):
                if config.variant.uses_coherence_guidance:
                    # The projected classical/quantum gradient interaction is
                    # logged directly by coherence_guided_generator_step.
                    pass
                elif config.variant.uses_relational_coverage:
                    # Component-specific diagnostics are produced by the
                    # relational calibration and post-run audit tools.
                    pass
                elif config.variant.uses_distribution_regularizer:
                    assert regularizer_model is not None
                    gradient_diagnostics = measure_distribution_gradient_diagnostics(
                        generator,
                        discriminator,
                        regularizer_model,
                        real_images,
                        generator_noise,
                        generator_labels,
                        regularizer_weight=config.regularizer_weight,
                        regularizer_gradient_ratio=config.regularizer_gradient_ratio,
                    )
                else:
                    gradient_diagnostics = measure_gradient_diagnostics(
                        generator,
                        discriminator,
                        regularizer_model,
                        generator_noise,
                        generator_labels,
                        regularizer_weight=config.regularizer_weight,
                        regularizer_temperature=config.regularizer_temperature,
                        regularizer_gradient_ratio=config.regularizer_gradient_ratio,
                    )
            if config.variant.uses_coherence_guidance:
                assert regularizer_model is not None
                generator_metrics = coherence_guided_generator_step(
                    generator,
                    discriminator,
                    regularizer_model,
                    generator_optimizer,
                    real_images,
                    generator_noise,
                    generator_labels,
                    regularizer_weight=config.regularizer_weight,
                    regularizer_gradient_ratio=config.regularizer_gradient_ratio,
                    coherence_gradient_ratio=config.coherence_gradient_ratio,
                )
            elif config.variant.uses_relational_coverage:
                assert regularizer_model is not None
                budget_target = None
                if budget_schedule is not None:
                    budget_target = budget_schedule.records[global_step]
                    if budget_target.step != next_step:
                        raise RuntimeError("coverage budget schedule was consumed out of order")
                generator_metrics = relational_coverage_generator_step(
                    generator,
                    discriminator,
                    regularizer_model,
                    generator_optimizer,
                    real_images,
                    generator_noise,
                    generator_labels,
                    kde_weight=config.regularizer_weight,
                    coverage_weight=config.coverage_weight,
                    budget_target=budget_target,
                    record_budget=(config.coverage_budget_mode is CoverageBudgetMode.RECORD),
                    match_angle_head_budget=config.match_angle_head_budget,
                )
                if config.coverage_budget_mode is not CoverageBudgetMode.FIXED:
                    _validate_reference_budget_metrics(generator_metrics, config)
                if config.coverage_budget_mode is CoverageBudgetMode.RECORD:
                    recorded_budget.append(
                        _budget_target_from_metrics(
                            next_step,
                            generator_metrics,
                            include_angle=config.match_angle_head_budget,
                        )
                    )
            elif config.variant.uses_distribution_regularizer:
                assert regularizer_model is not None
                generator_metrics = distribution_regularized_generator_step(
                    generator,
                    discriminator,
                    regularizer_model,
                    generator_optimizer,
                    real_images,
                    generator_noise,
                    generator_labels,
                    regularizer_weight=config.regularizer_weight,
                    regularizer_gradient_ratio=config.regularizer_gradient_ratio,
                )
            else:
                generator_metrics = generator_step(
                    generator,
                    discriminator,
                    regularizer_model,
                    generator_optimizer,
                    generator_noise,
                    generator_labels,
                    regularizer_weight=config.regularizer_weight,
                    regularizer_temperature=config.regularizer_temperature,
                    regularizer_gradient_ratio=config.regularizer_gradient_ratio,
                )

            global_step += 1
            last_record = {
                "epoch": epoch + 1,
                "step": global_step,
                "batch_size": batch_size,
                "generator": asdict(generator_metrics),
                "discriminator": asdict(discriminator_metrics),
            }
            if gradient_diagnostics is not None:
                last_record["gradient_diagnostics"] = asdict(gradient_diagnostics)
            if global_step % config.log_every_steps == 0:
                _append_json_line(log_path, last_record)
            if (
                config.checkpoint_every_steps > 0
                and global_step % config.checkpoint_every_steps == 0
            ):
                _save_checkpoint(
                    output_directory / f"checkpoint-step-{global_step:06d}.pt",
                    config,
                    global_step,
                    epoch,
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                )
            if config.max_steps is not None and global_step >= config.max_steps:
                break
        if config.max_steps is not None and global_step >= config.max_steps:
            break

    if budget_schedule is not None and global_step != len(budget_schedule.records):
        raise RuntimeError("coverage budget schedule was not consumed exactly once")
    if config.coverage_budget_mode is CoverageBudgetMode.RECORD:
        assert budget_schedule_path is not None
        schedule = CoverageBudgetSchedule(
            metadata=coverage_budget_metadata(config),
            records=tuple(recorded_budget),
        )
        if len(schedule.records) != config.max_steps:
            raise RuntimeError("reference run did not record the frozen budget horizon")
        schedule.write(budget_schedule_path)
        provenance["coverage_budget_schedule"] = str(budget_schedule_path)
        provenance["coverage_budget_schedule_sha256"] = file_sha256(budget_schedule_path)
        write_json(output_directory / "provenance.json", provenance)

    _save_checkpoint(
        output_directory / "checkpoint-final.pt",
        config,
        global_step,
        max(0, last_record.get("epoch", 1) - 1),
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
    )
    summary = {
        "run_name": config.run_name,
        "variant": config.variant.value,
        "device": str(device),
        "quantum_device": str(quantum_device) if quantum_device is not None else None,
        "completed_steps": global_step,
        "last_metrics": last_record,
        "coverage_budget_schedule": (
            str(budget_schedule_path) if budget_schedule_path is not None else None
        ),
        "coverage_budget_schedule_sha256": (
            file_sha256(budget_schedule_path) if budget_schedule_path is not None else None
        ),
    }
    write_json(output_directory / "summary.json", summary)
    return summary


def _build_data_loader(config: ExperimentConfig, device: torch.device) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=config.dataset_root,
        train=True,
        download=config.download_dataset,
        transform=transform,
    )
    if config.dataset_limit is not None:
        dataset = Subset(dataset, range(min(config.dataset_limit, len(dataset))))
    return create_seeded_data_loader(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )


def _build_models(
    config: ExperimentConfig,
    device: torch.device,
    quantum_device: torch.device | None,
) -> tuple[
    SharedQuantumGenerator,
    ACGANDiscriminator,
    torch.nn.Module | None,
]:
    circuit_spec = QuantumCircuitSpec(
        num_qubits=config.num_qubits,
        reps=config.ansatz_reps,
    )
    generator = SharedQuantumGenerator(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        image_channels=config.image_channels,
        num_angles=circuit_spec.num_parameters,
    ).to(device)
    discriminator = ACGANDiscriminator(
        num_classes=config.num_classes,
        image_channels=config.image_channels,
    ).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    regularizer_model = None
    if config.variant is ExperimentVariant.CLASSICAL_PROTOTYPE:
        regularizer_model = ClassicalPrototypeEnergy(
            num_classes=config.num_classes,
            num_angles=circuit_spec.num_parameters,
        ).to(device=device, dtype=torch.float32)
    elif config.variant in {
        ExperimentVariant.QUANTUM_CONTRASTIVE,
        ExperimentVariant.QUANTUM_PERMUTED,
    }:
        assert quantum_device is not None
        hamiltonian_spec = IsingHamiltonianSpec(
            coupling=config.ising_coupling,
            global_field=config.ising_field,
            num_classes=config.num_classes,
            family=HamiltonianFamily.CLASS_ENCODED,
        )
        base_backend = TorchStatevectorEnergy(circuit_spec, hamiltonian_spec)
        if quantum_device == device:
            regularizer_model = base_backend.to(device=device, dtype=torch.float32)
        else:
            regularizer_model = DeviceBridgedEnergy(base_backend, quantum_device)
        if config.variant is ExperimentVariant.QUANTUM_PERMUTED:
            regularizer_model = PermutedClassEnergy(
                regularizer_model,
                num_classes=config.num_classes,
            )
    elif config.variant in {
        ExperimentVariant.QUANTUM_MODULAR_FREE_ENERGY,
        ExperimentVariant.QUANTUM_MODULAR_DEPHASED,
        ExperimentVariant.QUANTUM_MODULAR_PRODUCT,
        ExperimentVariant.QUANTUM_MODULAR_ENERGY_ONLY,
    }:
        assert quantum_device is not None
        ablations = {
            ExperimentVariant.QUANTUM_MODULAR_FREE_ENERGY: ModularAblation.FULL,
            ExperimentVariant.QUANTUM_MODULAR_DEPHASED: ModularAblation.DEPHASED,
            ExperimentVariant.QUANTUM_MODULAR_PRODUCT: ModularAblation.PRODUCT,
            ExperimentVariant.QUANTUM_MODULAR_ENERGY_ONLY: ModularAblation.ENERGY_ONLY,
        }
        regularizer_model = QuantumModularFreeEnergy(
            circuit_spec,
            execution_device=quantum_device,
            angle_scale=config.modular_angle_scale,
            depolarization=config.modular_depolarization,
            ablation=ablations[config.variant],
        )
    elif config.variant is ExperimentVariant.QUANTUM_DENSITY_MMD:
        assert quantum_device is not None
        regularizer_model = QuantumDensityMMD(
            circuit_spec,
            execution_device=quantum_device,
            angle_scale=config.density_mmd_angle_scale,
        )
    elif config.variant is ExperimentVariant.CLASSICAL_RBF_MMD:
        execution_device = torch.device("cpu") if device.type == "mps" else device
        regularizer_model = ClassConditionalRBFMMD(
            execution_device=execution_device,
            sigma_squared=config.rbf_sigma_squared,
        )
    elif config.variant is ExperimentVariant.QUANTUM_COHERENCE_GUIDANCE:
        assert quantum_device is not None
        regularizer_model = RBFQuantumCoherenceGuidance(
            circuit_spec,
            execution_device=quantum_device,
            angle_scale=config.modular_angle_scale,
            depolarization=config.modular_depolarization,
            sigma_squared=config.rbf_sigma_squared,
        )
    elif config.variant is ExperimentVariant.QUANTUM_MODULAR_REFERENCE:
        assert quantum_device is not None
        regularizer_model = QuantumModularReference(
            circuit_spec,
            num_classes=config.num_classes,
            execution_device=quantum_device,
            angle_scale=config.modular_angle_scale,
            depolarization=config.modular_depolarization,
        )
    elif config.variant is ExperimentVariant.CLASSICAL_RBF_REFERENCE:
        execution_device = torch.device("cpu") if device.type == "mps" else device
        regularizer_model = ClassConditionalRBFReferenceMMD(
            num_classes=config.num_classes,
            execution_device=execution_device,
            sigma_squared=config.rbf_sigma_squared,
        )
    elif config.variant in {
        ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE,
        ExperimentVariant.CLASSICAL_LOG_KDE_SCALE_CONTROL,
    }:
        execution_device = torch.device("cpu") if device.type == "mps" else device
        regularizer_model = ClassConditionalLogKDEReference(
            num_classes=config.num_classes,
            execution_device=execution_device,
            sigma_squared=config.kde_sigma_squared,
            temperature=config.kde_temperature,
        )
    elif config.variant in {
        ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE,
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
    }:
        if config.variant is not ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE:
            assert quantum_device is not None
            execution_device = quantum_device
            kernels = {
                ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE: (RelationalKernel.QUANTUM_FULL),
                ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT: (
                    RelationalKernel.QUANTUM_PRODUCT
                ),
                ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED: (
                    RelationalKernel.QUANTUM_DEPHASED
                ),
            }
            kernel = kernels[config.variant]
        else:
            execution_device = torch.device("cpu") if device.type == "mps" else device
            kernel = RelationalKernel.CLASSICAL_PERIODIC_RBF
        _zero_angle_residual_output(generator)
        if config.freeze_angle_head:
            for parameter in generator.angle_head.parameters():
                parameter.requires_grad_(False)
        regularizer_model = KDERelationalCoverageReference(
            circuit_spec,
            num_classes=config.num_classes,
            execution_device=execution_device,
            angle_scale=config.contrastive_angle_scale,
            angle_residual_fraction=config.angle_residual_fraction,
            coverage_temperature=config.coverage_temperature,
            kde_sigma_squared=config.kde_sigma_squared,
            kde_temperature=config.kde_temperature,
            kernel=kernel,
        )
    elif config.variant in {
        ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE,
        ExperimentVariant.HYBRID_MODULAR_KDE_PRODUCT,
        ExperimentVariant.HYBRID_MODULAR_KDE_DEPHASED,
    }:
        assert quantum_device is not None
        ablations = {
            ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE: ModularAblation.FULL,
            ExperimentVariant.HYBRID_MODULAR_KDE_PRODUCT: ModularAblation.PRODUCT,
            ExperimentVariant.HYBRID_MODULAR_KDE_DEPHASED: ModularAblation.DEPHASED,
        }
        regularizer_model = HybridQuantumKDEContrastiveReference(
            circuit_spec,
            num_classes=config.num_classes,
            execution_device=quantum_device,
            angle_scale=config.contrastive_angle_scale,
            depolarization=config.modular_depolarization,
            quantum_temperature=config.contrastive_quantum_temperature,
            kde_sigma_squared=config.kde_sigma_squared,
            kde_temperature=config.kde_temperature,
            quantum_mixture_weight=config.quantum_mixture_weight,
            ablation=ablations[config.variant],
        )
    return generator, discriminator, regularizer_model


def _fit_reference_regularizer(
    regularizer: (
        QuantumModularReference
        | ClassConditionalRBFReferenceMMD
        | ClassConditionalLogKDEReference
        | HybridQuantumKDEContrastiveReference
        | KDERelationalCoverageReference
    ),
    dataset: Dataset,
    *,
    num_classes: int,
    samples_per_class: int,
) -> int:
    images, labels, actual_count = _balanced_reference_batch(
        dataset,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
    )
    regularizer.fit_reference(images, labels)
    return actual_count


def _zero_angle_residual_output(generator: SharedQuantumGenerator) -> None:
    """Start additive angle residuals at the deterministic image encoding."""

    output_layer = generator.angle_head[-2]
    if not isinstance(output_layer, torch.nn.Linear) or not getattr(
        output_layer,
        "is_angle_output",
        False,
    ):
        raise RuntimeError("generator angle output layer is not identifiable")
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.zero_()


def _balanced_reference_batch(
    dataset: Dataset,
    *,
    num_classes: int,
    samples_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    buckets: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = torch.zeros(num_classes, dtype=torch.long)
    reference_generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        generator=reference_generator,
    )
    for images, labels in loader:
        for class_index in range(num_classes):
            remaining = samples_per_class - counts[class_index].item()
            if remaining <= 0:
                continue
            selected = images[labels == class_index][:remaining]
            if selected.numel() == 0:
                continue
            buckets[class_index].append(selected)
            counts[class_index] += selected.shape[0]
        if torch.all(counts >= samples_per_class):
            break
    actual_count = int(counts.min().item())
    if actual_count <= 0:
        raise RuntimeError("dataset does not contain every class required by the reference bank")
    class_images = [torch.cat(bucket)[:actual_count] for bucket in buckets]
    images = torch.cat(class_images)
    labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(actual_count)
    return images, labels, actual_count


def _resolve_quantum_device(
    config: ExperimentConfig,
    training_device: torch.device,
) -> torch.device:
    if config.quantum_device == "cpu":
        return torch.device("cpu")
    if config.quantum_device == "same":
        return training_device
    # Tiny complex statevectors are faster on CPU than MPS because MPS dispatch dominates.
    return torch.device("cpu") if training_device.type == "mps" else training_device


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_generator_optimizer(
    config: ExperimentConfig,
    generator: SharedQuantumGenerator,
) -> torch.optim.Adam:
    settings = {
        "lr": config.learning_rate_generator,
        "betas": (config.beta1, config.beta2),
    }
    if config.coverage_budget_mode is CoverageBudgetMode.FIXED:
        return torch.optim.Adam(generator.parameters(), **settings)
    shared_parameters = (
        *generator.label_embedding.parameters(),
        *generator.input_projection.parameters(),
        *generator.image_decoder.parameters(),
    )
    groups: list[dict[str, Any]] = [{"params": shared_parameters, "name": "shared_image"}]
    if config.match_angle_head_budget:
        groups.append(
            {
                "params": tuple(generator.angle_head.parameters()),
                "name": "angle_head",
            }
        )
    return torch.optim.Adam(groups, **settings)


def _validate_reference_budget_metrics(
    metrics: Any,
    config: ExperimentConfig,
) -> None:
    shared_error = metrics.coverage_shared_ratio_relative_error
    if shared_error is None or not math.isfinite(shared_error):
        raise RuntimeError("shared coverage budget did not produce a finite error")
    if shared_error > 1e-5:
        raise RuntimeError("shared coverage budget exceeded its frozen tolerance")
    adam_error = metrics.coverage_shared_adam_ratio_relative_error
    if adam_error is None or not math.isfinite(adam_error) or adam_error > 1e-4:
        raise RuntimeError("shared Adam budget exceeded its frozen tolerance")
    adam_quantization_corrections = metrics.coverage_shared_adam_quantization_corrections
    adam_quantization_relative_norm = metrics.coverage_shared_adam_quantization_relative_norm
    if (
        adam_quantization_corrections is None
        or not 0 <= adam_quantization_corrections <= 128
        or adam_quantization_relative_norm is None
        or not math.isfinite(adam_quantization_relative_norm)
        or not 0 <= adam_quantization_relative_norm <= 0.02
    ):
        raise RuntimeError("shared Adam quantization repair exceeded its frozen limits")
    if config.coverage_budget_mode is CoverageBudgetMode.RECORD and (
        adam_quantization_corrections != 0 or adam_quantization_relative_norm != 0
    ):
        raise RuntimeError("full reference Adam update unexpectedly used quantization repair")
    proposal_error = metrics.coverage_shared_adam_proposal_relative_error
    if proposal_error is None or not math.isfinite(proposal_error) or proposal_error > 1e-5:
        raise RuntimeError("analytical and realized Adam updates disagree")
    if config.match_angle_head_budget:
        angle_gradient_error = metrics.coverage_angle_gradient_relative_error
        angle_update_error = metrics.coverage_angle_update_relative_error
        if (
            angle_gradient_error is None
            or not math.isfinite(angle_gradient_error)
            or angle_gradient_error > 1e-5
        ):
            raise RuntimeError("angle gradient budget exceeded its frozen tolerance")
        if (
            angle_update_error is None
            or not math.isfinite(angle_update_error)
            or angle_update_error > 1e-4
        ):
            raise RuntimeError("angle Adam update budget exceeded its frozen tolerance")
        angle_quantization_corrections = metrics.coverage_angle_update_quantization_corrections
        angle_quantization_relative_norm = metrics.coverage_angle_update_quantization_relative_norm
        if (
            angle_quantization_corrections is None
            or not 0 <= angle_quantization_corrections <= 128
            or angle_quantization_relative_norm is None
            or not math.isfinite(angle_quantization_relative_norm)
            or not 0 <= angle_quantization_relative_norm <= 0.02
        ):
            raise RuntimeError("angle Adam quantization repair exceeded its frozen limits")
        if config.coverage_budget_mode is CoverageBudgetMode.RECORD and (
            angle_quantization_corrections != 0 or angle_quantization_relative_norm != 0
        ):
            raise RuntimeError("full reference angle update unexpectedly used quantization repair")


def _budget_target_from_metrics(
    step: int,
    metrics: Any,
    *,
    include_angle: bool,
) -> CoverageBudgetTarget:
    required = {
        "shared_ratio": metrics.coverage_shared_target_ratio,
        "shared_anchor_gradient_norm": metrics.coverage_shared_anchor_gradient_norm,
        "shared_coverage_gradient_norm": metrics.coverage_shared_gradient_norm,
        "shared_weighted_coverage_gradient_norm": (metrics.coverage_shared_weighted_gradient_norm),
        "shared_adam_update_norm": metrics.coverage_shared_adam_update_norm,
        "shared_adam_auxiliary_ratio": metrics.coverage_shared_adam_auxiliary_ratio,
    }
    if any(value is None for value in required.values()):
        raise RuntimeError("reference step did not expose every shared budget metric")
    return CoverageBudgetTarget(
        step=step,
        **required,
        angle_gradient_norm=(
            metrics.coverage_angle_gradient_target_norm if include_angle else None
        ),
        angle_update_norm=(metrics.coverage_angle_update_target_norm if include_angle else None),
    )


def _save_checkpoint(
    path: Path,
    config: ExperimentConfig,
    global_step: int,
    epoch: int,
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
) -> None:
    payload = {
        "config": config.to_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
