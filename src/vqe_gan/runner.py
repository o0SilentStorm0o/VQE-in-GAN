"""End-to-end experiment runner for the corrected MNIST ACGAN."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.data import create_seeded_data_loader
from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator, initialize_weights
from vqe_gan.quantum.spec import (
    HamiltonianFamily,
    IsingHamiltonianSpec,
    QuantumCircuitSpec,
)
from vqe_gan.quantum.torch_backend import DeviceBridgedEnergy, TorchStatevectorEnergy
from vqe_gan.regularizers import ClassicalPrototypeEnergy, PermutedClassEnergy
from vqe_gan.reproducibility import (
    collect_provenance,
    resolve_device,
    seed_everything,
    write_json,
)
from vqe_gan.training import (
    discriminator_step,
    generator_step,
    measure_gradient_diagnostics,
)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train one configured run and return its final summary."""

    repository_root = Path(__file__).resolve().parents[2]
    output_directory = config.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    write_json(output_directory / "config.json", config.to_dict())
    provenance = collect_provenance(repository_root)
    provenance["energy_model"] = config.variant.value

    seed_everything(config.seed)
    device = resolve_device(config.device)
    quantum_device = (
        _resolve_quantum_device(config, device)
        if config.variant
        in {ExperimentVariant.QUANTUM_CONTRASTIVE, ExperimentVariant.QUANTUM_PERMUTED}
        else None
    )
    provenance["training_device"] = str(device)
    provenance["quantum_execution_device"] = (
        str(quantum_device) if quantum_device is not None else None
    )
    data_loader = _build_data_loader(config, device)
    generator, discriminator, energy_model = _build_models(config, device, quantum_device)
    provenance["regularizer_weight"] = config.regularizer_weight
    provenance["regularizer_gradient_ratio"] = config.regularizer_gradient_ratio
    write_json(output_directory / "provenance.json", provenance)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate_generator,
        betas=(config.beta1, config.beta2),
    )
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
            generator_labels = torch.randint(
                config.num_classes,
                (batch_size,),
                device=device,
            )
            next_step = global_step + 1
            gradient_diagnostics = None
            if (
                config.gradient_diagnostics_every_steps > 0
                and next_step % config.gradient_diagnostics_every_steps == 0
            ):
                gradient_diagnostics = measure_gradient_diagnostics(
                    generator,
                    discriminator,
                    energy_model,
                    generator_noise,
                    generator_labels,
                    regularizer_weight=config.regularizer_weight,
                    regularizer_temperature=config.regularizer_temperature,
                    regularizer_gradient_ratio=config.regularizer_gradient_ratio,
                )
            generator_metrics = generator_step(
                generator,
                discriminator,
                energy_model,
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
    TorchStatevectorEnergy | DeviceBridgedEnergy | None,
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

    energy_model = None
    if config.variant is ExperimentVariant.CLASSICAL_PROTOTYPE:
        energy_model = ClassicalPrototypeEnergy(
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
            energy_model = base_backend.to(device=device, dtype=torch.float32)
        else:
            energy_model = DeviceBridgedEnergy(base_backend, quantum_device)
        if config.variant is ExperimentVariant.QUANTUM_PERMUTED:
            energy_model = PermutedClassEnergy(
                energy_model,
                num_classes=config.num_classes,
            )
    return generator, discriminator, energy_model


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
