"""Benchmark complete discriminator-plus-generator steps with and without quantum loss."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass

import torch

from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator, initialize_weights
from vqe_gan.quantum.spec import HamiltonianFamily, IsingHamiltonianSpec
from vqe_gan.quantum.torch_backend import DeviceBridgedEnergy, TorchStatevectorEnergy
from vqe_gan.reproducibility import resolve_device, seed_everything
from vqe_gan.training import discriminator_step, generator_step


@dataclass
class BenchmarkVariant:
    generator: SharedQuantumGenerator
    discriminator: ACGANDiscriminator
    generator_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer
    energy_backend: TorchStatevectorEnergy | DeviceBridgedEnergy | None
    quantum_weight: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--quantum-device", choices=("same", "cpu", "both"), default="same")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_variant(
    device: torch.device,
    quantum_device: torch.device,
    *,
    with_quantum: bool,
) -> BenchmarkVariant:
    generator = SharedQuantumGenerator().to(device)
    discriminator = ACGANDiscriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    energy_backend = None
    if with_quantum:
        base_backend = TorchStatevectorEnergy(
            hamiltonian_spec=IsingHamiltonianSpec(family=HamiltonianFamily.CLASS_ENCODED)
        )
        energy_backend = (
            base_backend.to(device=device, dtype=torch.float32)
            if quantum_device == device
            else DeviceBridgedEnergy(base_backend, quantum_device)
        )
    return BenchmarkVariant(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
        discriminator_optimizer=torch.optim.Adam(
            discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
        ),
        energy_backend=energy_backend,
        quantum_weight=0.1 if with_quantum else 0.0,
    )


def run_step(
    variant: BenchmarkVariant,
    real_images: torch.Tensor,
    labels: torch.Tensor,
    discriminator_noise: torch.Tensor,
    generator_noise: torch.Tensor,
) -> None:
    discriminator_step(
        variant.generator,
        variant.discriminator,
        variant.discriminator_optimizer,
        real_images,
        labels,
        discriminator_noise,
        labels,
    )
    generator_step(
        variant.generator,
        variant.discriminator,
        variant.energy_backend,
        variant.generator_optimizer,
        generator_noise,
        labels,
        quantum_weight=variant.quantum_weight,
    )


def main() -> None:
    arguments = parse_arguments()
    seed_everything(43)
    device = resolve_device(arguments.device)
    quantum_devices = {
        "same": device,
        "cpu": torch.device("cpu"),
    }
    selected_quantum_devices = (
        quantum_devices
        if arguments.quantum_device == "both"
        else {arguments.quantum_device: quantum_devices[arguments.quantum_device]}
    )
    real_images = torch.randn(arguments.batch_size, 1, 28, 28, device=device).clamp(-1, 1)
    labels = torch.arange(arguments.batch_size, device=device) % 10
    discriminator_noise = torch.randn(arguments.batch_size, 100, device=device)
    generator_noise = torch.randn(arguments.batch_size, 100, device=device)
    variants = {"no_regularizer": build_variant(device, device, with_quantum=False)}
    variants.update(
        {
            f"quantum_contrastive_{name}": build_variant(
                device,
                quantum_device,
                with_quantum=True,
            )
            for name, quantum_device in selected_quantum_devices.items()
        }
    )

    for variant in variants.values():
        for _ in range(arguments.warmup):
            run_step(
                variant,
                real_images,
                labels,
                discriminator_noise,
                generator_noise,
            )
        synchronize(device)

    timings: dict[str, list[float]] = {name: [] for name in variants}
    for _ in range(arguments.repeats):
        for name, variant in variants.items():
            synchronize(device)
            start = time.perf_counter()
            run_step(
                variant,
                real_images,
                labels,
                discriminator_noise,
                generator_noise,
            )
            synchronize(device)
            timings[name].append((time.perf_counter() - start) * 1000)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    baseline = medians["no_regularizer"]
    overheads = {
        name: {
            "milliseconds": median - baseline,
            "percent": 100 * (median / baseline - 1),
        }
        for name, median in medians.items()
        if name != "no_regularizer"
    }
    result = {
        "device": str(device),
        "quantum_devices": {
            name: str(quantum_device)
            for name, quantum_device in selected_quantum_devices.items()
        },
        "batch_size": arguments.batch_size,
        "warmup": arguments.warmup,
        "repeats": arguments.repeats,
        "median_step_ms": medians,
        "quantum_overhead": overheads,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
