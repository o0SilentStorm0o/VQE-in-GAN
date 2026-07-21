"""Checkpoint evaluation using a frozen MNIST classifier and balanced samples."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vqe_gan.evaluation.classifier import load_mnist_classifier
from vqe_gan.evaluation.metrics import compute_distribution_metrics
from vqe_gan.models import SharedQuantumGenerator
from vqe_gan.reproducibility import (
    file_sha256,
    resolve_device,
    seed_everything,
    write_json,
)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    classifier_path: str | Path,
    *,
    dataset_root: str,
    output_path: str | Path,
    samples: int = 1_000,
    batch_size: int = 128,
    seed: int = 91_001,
    device_name: str = "auto",
    download_dataset: bool = True,
) -> dict[str, Any]:
    """Evaluate one generator checkpoint on a balanced real/generated MNIST sample."""

    if samples < 20 or samples % 10 != 0:
        raise ValueError("samples must be at least 20 and divisible by 10")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")

    seed_everything(seed)
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    num_angles = 2 * config["num_qubits"] * (config["ansatz_reps"] + 1)
    generator = SharedQuantumGenerator(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        image_channels=config["image_channels"],
        num_angles=num_angles,
    ).to(device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()
    classifier = load_mnist_classifier(classifier_path, device=device)

    real_images, real_labels = _balanced_real_samples(
        dataset_root,
        samples=samples,
        batch_size=batch_size,
        download=download_dataset,
    )
    real_logits, real_features = _classifier_outputs(
        classifier,
        real_images,
        batch_size=batch_size,
        device=device,
    )
    generated_images, conditioning_labels = _balanced_generated_samples(
        generator,
        samples=samples,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    generated_logits, generated_features = _classifier_outputs(
        classifier,
        generated_images,
        batch_size=batch_size,
        device=device,
    )
    predicted_real_labels = real_logits.argmax(dim=1)
    predicted_generated_labels = generated_logits.argmax(dim=1)
    metrics = compute_distribution_metrics(
        real_features,
        real_labels,
        generated_features,
        conditioning_labels,
        predicted_generated_labels,
    )
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "classifier": str(Path(classifier_path).resolve()),
        "classifier_sha256": file_sha256(classifier_path),
        "device": str(device),
        "samples": samples,
        "seed": seed,
        "real_classifier_accuracy": (
            predicted_real_labels == real_labels
        ).float().mean().item(),
        "metrics": asdict(metrics),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def _balanced_real_samples(
    dataset_root: str,
    *,
    samples: int,
    batch_size: int,
    download: bool,
) -> tuple[Tensor, Tensor]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(
        root=dataset_root,
        train=False,
        download=download,
        transform=transform,
    )
    target_per_class = samples // 10
    counts = torch.zeros(10, dtype=torch.long)
    selected_images: list[Tensor] = []
    selected_labels: list[Tensor] = []
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        for class_index in range(10):
            remaining = target_per_class - counts[class_index].item()
            if remaining <= 0:
                continue
            class_images = images[labels == class_index][:remaining]
            if class_images.numel() == 0:
                continue
            selected_images.append(class_images)
            selected_labels.append(
                torch.full((class_images.shape[0],), class_index, dtype=torch.long)
            )
            counts[class_index] += class_images.shape[0]
        if torch.all(counts == target_per_class):
            break
    if not torch.all(counts == target_per_class):
        raise RuntimeError("MNIST test set did not provide enough balanced samples")
    return torch.cat(selected_images), torch.cat(selected_labels)


def _balanced_generated_samples(
    generator: SharedQuantumGenerator,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    labels = torch.arange(samples, dtype=torch.long) % generator.num_classes
    noise_generator = torch.Generator().manual_seed(seed + 1)
    noise = torch.randn(samples, generator.latent_dim, generator=noise_generator)
    generated_batches = []
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            end = min(samples, start + batch_size)
            generated_batches.append(
                generator(noise[start:end].to(device), labels[start:end].to(device)).cpu()
            )
    return torch.cat(generated_batches), labels


def _classifier_outputs(
    classifier: torch.nn.Module,
    images: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    logits_batches = []
    feature_batches = []
    with torch.no_grad():
        for image_batch in images.split(batch_size):
            logits, features = classifier.forward_with_features(image_batch.to(device))
            logits_batches.append(logits.cpu())
            feature_batches.append(features.cpu())
    return torch.cat(logits_batches), torch.cat(feature_batches)
