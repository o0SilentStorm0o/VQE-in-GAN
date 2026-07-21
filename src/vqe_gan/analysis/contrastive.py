"""Forensic diagnostics for the failed contrastive-reference experiment."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torchvision import datasets, transforms

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.evaluation.classifier import load_mnist_classifier
from vqe_gan.evaluation.metrics import frechet_distance
from vqe_gan.evaluation.run import (
    _balanced_generated_samples,
    _balanced_real_samples,
    _classifier_outputs,
)
from vqe_gan.models import ACGANDiscriminator, SharedQuantumGenerator
from vqe_gan.quantum.spec import QuantumCircuitSpec
from vqe_gan.regularizers import HybridQuantumKDEContrastiveReference
from vqe_gan.reproducibility import file_sha256, seed_everything
from vqe_gan.runner import (
    _build_data_loader,
    _build_models,
    _fit_reference_regularizer,
)
from vqe_gan.training.steps import discriminator_step

GradientTuple = tuple[Tensor, ...]

_CLASSICAL_VARIANT = ExperimentVariant.CLASSICAL_LOG_KDE_CONTRASTIVE.value
_HYBRID_VARIANT = ExperimentVariant.HYBRID_MODULAR_KDE_CONTRASTIVE.value
_MODULE_PREFIXES = (
    "label_embedding",
    "input_projection",
    "image_decoder",
    "angle_head",
)
_PAIRWISE_CONFIG_FIELDS = (
    "seed",
    "batch_size",
    "latent_dim",
    "num_classes",
    "image_channels",
    "learning_rate_generator",
    "learning_rate_discriminator",
    "beta1",
    "beta2",
    "real_label_smoothing",
    "num_qubits",
    "ansatz_reps",
    "regularizer_weight",
    "regularizer_gradient_ratio",
    "modular_depolarization",
    "contrastive_reference_samples_per_class",
    "contrastive_angle_scale",
    "contrastive_quantum_temperature",
    "kde_sigma_squared",
    "kde_temperature",
    "quantum_mixture_weight",
)


def diagnose_contrastive_failure(
    classical_checkpoints: Sequence[str | Path],
    hybrid_checkpoints: Sequence[str | Path],
    classifier_path: str | Path,
    *,
    dataset_root: str,
    evaluation_samples: int = 5_000,
    score_samples: int = 1_000,
    gradient_samples: int = 100,
    evaluation_seed: int = 91_001,
    gradient_seed: int = 123_456,
) -> dict[str, Any]:
    """Diagnose paired classical/hybrid checkpoints without changing either checkpoint.

    The returned dictionary contains architecture reachability, per-class outcomes, teacher-score
    behavior, exact gradient decomposition, the first Adam update, and logged trajectory
    divergence. All execution is deterministic CPU analysis.
    """

    if len(classical_checkpoints) != len(hybrid_checkpoints) or not classical_checkpoints:
        raise ValueError("classical and hybrid checkpoint lists must have equal positive length")
    for name, value in (
        ("evaluation_samples", evaluation_samples),
        ("score_samples", score_samples),
        ("gradient_samples", gradient_samples),
    ):
        if value < 20 or value % 10 != 0:
            raise ValueError(f"{name} must be at least 20 and divisible by 10")
    if score_samples > evaluation_samples:
        raise ValueError("score_samples cannot exceed evaluation_samples")

    device = torch.device("cpu")
    pairs = [
        _load_and_validate_pair(classical_path, hybrid_path, device)
        for classical_path, hybrid_path in zip(
            classical_checkpoints,
            hybrid_checkpoints,
            strict=True,
        )
    ]
    _validate_cross_pair_configuration(pairs)
    first_config = pairs[0]["hybrid_checkpoint"]["config"]
    regularizer, reference_count = _build_reference_regularizer(
        first_config,
        dataset_root=dataset_root,
        device=device,
    )
    classifier = load_mnist_classifier(classifier_path, device=device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    real_images, real_labels = _balanced_real_samples(
        dataset_root,
        samples=evaluation_samples,
        batch_size=128,
        download=False,
    )
    _, real_features = _classifier_outputs(
        classifier,
        real_images,
        batch_size=128,
        device=device,
    )
    real_centroids = torch.stack(
        [real_features[real_labels == class_index].mean(dim=0) for class_index in range(10)]
    )

    pair_reports = []
    for pair in pairs:
        pair_reports.append(
            _diagnose_pair(
                pair,
                regularizer,
                classifier,
                real_features,
                real_labels,
                real_centroids,
                dataset_root=dataset_root,
                evaluation_samples=evaluation_samples,
                score_samples=score_samples,
                gradient_samples=gradient_samples,
                evaluation_seed=evaluation_seed,
                gradient_seed=gradient_seed,
                device=device,
            )
        )

    return {
        "schema_version": 1,
        "analysis": "post_hoc_contrastive_failure_diagnostic",
        "interpretation": "mechanistic_only_not_confirmatory",
        "device": str(device),
        "evaluation_samples": evaluation_samples,
        "score_samples": score_samples,
        "gradient_samples": gradient_samples,
        "evaluation_seed": evaluation_seed,
        "gradient_seed": gradient_seed,
        "classifier": str(Path(classifier_path).resolve()),
        "classifier_sha256": file_sha256(classifier_path),
        "reference_samples_per_class": reference_count,
        "regularizer_trainable_parameters": sum(
            parameter.numel() for parameter in regularizer.parameters()
        ),
        "pairs": pair_reports,
        "cross_seed_summary": _cross_seed_summary(pair_reports),
    }


def decompose_hybrid_gradients(
    kde_loss: Tensor,
    scaled_kde_loss: Tensor,
    hybrid_loss: Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> dict[str, GradientTuple]:
    """Separate hybrid-minus-KDE gradients into scale and quantum-addition effects."""

    kde_gradient = _gradients(kde_loss, parameters, retain_graph=True)
    scaled_gradient = _gradients(scaled_kde_loss, parameters, retain_graph=True)
    hybrid_gradient = _gradients(hybrid_loss, parameters, retain_graph=True)
    total_delta = _subtract_gradients(hybrid_gradient, kde_gradient)
    scale_effect = _subtract_gradients(scaled_gradient, kde_gradient)
    quantum_addition = _subtract_gradients(hybrid_gradient, scaled_gradient)
    identity_residual = _subtract_gradients(
        total_delta,
        _add_gradients(scale_effect, quantum_addition),
    )
    return {
        "kde": kde_gradient,
        "scaled_kde": scaled_gradient,
        "hybrid": hybrid_gradient,
        "total_delta": total_delta,
        "scale_effect": scale_effect,
        "quantum_addition": quantum_addition,
        "identity_residual": identity_residual,
    }


def classwise_feature_diagnostics(
    real_features: Tensor,
    real_labels: Tensor,
    generated_features: Tensor,
    conditioning_labels: Tensor,
    classifier_logits: Tensor,
) -> list[dict[str, Any]]:
    """Return class-level semantic, FID-component, and diversity diagnostics."""

    classes = torch.unique(real_labels, sorted=True)
    if not torch.equal(classes, torch.unique(conditioning_labels, sorted=True)):
        raise ValueError("real and generated labels must contain the same classes")
    if classifier_logits.shape[0] != generated_features.shape[0]:
        raise ValueError("classifier logits must match generated samples")
    predictions = classifier_logits.argmax(dim=1)
    probabilities = classifier_logits.softmax(dim=1)
    log_probabilities = classifier_logits.log_softmax(dim=1)
    rows: list[dict[str, Any]] = []
    for class_tensor in classes:
        class_index = int(class_tensor.item())
        real_mask = real_labels == class_index
        generated_mask = conditioning_labels == class_index
        real_class = real_features[real_mask]
        generated_class = generated_features[generated_mask]
        class_fid = frechet_distance(real_class, generated_class)
        centroid_component = (
            real_class.mean(dim=0) - generated_class.mean(dim=0)
        ).square().sum().item()
        prediction_counts = torch.bincount(
            predictions[generated_mask],
            minlength=classifier_logits.shape[1],
        )
        prediction_counts[class_index] = -1
        top_wrong_class = int(prediction_counts.argmax().item())
        top_wrong_fraction = (
            prediction_counts[top_wrong_class].float() / generated_mask.sum()
        ).item()
        rows.append(
            {
                "class": class_index,
                "conditional_accuracy": (
                    predictions[generated_mask] == class_index
                ).float().mean().item(),
                "target_probability": probabilities[generated_mask, class_index].mean().item(),
                "target_nll": -log_probabilities[generated_mask, class_index].mean().item(),
                "class_fid": class_fid,
                "fid_centroid_component": centroid_component,
                "fid_covariance_component": class_fid - centroid_component,
                "intra_class_diversity": torch.pdist(generated_class).mean().item(),
                "top_wrong_class": top_wrong_class,
                "top_wrong_fraction": top_wrong_fraction,
            }
        )
    return rows


def _load_and_validate_pair(
    classical_path: str | Path,
    hybrid_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    classical_path = Path(classical_path)
    hybrid_path = Path(hybrid_path)
    classical = torch.load(classical_path, map_location=device, weights_only=True)
    hybrid = torch.load(hybrid_path, map_location=device, weights_only=True)
    classical_config = classical["config"]
    hybrid_config = hybrid["config"]
    if classical_config["variant"] != _CLASSICAL_VARIANT:
        raise ValueError(f"unexpected classical variant: {classical_config['variant']}")
    if hybrid_config["variant"] != _HYBRID_VARIANT:
        raise ValueError(f"unexpected hybrid variant: {hybrid_config['variant']}")
    mismatches = [
        field
        for field in _PAIRWISE_CONFIG_FIELDS
        if classical_config[field] != hybrid_config[field]
    ]
    if mismatches:
        raise ValueError(f"paired checkpoint configurations differ in: {mismatches}")
    if classical["global_step"] != hybrid["global_step"]:
        raise ValueError("paired checkpoints must have the same training horizon")
    return {
        "seed": classical_config["seed"],
        "classical_path": classical_path,
        "hybrid_path": hybrid_path,
        "classical_checkpoint": classical,
        "hybrid_checkpoint": hybrid,
    }


def _validate_cross_pair_configuration(pairs: Sequence[dict[str, Any]]) -> None:
    first = pairs[0]["hybrid_checkpoint"]["config"]
    frozen_fields = tuple(field for field in _PAIRWISE_CONFIG_FIELDS if field != "seed")
    for pair in pairs[1:]:
        config = pair["hybrid_checkpoint"]["config"]
        mismatches = [field for field in frozen_fields if first[field] != config[field]]
        if mismatches:
            raise ValueError(f"cross-seed frozen configuration differs in: {mismatches}")


def _build_reference_regularizer(
    config: dict[str, Any],
    *,
    dataset_root: str,
    device: torch.device,
) -> tuple[HybridQuantumKDEContrastiveReference, int]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(
        root=dataset_root,
        train=True,
        download=False,
        transform=transform,
    )
    regularizer = HybridQuantumKDEContrastiveReference(
        QuantumCircuitSpec(
            num_qubits=config["num_qubits"],
            reps=config["ansatz_reps"],
        ),
        num_classes=config["num_classes"],
        execution_device=device,
        angle_scale=config["contrastive_angle_scale"],
        depolarization=config["modular_depolarization"],
        quantum_temperature=config["contrastive_quantum_temperature"],
        kde_sigma_squared=config["kde_sigma_squared"],
        kde_temperature=config["kde_temperature"],
        quantum_mixture_weight=config["quantum_mixture_weight"],
    )
    count = _fit_reference_regularizer(
        regularizer,
        dataset,
        num_classes=config["num_classes"],
        samples_per_class=config["contrastive_reference_samples_per_class"],
    )
    return regularizer, count


def _diagnose_pair(
    pair: dict[str, Any],
    regularizer: HybridQuantumKDEContrastiveReference,
    classifier: nn.Module,
    real_features: Tensor,
    real_labels: Tensor,
    real_centroids: Tensor,
    *,
    dataset_root: str,
    evaluation_samples: int,
    score_samples: int,
    gradient_samples: int,
    evaluation_seed: int,
    gradient_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    seed = pair["seed"]
    classical_checkpoint = pair["classical_checkpoint"]
    hybrid_checkpoint = pair["hybrid_checkpoint"]
    config = hybrid_checkpoint["config"]
    classical_generator, classical_discriminator = _models_from_checkpoint(
        classical_checkpoint,
        device,
    )
    hybrid_generator, hybrid_discriminator = _models_from_checkpoint(
        hybrid_checkpoint,
        device,
    )

    classical_images, conditioning_labels = _balanced_generated_samples(
        classical_generator,
        samples=evaluation_samples,
        batch_size=128,
        seed=evaluation_seed,
        device=device,
    )
    hybrid_images, hybrid_labels = _balanced_generated_samples(
        hybrid_generator,
        samples=evaluation_samples,
        batch_size=128,
        seed=evaluation_seed,
        device=device,
    )
    if not torch.equal(conditioning_labels, hybrid_labels):
        raise RuntimeError("paired generators did not receive identical conditioning labels")
    classical_logits, classical_features = _classifier_outputs(
        classifier,
        classical_images,
        batch_size=128,
        device=device,
    )
    hybrid_logits, hybrid_features = _classifier_outputs(
        classifier,
        hybrid_images,
        batch_size=128,
        device=device,
    )
    classical_classwise = classwise_feature_diagnostics(
        real_features,
        real_labels,
        classical_features,
        conditioning_labels,
        classical_logits,
    )
    hybrid_classwise = classwise_feature_diagnostics(
        real_features,
        real_labels,
        hybrid_features,
        conditioning_labels,
        hybrid_logits,
    )
    outcome = _paired_outcome_report(
        classical_classwise,
        hybrid_classwise,
        classical_images,
        hybrid_images,
        classical_logits,
        hybrid_logits,
    )
    evaluation_verification = {
        "classical": _verify_stored_evaluation(
            pair["classical_path"].with_name("evaluation.json"),
            classical_classwise,
        ),
        "hybrid": _verify_stored_evaluation(
            pair["hybrid_path"].with_name("evaluation.json"),
            hybrid_classwise,
        ),
    }

    teacher_scores = {
        "classical_generator": _teacher_score_report(
            regularizer,
            classical_images[:score_samples],
            conditioning_labels[:score_samples],
            classical_logits[:score_samples].argmax(dim=1),
        ),
        "hybrid_generator": _teacher_score_report(
            regularizer,
            hybrid_images[:score_samples],
            conditioning_labels[:score_samples],
            hybrid_logits[:score_samples].argmax(dim=1),
        ),
    }

    gradient_labels = torch.arange(gradient_samples, dtype=torch.long) % config["num_classes"]
    gradient_noise = torch.randn(
        gradient_samples,
        config["latent_dim"],
        generator=torch.Generator().manual_seed(gradient_seed),
    )
    final_gradients = {
        "classical_generator": _gradient_report(
            classical_generator,
            classical_discriminator,
            regularizer,
            classifier,
            real_centroids,
            gradient_noise,
            gradient_labels,
            regularizer_weight=config["regularizer_weight"],
            generator_training=False,
        ),
        "hybrid_generator": _gradient_report(
            hybrid_generator,
            hybrid_discriminator,
            regularizer,
            classifier,
            real_centroids,
            gradient_noise,
            gradient_labels,
            regularizer_weight=config["regularizer_weight"],
            generator_training=False,
        ),
    }

    initial = _reconstruct_initial_pair(
        ExperimentConfig(**config),
        classifier,
        real_centroids,
        dataset_root=dataset_root,
        device=device,
    )
    architecture = _architecture_report(
        initial["generator_before_update"],
        classical_generator,
        hybrid_generator,
        regularizer,
        initial["first_noise"],
        initial["first_labels"],
    )
    first_update = initial["first_update"]
    first_difference_norm = first_update["difference_norms"]["hybrid_minus_kde"]
    final_parameter_distance = _parameter_distance(
        classical_generator,
        hybrid_generator,
    )
    trajectory = _trajectory_report(
        pair["classical_path"].with_name("metrics.jsonl"),
        pair["hybrid_path"].with_name("metrics.jsonl"),
        final_parameter_distance=final_parameter_distance,
        first_update_difference=first_difference_norm,
    )

    return {
        "seed": seed,
        "global_step": classical_checkpoint["global_step"],
        "checkpoints": {
            "classical": {
                "path": str(pair["classical_path"].resolve()),
                "sha256": file_sha256(pair["classical_path"]),
            },
            "hybrid": {
                "path": str(pair["hybrid_path"].resolve()),
                "sha256": file_sha256(pair["hybrid_path"]),
            },
        },
        "architecture": architecture,
        "outcome": outcome,
        "evaluation_verification": evaluation_verification,
        "teacher_scores": teacher_scores,
        "gradients": {
            "actual_first_generator_batch": initial["gradient_report"],
            "fixed_balanced_final_batches": final_gradients,
        },
        "first_adam_update": first_update,
        "trajectory_divergence": trajectory,
    }


def _models_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[SharedQuantumGenerator, ACGANDiscriminator]:
    config = checkpoint["config"]
    num_angles = 2 * config["num_qubits"] * (config["ansatz_reps"] + 1)
    generator = SharedQuantumGenerator(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        image_channels=config["image_channels"],
        num_angles=num_angles,
    ).to(device)
    discriminator = ACGANDiscriminator(
        num_classes=config["num_classes"],
        image_channels=config["image_channels"],
    ).to(device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
    generator.eval()
    discriminator.eval()
    return generator, discriminator


def _paired_outcome_report(
    classical_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    classical_images: Tensor,
    hybrid_images: Tensor,
    classical_logits: Tensor,
    hybrid_logits: Tensor,
) -> dict[str, Any]:
    delta_fields = (
        "conditional_accuracy",
        "target_probability",
        "target_nll",
        "class_fid",
        "fid_centroid_component",
        "fid_covariance_component",
        "intra_class_diversity",
    )
    deltas = []
    for classical, hybrid in zip(classical_rows, hybrid_rows, strict=True):
        if classical["class"] != hybrid["class"]:
            raise RuntimeError("paired class diagnostics are misaligned")
        deltas.append(
            {
                "class": classical["class"],
                **{field: hybrid[field] - classical[field] for field in delta_fields},
            }
        )
    mean_delta = {
        field: sum(row[field] for row in deltas) / len(deltas) for field in delta_fields
    }
    class_fid_delta = mean_delta["class_fid"]
    centroid_fraction = (
        mean_delta["fid_centroid_component"] / class_fid_delta
        if abs(class_fid_delta) > 1e-12
        else None
    )
    return {
        "classical": classical_rows,
        "hybrid": hybrid_rows,
        "hybrid_minus_classical": deltas,
        "mean_hybrid_minus_classical": mean_delta,
        "counts": {
            "accuracy_improved_classes": [
                row["class"] for row in deltas if row["conditional_accuracy"] > 0
            ],
            "class_fid_improved_classes": [
                row["class"] for row in deltas if row["class_fid"] < 0
            ],
            "diversity_improved_classes": [
                row["class"] for row in deltas if row["intra_class_diversity"] > 0
            ],
        },
        "class_fid_delta_fraction_from_centroid": centroid_fraction,
        "paired_image_rmse": (hybrid_images - classical_images).square().mean().sqrt().item(),
        "independent_classifier_prediction_disagreement": (
            hybrid_logits.argmax(dim=1) != classical_logits.argmax(dim=1)
        ).float().mean().item(),
    }


def _verify_stored_evaluation(
    evaluation_path: Path,
    classwise_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not evaluation_path.is_file():
        return {"available": False}
    stored = json.loads(evaluation_path.read_text(encoding="utf-8"))["metrics"]
    recomputed = {
        "conditional_accuracy": _mean(
            row["conditional_accuracy"] for row in classwise_rows
        ),
        "class_conditional_feature_fid_64": _mean(
            row["class_fid"] for row in classwise_rows
        ),
        "generated_intra_class_diversity": _mean(
            row["intra_class_diversity"] for row in classwise_rows
        ),
    }
    differences = {name: recomputed[name] - stored[name] for name in recomputed}
    return {
        "available": True,
        "path": str(evaluation_path.resolve()),
        "stored": {name: stored[name] for name in recomputed},
        "recomputed": recomputed,
        "differences": differences,
        "maximum_absolute_difference": max(abs(value) for value in differences.values()),
    }


def _teacher_score_report(
    regularizer: HybridQuantumKDEContrastiveReference,
    images: Tensor,
    labels: Tensor,
    independent_predictions: Tensor,
) -> dict[str, Any]:
    kde_batches = []
    quantum_batches = []
    with torch.no_grad():
        for image_batch in images.split(64):
            kde_batches.append(regularizer.classical.class_logits(image_batch))
            quantum_batches.append(regularizer.quantum.class_logits(image_batch))
    kde_logits = torch.cat(kde_batches)
    quantum_logits = torch.cat(quantum_batches)
    mixture = regularizer.quantum_mixture_weight
    hybrid_logits = (1 - mixture) * kde_logits + mixture * quantum_logits
    kde_predictions = kde_logits.argmax(dim=1)
    quantum_predictions = quantum_logits.argmax(dim=1)
    hybrid_predictions = hybrid_logits.argmax(dim=1)
    changed = hybrid_predictions != kde_predictions
    kde_centered = kde_logits - kde_logits.mean(dim=1, keepdim=True)
    quantum_centered = quantum_logits - quantum_logits.mean(dim=1, keepdim=True)
    return {
        "kde": _logit_report(kde_logits, labels),
        "quantum": _logit_report(quantum_logits, labels),
        "hybrid": _logit_report(hybrid_logits, labels),
        "kde_quantum_prediction_agreement": (
            kde_predictions == quantum_predictions
        ).float().mean().item(),
        "hybrid_changed_kde_fraction": changed.float().mean().item(),
        "hybrid_change_beneficial_for_requested_label": (
            (kde_predictions != labels) & (hybrid_predictions == labels)
        ).float().mean().item(),
        "hybrid_change_harmful_for_requested_label": (
            (kde_predictions == labels) & (hybrid_predictions != labels)
        ).float().mean().item(),
        "hybrid_changed_kde_and_matches_independent_classifier": (
            changed & (hybrid_predictions == independent_predictions)
        ).float().mean().item(),
        "effective_centered_logit_perturbation_ratio": (
            mixture * quantum_centered.norm()
            / ((1 - mixture) * kde_centered.norm()).clamp_min(torch.finfo(torch.float32).eps)
        ).item(),
    }


def _logit_report(logits: Tensor, labels: Tensor) -> dict[str, float]:
    centered = logits - logits.mean(dim=1, keepdim=True)
    target = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels.unsqueeze(1), -torch.inf)
    margin = target - competitors.max(dim=1).values
    return {
        "cross_entropy": functional.cross_entropy(logits, labels).item(),
        "requested_class_accuracy": (logits.argmax(dim=1) == labels).float().mean().item(),
        "mean_target_margin": margin.mean().item(),
        "centered_logit_rms": centered.square().mean().sqrt().item(),
    }


def _gradient_report(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer: HybridQuantumKDEContrastiveReference,
    classifier: nn.Module,
    real_centroids: Tensor,
    noise: Tensor,
    labels: Tensor,
    *,
    regularizer_weight: float,
    generator_training: bool,
) -> dict[str, Any]:
    generator = deepcopy(generator)
    discriminator = deepcopy(discriminator)
    generator.train(generator_training)
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    blocks = (
        ("label_embedding", tuple(generator.label_embedding.parameters())),
        ("input_projection", tuple(generator.input_projection.parameters())),
        ("image_decoder", tuple(generator.image_decoder.parameters())),
    )
    parameters = tuple(parameter for _, block in blocks for parameter in block)
    images = generator(noise, labels)
    kde_logits = regularizer.classical.class_logits(images)
    quantum_logits = regularizer.quantum.class_logits(images)
    mixture = regularizer.quantum_mixture_weight
    scaled_kde_logits = (1 - mixture) * kde_logits
    hybrid_logits = scaled_kde_logits + mixture * quantum_logits
    adversarial_logits, auxiliary_logits = discriminator(images)
    classifier_logits, classifier_features = classifier.forward_with_features(images)

    losses = {
        "kde": functional.cross_entropy(kde_logits, labels),
        "scaled_kde": functional.cross_entropy(scaled_kde_logits, labels),
        "quantum": functional.cross_entropy(quantum_logits, labels),
        "hybrid": functional.cross_entropy(hybrid_logits, labels),
        "adversarial": functional.binary_cross_entropy_with_logits(
            adversarial_logits,
            torch.ones_like(adversarial_logits),
        ),
        "auxiliary": functional.cross_entropy(auxiliary_logits, labels),
        "independent_semantic": functional.cross_entropy(classifier_logits, labels),
        "centroid_fidelity_proxy": (
            classifier_features - real_centroids[labels]
        ).square().sum(dim=1).mean(),
        "diversity_promotion_proxy": _diversity_promotion_loss(
            classifier_features,
            labels,
        ),
    }
    decomposition = decompose_hybrid_gradients(
        losses["kde"],
        losses["scaled_kde"],
        losses["hybrid"],
        parameters,
    )
    quantum_gradient = _gradients(losses["quantum"], parameters, retain_graph=True)
    adversarial_gradient = _gradients(losses["adversarial"], parameters, retain_graph=True)
    auxiliary_gradient = _gradients(losses["auxiliary"], parameters, retain_graph=True)
    semantic_gradient = _gradients(
        losses["independent_semantic"],
        parameters,
        retain_graph=True,
    )
    fidelity_gradient = _gradients(
        losses["centroid_fidelity_proxy"],
        parameters,
        retain_graph=True,
    )
    diversity_gradient = _gradients(
        losses["diversity_promotion_proxy"],
        parameters,
        retain_graph=True,
    )
    gan_gradient = _add_gradients(adversarial_gradient, auxiliary_gradient)
    gradients = {
        **decomposition,
        "standalone_quantum": quantum_gradient,
        "adversarial": adversarial_gradient,
        "auxiliary": auxiliary_gradient,
        "gan": gan_gradient,
        "independent_semantic": semantic_gradient,
        "centroid_fidelity_proxy": fidelity_gradient,
        "diversity_promotion_proxy": diversity_gradient,
    }
    gan_norm = _gradient_norm(gan_gradient)
    comparison_targets = (
        "kde",
        "standalone_quantum",
        "gan",
        "adversarial",
        "auxiliary",
        "independent_semantic",
        "centroid_fidelity_proxy",
        "diversity_promotion_proxy",
    )
    return {
        "generator_mode": "train" if generator_training else "eval",
        "batch_size": labels.shape[0],
        "classes_present": [int(value) for value in labels.unique(sorted=True)],
        "losses": {name: value.item() for name, value in losses.items()},
        "gradient_norms": {
            name: _gradient_norm(gradient).item() for name, gradient in gradients.items()
        },
        "weighted_to_gan_norm_ratios": {
            name: _safe_ratio(regularizer_weight * _gradient_norm(gradients[name]), gan_norm)
            for name in (
                "kde",
                "scaled_kde",
                "hybrid",
                "total_delta",
                "scale_effect",
                "quantum_addition",
            )
        },
        "decomposition": {
            "scale_effect_over_total_delta": _safe_ratio(
                _gradient_norm(decomposition["scale_effect"]),
                _gradient_norm(decomposition["total_delta"]),
            ),
            "quantum_addition_over_total_delta": _safe_ratio(
                _gradient_norm(decomposition["quantum_addition"]),
                _gradient_norm(decomposition["total_delta"]),
            ),
            "standalone_quantum_over_kde": _safe_ratio(
                _gradient_norm(quantum_gradient),
                _gradient_norm(decomposition["kde"]),
            ),
            "identity_residual_norm": _gradient_norm(
                decomposition["identity_residual"]
            ).item(),
            "total_delta_scale_effect_cosine": _gradient_cosine(
                decomposition["total_delta"],
                decomposition["scale_effect"],
            ),
            "total_delta_quantum_addition_cosine": _gradient_cosine(
                decomposition["total_delta"],
                decomposition["quantum_addition"],
            ),
            "scale_effect_quantum_addition_cosine": _gradient_cosine(
                decomposition["scale_effect"],
                decomposition["quantum_addition"],
            ),
        },
        "cosines": {
            source: {
                target: _gradient_cosine(gradients[source], gradients[target])
                for target in comparison_targets
            }
            for source in ("total_delta", "scale_effect", "quantum_addition")
        },
        "module_gradient_norms": {
            source: _module_gradient_norms(gradients[source], blocks)
            for source in ("kde", "total_delta", "scale_effect", "quantum_addition")
        },
    }


def _diversity_promotion_loss(features: Tensor, labels: Tensor) -> Tensor:
    class_variances = []
    for class_index in labels.unique(sorted=True):
        class_features = features[labels == class_index]
        if class_features.shape[0] >= 2:
            class_variances.append(
                (class_features - class_features.mean(dim=0)).square().sum(dim=1).mean()
            )
    if not class_variances:
        raise ValueError("diversity proxy requires at least two samples in one class")
    return -torch.stack(class_variances).mean()


def _reconstruct_initial_pair(
    config: ExperimentConfig,
    classifier: nn.Module,
    real_centroids: Tensor,
    *,
    dataset_root: str,
    device: torch.device,
) -> dict[str, Any]:
    config_payload = config.to_dict()
    config_payload["dataset_root"] = dataset_root
    config = ExperimentConfig(**config_payload)
    seed_everything(config.seed)
    data_loader = _build_data_loader(config, device)
    generator, discriminator, regularizer = _build_models(config, device, device)
    if not isinstance(regularizer, HybridQuantumKDEContrastiveReference):
        raise RuntimeError("initial reconstruction did not build the hybrid regularizer")
    _fit_reference_regularizer(
        regularizer,
        data_loader.dataset,
        num_classes=config.num_classes,
        samples_per_class=config.contrastive_reference_samples_per_class,
    )
    initial_generator = deepcopy(generator)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate_discriminator,
        betas=(config.beta1, config.beta2),
    )
    real_images, real_labels = next(iter(data_loader))
    discriminator_noise = torch.randn(config.batch_size, config.latent_dim, device=device)
    discriminator_labels = torch.randint(
        config.num_classes,
        (config.batch_size,),
        device=device,
    )
    discriminator_step(
        generator,
        discriminator,
        discriminator_optimizer,
        real_images,
        real_labels,
        discriminator_noise,
        discriminator_labels,
        real_label_smoothing=config.real_label_smoothing,
    )
    generator_noise = torch.randn(config.batch_size, config.latent_dim, device=device)
    gradient_report = _gradient_report(
        generator,
        discriminator,
        regularizer,
        classifier,
        real_centroids,
        generator_noise,
        real_labels,
        regularizer_weight=config.regularizer_weight,
        generator_training=True,
    )
    first_update = _first_update_report(
        generator,
        discriminator,
        regularizer,
        generator_noise,
        real_labels,
        config,
    )
    return {
        "generator_before_update": initial_generator,
        "first_noise": generator_noise,
        "first_labels": real_labels,
        "gradient_report": gradient_report,
        "first_update": first_update,
    }


def _first_update_report(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer: HybridQuantumKDEContrastiveReference,
    noise: Tensor,
    labels: Tensor,
    config: ExperimentConfig,
) -> dict[str, Any]:
    base_parameters = _parameter_vector(generator)
    models: dict[str, SharedQuantumGenerator] = {}
    losses: dict[str, dict[str, float]] = {}
    for mode in ("kde", "scaled_kde", "hybrid"):
        model = deepcopy(generator)
        model_discriminator = deepcopy(discriminator)
        losses[mode] = _apply_first_generator_update(
            model,
            model_discriminator,
            regularizer,
            noise,
            labels,
            config,
            mode=mode,
        )
        models[mode] = model
    updates = {
        mode: _parameter_vector(model) - base_parameters for mode, model in models.items()
    }
    hybrid_minus_kde = updates["hybrid"] - updates["kde"]
    scaled_minus_kde = updates["scaled_kde"] - updates["kde"]
    hybrid_minus_scaled = updates["hybrid"] - updates["scaled_kde"]
    return {
        "losses": losses,
        "update_norms": {mode: vector.norm().item() for mode, vector in updates.items()},
        "difference_norms": {
            "hybrid_minus_kde": hybrid_minus_kde.norm().item(),
            "scaled_kde_minus_kde": scaled_minus_kde.norm().item(),
            "hybrid_minus_scaled_kde": hybrid_minus_scaled.norm().item(),
        },
        "scaling_explanation": {
            "hybrid_kde_difference_vs_scaling_difference_cosine": _vector_cosine(
                hybrid_minus_kde,
                scaled_minus_kde,
            ),
            "scaling_difference_over_hybrid_kde_difference": _safe_ratio(
                scaled_minus_kde.norm(),
                hybrid_minus_kde.norm(),
            ),
            "quantum_residual_over_hybrid_kde_difference": _safe_ratio(
                hybrid_minus_scaled.norm(),
                hybrid_minus_kde.norm(),
            ),
        },
    }


def _apply_first_generator_update(
    generator: SharedQuantumGenerator,
    discriminator: ACGANDiscriminator,
    regularizer: HybridQuantumKDEContrastiveReference,
    noise: Tensor,
    labels: Tensor,
    config: ExperimentConfig,
    *,
    mode: str,
) -> dict[str, float]:
    generator.train()
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate_generator,
        betas=(config.beta1, config.beta2),
    )
    optimizer.zero_grad(set_to_none=True)
    images = generator(noise, labels)
    kde_logits = regularizer.classical.class_logits(images)
    quantum_logits = regularizer.quantum.class_logits(images)
    mixture = regularizer.quantum_mixture_weight
    if mode == "kde":
        regularizer_loss = functional.cross_entropy(kde_logits, labels)
    elif mode == "scaled_kde":
        regularizer_loss = functional.cross_entropy((1 - mixture) * kde_logits, labels)
    elif mode == "hybrid":
        regularizer_loss = functional.cross_entropy(
            (1 - mixture) * kde_logits + mixture * quantum_logits,
            labels,
        )
    else:
        raise ValueError(f"unknown first-update mode: {mode}")
    adversarial_logits, auxiliary_logits = discriminator(images)
    gan_loss = functional.binary_cross_entropy_with_logits(
        adversarial_logits,
        torch.ones_like(adversarial_logits),
    ) + functional.cross_entropy(auxiliary_logits, labels)
    total = gan_loss + config.regularizer_weight * regularizer_loss
    total.backward()
    optimizer.step()
    return {
        "gan": gan_loss.detach().item(),
        "regularizer": regularizer_loss.detach().item(),
        "total": total.detach().item(),
    }


def _architecture_report(
    initial_generator: SharedQuantumGenerator,
    classical_generator: SharedQuantumGenerator,
    hybrid_generator: SharedQuantumGenerator,
    regularizer: HybridQuantumKDEContrastiveReference,
    noise: Tensor,
    labels: Tensor,
) -> dict[str, Any]:
    probe_generator = deepcopy(initial_generator)
    probe_generator.eval()
    images = probe_generator(noise, labels)
    hybrid_loss = functional.cross_entropy(regularizer.class_logits(images), labels)
    image_parameters = (
        *probe_generator.label_embedding.parameters(),
        *probe_generator.input_projection.parameters(),
        *probe_generator.image_decoder.parameters(),
    )
    angle_parameters = tuple(probe_generator.angle_head.parameters())
    image_gradients = torch.autograd.grad(
        hybrid_loss,
        image_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    angle_gradients = torch.autograd.grad(
        hybrid_loss,
        angle_parameters,
        retain_graph=False,
        allow_unused=True,
    )
    return {
        "regularizer_trainable_parameters": sum(
            parameter.numel() for parameter in regularizer.parameters()
        ),
        "generator_image_path_parameters": sum(
            parameter.numel() for parameter in image_parameters
        ),
        "generator_angle_head_parameters": sum(
            parameter.numel() for parameter in angle_parameters
        ),
        "hybrid_gradient_reaches_image_path": any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in image_gradients
        ),
        "hybrid_gradient_reaches_angle_head": any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in angle_gradients
        ),
        "final_parameter_change_from_initialization": {
            "classical": _module_change_report(initial_generator, classical_generator),
            "hybrid": _module_change_report(initial_generator, hybrid_generator),
        },
        "final_hybrid_minus_classical": _parameter_distance(
            classical_generator,
            hybrid_generator,
        ),
    }


def _module_change_report(
    initial: SharedQuantumGenerator,
    final: SharedQuantumGenerator,
) -> dict[str, Any]:
    initial_parameters = dict(initial.named_parameters())
    final_parameters = dict(final.named_parameters())
    report = {}
    for prefix in _MODULE_PREFIXES:
        differences = torch.cat(
            [
                (final_parameters[name].detach() - value.detach()).flatten().float()
                for name, value in initial_parameters.items()
                if name.startswith(prefix)
            ]
        )
        report[prefix] = {
            "values": differences.numel(),
            "changed_values": int(torch.count_nonzero(differences).item()),
            "l2": differences.norm().item(),
            "max_absolute": differences.abs().max().item(),
        }
    return report


def _parameter_distance(
    first: SharedQuantumGenerator,
    second: SharedQuantumGenerator,
) -> dict[str, float]:
    first_vector = _parameter_vector(first)
    second_vector = _parameter_vector(second)
    difference = second_vector - first_vector
    return {
        "l2": difference.norm().item(),
        "relative_to_classical_parameter_norm": _safe_ratio(
            difference.norm(),
            first_vector.norm(),
        ),
        "parameter_cosine": _vector_cosine(first_vector, second_vector),
    }


def _trajectory_report(
    classical_log_path: Path,
    hybrid_log_path: Path,
    *,
    final_parameter_distance: dict[str, float],
    first_update_difference: float,
) -> dict[str, Any]:
    classical_rows = [
        json.loads(line) for line in classical_log_path.read_text(encoding="utf-8").splitlines()
    ]
    hybrid_rows = [
        json.loads(line) for line in hybrid_log_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(classical_rows) != len(hybrid_rows):
        raise ValueError("paired metric logs have different lengths")
    fields = (
        ("generator", "adversarial"),
        ("generator", "auxiliary"),
        ("generator", "regularizer"),
        ("discriminator", "total"),
    )
    metrics = {}
    for section, field in fields:
        differences = [
            abs(classical[section][field] - hybrid[section][field])
            for classical, hybrid in zip(classical_rows, hybrid_rows, strict=True)
        ]
        name = f"{section}.{field}"
        metrics[name] = {
            "first_difference_step": next(
                (index + 1 for index, difference in enumerate(differences) if difference != 0),
                None,
            ),
            "maximum_absolute_difference": max(differences),
            "root_mean_square_difference": (
                sum(value * value for value in differences) / len(differences)
            )
            ** 0.5,
            "step_1": {
                "classical": classical_rows[0][section][field],
                "hybrid": hybrid_rows[0][section][field],
                "absolute_difference": differences[0],
            },
        }
    return {
        "logged_steps": len(classical_rows),
        "metrics": metrics,
        "final_parameter_distance": final_parameter_distance,
        "first_update_hybrid_minus_kde_l2": first_update_difference,
        "final_to_first_difference_amplification": (
            final_parameter_distance["l2"] / first_update_difference
            if first_update_difference > 0
            else None
        ),
    }


def _cross_seed_summary(pair_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    initial_reports = [
        report["gradients"]["actual_first_generator_batch"] for report in pair_reports
    ]
    first_updates = [report["first_adam_update"] for report in pair_reports]
    return {
        "seeds": [report["seed"] for report in pair_reports],
        "all_angle_heads_unchanged": all(
            report["architecture"]["final_parameter_change_from_initialization"][variant][
                "angle_head"
            ]["changed_values"]
            == 0
            for report in pair_reports
            for variant in ("classical", "hybrid")
        ),
        "mean_initial_scale_effect_over_total_delta": _mean(
            report["decomposition"]["scale_effect_over_total_delta"]
            for report in initial_reports
        ),
        "mean_initial_quantum_addition_over_total_delta": _mean(
            report["decomposition"]["quantum_addition_over_total_delta"]
            for report in initial_reports
        ),
        "mean_initial_weighted_quantum_addition_to_gan": _mean(
            report["weighted_to_gan_norm_ratios"]["quantum_addition"]
            for report in initial_reports
        ),
        "mean_first_update_scaling_alignment": _mean(
            report["scaling_explanation"][
                "hybrid_kde_difference_vs_scaling_difference_cosine"
            ]
            for report in first_updates
        ),
        "mean_first_update_quantum_residual_fraction": _mean(
            report["scaling_explanation"][
                "quantum_residual_over_hybrid_kde_difference"
            ]
            for report in first_updates
        ),
        "final_to_first_difference_amplification": {
            str(report["seed"]): report["trajectory_divergence"][
                "final_to_first_difference_amplification"
            ]
            for report in pair_reports
        },
        "per_seed_outcome": {
            str(report["seed"]): report["outcome"]["mean_hybrid_minus_classical"]
            for report in pair_reports
        },
    }


def _gradients(
    loss: Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> GradientTuple:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        gradient if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(gradients, parameters, strict=True)
    )


def _gradient_norm(gradients: GradientTuple) -> Tensor:
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()


def _gradient_dot(first: GradientTuple, second: GradientTuple) -> Tensor:
    return torch.stack(
        [(left * right).sum() for left, right in zip(first, second, strict=True)]
    ).sum()


def _gradient_cosine(first: GradientTuple, second: GradientTuple) -> float | None:
    denominator = _gradient_norm(first) * _gradient_norm(second)
    if denominator.item() <= torch.finfo(denominator.dtype).eps:
        return None
    return (_gradient_dot(first, second) / denominator).item()


def _add_gradients(first: GradientTuple, second: GradientTuple) -> GradientTuple:
    return tuple(left + right for left, right in zip(first, second, strict=True))


def _subtract_gradients(first: GradientTuple, second: GradientTuple) -> GradientTuple:
    return tuple(left - right for left, right in zip(first, second, strict=True))


def _module_gradient_norms(
    gradients: GradientTuple,
    blocks: tuple[tuple[str, tuple[nn.Parameter, ...]], ...],
) -> dict[str, float]:
    report = {}
    start = 0
    for name, parameters in blocks:
        end = start + len(parameters)
        report[name] = _gradient_norm(gradients[start:end]).item()
        start = end
    return report


def _parameter_vector(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().flatten().float() for parameter in model.parameters()])


def _vector_cosine(first: Tensor, second: Tensor) -> float | None:
    denominator = first.norm() * second.norm()
    if denominator.item() <= torch.finfo(denominator.dtype).eps:
        return None
    return torch.dot(first, second).div(denominator).item()


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> float:
    if denominator.item() <= torch.finfo(denominator.dtype).eps:
        return 0.0
    return (numerator / denominator).item()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)
