"""Domain-specific feature distribution metrics for conditional MNIST generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DistributionMetrics:
    conditional_accuracy: float
    feature_fid_64: float
    class_conditional_feature_fid_64: float
    feature_precision: float
    feature_recall: float
    generated_intra_class_diversity: float
    real_intra_class_diversity: float
    diversity_ratio: float


def compute_distribution_metrics(
    real_features: Tensor,
    real_labels: Tensor,
    generated_features: Tensor,
    conditioning_labels: Tensor,
    predicted_generated_labels: Tensor,
    *,
    manifold_neighbors: int = 3,
) -> DistributionMetrics:
    """Compute semantic accuracy, feature FID, manifold coverage, and diversity."""

    _validate_feature_inputs(
        real_features,
        real_labels,
        generated_features,
        conditioning_labels,
        predicted_generated_labels,
    )
    conditional_accuracy = (
        predicted_generated_labels == conditioning_labels
    ).float().mean().item()
    feature_fid = frechet_distance(real_features, generated_features)
    conditional_feature_fid = class_conditional_frechet_distance(
        real_features,
        real_labels,
        generated_features,
        conditioning_labels,
    )
    precision, recall = feature_precision_recall(
        real_features,
        generated_features,
        neighbors=manifold_neighbors,
    )
    generated_diversity = intra_class_diversity(
        generated_features,
        conditioning_labels,
    )
    real_diversity = intra_class_diversity(real_features, real_labels)
    diversity_ratio = generated_diversity / real_diversity if real_diversity > 0 else 0.0
    return DistributionMetrics(
        conditional_accuracy=conditional_accuracy,
        feature_fid_64=feature_fid,
        class_conditional_feature_fid_64=conditional_feature_fid,
        feature_precision=precision,
        feature_recall=recall,
        generated_intra_class_diversity=generated_diversity,
        real_intra_class_diversity=real_diversity,
        diversity_ratio=diversity_ratio,
    )


def frechet_distance(first_features: Tensor, second_features: Tensor) -> float:
    first = first_features.detach().to(device="cpu", dtype=torch.float64)
    second = second_features.detach().to(device="cpu", dtype=torch.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("feature matrices must be two-dimensional with equal feature width")
    if min(first.shape[0], second.shape[0]) < 2:
        raise ValueError("at least two samples are required for feature FID")

    first_mean = first.mean(dim=0)
    second_mean = second.mean(dim=0)
    first_covariance = torch.cov(first.transpose(0, 1))
    second_covariance = torch.cov(second.transpose(0, 1))
    first_sqrt = _symmetric_matrix_sqrt(first_covariance)
    covariance_product = first_sqrt @ second_covariance @ first_sqrt
    product_eigenvalues = torch.linalg.eigvalsh(
        0.5 * (covariance_product + covariance_product.transpose(0, 1))
    ).clamp_min(0)
    trace_sqrt_product = product_eigenvalues.sqrt().sum()
    mean_distance = (first_mean - second_mean).square().sum()
    fid = (
        mean_distance
        + torch.trace(first_covariance)
        + torch.trace(second_covariance)
        - 2 * trace_sqrt_product
    )
    return max(0.0, fid.item())


def class_conditional_frechet_distance(
    real_features: Tensor,
    real_labels: Tensor,
    generated_features: Tensor,
    conditioning_labels: Tensor,
) -> float:
    """Average feature FID across requested classes, weighted equally by class."""

    classes = torch.unique(real_labels, sorted=True)
    if not torch.equal(classes, torch.unique(conditioning_labels, sorted=True)):
        raise ValueError("real and conditioning labels must contain the same classes")
    class_distances = []
    for class_index in classes:
        class_distances.append(
            frechet_distance(
                real_features[real_labels == class_index],
                generated_features[conditioning_labels == class_index],
            )
        )
    return sum(class_distances) / len(class_distances)


def feature_precision_recall(
    real_features: Tensor,
    generated_features: Tensor,
    *,
    neighbors: int = 3,
) -> tuple[float, float]:
    real = real_features.detach().to(device="cpu", dtype=torch.float32)
    generated = generated_features.detach().to(device="cpu", dtype=torch.float32)
    if neighbors < 1 or neighbors >= min(real.shape[0], generated.shape[0]):
        raise ValueError("neighbors must be positive and smaller than both sample counts")

    real_radii = _manifold_radii(real, neighbors)
    generated_radii = _manifold_radii(generated, neighbors)
    cross_distances = torch.cdist(real, generated)
    precision = (cross_distances <= real_radii.unsqueeze(1)).any(dim=0).float().mean()
    recall = (cross_distances <= generated_radii.unsqueeze(0)).any(dim=1).float().mean()
    return precision.item(), recall.item()


def intra_class_diversity(features: Tensor, labels: Tensor) -> float:
    class_diversities = []
    for class_index in labels.unique(sorted=True):
        class_features = features[labels == class_index].detach().to(device="cpu")
        if class_features.shape[0] >= 2:
            class_diversities.append(torch.pdist(class_features).mean())
    if not class_diversities:
        raise ValueError("at least one class must contain two samples")
    return torch.stack(class_diversities).mean().item()


def _symmetric_matrix_sqrt(matrix: Tensor) -> Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T


def _manifold_radii(features: Tensor, neighbors: int) -> Tensor:
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(torch.inf)
    return distances.kthvalue(neighbors, dim=1).values


def _validate_feature_inputs(
    real_features: Tensor,
    real_labels: Tensor,
    generated_features: Tensor,
    conditioning_labels: Tensor,
    predicted_generated_labels: Tensor,
) -> None:
    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError("features must have shape (samples, feature_dim)")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("real and generated feature dimensions must match")
    if real_labels.shape != (real_features.shape[0],):
        raise ValueError("real_labels must match real_features")
    expected_generated_shape = (generated_features.shape[0],)
    if conditioning_labels.shape != expected_generated_shape:
        raise ValueError("conditioning_labels must match generated_features")
    if predicted_generated_labels.shape != expected_generated_shape:
        raise ValueError("predicted labels must match generated_features")
