from __future__ import annotations

import torch

from vqe_gan.evaluation.metrics import (
    compute_distribution_metrics,
    feature_precision_recall,
    frechet_distance,
)


def test_identical_features_have_zero_fid_and_full_manifold_overlap() -> None:
    torch.manual_seed(71)
    features = torch.randn(30, 8)

    assert frechet_distance(features, features) < 1e-10
    precision, recall = feature_precision_recall(features, features, neighbors=3)
    assert precision == 1
    assert recall == 1


def test_shifted_features_have_positive_fid() -> None:
    torch.manual_seed(73)
    features = torch.randn(40, 6)

    assert frechet_distance(features, features + 2) > 20


def test_distribution_metrics_use_conditioning_labels_for_accuracy_and_diversity() -> None:
    torch.manual_seed(79)
    real_features = torch.randn(20, 5)
    generated_features = real_features.clone()
    labels = torch.arange(20) % 10
    predictions = labels.clone()

    metrics = compute_distribution_metrics(
        real_features,
        labels,
        generated_features,
        labels,
        predictions,
    )

    assert metrics.conditional_accuracy == 1
    assert metrics.feature_fid_64 < 1e-10
    assert metrics.class_conditional_feature_fid_64 < 1e-10
    assert metrics.feature_precision == 1
    assert metrics.feature_recall == 1
    assert abs(metrics.diversity_ratio - 1) < 1e-6
