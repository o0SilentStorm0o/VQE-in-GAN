"""Deterministic data-loading helpers shared by every experiment variant."""

from __future__ import annotations

import random
from typing import TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Sample = TypeVar("Sample")


def create_seeded_data_loader(
    dataset: Dataset[Sample],
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Sample]:
    """Create a shuffled loader whose sample order is isolated from global RNG state."""

    if len(dataset) < batch_size:
        raise ValueError("dataset must contain at least one full batch")
    data_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_data_worker if num_workers > 0 else None,
        generator=data_generator,
        persistent_workers=num_workers > 0,
    )


def _seed_data_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
