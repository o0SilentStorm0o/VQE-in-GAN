"""Determinism and provenance helpers for experiment runs."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import qiskit
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def collect_provenance(repository_root: Path) -> dict[str, Any]:
    lock_path = repository_root / "uv.lock"
    revision = _git_output(repository_root, "rev-parse", "HEAD")
    status = _git_output(repository_root, "status", "--porcelain")
    return {
        "source_revision": revision,
        "source_dirty": bool(status),
        "uv_lock_sha256": _sha256(lock_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "qiskit": qiskit.__version__,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
