"""Helpers for PCsInit (Principal Components-based Initialization).

The first layer of a per-module MLP is initialised with the leading principal
directions of the standardised training inputs instead of running PCA beforehand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pca_projection(train_matrix, variance_threshold=0.95, subset_fraction=1.0, seed=0):
    """Return an (r, d) array whose rows are the leading principal directions.

    r is the smallest number of components reaching `variance_threshold`.
    subset_fraction < 1 fits on a random row subset (PCsInit-Sub).
    Returns None when PCA is not possible (too few rows or zero variance).
    """
    x = np.nan_to_num(np.asarray(train_matrix, dtype=float))

    if 0.0 < subset_fraction < 1.0:
        rng = np.random.default_rng(seed)
        n_rows = max(2, int(len(x) * subset_fraction))
        x = x[rng.choice(len(x), size=min(n_rows, len(x)), replace=False)]

    if x.shape[0] < 2:
        return None

    x = x - x.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    variance = singular_values ** 2
    total = variance.sum()
    if total <= 0:
        return None

    n_components = int(np.searchsorted(np.cumsum(variance) / total, variance_threshold)) + 1
    return vt[: min(n_components, vt.shape[0])]


def module_pca_applied(meta_path: Path, module_name: str):
    """True/False as recorded by the upstream scripts; None if unknown."""
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("modules", {}).get(module_name, {}).get("pca_applied")
