from __future__ import annotations

import numpy as np


def batch_mean_error(x: np.ndarray, n_batches: int = 64) -> tuple[float, float]:
    """Mean and batch-means standard error (independent of critmc.stats on purpose)."""
    x = np.asarray(x, dtype=np.float64)
    size = len(x) // n_batches
    b = x[len(x) - size * n_batches :].reshape(n_batches, size).mean(axis=1)
    return float(x.mean()), float(b.std(ddof=1) / np.sqrt(n_batches))


def assert_consistent(sample: np.ndarray, exact: float, z_max: float = 4.5, n_batches: int = 64):
    mean, err = batch_mean_error(sample, n_batches)
    z = (mean - exact) / err if err > 0 else 0.0
    assert abs(z) < z_max, f"mean {mean:.6f} +- {err:.2e} vs exact {exact:.6f} (z = {z:.2f})"
