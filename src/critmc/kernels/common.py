"""Shared constants and helpers for the compiled kernels.

Update codes
------------
Every model exposes a single ``run`` kernel that executes a *schedule*: an
``(k, 2)`` integer array of ``(update_code, repetitions)`` rows that is
applied once per measurement.  Keeping the whole Markov chain inside one
compiled call removes Python overhead even for tiny lattices.
"""

from __future__ import annotations

import numpy as np
from numba import njit

METROPOLIS = 0  # sequential (typewriter) single-site Metropolis sweep
METROPOLIS_RANDOM = 1  # N single-site Metropolis attempts at random sites
HEATBATH = 2  # sequential heat-bath (Glauber) sweep
WOLFF = 3  # one Wolff single-cluster update
SWENDSEN_WANG = 4  # one Swendsen-Wang multi-cluster update
OVERRELAX = 5  # microcanonical over-relaxation sweep (continuous spins)

UPDATE_CODES = {
    "metropolis": METROPOLIS,
    "metropolis-random": METROPOLIS_RANDOM,
    "heatbath": HEATBATH,
    "wolff": WOLFF,
    "sw": SWENDSEN_WANG,
    "swendsen-wang": SWENDSEN_WANG,
    "overrelax": OVERRELAX,
}

CLUSTER_CODES = (WOLFF, SWENDSEN_WANG)

# columns of the per-measurement diagnostics array returned by every run kernel
AUX_ACCEPTED = 0  # accepted single-site moves
AUX_WOLFF_SIZE = 1  # summed size of Wolff clusters
AUX_SW_C2 = 2  # sum_C |C|^2 of the last Swendsen-Wang decomposition (improved estimator)
AUX_SW_NCLUSTERS = 3  # number of clusters in the last Swendsen-Wang decomposition
N_AUX = 4


@njit(cache=True, inline="always")
def randbelow(rng, n):
    """Uniform integer in ``[0, n)`` from a single double.

    About five times faster than ``Generator.integers`` inside numba; the bias
    is below ``n / 2**53`` and therefore irrelevant for lattice sizes.
    """
    k = int(rng.random() * n)
    return k if k < n else n - 1


@njit(cache=True, inline="always")
def find_root(parent, i):
    """Union-find root with path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True, inline="always")
def union(parent, size, a, b):
    """Union by size; returns the new root."""
    ra = find_root(parent, a)
    rb = find_root(parent, b)
    if ra == rb:
        return ra
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]
    return ra


class Workspace:
    """Scratch buffers reused by the cluster kernels (allocated once per simulation)."""

    def __init__(self, n_sites: int):
        self.queue = np.empty(n_sites, dtype=np.int64)
        self.mark = np.full(n_sites, -1, dtype=np.int64)
        self.parent = np.empty(n_sites, dtype=np.int64)
        self.size = np.empty(n_sites, dtype=np.int64)
        self.flag = np.empty(n_sites, dtype=np.int64)
        self.proj = np.empty(n_sites, dtype=np.float64)
        self.tag = 0
