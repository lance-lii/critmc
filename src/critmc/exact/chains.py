"""Exact transfer-operator solutions of periodic O(n) chains.

For a ring of ``N`` spins with ``H = -sum_i S_i . S_{i+1}`` the partition
function is ``Z = sum_l d_l lambda_l^N`` where the transfer operator
``exp(beta S . S')`` is diagonalised by

* XY (n = 2):        Fourier modes, ``lambda_m = 2 pi I_m(beta)``, ``d_m = 1``;
* Heisenberg (n = 3): spherical harmonics (Funk-Hecke),
  ``lambda_l = 4 pi i_l(beta)``, ``d_l = 2l + 1``,

with ``I_m`` the modified Bessel function and ``i_l`` its spherical variant.
These give exact references for the continuous-spin kernels at any ``N``.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["on_ring_energy", "on_ring_specific_heat"]


def _spectrum(beta: float, n: int, l_max: int):
    """Scaled eigenvalues, their first two beta-derivatives and degeneracies.

    A common factor ``exp(beta)`` is removed from every eigenvalue; it cancels
    in all normalised averages.
    """
    if n == 2:
        m = np.arange(-l_max, l_max + 1)
        lam = special.ive(m, beta)
        # I_m' = (I_{m-1} + I_{m+1}) / 2,  I_m'' = (I_{m-2} + 2 I_m + I_{m+2}) / 4
        d1 = 0.5 * (special.ive(m - 1, beta) + special.ive(m + 1, beta))
        d2 = 0.25 * (special.ive(m - 2, beta) + 2 * lam + special.ive(m + 2, beta))
        deg = np.ones_like(lam)
    elif n == 3:
        l = np.arange(0, l_max + 1)
        scale = np.exp(-beta)
        lam = special.spherical_in(l, beta) * scale
        d1 = special.spherical_in(l, beta, derivative=True) * scale
        # i_l'' from the modified spherical Bessel equation
        with np.errstate(divide="ignore", invalid="ignore"):
            d2 = lam * (1.0 + l * (l + 1) / beta**2) - 2.0 / beta * d1
        deg = 2.0 * l + 1.0
    else:
        raise ValueError("only n = 2 (XY) and n = 3 (Heisenberg) are supported")
    return lam, d1, d2, deg


def _moments(beta: float, n_sites: int, n: int, l_max: int):
    lam, d1, d2, deg = _spectrum(beta, n, l_max)
    lead = lam.max()
    r = lam / lead
    w = deg * r**n_sites
    z = w.sum()
    # d/dbeta ln(lambda^N) = N lambda'/lambda and
    # d^2/dbeta^2 lambda^N = N lambda^(N-2) [(N-1) lambda'^2 + lambda lambda'']
    g1 = n_sites * d1 / lam
    g2 = n_sites * ((n_sites - 1) * (d1 / lam) ** 2 + d2 / lam)
    first = np.sum(w * g1) / z
    second = np.sum(w * g2) / z
    return first, second


def on_ring_energy(beta: float, n_sites: int, n: int, l_max: int = 60) -> float:
    """Exact ``<E>/N`` of a periodic ring of ``n_sites`` O(n) spins."""
    if n_sites < 3:
        raise ValueError("ring needs at least 3 sites")
    first, _ = _moments(float(beta), n_sites, n, l_max)
    return -first / n_sites


def on_ring_specific_heat(beta: float, n_sites: int, n: int, l_max: int = 60) -> float:
    """Exact specific heat per site ``beta^2 (<E^2> - <E>^2) / N`` of the ring."""
    if n_sites < 3:
        raise ValueError("ring needs at least 3 sites")
    first, second = _moments(float(beta), n_sites, n, l_max)
    return beta**2 * (second - first**2) / n_sites
