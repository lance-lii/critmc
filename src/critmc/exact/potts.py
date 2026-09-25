"""Exact results for the square-lattice q-state Potts model.

* Critical point from self-duality: ``beta_c = ln(1 + sqrt q)``.
* Energy per site at ``beta_c``: ``e_c = -(1 + 1/sqrt q)`` (for ``q > 4`` this is
  the mean of the ordered and disordered energies).
* Latent heat for ``q > 4`` [R. J. Baxter, J. Phys. C 6, L445 (1973)]:
  ``Delta e = 2 (1 + 1/sqrt q) tanh(theta/2) prod_{n>=1} tanh^2(n theta)``
  with ``2 cosh theta = sqrt q``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["beta_c", "critical_energy", "latent_heat", "phase_energies"]


def beta_c(q: int) -> float:
    return float(np.log1p(np.sqrt(q)))


def critical_energy(q: int) -> float:
    """Energy per site at the transition (ferromagnet, two bonds per site)."""
    return -(1.0 + 1.0 / np.sqrt(q))


def latent_heat(q: int, n_terms: int = 200) -> float:
    """Latent heat per site; zero for the continuous transitions ``q <= 4``."""
    if q <= 4:
        return 0.0
    theta = np.arccosh(np.sqrt(q) / 2.0)
    n = np.arange(1, n_terms + 1)
    prod = np.prod(np.tanh(n * theta) ** 2)
    return float(2.0 * (1.0 + 1.0 / np.sqrt(q)) * np.tanh(theta / 2.0) * prod)


def phase_energies(q: int) -> tuple[float, float]:
    """``(e_ordered, e_disordered)`` coexisting at ``beta_c``."""
    ec = critical_energy(q)
    half = 0.5 * latent_heat(q)
    return ec - half, ec + half
