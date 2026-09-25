"""Onsager's solution of the square-lattice Ising model (thermodynamic limit).

L. Onsager, Phys. Rev. 65, 117 (1944); C. N. Yang, Phys. Rev. 85, 808 (1952).
All quantities are per site with ``J = k_B = 1``.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

__all__ = [
    "BETA_C",
    "T_C",
    "free_energy",
    "energy",
    "specific_heat",
    "magnetization",
    "heat_capacity_amplitude",
]

BETA_C = 0.5 * np.log1p(np.sqrt(2.0))
"""Critical coupling ``K_c = ln(1 + sqrt 2) / 2``."""

T_C = 1.0 / BETA_C
"""Critical temperature ``2 / ln(1 + sqrt 2) = 2.269185...``."""


def _as_beta(T):
    T = np.asarray(T, dtype=np.float64)
    if np.any(T <= 0):
        raise ValueError("temperatures must be positive")
    return 1.0 / T


def free_energy(T):
    """Free energy per site ``f(T)``."""
    beta = np.atleast_1d(_as_beta(T))
    out = np.empty_like(beta)
    for idx, b in enumerate(beta):
        k = 1.0 / np.sinh(2.0 * b) ** 2
        c2 = np.cosh(2.0 * b) ** 2

        def integrand(theta, k=k, c2=c2):
            return np.log(c2 + np.sqrt(1.0 + k * k - 2.0 * k * np.cos(2.0 * theta)) / k)

        val, _ = integrate.quad(integrand, 0.0, np.pi, limit=200, points=[np.pi / 2])
        out[idx] = -(0.5 * np.log(2.0) + val / (2.0 * np.pi)) / b
    return out if np.ndim(T) else float(out[0])


def energy(T):
    """Internal energy per site ``u = -coth 2K [1 + (2/pi)(2 tanh^2 2K - 1) K(kappa)]``."""
    beta = _as_beta(T)
    t2 = np.tanh(2.0 * beta) ** 2
    kappa2 = (2.0 * np.sinh(2.0 * beta) / np.cosh(2.0 * beta) ** 2) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ell = special.ellipk(kappa2)
        term = np.where(np.isclose(kappa2, 1.0, rtol=0, atol=1e-15), 0.0, (2 * t2 - 1) * ell)
    return -1.0 / np.tanh(2.0 * beta) * (1.0 + 2.0 / np.pi * term)


def specific_heat(T):
    """Specific heat per site ``c = -K^2 du/dK``; diverges logarithmically at ``T_C``.

    Obtained by differentiating :func:`energy` analytically, using
    ``dK(k)/dk = E(k) / (k (1 - k^2)) - K(k) / k`` for the complete elliptic
    integrals of modulus ``kappa = 2 sinh 2K / cosh^2 2K``.
    """
    beta = _as_beta(T)
    s = np.sinh(2.0 * beta)
    ch = np.cosh(2.0 * beta)
    t = s / ch
    a = 2.0 * t * t - 1.0
    kappa = 2.0 * s / ch**2
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = special.ellipk(kappa**2)
        e1 = special.ellipe(kappa**2)
        # a * dK/dkappa * dkappa/dK; the 0 * inf at T_c is resolved by the limit a -> 0
        dk1 = e1 / (kappa * (1.0 - kappa**2)) - k1 / kappa
        dkappa = 4.0 * (1.0 - s * s) / ch**3
        a_dk = np.where(np.isclose(a, 0.0, atol=1e-13), 0.0, a * dk1 * dkappa)
        d = -2.0 / s**2 * a * k1 + (ch / s) * 8.0 * t * (1.0 - t * t) * k1 + (ch / s) * a_dk
        du = 2.0 / s**2 - 2.0 / np.pi * d
    return -(beta**2) * du


def magnetization(T):
    """Spontaneous magnetisation ``(1 - sinh^-4 2K)^(1/8)`` below ``T_C``, else 0."""
    beta = _as_beta(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = 1.0 - np.sinh(2.0 * beta) ** -4
    return np.where(beta > BETA_C, np.clip(inner, 0.0, None) ** 0.125, 0.0)


def heat_capacity_amplitude() -> float:
    """Amplitude ``A`` of the finite-size peak ``c_max(L) ~ A ln L`` on the torus.

    Ferdinand & Fisher, Phys. Rev. 185, 832 (1969): ``A = 8 K_c^2 / pi``.
    """
    return 8.0 * BETA_C**2 / np.pi
