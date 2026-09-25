r"""Exact partition function of the finite square-lattice Ising torus.

B. Kaufman, Phys. Rev. 76, 1232 (1949), in the form used by
P. D. Beale, Phys. Rev. Lett. 76, 78 (1996).  For an ``m x n`` torus with
``K = beta J``

.. math::

    Z = \tfrac12 (2\sinh 2K)^{mn/2} \sum_{i=1}^{4} Z_i,

    Z_1 = \prod_{r=0}^{n-1} 2\cosh\tfrac{m\gamma_{2r+1}}{2},\quad
    Z_2 = \prod_{r=0}^{n-1} 2\sinh\tfrac{m\gamma_{2r+1}}{2},

    Z_3 = \prod_{r=0}^{n-1} 2\cosh\tfrac{m\gamma_{2r}}{2},\quad
    Z_4 = \prod_{r=0}^{n-1} 2\sinh\tfrac{m\gamma_{2r}}{2},

with :math:`\cosh\gamma_l = \cosh 2K \coth 2K - \cos(\pi l / n)` for
:math:`l \ge 1` and the *signed* :math:`\gamma_0 = 2K + \ln\tanh K`, which
changes sign at the critical point.

Everything is evaluated in log space and differentiated analytically, giving
``ln Z``, the energy and the specific heat to machine precision for lattices
of any size.  The only delicate factor is :math:`2\sinh(m\gamma_0/2)`, which
vanishes at ``K_c`` and is treated separately so that no ``0 * inf`` or
catastrophic cancellation can occur.
"""

from __future__ import annotations

import numpy as np

__all__ = ["log_partition", "energy", "specific_heat", "free_energy", "thermodynamics"]


def _ln2cosh(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax))


def _ln2sinh_abs(x):
    ax = np.abs(x)
    with np.errstate(divide="ignore"):
        return ax + np.log1p(-np.exp(-2.0 * ax))


def _factor_sums(x, x1, x2, hyperbolic):
    """``(sum ln|g|, A = sum g'/g, B = g''/g of the product)`` for g = 2cosh(x) or 2sinh(x)."""
    if hyperbolic == "cosh":
        f = _ln2cosh(x)
        t = np.tanh(x)
    else:
        f = _ln2sinh_abs(x)
        t = 1.0 / np.tanh(x)
    a = t * x1
    b = x1 * x1 + t * x2  # g''/g for a single factor
    big_a = a.sum(axis=1)
    big_b = b.sum(axis=1) + big_a**2 - (a * a).sum(axis=1)
    return f.sum(axis=1), big_a, big_b


def _derivatives(beta, m, n):
    """``(ln Z, d lnZ/dK, d^2 lnZ/dK^2)`` for an array of couplings."""
    K = np.atleast_1d(np.asarray(beta, dtype=np.float64))[:, None]
    if np.any(K <= 0):
        raise ValueError("beta must be positive")
    l = np.arange(2 * n)[None, :]
    sh = np.sinh(2.0 * K)
    ch = np.cosh(2.0 * K)
    c1 = 2.0 * ch * (1.0 - 1.0 / sh**2)  # d/dK of cosh2K coth2K
    c2 = 4.0 * (sh - 1.0 / sh + 2.0 * ch**2 / sh**3)

    # y - 1 = cosh(gamma) - 1 written without cancellation
    ym1 = (sh - 1.0) ** 2 / sh + 2.0 * np.sin(np.pi * l / (2.0 * n)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        sg = np.sqrt(ym1 * (ym1 + 2.0))  # sinh(gamma)
        gam = np.log1p(ym1 + sg)
        g1 = c1 / sg
        g2 = (c2 - (1.0 + ym1) * g1**2) / sg
    gam[:, 0] = (2.0 * K + np.log(np.tanh(K)))[:, 0]
    g1[:, 0] = (2.0 + 2.0 / sh)[:, 0]
    g2[:, 0] = (-4.0 * ch / sh**2)[:, 0]

    x, x1, x2 = 0.5 * m * gam, 0.5 * m * g1, 0.5 * m * g2
    odd = slice(1, None, 2)
    rest = slice(2, None, 2)  # even l >= 2

    f1, a1, b1 = _factor_sums(x[:, odd], x1[:, odd], x2[:, odd], "cosh")
    f2, a2, b2 = _factor_sums(x[:, odd], x1[:, odd], x2[:, odd], "sinh")
    f3r, a3r, b3r = _factor_sums(x[:, rest], x1[:, rest], x2[:, rest], "cosh")
    f4r, a4r, b4r = _factor_sums(x[:, rest], x1[:, rest], x2[:, rest], "sinh")

    # Z3 includes 2cosh(x0) like any other factor
    x0, x01, x02 = x[:, 0], x1[:, 0], x2[:, 0]
    t0 = np.tanh(x0)
    f3 = f3r + _ln2cosh(x0)
    a3 = a3r + t0 * x01
    b3 = b3r + x01**2 + t0 * x02 + 2.0 * t0 * x01 * a3r

    # Z4 = 2 sinh(x0) * R.  Work with w4 = Z4/S and w4hat = w4 * coth(x0) = 2cosh(x0) R / S.
    f4 = f4r + _ln2sinh_abs(x0)
    sign4 = np.sign(x0)
    f4hat = f4r + _ln2cosh(x0)

    fmax = np.maximum(np.maximum(f1, f2), np.maximum(f3, f4))
    e1, e2, e3 = np.exp(f1 - fmax), np.exp(f2 - fmax), np.exp(f3 - fmax)
    e4 = sign4 * np.exp(f4 - fmax)
    e4hat = np.exp(f4hat - fmax)
    total = e1 + e2 + e3 + e4
    w1, w2, w3, w4, w4hat = e1 / total, e2 / total, e3 / total, e4 / total, e4hat / total

    s1 = w1 * a1 + w2 * a2 + w3 * a3 + (w4hat * x01 + w4 * a4r)
    s2 = (
        w1 * b1
        + w2 * b2
        + w3 * b3
        + (w4 * x01**2 + w4hat * x02 + 2.0 * w4hat * x01 * a4r + w4 * b4r)
    )

    n_sites = m * n
    sh0, ch0 = sh[:, 0], ch[:, 0]
    ln_z = -np.log(2.0) + 0.5 * n_sites * np.log(2.0 * sh0) + fmax + np.log(total)
    d1 = n_sites * ch0 / sh0 + s1
    d2 = -2.0 * n_sites / sh0**2 + s2 - s1**2
    return ln_z, d1, d2


def _shape(L, M):
    m = int(L)
    n = int(L if M is None else M)
    if m < 2 or n < 2:
        raise ValueError("lattice extents must be >= 2")
    return m, n


def _restore(beta, arr):
    return arr if np.ndim(beta) else float(arr[0])


def log_partition(beta, L, M=None):
    """``ln Z`` of the ``L x M`` torus (``M = L`` by default)."""
    m, n = _shape(L, M)
    return _restore(beta, _derivatives(beta, m, n)[0])


def free_energy(beta, L, M=None):
    """Free energy per site ``-ln Z / (beta N)``."""
    m, n = _shape(L, M)
    ln_z = _derivatives(beta, m, n)[0]
    return _restore(beta, -ln_z / (np.atleast_1d(beta) * m * n))


def energy(beta, L, M=None):
    """Internal energy per site ``-(1/N) d ln Z / d beta``."""
    m, n = _shape(L, M)
    d1 = _derivatives(beta, m, n)[1]
    return _restore(beta, -d1 / (m * n))


def specific_heat(beta, L, M=None):
    """Specific heat per site ``(beta^2 / N) d^2 ln Z / d beta^2``."""
    m, n = _shape(L, M)
    d2 = _derivatives(beta, m, n)[2]
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    return _restore(beta, b * b * d2 / (m * n))


def thermodynamics(beta, L, M=None) -> dict[str, np.ndarray]:
    """``ln Z``, ``f``, ``u`` and ``c`` per site in one pass."""
    m, n = _shape(L, M)
    ln_z, d1, d2 = _derivatives(beta, m, n)
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    n_sites = m * n
    return {
        "lnZ": ln_z,
        "f": -ln_z / (b * n_sites),
        "u": -d1 / n_sites,
        "c": b * b * d2 / n_sites,
    }
