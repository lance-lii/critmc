r"""Exact density of states of the square-lattice Ising torus.

On an ``L x M`` torus (``N = LM`` sites, ``2N`` bonds) the bond energy takes the
values ``E_j = -2N + 4j``, ``j = 0..N``, so

.. math::

    Z(K) = e^{2NK} P(x), \qquad P(x) = \sum_{j=0}^{N} g_j x^j, \qquad x = e^{-4K}.

The integer coefficients ``g_j`` - the exact density of states - are
recovered by evaluating ``P`` on a twisted grid of the unit circle,
``x_k = exp(2 pi i (k + 1/2) / S)``, and inverting the discrete Fourier
transform.  ``P(x_k)`` is obtained from Kaufman's formula continued to complex
coupling ``K = -ln(x) / 4``.  For even ``L`` and ``M`` every factor can be
written in a manifestly single-valued form:

* the factors of index ``l`` and ``2M - l`` coincide and are combined into
  ``(2 cosh(L gamma_l / 2))^2`` and ``(2 sinh(L gamma_l / 2))^2``, which are
  even in ``gamma_l`` and invariant under ``gamma_l -> gamma_l + 2 pi i``;
* the unpaired ``gamma_0`` enters through ``e^{gamma_0} = e^{2K} tanh K``;
* the unpaired ``gamma_M`` enters through ``cosh gamma_M = c + 1`` and the
  analytic ``sinh gamma_M = cosh 2K (1 + sinh 2K) / sinh 2K``.

Hence no branch choices are needed anywhere.  With ``~0.3 N`` significant
digits (mpmath) the transform returns integers to within ``10^-20``; the
result is verified against the sum rule ``sum_j g_j = 2^N``, the symmetry
``g_j = g_{N-j}`` and ``g_0 = 2``, ``g_1 = 0``, ``g_2 = 2N``.

This reproduces the method of P. D. Beale, PRL 76, 78 (1996) with a
numerical rather than symbolic polynomial expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import logsumexp

__all__ = ["DensityOfStates", "ising_dos", "partition_polynomial_value"]


def _pow(z, k: int):
    return z**k


def partition_polynomial_value(x, L: int, M: int | None = None):
    """``P(x) = Z(K) e^{-2NK}`` at complex ``x = e^{-4K}`` (mpmath, current precision)."""
    m = int(L)
    n = int(L if M is None else M)
    if m % 2 or n % 2:
        raise ValueError("the analytic continuation requires even L and M")
    x = mp.mpc(x)
    K = -mp.log(x) / 4
    s2 = mp.sinh(2 * K)
    c2 = mp.cosh(2 * K)
    c = c2 * c2 / s2
    half = m // 2

    z1 = mp.mpc(1)
    z2 = mp.mpc(1)
    for l in range(1, n, 2):  # odd l < n, paired with 2n - l
        y = c - mp.cospi(mp.mpf(l) / n)
        w = _pow(y + mp.sqrt(y * y - 1), half)  # e^{m gamma / 2} on some branch
        wi = 1 / w
        z1 *= (w + wi) ** 2
        z2 *= (w - wi) ** 2

    xg = mp.exp(2 * K) * mp.tanh(K)  # e^{gamma_0}, signed gamma_0
    w0 = _pow(xg, half)
    yn = c + 1
    sinh_gn = c2 * (1 + s2) / s2
    wn = _pow(yn + sinh_gn, half)
    z3 = (w0 + 1 / w0) * (wn + 1 / wn)
    z4 = (w0 - 1 / w0) * (wn - 1 / wn)
    for l in range(2, n - 1, 2):  # even 2 <= l <= n - 2, paired with 2n - l
        y = c - mp.cospi(mp.mpf(l) / n)
        w = _pow(y + mp.sqrt(y * y - 1), half)
        wi = 1 / w
        z3 *= (w + wi) ** 2
        z4 *= (w - wi) ** 2

    n_sites = m * n
    z = (2 * s2) ** (n_sites // 2) * (z1 + z2 + z3 + z4) / 2
    return z * mp.exp(-2 * n_sites * K)


def _fft(values: list, inverse: bool = False) -> list:
    """Iterative radix-2 FFT on a list of mpmath complex numbers."""
    size = len(values)
    if size & (size - 1):
        raise ValueError("FFT length must be a power of two")
    a = list(values)
    j = 0
    for i in range(1, size):  # bit-reversal permutation
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    sign = 1 if inverse else -1
    length = 2
    while length <= size:
        w_len = mp.expjpi(sign * mp.mpf(2) / length)
        half = length // 2
        twiddles = [mp.mpc(1)]
        for _ in range(half - 1):
            twiddles.append(twiddles[-1] * w_len)
        for start in range(0, size, length):
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * twiddles[k]
                a[start + k] = u + v
                a[start + k + half] = u - v
        length <<= 1
    return a


def _exact_coefficients(L: int, M: int, guard_digits: int = 40) -> list[int]:
    n_sites = L * M
    size = 1
    while size < n_sites + 1:
        size <<= 1
    dps = int(0.30103 * n_sites) + guard_digits + 5 * int(np.log10(size) + 1)
    with mp.workdps(dps):
        half = size // 2
        values: list = [None] * size
        for k in range(half):
            xk = mp.expjpi(mp.mpf(2 * k + 1) / size)
            pk = partition_polynomial_value(xk, L, M)
            values[k] = pk
            values[size - 1 - k] = mp.conj(pk)
        # g_j = (1/S) sum_k P(x_k) exp(-2 pi i j (k + 1/2) / S): undo the half-step twist
        spec = _fft(values)
        coeffs: list[int] = []
        tol = mp.mpf(10) ** -10
        for jj in range(n_sites + 1):
            val = spec[jj] * mp.expjpi(-mp.mpf(jj) / size) / size
            nearest = int(mp.nint(val.real))
            if abs(val.real - nearest) > tol or abs(val.imag) > tol:
                raise ArithmeticError(
                    f"precision loss extracting g_{jj} (L={L}); increase guard_digits"
                )
            coeffs.append(nearest)
    return coeffs


def _verify(coeffs: list[int], n_sites: int) -> None:
    if sum(coeffs) != 2**n_sites:
        raise ArithmeticError("sum rule sum_j g_j = 2^N violated")
    if coeffs != coeffs[::-1]:
        raise ArithmeticError("symmetry g_j = g_{N-j} violated")
    if coeffs[0] != 2 or coeffs[1] != 0 or coeffs[2] != 2 * n_sites:
        raise ArithmeticError("low-energy degeneracies are wrong")
    if any(g < 0 for g in coeffs):
        raise ArithmeticError("negative density of states")


@dataclass(frozen=True)
class DensityOfStates:
    """Exact density of states ``g(E_j)`` with ``E_j = -2N + 4j``."""

    L: int
    M: int
    g: tuple[int, ...]

    @property
    def n_sites(self) -> int:
        return self.L * self.M

    @property
    def energies(self) -> np.ndarray:
        """Bond energies of all levels (including the empty ones, ``g = 0``)."""
        n = self.n_sites
        return -2.0 * n + 4.0 * np.arange(n + 1)

    @property
    def log_g(self) -> np.ndarray:
        """``ln g`` as float64 (``-inf`` for empty levels)."""
        return np.array([float(mp.log(v)) if v > 0 else -np.inf for v in self.g])

    def log_z(self, beta) -> np.ndarray:
        beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        a = self.log_g[None, :] - beta[:, None] * self.energies[None, :]
        return logsumexp(a, axis=1)

    def canonical(self, beta) -> dict[str, np.ndarray]:
        """Energy per site, specific heat per site and ``ln Z`` at each ``beta``."""
        beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        e = self.energies
        a = self.log_g[None, :] - beta[:, None] * e[None, :]
        lz = logsumexp(a, axis=1)
        p = np.exp(a - lz[:, None])
        mean = p @ e
        var = p @ (e * e) - mean**2
        n = self.n_sites
        return {"lnZ": lz, "u": mean / n, "c": beta**2 * var / n}

    def energy_distribution(self, beta: float) -> np.ndarray:
        """Canonical probability of each energy level at ``beta``."""
        a = self.log_g - beta * self.energies
        return np.exp(a - logsumexp(a))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{self.L} {self.M}\n" + "\n".join(str(v) for v in self.g) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> DensityOfStates:
        lines = Path(path).read_text().split()
        L, M = int(lines[0]), int(lines[1])
        g = tuple(int(v) for v in lines[2:])
        _verify(list(g), L * M)
        return cls(L, M, g)


def ising_dos(L: int, M: int | None = None, cache_dir: str | Path | None = None) -> DensityOfStates:
    """Exact density of states of the ``L x M`` Ising torus (``L``, ``M`` even).

    Results are verified and optionally cached as plain-text integer lists.
    Runtime grows roughly like ``N^2``: ``L = 32`` takes seconds, ``L = 64``
    about a minute.
    """
    M = L if M is None else M
    if L % 2 or M % 2:
        raise ValueError("L and M must be even")
    cache = Path(cache_dir) / f"ising_dos_{L}x{M}.txt" if cache_dir is not None else None
    if cache is not None and cache.exists():
        return DensityOfStates.load(cache)
    coeffs = _exact_coefficients(L, M)
    _verify(coeffs, L * M)
    dos = DensityOfStates(L, M, tuple(coeffs))
    if cache is not None:
        dos.save(cache)
    return dos
