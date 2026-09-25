r"""Fisher zeros of the square-lattice Ising partition function.

In Fisher's variable ``u = e^{-2K}`` the partition function of the ``L x L``
torus is ``Z = u^{-N} P(u^2)`` with the integer polynomial ``P`` of
:mod:`critmc.exact.dos`.  In the thermodynamic limit the zeros condense on the
circles ``|u +- 1| = sqrt 2`` [M. E. Fisher, Lectures in Theoretical Physics
VIIC (1965)], which cut the positive real axis at ``u_c = sqrt 2 - 1``.  The
zero closest to ``u_c`` approaches it as ``|u_1(L) - u_c| ~ L^{-1/nu}``, which
gives an estimate of ``nu`` from exact data alone.

Two routes are provided: all zeros of ``P`` (``mpmath.polyroots``, feasible
up to ``L ~ 12``), and the leading zero alone by Muller iteration on the
analytically continued Kaufman formula (any even ``L``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

from .dos import DensityOfStates, partition_polynomial_value

__all__ = ["U_C", "fisher_zeros", "leading_zero", "leading_zeros", "LeadingZero"]

U_C = np.sqrt(2.0) - 1.0
"""Critical point in Fisher's variable ``u = e^{-2 K_c}``."""


def fisher_zeros(dos: DensityOfStates, extraprec: int | None = None) -> np.ndarray:
    """All ``2N`` zeros of ``Z`` in the complex ``u = e^{-2K}`` plane.

    The roots of ``P(x)`` are found with Durand-Kerner iteration in extended
    precision and mapped to ``u = +- sqrt(x)``.
    """
    coeffs = list(dos.g)  # ascending powers; g_0 = g_N = 2, so the degree is exactly N
    n = len(coeffs) - 1
    extraprec = extraprec if extraprec is not None else 4 * n + 50
    with mp.workdps(30):
        try:
            roots_x = mp.polyroots(coeffs, maxsteps=400, extraprec=extraprec, asc=True)
        except TypeError:  # mpmath < 1.4 only accepts descending order
            roots_x = mp.polyroots(coeffs[::-1], maxsteps=400, extraprec=extraprec)
    xs = np.array([complex(r) for r in roots_x])
    us = np.sqrt(xs)
    return np.concatenate([us, -us])


@dataclass(frozen=True)
class LeadingZero:
    L: int
    u: complex

    @property
    def K(self) -> complex:
        """The zero in the complex coupling plane, ``K = -ln(u) / 2``."""
        return -0.5 * np.log(self.u)

    @property
    def distance(self) -> float:
        return float(abs(self.u - U_C))


def leading_zero(L: int, guess: complex, dps: int = 40, max_iter: int = 40) -> LeadingZero:
    """Refine the Fisher zero near ``guess`` by Newton's method.

    The zeros of a large torus are dense and ``P`` is extremely steep, so the
    basin of attraction is only ``~0.1 / L`` wide; ``guess`` must come from a
    good extrapolation (see :func:`leading_zeros`).  Convergence is checked
    on the step size and the normalised residual.
    """
    with mp.workdps(dps + L // 4):
        u = mp.mpc(guess)
        # P has positive coefficients, so P(|x|) bounds |P(x)| and normalises the residual
        scale = partition_polynomial_value(abs(u) ** 2, L).real

        def f(v):
            return partition_polynomial_value(v * v, L) / scale

        tol = mp.mpf(10) ** (-dps // 2)
        for _ in range(max_iter):
            step = f(u) / mp.diff(f, u)
            u -= step
            if abs(step) < tol:
                break
        else:
            raise ArithmeticError(f"Newton iteration for the L={L} zero did not converge")
        if abs(u - mp.mpc(guess)) > 0.5 / L:
            raise ArithmeticError(f"Newton iteration for L={L} left the neighbourhood of the guess")
    return LeadingZero(int(L), complex(u))


def leading_zeros(sizes, dps: int = 40) -> list[LeadingZero]:
    """Leading zeros for increasing even ``L``.

    The smallest size is solved from its complete zero set.  Later guesses
    extrapolate ``u_1(L) = u_c + a/L + b/L^2`` through the two previous zeros
    (plain ``1/L`` scaling for the second size), which keeps every Newton
    start well inside its basin of attraction.
    """
    sizes = sorted(int(L) for L in sizes)
    if any(L % 2 for L in sizes):
        raise ValueError("sizes must be even")
    from .dos import ising_dos

    first = sizes[0]
    zs = fisher_zeros(ising_dos(first))
    upper = zs[(zs.imag > 1e-12) & (zs.real > 0)]
    start = upper[np.argmin(np.abs(upper - U_C))]
    out = [leading_zero(first, start, dps)]
    for L in sizes[1:]:
        if len(out) == 1:
            guess = U_C + (out[0].u - U_C) * out[0].L / L
        else:
            (l1, u1), (l2, u2) = (out[-2].L, out[-2].u - U_C), (out[-1].L, out[-1].u - U_C)
            # solve u_i = a / l_i + b / l_i^2 for (a, b)
            mat = np.array([[1 / l1, 1 / l1**2], [1 / l2, 1 / l2**2]])
            a, b = np.linalg.solve(mat, np.array([u1, u2]))
            guess = U_C + a / L + b / L**2
        out.append(leading_zero(L, guess, dps))
    return out
