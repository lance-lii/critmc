r"""Histogram reweighting.

Single histogram [A. M. Ferrenberg & R. H. Swendsen, PRL 61, 2635 (1988)]:
samples drawn at ``beta_0`` estimate averages at a nearby ``beta`` through

.. math:: \langle O\rangle_\beta =
    \frac{\sum_t O_t\, e^{-(\beta-\beta_0)E_t}}{\sum_t e^{-(\beta-\beta_0)E_t}} .

Multiple histograms [Ferrenberg & Swendsen, PRL 63, 1195 (1989)] are
combined with the MBAR estimator [M. R. Shirts & J. D. Chodera, J. Chem. Phys.
129, 124105 (2008)], whose dimensionless free energies ``f_k = -ln Z(beta_k)``
minimise the convex function

.. math:: F(f) = \sum_n \ln \sum_k N_k e^{f_k - \beta_k E_n} - \sum_k N_k f_k .

We solve it by Newton's method with a backtracking line search in the gauge
``f_0 = 0``.  Correlated data are handled through statistical inefficiencies
``g_k = 2 tau_k``: each sample from run ``k`` carries weight ``1/g_k`` and
``N_k`` is replaced by the effective size ``N_k / g_k``.  All sums are done in
log space and, for discrete energies, over distinct energy values only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "Reweighter",
    "MultiHistogram",
    "canonical_observables",
    "observables_from_weights",
    "kish_ess",
]


def kish_ess(log_w: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` from log-weights."""
    return float(np.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w)))


class Reweighter:
    """Single-histogram reweighting of one canonical time series.

    Parameters
    ----------
    energy:
        Total energy of every sample.
    beta0:
        Inverse temperature at which the samples were generated.
    observables:
        Optional mapping of named per-sample arrays.
    """

    def __init__(self, energy, beta0: float, observables: Mapping[str, np.ndarray] | None = None):
        self.energy = np.asarray(energy, dtype=np.float64)
        self.beta0 = float(beta0)
        self.obs = {k: np.asarray(v, dtype=np.float64) for k, v in (observables or {}).items()}
        for k, v in self.obs.items():
            if len(v) != len(self.energy):
                raise ValueError(f"observable {k!r} has the wrong length")
        self._shift = float(self.energy.mean())

    def _get(self, obs) -> np.ndarray:
        return self.obs[obs] if isinstance(obs, str) else np.asarray(obs, dtype=np.float64)

    def log_weights(self, beta: float) -> np.ndarray:
        return -(float(beta) - self.beta0) * (self.energy - self._shift)

    def weights(self, beta: float) -> np.ndarray:
        lw = self.log_weights(beta)
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def mean(self, obs, beta: float) -> float:
        return float(self.weights(beta) @ self._get(obs))

    def derivative(self, obs, beta: float) -> float:
        """``d<O>/dbeta = <O><E> - <O E>``."""
        w = self.weights(beta)
        o = self._get(obs)
        return float((w @ o) * (w @ self.energy) - w @ (o * self.energy))

    def ess(self, beta: float) -> float:
        """Kish effective sample size of the reweighted ensemble."""
        return kish_ess(self.log_weights(beta))

    def reliable_range(self, min_fraction: float = 0.1) -> tuple[float, float]:
        """Interval of ``beta`` where the effective sample size stays above ``min_fraction * n``.

        Uses the Gaussian approximation ``ESS/n = exp(-(dbeta sigma_E)^2)``.
        """
        sigma = float(self.energy.std())
        if sigma == 0:
            return (self.beta0, self.beta0)
        half = np.sqrt(-np.log(min_fraction)) / sigma
        return (self.beta0 - half, self.beta0 + half)


def canonical_observables(
    energy, absm, m2, m4, beta0: float, beta: float, n_sites: int
) -> dict[str, float]:
    """Standard thermodynamic and order-parameter estimators reweighted to ``beta``.

    Returns the energy ``e`` and specific heat ``c`` per site, ``absm``, ``m2``,
    ``m4``, the connected susceptibility ``chi = beta N (<m^2> - <|m|>^2)``, the
    full susceptibility ``chi0 = beta N <m^2>``, the Binder cumulant
    ``U = 1 - <m^4>/(3<m^2>^2)``, and the beta-derivatives ``dU`` of ``U`` and
    ``dlogm``, ``dlogm2`` of ``ln<|m|>`` and ``ln<m^2>``, whose magnitudes grow like
    ``L^{1/nu}`` at criticality.
    """
    e = np.asarray(energy, dtype=np.float64)
    lw = -(beta - beta0) * (e - e.mean())
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return observables_from_weights(w, e, absm, m2, m4, beta, n_sites)


def observables_from_weights(
    w, energy, absm, m2, m4, beta: float, n_sites: int
) -> dict[str, float]:
    """The estimators of :func:`canonical_observables` for given normalised weights."""
    e = energy
    me = w @ e
    me2 = w @ (e * e)
    a = w @ absm
    b = w @ m2
    d = w @ m4
    ae = w @ (absm * e)
    be = w @ (m2 * e)
    de = w @ (m4 * e)
    db = b * me - be
    dd = d * me - de
    n = float(n_sites)
    return {
        "e": me / n,
        "c": beta * beta * (me2 - me * me) / n,
        "absm": a,
        "m2": b,
        "m4": d,
        "chi": beta * n * (b - a * a),
        "chi0": beta * n * b,
        "U": 1.0 - d / (3.0 * b * b),
        "dU": -(dd / (b * b) - 2.0 * d * db / b**3) / 3.0,
        "dlogm": me - ae / a,
        "dlogm2": me - be / b,
    }


class MultiHistogram:
    """Multi-histogram (MBAR) combination of canonical runs at several temperatures.

    Parameters
    ----------
    energies:
        One array of sample energies per run.
    betas:
        Inverse temperature of each run.
    observables:
        Optional list (one mapping per run) of named per-sample arrays.
    inefficiency:
        Statistical inefficiency ``g_k = 2 tau_int`` of each run (default 1).
    """

    def __init__(
        self,
        energies: Sequence[np.ndarray],
        betas: Sequence[float],
        observables: Sequence[Mapping[str, np.ndarray]] | None = None,
        inefficiency: Sequence[float] | None = None,
    ):
        if len(energies) != len(betas) or len(betas) < 1:
            raise ValueError("need one energy array per beta")
        self.betas = np.asarray(betas, dtype=np.float64)
        self.n_runs = len(self.betas)
        self.counts = np.array([len(e) for e in energies], dtype=np.float64)
        g = np.ones(self.n_runs) if inefficiency is None else np.asarray(inefficiency, dtype=float)
        if np.any(g < 1.0 - 1e-12):
            raise ValueError("statistical inefficiencies must be >= 1")
        self.g = g
        self.energy = np.concatenate([np.asarray(e, dtype=np.float64) for e in energies])
        self.run_index = np.repeat(np.arange(self.n_runs), self.counts.astype(int))
        self.sample_logw = -np.log(g)[self.run_index]
        self.obs: dict[str, np.ndarray] = {}
        if observables is not None:
            names = set(observables[0])
            for o in observables:
                if set(o) != names:
                    raise ValueError("all runs must provide the same observables")
            for name in names:
                self.obs[name] = np.concatenate(
                    [np.asarray(o[name], dtype=np.float64) for o in observables]
                )
        self._shift = float(self.energy.mean())
        self._fs = np.zeros(self.n_runs)  # free energies for the shifted energies E - shift
        self._compress()
        self.converged = False
        self.solve()

    # -- internal representation -------------------------------------------
    def _compress(self) -> None:
        """Group samples with identical energy (and run weight) for discrete models."""
        key = np.stack([self.energy, self.sample_logw], axis=1)
        uniq, inverse, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        self._u_energy = uniq[:, 0] - self._shift
        self._u_logmult = np.log(cnt) + uniq[:, 1]
        self._inverse = inverse.ravel()

    def _log_denominator(self, f: np.ndarray) -> np.ndarray:
        """``ln D(E) = ln sum_k (N_k/g_k) exp(f_k - beta_k E)`` for each distinct energy."""
        log_n = np.log(self.counts / self.g)
        a = log_n[None, :] + f[None, :] - self.betas[None, :] * self._u_energy[:, None]
        return logsumexp(a, axis=1)

    def _objective(self, f: np.ndarray):
        log_n = np.log(self.counts / self.g)
        ld = self._log_denominator(f)
        value = np.sum(np.exp(self._u_logmult) * ld) - np.sum(np.exp(log_n) * f)
        # W[u, k] = (N_k/g_k) exp(f_k - beta_k E_u) / D(E_u), times multiplicity
        log_w = (
            log_n[None, :]
            + f[None, :]
            - self.betas[None, :] * self._u_energy[:, None]
            - ld[:, None]
        )
        w = np.exp(log_w)
        mult = np.exp(self._u_logmult)
        col = mult @ w
        grad = col - np.exp(log_n)
        hess = np.diag(col) - (w * mult[:, None]).T @ w
        return value, grad, hess

    def _initial_guess(self) -> np.ndarray:
        """Chain single-histogram free-energy differences between neighbouring betas."""
        order = np.argsort(self.betas)
        f = np.zeros(self.n_runs)
        for a, b in zip(order[:-1], order[1:]):
            e = self.energy[self.run_index == a] - self._shift
            # f_b - f_a = -ln <exp(-(beta_b - beta_a) E)>_a
            f[b] = f[a] - (logsumexp(-(self.betas[b] - self.betas[a]) * e) - np.log(len(e)))
        return f - f[0]

    def solve(self, rtol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
        """Newton iterations for the free energies ``f_k`` (gauge ``f_0 = 0``).

        Converged when every gradient component is below ``rtol`` times the
        mean effective run size.
        """
        if self.n_runs == 1:
            self.converged = True
            return self.f
        f = self._initial_guess()
        scale = float(np.mean(self.counts / self.g))
        for _ in range(max_iter):
            value, grad, hess = self._objective(f)
            if np.max(np.abs(grad[1:])) < rtol * scale:
                self.converged = True
                break
            step = np.zeros_like(f)
            step[1:] = -np.linalg.solve(hess[1:, 1:], grad[1:])
            t = 1.0
            slope = float(grad @ step)
            while True:
                trial = f + t * step
                if self._objective(trial)[0] <= value + 1e-4 * t * slope or t < 1e-12:
                    break
                t *= 0.5
            f = trial
        self._fs = f - f[0]
        return self.f

    @property
    def f(self) -> np.ndarray:
        """Dimensionless free energies ``f_k = -ln(Z(beta_k) / Z(beta_0))``."""
        return self._fs + (self.betas - self.betas[0]) * self._shift

    # -- estimates ------------------------------------------------------------
    def _sample_log_weights(self, beta: float) -> np.ndarray:
        ld = self._log_denominator(self._fs)
        per_unique = -beta * self._u_energy - ld
        return per_unique[self._inverse] + self.sample_logw

    def log_z(self, beta) -> np.ndarray:
        """``ln Z(beta)`` relative to ``Z(beta_0)``, i.e. ``-f(beta)``."""
        beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        ld = self._log_denominator(self._fs)
        mult = self._u_logmult
        out = np.array([logsumexp(mult - b * self._u_energy - ld) for b in beta])
        # undo the energy shift; normalised so that ln Z(beta_0) = 0
        return out - (beta - self.betas[0]) * self._shift

    def weights(self, beta: float) -> np.ndarray:
        lw = self._sample_log_weights(beta)
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def mean(self, obs, beta: float) -> float:
        o = self.obs[obs] if isinstance(obs, str) else np.asarray(obs, dtype=np.float64)
        return float(self.weights(beta) @ o)

    def canonical(self, beta, n_sites: int) -> dict[str, np.ndarray]:
        """Energy and specific heat per site at each ``beta``."""
        beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        u = np.empty_like(beta)
        c = np.empty_like(beta)
        for i, b in enumerate(beta):
            w = self.weights(b)
            m1 = w @ self.energy
            m2 = w @ (self.energy**2)
            u[i] = m1 / n_sites
            c[i] = b * b * (m2 - m1 * m1) / n_sites
        return {"u": u, "c": c}

    def ess(self, beta: float) -> float:
        return kish_ess(self._sample_log_weights(beta))
