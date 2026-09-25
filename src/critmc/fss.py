"""Finite-size-scaling analysis with jackknife error propagation.

All canonical runs of one lattice size are combined by multi-histogram
reweighting (:class:`SizeData`).  Every estimator is a function of that
combined ensemble; its statistical error comes from a delete-one-block
jackknife in which block ``b`` is removed from *every* run of the size and the
reweighting equations are solved again.  Because runs of different sizes are
independent, estimators involving two sizes (e.g. Binder crossings) add the
jackknife variances of both sizes.

Scaling forms used (``t = beta - beta_c``):

* ``U(beta_c, L) -> U*``, crossings ``beta*(L, 2L) - beta_c ~ L^{-1/nu - omega}``;
* ``|dU/dbeta|``, ``d ln<|m|^k>/dbeta ~ L^{1/nu}``;
* ``chi(beta_c) ~ L^{gamma/nu}``, ``<|m|>(beta_c) ~ L^{-beta/nu}``;
* data collapse ``O L^{-kappa} = F(t L^{1/nu})`` scored by the quality
  function of J. Houdayer & A. K. Hartmann, PRB 70, 014418 (2004).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.optimize import brentq

from .reweight import MultiHistogram, observables_from_weights
from .sim import TimeSeries

__all__ = [
    "Estimate",
    "SizeData",
    "binder_crossing",
    "peak",
    "FitResult",
    "fit",
    "fit_power_law",
    "fit_log",
    "fit_shift",
    "collapse_quality",
    "data_collapse",
    "CollapseResult",
]


@dataclass(frozen=True)
class Estimate:
    value: float
    error: float

    def __iter__(self):
        yield self.value
        yield self.error

    def __format__(self, spec: str) -> str:
        return format_value(self.value, self.error)

    def __str__(self) -> str:
        return format_value(self.value, self.error)


def format_value(value: float, error: float, digits: int = 2) -> str:
    """``1.23456 +- 0.00321 -> '1.2346(32)'`` (parenthesis notation)."""
    if not np.isfinite(error) or error <= 0:
        return f"{value:.6g}"
    exponent = int(np.floor(np.log10(error))) - (digits - 1)
    decimals = max(0, -exponent)
    err_digits = int(round(error / 10.0**exponent))
    if err_digits >= 10**digits:  # rounding overflow, e.g. 99.6 -> 100
        exponent += 1
        decimals = max(0, -exponent)
        err_digits = int(round(error / 10.0**exponent))
    if exponent > 0:
        return f"{round(value, -exponent):.0f}({err_digits * 10**exponent})"
    return f"{value:.{decimals}f}({err_digits})"


def _jackknife(full, reps) -> tuple[np.ndarray, np.ndarray]:
    reps = np.asarray(reps, dtype=np.float64)
    full = np.asarray(full, dtype=np.float64)
    n = len(reps)
    mean = reps.mean(axis=0)
    value = n * full - (n - 1) * mean
    err = np.sqrt((n - 1) / n * np.sum((reps - mean) ** 2, axis=0))
    return value, err


@dataclass
class SizeData:
    """Canonical runs of a single lattice size, ready for reweighting."""

    L: int
    n_sites: int
    betas: np.ndarray
    energy: list[np.ndarray]
    absm: list[np.ndarray]
    m2: list[np.ndarray]
    m4: list[np.ndarray]
    n_blocks: int = 32
    inefficiency: np.ndarray | None = None
    _full: MultiHistogram | None = field(default=None, repr=False)
    _reps: list[MultiHistogram] | None = field(default=None, repr=False)

    @classmethod
    def from_runs(
        cls,
        runs: Sequence[TimeSeries],
        n_blocks: int = 32,
        inefficiency: Sequence[float] | None = None,
    ) -> SizeData:
        if not runs:
            raise ValueError("need at least one run")
        L = runs[0].L
        n = runs[0].n_sites
        if any(r.L != L or r.n_sites != n for r in runs):
            raise ValueError("all runs must share the lattice size")
        return cls(
            L=L,
            n_sites=n,
            betas=np.array([r.beta for r in runs]),
            energy=[np.asarray(r["E"], dtype=np.float64) for r in runs],
            absm=[np.asarray(r["absm"]) for r in runs],
            m2=[np.asarray(r["m2"]) for r in runs],
            m4=[np.asarray(r["m4"]) for r in runs],
            n_blocks=n_blocks,
            inefficiency=None if inefficiency is None else np.asarray(inefficiency, dtype=float),
        )

    # -- reweighting ensembles ------------------------------------------------
    def _build(self, keep: list[np.ndarray] | None) -> MultiHistogram:
        sel = (lambda a, i: a) if keep is None else (lambda a, i: a[keep[i]])
        energies = [sel(e, i) for i, e in enumerate(self.energy)]
        obs = [
            {"absm": sel(self.absm[i], i), "m2": sel(self.m2[i], i), "m4": sel(self.m4[i], i)}
            for i in range(len(self.energy))
        ]
        return MultiHistogram(energies, self.betas, obs, self.inefficiency)

    @property
    def full(self) -> MultiHistogram:
        if self._full is None:
            self._full = self._build(None)
        return self._full

    @property
    def replicates(self) -> list[MultiHistogram]:
        if self._reps is None:
            reps = []
            for b in range(self.n_blocks):
                keep = []
                for e in self.energy:
                    n = len(e)
                    size = n // self.n_blocks
                    start = n - size * self.n_blocks  # drop the oldest remainder
                    mask = np.zeros(n, dtype=bool)
                    mask[start:] = True
                    mask[start + b * size : start + (b + 1) * size] = False
                    keep.append(mask)
                reps.append(self._build(keep))
            self._reps = reps
        return self._reps

    def reliable_window(self, min_fraction: float = 0.05, n_grid: int = 241) -> tuple[float, float]:
        """Interval of ``beta`` around the runs where reweighting stays trustworthy.

        The Kish effective sample size of the combined ensemble is scanned on a
        grid; the window is the contiguous region around its maximum where it
        stays above ``min_fraction`` of that maximum.  Estimators that search
        in ``beta`` (crossings, peaks, collapses) should stay inside it.
        """
        b = self.betas
        e = np.concatenate(self.energy)
        pad = max(b.max() - b.min(), 3.0 / max(float(e.std()), 1e-12))
        grid = np.linspace(b.min() - pad, b.max() + pad, n_grid)
        ess = np.array([self.full.ess(x) for x in grid])
        ok = ess >= min_fraction * ess.max()
        i = int(np.argmax(ess))
        lo = i
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        hi = i
        while hi < n_grid - 1 and ok[hi + 1]:
            hi += 1
        return float(grid[lo]), float(grid[hi])

    def observables(self, beta: float, mh: MultiHistogram | None = None) -> dict[str, float]:
        mh = self.full if mh is None else mh
        w = mh.weights(beta)
        return observables_from_weights(
            w, mh.energy, mh.obs["absm"], mh.obs["m2"], mh.obs["m4"], beta, self.n_sites
        )

    # -- estimators -----------------------------------------------------------
    def replicate_values(self, fn: Callable[[MultiHistogram], object]):
        """``(fn(full ensemble), [fn(replicate) ...])``."""
        return fn(self.full), [fn(mh) for mh in self.replicates]

    def estimate(self, fn: Callable[[MultiHistogram], float]) -> Estimate:
        full, reps = self.replicate_values(fn)
        value, err = _jackknife(full, reps)
        return Estimate(float(value), float(err))

    def at(self, beta: float, keys: Sequence[str] | None = None) -> dict[str, Estimate]:
        """Jackknife estimates of the standard observables at ``beta``."""
        keys = list(keys) if keys is not None else list(self.observables(beta))
        full, reps = self.replicate_values(lambda mh: self.observables(beta, mh))
        out = {}
        for k in keys:
            value, err = _jackknife(full[k], [r[k] for r in reps])
            out[k] = Estimate(float(value), float(err))
        return out

    def curve(self, betas: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(value, error, replicate matrix)`` of observable ``key`` on a beta grid."""

        def fn(mh):
            return np.array([self.observables(b, mh)[key] for b in betas])

        full, reps = self.replicate_values(fn)
        value, err = _jackknife(full, reps)
        return value, err, np.asarray(reps)


def _root(fn, bracket, expand: int = 6):
    a, b = bracket
    fa, fb = fn(a), fn(b)
    width = b - a
    for _ in range(expand):
        if np.sign(fa) != np.sign(fb):
            return brentq(fn, a, b, xtol=1e-12)
        a, b = a - width / 2, b + width / 2
        width = b - a
        fa, fb = fn(a), fn(b)
    raise ValueError("no sign change found; widen the bracket")


def binder_crossing(
    d1: SizeData, d2: SizeData, bracket: tuple[float, float], key: str = "U"
) -> dict[str, Estimate]:
    """Crossing ``beta*`` of ``key`` (default the Binder cumulant) between two sizes.

    Returns jackknife estimates of ``beta*`` and of the cumulant at the crossing.
    """

    def solve(mh1, mh2):
        diff = lambda b: d1.observables(b, mh1)[key] - d2.observables(b, mh2)[key]  # noqa: E731
        b = _root(diff, bracket)
        return np.array([b, d1.observables(b, mh1)[key]])

    full = solve(d1.full, d2.full)
    var = np.zeros(2)
    corr = np.zeros(2)
    for side in (0, 1):
        reps = [
            solve(mh, d2.full) if side == 0 else solve(d1.full, mh)
            for mh in (d1.replicates if side == 0 else d2.replicates)
        ]
        value, err = _jackknife(full, reps)
        var += err**2
        corr += value - full
    value = full + corr
    return {
        "beta": Estimate(float(value[0]), float(np.sqrt(var[0]))),
        key: Estimate(float(value[1]), float(np.sqrt(var[1]))),
    }


def peak(data: SizeData, key: str, bracket: tuple[float, float]) -> dict[str, Estimate]:
    """Location and height of the maximum of observable ``key`` inside ``bracket``."""

    def solve(mh):
        f = lambda b: -data.observables(b, mh)[key]  # noqa: E731
        grid = np.linspace(*bracket, 41)
        vals = [f(b) for b in grid]
        i = int(np.clip(np.argmin(vals), 1, len(grid) - 2))
        # bounded Brent search in the two grid cells around the coarse maximum
        res = optimize.minimize_scalar(
            f, bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-12}
        )
        return np.array([res.x, -res.fun])

    full, reps = data.replicate_values(solve)
    value, err = _jackknife(full, reps)
    return {
        "beta": Estimate(float(value[0]), float(err[0])),
        key: Estimate(float(value[1]), float(err[1])),
    }


@dataclass(frozen=True)
class FitResult:
    """Weighted least-squares fit with parameter covariance and goodness of fit."""

    params: np.ndarray
    errors: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    names: tuple[str, ...]

    @property
    def p_value(self) -> float:
        return float(stats.chi2.sf(self.chi2, self.dof)) if self.dof > 0 else float("nan")

    @property
    def chi2_dof(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def __getitem__(self, name: str) -> Estimate:
        i = self.names.index(name)
        return Estimate(float(self.params[i]), float(self.errors[i]))

    def __str__(self) -> str:
        body = ", ".join(
            f"{n}={format_value(p, e)}" for n, p, e in zip(self.names, self.params, self.errors)
        )
        return f"{body}; chi2/dof={self.chi2:.2f}/{self.dof}"


def fit(model: Callable, x, y, dy, p0, names: Sequence[str]) -> FitResult:
    """Fit ``y = model(x, *p)`` with standard errors ``dy`` (absolute sigma)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    if y.size < len(p0):  # count observations, not rows of a multi-dimensional x
        raise ValueError("fewer data points than parameters")
    popt, pcov = optimize.curve_fit(model, x, y, p0=p0, sigma=dy, absolute_sigma=True, maxfev=20000)
    resid = (y - model(x, *popt)) / dy
    return FitResult(
        popt, np.sqrt(np.diag(pcov)), pcov, float(resid @ resid), y.size - len(p0), tuple(names)
    )


def fit_power_law(L, y, dy, omega: float | None = None) -> FitResult:
    """``y = a L^b`` or, with ``omega``, ``y = a L^b (1 + c L^-omega)``."""
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(np.log(L), np.log(np.abs(y)), 1)
    a0 = np.sign(y[0]) * np.exp(intercept)
    if omega is None:
        return fit(lambda x, a, b: a * x**b, L, y, dy, [a0, slope], ("a", "exponent"))
    return fit(
        lambda x, a, b, c: a * x**b * (1 + c * x ** (-omega)),
        L,
        y,
        dy,
        [a0, slope, 0.0],
        ("a", "exponent", "c"),
    )


def fit_log(L, y, dy) -> FitResult:
    """``y = a ln L + b`` (e.g. the specific-heat peak of the 2D Ising model)."""
    return fit(lambda x, a, b: a * np.log(x) + b, L, y, dy, [1.0, 0.0], ("a", "b"))


def fit_shift(L, beta, dbeta, exponent: float | None = None) -> FitResult:
    """``beta(L) = beta_c + a L^-x`` with ``x`` free, or fixed to ``exponent``."""
    L = np.asarray(L, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if exponent is None:
        return fit(
            lambda x, bc, a, e: bc + a * x ** (-e),
            L,
            beta,
            dbeta,
            [beta[-1], (beta[0] - beta[-1]) * L[0], 1.0],
            ("beta_c", "a", "x"),
        )
    return fit(
        lambda x, bc, a: bc + a * x ** (-exponent),
        L,
        beta,
        dbeta,
        [beta[-1], (beta[0] - beta[-1]) * L[0] ** exponent],
        ("beta_c", "a"),
    )


# ---------------------------------------------------------------------------
# data collapse
# ---------------------------------------------------------------------------
def _scaled(curves, beta_c, inv_nu, kappa):
    out = []
    for L, (b, y, dy) in sorted(curves.items()):
        x = (np.asarray(b) - beta_c) * L**inv_nu
        s = float(L) ** (-kappa)
        order = np.argsort(x)
        out.append((x[order], np.asarray(y)[order] * s, np.asarray(dy)[order] * s))
    return out


def collapse_quality(curves: dict[int, tuple], beta_c: float, inv_nu: float, kappa: float) -> float:
    """Houdayer-Hartmann quality ``S`` of ``y L^-kappa`` against ``(beta - beta_c) L^inv_nu``.

    ``curves`` maps ``L -> (betas, y, dy)``.  ``S ~ 1`` for a collapse consistent
    with the error bars.
    """
    scaled = _scaled(curves, beta_c, inv_nu, kappa)
    total = 0.0
    count = 0
    for p, (xp, yp, dyp) in enumerate(scaled):
        for i in range(len(xp)):
            xi = xp[i]
            px, py, pw = [], [], []
            for q, (xq, yq, dyq) in enumerate(scaled):
                if q == p or xi < xq[0] or xi > xq[-1]:
                    continue
                j = int(np.clip(np.searchsorted(xq, xi) - 1, 0, len(xq) - 2))
                for k in (j, j + 1):
                    px.append(xq[k])
                    py.append(yq[k])
                    pw.append(1.0 / dyq[k] ** 2)
            if len(px) < 2:
                continue
            px, py, pw = map(np.asarray, (px, py, pw))
            k0, kx, ky = pw.sum(), pw @ px, pw @ py
            kxx, kxy = pw @ (px * px), pw @ (px * py)
            det = k0 * kxx - kx * kx
            if det <= 0:
                continue
            yfit = (kxx * ky - kx * kxy) / det + xi * (k0 * kxy - kx * ky) / det
            var_fit = (kxx - 2 * xi * kx + xi * xi * k0) / det
            total += (yp[i] - yfit) ** 2 / (dyp[i] ** 2 + var_fit)
            count += 1
    return total / count if count else np.inf


@dataclass(frozen=True)
class CollapseResult:
    beta_c: float
    inv_nu: float
    kappa: float
    quality: float


def data_collapse(
    curves: dict[int, tuple], beta_c: float, inv_nu: float, kappa: float, fixed: Sequence[str] = ()
) -> CollapseResult:
    """Minimise the collapse quality over ``beta_c``, ``inv_nu`` and ``kappa`` (Nelder-Mead)."""
    names = ["beta_c", "inv_nu", "kappa"]
    start = {"beta_c": beta_c, "inv_nu": inv_nu, "kappa": kappa}
    free = [n for n in names if n not in fixed]

    def unpack(p):
        vals = dict(start)
        vals.update(zip(free, p))
        return vals

    def objective(p):
        v = unpack(p)
        return collapse_quality(curves, v["beta_c"], v["inv_nu"], v["kappa"])

    x0 = np.array([start[n] for n in free])
    simplex = [x0] + [
        x0 + np.eye(len(x0))[i] * max(1e-3, 0.02 * abs(x0[i])) for i in range(len(x0))
    ]
    res = optimize.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": np.array(simplex),
            "xatol": 1e-8,
            "fatol": 1e-10,
            "maxiter": 4000,
        },
    )
    v = unpack(res.x)
    return CollapseResult(v["beta_c"], v["inv_nu"], v["kappa"], float(res.fun))
