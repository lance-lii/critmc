"""Autocorrelation analysis of Markov-chain time series.

Successive Monte Carlo measurements are correlated.  For a stationary series
with autocovariance ``Gamma(t)`` the variance of the sample mean is
``2 tau_int Gamma(0) / N`` with the integrated autocorrelation time

    tau_int = 1/2 + sum_{t >= 1} rho(t),        rho(t) = Gamma(t) / Gamma(0),

which has to be estimated with a finite summation window ``W``, because the
noise of the estimated ``rho(t)`` does not decay with ``t``.  This module provides

* FFT estimators of ``Gamma(t)`` and ``rho(t)`` (Wiener-Khinchin theorem; zero
  padding to at least ``2N`` removes the circular wrap-around);
* :func:`tau_int_sokal`, the self-consistent window ``W >= c tau_int(W)`` of
  N. Madras and A. D. Sokal, J. Stat. Phys. 50, 109 (1988), with the error
  ``var(tau_int) ~ 2 (2W + 1) tau_int^2 / N`` (A. D. Sokal, "Monte Carlo methods
  in statistical mechanics: foundations and new algorithms", Cargese lectures
  1996, in Functional Integration, NATO ASI Series B 361, Plenum, 1997);
* :func:`gamma_method`, the Gamma-method of U. Wolff, "Monte Carlo errors with
  less errors", Comput. Phys. Commun. 156, 143 (2004) [Erratum: 176, 383 (2007)],
  hep-lat/0306017.  It follows his ``UWerr`` routine: errors of derived
  quantities ``f(A_1, A_2, ...)`` from the fluctuations projected onto the
  gradient of ``f``, automatic windowing with parameter ``S``, the bias
  correction of ``Gamma_F`` and the combination of independent replicas.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaincc

__all__ = [
    "AutocorrTime",
    "GammaResult",
    "autocovariance",
    "autocorrelation",
    "tau_int_sokal",
    "gamma_method",
    "effective_sample_size",
]

# tau_W used when the summed autocorrelation is not positive (UWerr uses eps as well)
_TINY = float(np.finfo(float).eps)


@dataclass(frozen=True, eq=False)
class AutocorrTime:
    """Integrated autocorrelation time from :func:`tau_int_sokal`.

    Attributes
    ----------
    tau:
        ``tau_int`` at the selected window.
    dtau:
        Statistical error of ``tau``, ``|tau| sqrt(2 (2W + 1) / N)``.
    window:
        Selected summation window ``W``.
    rho:
        Normalised autocorrelation function ``rho(t)`` for ``t = 0..max_lag``.
    tau_curve:
        ``tau_int(W) = 1/2 + sum_{t=1}^{W} rho(t)`` for ``W = 0..max_lag``.
    converged:
        ``False`` if no ``W <= max_lag`` satisfied ``W >= c tau_int(W)``; the
        window is then ``max_lag`` and ``tau`` is unreliable.
    """

    tau: float
    dtau: float
    window: int
    rho: np.ndarray
    tau_curve: np.ndarray
    converged: bool


@dataclass(frozen=True, eq=False)
class GammaResult:
    """Result of Wolff's Gamma-method, :func:`gamma_method`.

    Attributes
    ----------
    value:
        Estimate of ``F = f(A)``; with several replicas the ``O(1/N)`` bias
        estimated from the replica means is subtracted.
    error:
        Statistical error ``sigma_F`` (NaN if the estimated ``C_F`` is not
        positive, which only happens for pathologically anti-correlated data).
    error_of_error:
        Uncertainty of ``error``, ``sigma_F sqrt((W + 1/2) / N)``.
    tau_int:
        Integrated autocorrelation time of the projected fluctuations.
    dtau_int:
        Error of ``tau_int``, ``2 tau_int sqrt((W - tau_int + 1/2) / N)``.
    window:
        Optimal summation window ``W_opt``.
    rho:
        Bias-corrected normalised autocorrelation ``Gamma_F(t) / Gamma_F(0)`` for
        ``t = 0..min(N_min // 2, max(2 W_opt, 10))``.
    tau_curve:
        ``tau_int(W)`` for the same range of ``W``.
    n:
        Total number of samples ``N`` (all replicas).
    n_replicas:
        Number of replicas ``R``.
    converged:
        ``False`` if the windowing criterion was never met; ``window`` is then
        the search cap and the error is probably underestimated.
    q_value:
        Goodness-of-fit probability of the spread of the replica estimates
        (``None`` for a single replica).
    names:
        Labels of the primary observables, if given.
    """

    value: float
    error: float
    error_of_error: float
    tau_int: float
    dtau_int: float
    window: int
    rho: np.ndarray
    tau_curve: np.ndarray
    n: int
    n_replicas: int
    converged: bool
    q_value: float | None = None
    names: tuple[str, ...] | None = None


# --------------------------------------------------------------------------------------
# input validation (also used by the blocking and resampling modules)
# --------------------------------------------------------------------------------------


def _as_int(value: Any, name: str, minimum: int | None = None) -> int:
    """Validate an integer argument; bools and floats are rejected."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        out = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _as_real(value: Any, name: str) -> float:
    """Validate a finite real scalar argument."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def _as_series(x: Any, name: str = "x", *, min_len: int = 2, nd: bool = False) -> np.ndarray:
    """Finite float array with samples along axis 0 (1-D unless ``nd``)."""
    try:
        arr = np.asarray(x)
    except ValueError as exc:  # ragged nested sequences
        raise ValueError(f"{name} must be a rectangular array of numbers") from exc
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numbers, got dtype {arr.dtype}")
    if arr.ndim == 0 or (arr.ndim > 1 and not nd):
        expected = "an array with samples along axis 0" if nd else "a 1-D array"
        raise ValueError(f"{name} must be {expected}, got shape {arr.shape}")
    if arr.shape[0] < min_len:
        raise ValueError(f"{name} needs at least {min_len} samples, got {arr.shape[0]}")
    arr = arr.astype(float, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


# --------------------------------------------------------------------------------------
# autocovariance
# --------------------------------------------------------------------------------------


def _centred(x: np.ndarray) -> np.ndarray:
    """Deviations from the mean along axis 0, exactly zero for constant columns."""
    return np.where(np.ptp(x, axis=0) == 0, 0.0, x - x.mean(axis=0))


def _lagged_sums(x: np.ndarray, max_lag: int) -> np.ndarray:
    """``s(t) = sum_i x[i] x[i + t]`` for ``t = 0..max_lag`` along axis 0 (no centring)."""
    n = x.shape[0]
    n_fft = 1 << (2 * n - 1).bit_length()  # power of two >= 2n: no circular aliasing
    spec = np.fft.rfft(x, n=n_fft, axis=0)
    return np.fft.irfft(spec.real**2 + spec.imag**2, n=n_fft, axis=0)[: max_lag + 1]


def autocovariance(x: ArrayLike, max_lag: int | None = None) -> np.ndarray:
    """Autocovariance of a time series, computed by FFT.

    ``Gamma(t) = 1/(N - t) sum_{i=0}^{N-1-t} (x_i - xbar) (x_{i+t} - xbar)``.

    Parameters
    ----------
    x:
        Samples along axis 0, shape ``(N,)`` or ``(N, ...)``; columns are treated
        independently.
    max_lag:
        Largest lag, ``0 <= max_lag <= N - 1``.  The default ``N // 2`` leaves out
        the tail, where each estimate averages only a few products and is pure
        noise.

    Returns
    -------
    numpy.ndarray
        ``Gamma(t)`` for ``t = 0..max_lag``, shape ``(max_lag + 1,) + x.shape[1:]``.
    """
    arr = _as_series(x, nd=True)
    n = arr.shape[0]
    max_lag = n // 2 if max_lag is None else _as_int(max_lag, "max_lag", 0)
    if max_lag > n - 1:
        raise ValueError(f"max_lag must be <= N - 1 = {n - 1}, got {max_lag}")
    sums = _lagged_sums(_centred(arr), max_lag)
    norm = n - np.arange(max_lag + 1)
    return sums / norm.reshape((-1,) + (1,) * (arr.ndim - 1))


def autocorrelation(x: ArrayLike, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation ``rho(t) = Gamma(t) / Gamma(0)``.

    Parameters are those of :func:`autocovariance`.  Raises ``ValueError`` for a
    constant series, whose autocorrelation is undefined.
    """
    gamma = autocovariance(x, max_lag)
    if np.any(gamma[0] <= 0):
        raise ValueError("x is constant (zero variance); its autocorrelation is undefined")
    return gamma / gamma[0]


# --------------------------------------------------------------------------------------
# Madras-Sokal windowing
# --------------------------------------------------------------------------------------


def tau_int_sokal(x: ArrayLike, c: float = 6.0, *, max_lag: int | None = None) -> AutocorrTime:
    """Integrated autocorrelation time with the Madras-Sokal automatic window.

    ``tau_int(W) = 1/2 + sum_{t=1}^{W} rho(t)`` is evaluated for all ``W`` up to
    ``max_lag`` and the window is the smallest ``W`` with ``W >= c tau_int(W)``,
    which balances the truncation bias ``~ exp(-W / tau)`` against the
    statistical error ``tau sqrt(2 (2W + 1) / N)``.

    Parameters
    ----------
    x:
        1-D time series.
    c:
        Window factor; ``c ~ 6`` suits a roughly exponential decay of ``rho``.
    max_lag:
        Largest lag considered, default ``N // 2``.

    Returns
    -------
    AutocorrTime

    Raises
    ------
    ValueError
        For invalid input, including a constant series.
    """
    arr = _as_series(x)
    c = _as_real(c, "c")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    rho = autocorrelation(arr, max_lag)
    tau_curve = np.cumsum(rho) - 0.5  # tau(W) = sum_{t=0}^{W} rho(t) - 1/2
    hits = np.flatnonzero(np.arange(rho.size) >= c * tau_curve)
    converged = hits.size > 0
    window = int(hits[0]) if converged else rho.size - 1
    tau = float(tau_curve[window])
    dtau = abs(tau) * np.sqrt(2.0 * (2 * window + 1) / arr.size)
    return AutocorrTime(
        tau=tau, dtau=float(dtau), window=window, rho=rho, tau_curve=tau_curve, converged=converged
    )


# --------------------------------------------------------------------------------------
# Wolff's Gamma-method
# --------------------------------------------------------------------------------------


def _is_number(item: Any) -> bool:
    return np.isscalar(item) or (isinstance(item, np.ndarray) and item.ndim == 0)


def _as_replicas(data: Any) -> list[np.ndarray]:
    """Normalise ``data`` to a list of ``(N_r, n_primary)`` float arrays."""
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("data is empty")
        # a plain sequence of numbers is one series; otherwise every item is a replica
        items = [data] if all(_is_number(item) for item in data) else list(data)
    else:
        items = [data]
    replicas = []
    for r, item in enumerate(items):
        name = "data" if len(items) == 1 else f"replica {r}"
        arr = _as_series(item, name, nd=True)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError(f"{name} must have shape (N,) or (N, n_primary), got {arr.shape}")
        if arr.shape[1] == 0:
            raise ValueError(f"{name} contains no primary observables")
        replicas.append(arr)
    n_primary = sorted({arr.shape[1] for arr in replicas})
    if len(n_primary) > 1:
        raise ValueError(f"replicas differ in their number of primary observables: {n_primary}")
    return replicas


def _first(a: np.ndarray) -> float:
    return a[0]


def _eval_scalar(f: Callable[[np.ndarray], Any], a: np.ndarray) -> float:
    """Evaluate ``f`` at primary means ``a``; it must return a finite scalar."""
    res = f(a.copy())
    try:
        out = np.asarray(res, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"f must return a real scalar, got {res!r}") from None
    if out.size != 1:
        raise ValueError(f"f must return a scalar, got an array of shape {out.shape}")
    val = float(out.reshape(()))
    if not np.isfinite(val):
        raise ValueError(f"f returned {val} at primary means {a}")
    return val


def _gradient(
    f: Callable[[np.ndarray], Any], abar: np.ndarray, var: np.ndarray, n: int
) -> np.ndarray:
    """Symmetric finite-difference gradient of ``f`` at ``abar``, steps ``sqrt(var / N)``."""
    h = np.sqrt(var / n)
    # fallback step for primaries without fluctuations (or below floating-point resolution)
    unresolved = (abar + h) - (abar - h) <= 0
    h = np.where(unresolved, 1e-8 * np.maximum(1.0, np.abs(abar)), h)
    grad = np.empty(abar.size)
    for alpha in range(abar.size):
        up, down = abar.copy(), abar.copy()
        up[alpha] += h[alpha]
        down[alpha] -= h[alpha]
        grad[alpha] = (_eval_scalar(f, up) - _eval_scalar(f, down)) / (up[alpha] - down[alpha])
    return grad


def _wolff_window(gamma: np.ndarray, S: float, n: int) -> tuple[int, bool]:
    """First ``W`` with ``g(W) = exp(-W / tau_W) - tau_W / sqrt(W N) < 0``."""
    w = np.arange(1, gamma.size)
    g_int = np.cumsum(gamma[1:]) / gamma[0]
    # tau_W = S tau, where exp(-t / tau) is the exponential decay whose sum over t >= 1
    # equals the measured G(W) = sum_{t=1}^{W} rho(t)
    tau_w = np.full(w.size, _TINY)
    pos = g_int > 0
    with np.errstate(divide="ignore", over="ignore"):
        tau_w[pos] = S / np.log1p(1.0 / g_int[pos])
    tau_w = np.maximum(tau_w, _TINY)
    g = np.exp(-w / tau_w) - tau_w / np.sqrt(w * float(n))
    hits = np.flatnonzero(g < 0)
    if hits.size:
        return int(w[hits[0]]), True
    return (int(w[-1]) if w.size else 0), False


def gamma_method(
    data: ArrayLike | Sequence[ArrayLike],
    f: Callable[[np.ndarray], float] | None = None,
    *,
    S: float = 1.5,
    names: Sequence[str] | None = None,
    max_window: int | None = None,
) -> GammaResult:
    """Error and autocorrelation time of ``F = f(A_1, ..., A_n)`` by Wolff's Gamma-method.

    The fluctuations of the primary observables are projected onto the gradient
    of ``f``, ``delta_F(i) = sum_a f_a (a_a^i - abar_a)``, whose autocorrelation
    function ``Gamma_F(t)`` (summed over replicas) is integrated up to the first
    window ``W`` with ``g(W) = exp(-W / tau_W) - tau_W / sqrt(W N) < 0``.  This
    minimises the sum of the truncation error ``~ exp(-W / tau_W)`` and the
    statistical error ``~ sqrt(W / N)``, where ``tau_W = S tau`` and ``tau`` is the
    time constant of the exponential decay with the measured ``sum_{t=1}^{W} rho(t)``.
    ``Gamma_F`` is then corrected for the bias caused by subtracting the sample
    mean, ``Gamma_F(t) += C_F / N``.

    Parameters
    ----------
    data:
        Primary observables, shape ``(N,)`` or ``(N, n_primary)``, or a list of such
        arrays holding independent replicas (possibly of different lengths) of
        the same Markov process.
    f:
        Derived quantity: called with the 1-D array of primary means, returns a
        scalar.  Default: the mean of the single primary observable.
    S:
        Windowing parameter, ``S ~ 1..2`` (Wolff recommends 1.5); ``S = 0`` assumes
        no autocorrelations (``W_opt = 0``), as in ``UWerr``.
    names:
        Optional labels of the primary observables, stored in the result.
    max_window:
        Optional cap on the window search, which otherwise runs up to half the
        length of the shortest replica.

    Returns
    -------
    GammaResult

    Examples
    --------
    Binder-type ratio ``<m^4> / <m^2>^2`` of a magnetisation time series ``m``::

        res = gamma_method(np.column_stack((m**2, m**4)), lambda a: a[1] / a[0] ** 2)
    """
    replicas = _as_replicas(data)
    n_primary = replicas[0].shape[1]
    n_rep = len(replicas)
    lengths = np.array([r.shape[0] for r in replicas])
    n_tot = int(lengths.sum())
    if f is None and n_primary != 1:
        raise ValueError(f"f is required for data with {n_primary} primary observables")
    if f is not None and not callable(f):
        raise ValueError("f must be a callable taking the array of primary means")
    S = _as_real(S, "S")
    if S < 0:
        raise ValueError(f"S must be >= 0, got {S}")
    if max_window is not None:
        max_window = _as_int(max_window, "max_window", 1)
    if names is not None:
        names = (names,) if isinstance(names, str) else tuple(str(s) for s in names)
        if len(names) != n_primary:
            raise ValueError(f"got {len(names)} names for {n_primary} primary observables")

    x = replicas[0] if n_rep == 1 else np.concatenate(replicas, axis=0)
    const = np.ptp(x, axis=0) == 0
    abar = np.where(const, x[0], x.mean(axis=0))  # grand means
    dev = np.where(const, 0.0, x - abar)
    func = _first if f is None else f
    f_bar = _eval_scalar(func, abar)
    grad = np.ones(1) if f is None else _gradient(func, abar, np.mean(dev**2, axis=0), n_tot)
    delta = dev @ grad  # projected fluctuations delta_F(i)

    # Gamma_F(t) = sum_r sum_i delta_r(i) delta_r(i + t) / (N - R t)
    t_max = int(lengths.min()) // 2
    gamma = np.zeros(t_max + 1)
    for part in np.split(delta, np.cumsum(lengths)[:-1]):
        gamma += _lagged_sums(part, t_max)
    gamma /= n_tot - n_rep * np.arange(t_max + 1)

    if not gamma[0] > 0:  # no fluctuations: the result is exact
        return GammaResult(
            value=f_bar,
            error=0.0,
            error_of_error=0.0,
            tau_int=0.5,
            dtau_int=0.0,
            window=0,
            rho=np.ones(1),
            tau_curve=np.full(1, 0.5),
            n=n_tot,
            n_replicas=n_rep,
            converged=True,
            names=names,
        )

    if S == 0:
        w_opt, converged = 0, True
    else:
        w_cap = t_max if max_window is None else min(t_max, max_window)
        w_opt, converged = _wolff_window(gamma[: w_cap + 1], S, n_tot)

    c_f = gamma[0] + 2.0 * gamma[1 : w_opt + 1].sum()
    # subtracting the sample mean biases every Gamma_F(t) by about -C_F / N
    gamma = gamma + c_f / n_tot
    c_f = gamma[0] + 2.0 * gamma[1 : w_opt + 1].sum()
    tau_int = c_f / (2.0 * gamma[0])
    error = float(np.sqrt(c_f / n_tot)) if c_f > 0 else float("nan")
    n_out = min(t_max, max(2 * w_opt, 10))
    rho = gamma[: n_out + 1] / gamma[0]

    value, q_value = f_bar, None
    if n_rep > 1:
        rep_means = [np.where(const, abar, r.mean(axis=0)) for r in replicas]
        f_rep = np.array([_eval_scalar(func, a) for a in rep_means])
        f_rep_mean = float(lengths @ f_rep) / n_tot
        # replica means carry a bias ~ B / N_r, the grand mean ~ B / N
        bias = (f_rep_mean - f_bar) / (n_rep - 1)
        value = f_bar - bias
        if c_f > 0:
            spread = (f_rep - bias * n_tot / lengths) - (f_rep_mean - bias * n_rep)
            chi2 = float(lengths @ spread**2) / c_f
            q_value = float(gammaincc(0.5 * (n_rep - 1), 0.5 * chi2))

    return GammaResult(
        value=float(value),
        error=error,
        error_of_error=error * float(np.sqrt((w_opt + 0.5) / n_tot)),
        tau_int=float(tau_int),
        dtau_int=float(2.0 * abs(tau_int) * np.sqrt(max(w_opt - tau_int + 0.5, 0.0) / n_tot)),
        window=w_opt,
        rho=rho,
        tau_curve=np.cumsum(rho) - 0.5,
        n=n_tot,
        n_replicas=n_rep,
        converged=converged,
        q_value=q_value,
        names=names,
    )


def effective_sample_size(x: ArrayLike | Sequence[ArrayLike], *, S: float = 1.5) -> float:
    """Number of effectively independent samples, ``N / (2 tau_int)``.

    ``tau_int`` comes from :func:`gamma_method`; ``x`` is a 1-D series or a list of
    replicas.  Returns NaN if the estimated ``tau_int`` is not positive.
    """
    res = gamma_method(x, S=S)
    return res.n / (2.0 * res.tau_int) if res.tau_int > 0 else float("nan")
