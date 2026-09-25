"""Resampling estimates of statistical errors: jackknife and bootstrap.

The jackknife (M. H. Quenouille, Biometrika 43, 353 (1956); J. W. Tukey, Ann.
Math. Stat. 29, 614 (1958)) re-evaluates an estimator with one block of data
left out.  Deleting blocks of consecutive samples that are much longer than
the autocorrelation time makes it valid for Markov-chain time series, and it
propagates errors through arbitrary non-linear functions of several correlated
observables; see B. A. Berg, "Markov Chain Monte Carlo Simulations and Their
Statistical Analysis" (World Scientific, 2004).

The bootstrap (B. Efron, Ann. Stat. 7, 1 (1979)) resamples with replacement.
For time series whole blocks are resampled: circular moving blocks of fixed
length (H. R. Kuensch, Ann. Stat. 17, 1217 (1989); D. N. Politis and J. P.
Romano, in "Exploring the Limits of Bootstrap", Wiley, 1992) or blocks of
geometrically distributed length, the stationary bootstrap of D. N. Politis and
J. P. Romano, J. Am. Stat. Assoc. 89, 1303 (1994).  Confidence intervals include
the bias-corrected and accelerated (BCa) interval of B. Efron, J. Am. Stat.
Assoc. 82, 171 (1987); see also B. Efron and R. J. Tibshirani, "An Introduction
to the Bootstrap" (Chapman & Hall, 1993).

Samples lie along axis 0 of every data array.  Where the data are split into
equal blocks, the surplus samples are dropped from the start of the series,
where a Markov chain is least equilibrated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from .autocorr import _as_int, _as_real

__all__ = [
    "JackknifeResult",
    "BootstrapResult",
    "block_means",
    "jackknife",
    "jackknife_means",
    "jackknife_independent",
    "bootstrap",
    "stationary_bootstrap_indices",
]

_DEFAULT_BLOCKS = 64  # jackknife blocks: relative error of the error ~ 1/sqrt(2 * 63) ~ 9 %
_BCA_MAX_GROUPS = 1000  # cap on the jackknife evaluations for the BCa acceleration
_IID_BATCH = 2**16  # indices drawn per Generator call in the i.i.d. bootstrap
_METHODS = ("percentile", "basic", "normal", "bca")


@dataclass(frozen=True, eq=False)
class JackknifeResult:
    """Jackknife estimate; array-valued fields have the shape of ``f``'s output.

    Attributes
    ----------
    value:
        Bias-corrected estimate ``n theta_full - (n - 1) mean_i theta_i``.
    error:
        Standard error ``sqrt((n - 1) / n sum_i (theta_i - mean_i theta_i)^2)``.
    bias:
        Estimated bias of ``naive``, so that ``value = naive - bias``.
    naive:
        ``theta_full``: ``f`` evaluated on all (retained) data.
    replicates:
        Leave-one-block-out estimates ``theta_i``, shape ``(n_blocks, ...)``; for
        :func:`jackknife_independent` a tuple with one such array per data set.
    """

    value: float | np.ndarray
    error: float | np.ndarray
    bias: float | np.ndarray
    naive: float | np.ndarray
    replicates: np.ndarray | tuple[np.ndarray, ...]


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Bootstrap estimate; array-valued fields have the shape of ``f``'s output.

    Attributes
    ----------
    value:
        ``f`` evaluated on the original data (not bias-corrected).
    error:
        Standard deviation of the replicates (``ddof=1``).
    ci:
        ``(lower, upper)`` confidence limits.
    replicates:
        Bootstrap replicates, shape ``(n_boot, ...)``.
    method:
        Interval method: ``"percentile"``, ``"basic"``, ``"normal"`` or ``"bca"``.
    confidence:
        Confidence level of ``ci``.
    """

    value: float | np.ndarray
    error: float | np.ndarray
    ci: tuple[float | np.ndarray, float | np.ndarray]
    replicates: np.ndarray
    method: str
    confidence: float


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _as_arrays(
    arrays: Sequence[Any], label: str = "", min_len: int = 2
) -> tuple[list[np.ndarray], int]:
    """Convert data arrays and check that they share the length of axis 0."""
    if len(arrays) == 0:
        raise ValueError(f"{label}at least one data array is required")
    out = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in out):
        raise ValueError(f"{label}data arrays need samples along axis 0, got a scalar")
    lengths = [a.shape[0] for a in out]
    if len(set(lengths)) > 1:
        raise ValueError(f"{label}data arrays differ in length along axis 0: {lengths}")
    if lengths[0] < min_len:
        raise ValueError(f"{label}need at least {min_len} samples, got {lengths[0]}")
    return out, lengths[0]


def _evaluate(
    f: Callable[..., Any], args: Sequence[Any], shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """``f(*args)`` as a float array, optionally checked against the expected shape."""
    res = f(*args)
    try:
        out = np.asarray(res, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"f must return real numbers, got {type(res).__name__}") from None
    if shape is not None and out.shape != shape:
        raise ValueError(f"f returned shape {out.shape} on resampled data, {shape} on all data")
    return out


def _unwrap(a: np.ndarray) -> float | np.ndarray:
    return float(a) if a.ndim == 0 else a


def _drop(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``a`` without the samples ``lo:hi`` of axis 0."""
    if lo == 0:
        return a[hi:]
    if hi >= a.shape[0]:
        return a[:lo]
    return np.concatenate((a[:lo], a[hi:]), axis=0)


def _n_blocks(n_blocks: Any, n: int, label: str = "") -> int:
    if n_blocks is None:
        return min(n, _DEFAULT_BLOCKS)
    nb = _as_int(n_blocks, f"{label}n_blocks", 2)
    if nb > n:
        raise ValueError(f"{label}n_blocks={nb} exceeds the number of samples ({n})")
    return nb


def _jackknife_moments(theta: np.ndarray, reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bias and variance from leave-one-out replicates."""
    n = reps.shape[0]
    mean = reps.mean(axis=0)
    return (n - 1) * (mean - theta), (n - 1) / n * np.sum((reps - mean) ** 2, axis=0)


def _as_generator(rng: Any) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    seedlike = (np.random.SeedSequence, np.random.BitGenerator, int, np.integer)
    if rng is None or (isinstance(rng, seedlike) and not isinstance(rng, bool)):
        return np.random.default_rng(rng)
    raise ValueError(f"rng must be None, an integer seed or a numpy Generator, got {rng!r}")


def _mean_block(value: Any, name: str) -> float:
    mean_block = _as_real(value, name)
    if mean_block < 1:
        raise ValueError(f"{name} must be >= 1, got {mean_block}")
    return mean_block


# --------------------------------------------------------------------------------------
# jackknife
# --------------------------------------------------------------------------------------


def block_means(x: ArrayLike, n_blocks: int) -> np.ndarray:
    """Means of ``n_blocks`` equal consecutive blocks along axis 0.

    The first ``N mod n_blocks`` (oldest) samples are discarded.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_blocks,) + x.shape[1:]``.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError("x needs samples along axis 0, got a scalar")
    n = arr.shape[0]
    nb = _as_int(n_blocks, "n_blocks", 1)
    if nb > n:
        raise ValueError(f"n_blocks={nb} exceeds the number of samples ({n})")
    size = n // nb
    return arr[n - nb * size :].reshape(nb, size, *arr.shape[1:]).mean(axis=1)


def jackknife(
    f: Callable[..., Any], *arrays: ArrayLike, n_blocks: int | None = None
) -> JackknifeResult:
    """Delete-one-block jackknife of an arbitrary estimator.

    Parameters
    ----------
    f:
        ``f(*arrays)`` returning a scalar or an array.  It receives the retained
        *samples* (all blocks but one, in their original order).
    *arrays:
        Data arrays sharing axis 0, e.g. energy and magnetisation time series.
    n_blocks:
        Number of blocks, ``2 <= n_blocks <= N``; default ``min(N, 64)``.  The
        blocks must be much longer than the autocorrelation time, and ``n``
        blocks determine the error to about ``1 / sqrt(2 (n - 1))`` (9 % for 64).
        ``n_blocks = N`` is the ordinary delete-one jackknife.

    Returns
    -------
    JackknifeResult
    """
    data, n = _as_arrays(arrays)
    nb = _n_blocks(n_blocks, n)
    size = n // nb
    data = [a[n - nb * size :] for a in data]
    theta = _evaluate(f, data)
    reps = np.empty((nb, *theta.shape))
    for i in range(nb):
        reps[i] = _evaluate(f, [_drop(a, i * size, (i + 1) * size) for a in data], theta.shape)
    bias, var = _jackknife_moments(theta, reps)
    return JackknifeResult(
        value=_unwrap(theta - bias),
        error=_unwrap(np.sqrt(var)),
        bias=_unwrap(bias),
        naive=_unwrap(theta),
        replicates=reps,
    )


def jackknife_means(f: Callable[..., Any], *block_mean_arrays: ArrayLike) -> JackknifeResult:
    """Jackknife of a function of means, from block means.

    Much faster than :func:`jackknife` when the estimator depends on the data
    only through means: ``f`` receives, for every array, the mean over the
    retained blocks, obtained as ``(total - block) / (n - 1)``.

    Parameters
    ----------
    f:
        ``f(*means)`` returning a scalar or an array.
    *block_mean_arrays:
        Means of equally sized blocks (see :func:`block_means`), sharing axis 0
        of length ``n >= 2``.

    Returns
    -------
    JackknifeResult
    """
    data, n = _as_arrays(block_mean_arrays)
    data = [np.asarray(a, dtype=float) for a in data]
    totals = [a.sum(axis=0) for a in data]
    theta = _evaluate(f, [t / n for t in totals])
    loo = [(t - a) / (n - 1) for t, a in zip(totals, data, strict=True)]
    reps = np.empty((n, *theta.shape))
    for i in range(n):
        reps[i] = _evaluate(f, [m[i] for m in loo], theta.shape)
    bias, var = _jackknife_moments(theta, reps)
    return JackknifeResult(
        value=_unwrap(theta - bias),
        error=_unwrap(np.sqrt(var)),
        bias=_unwrap(bias),
        naive=_unwrap(theta),
        replicates=reps,
    )


def jackknife_independent(
    f: Callable[[list[Any]], Any],
    datasets: Sequence[Any],
    n_blocks: int | Sequence[int] | None = None,
) -> JackknifeResult:
    """Jackknife of a function of several statistically independent data sets.

    Each data set (e.g. the simulation at one lattice size) is jackknifed in
    turn while the others are kept whole.  Because the data sets are
    independent, their variances and biases add:
    ``error^2 = sum_d error_d^2`` and ``bias = sum_d bias_d``.

    Parameters
    ----------
    f:
        ``f(datasets)`` returning a scalar or an array.  It receives a list with,
        for every data set, a tuple of its (block-deleted) arrays; a data set
        given as a bare array is passed as a bare array.
    datasets:
        One tuple of arrays (sharing axis 0) per independent data set.
    n_blocks:
        Blocks per data set: ``None`` (``min(N_d, 64)`` for each), one integer for
        all data sets, or a sequence with one integer per data set.

    Returns
    -------
    JackknifeResult
        ``replicates`` is a tuple with the replicates of each data set.
    """
    if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty list with one tuple of arrays per data set")
    if n_blocks is None or np.ndim(n_blocks) == 0:
        per_set = [n_blocks] * len(datasets)
    else:
        per_set = list(n_blocks)
        if len(per_set) != len(datasets):
            raise ValueError(f"got {len(per_set)} n_blocks for {len(datasets)} data sets")

    parts, bare, blocks = [], [], []
    for d, (ds, requested) in enumerate(zip(datasets, per_set, strict=True)):
        label = f"data set {d}: "
        if isinstance(ds, np.ndarray):
            arrays, is_bare = [ds], True
        elif isinstance(ds, (list, tuple)):
            arrays, is_bare = list(ds), False
        else:
            raise ValueError(f"{label}expected a tuple of arrays, got {type(ds).__name__}")
        arrays, n = _as_arrays(arrays, label)
        nb = _n_blocks(requested, n, label)
        size = n // nb
        parts.append([a[n - nb * size :] for a in arrays])
        bare.append(is_bare)
        blocks.append((nb, size))

    def pack(sets: list[list[np.ndarray]]) -> list[Any]:
        return [s[0] if b else tuple(s) for s, b in zip(sets, bare, strict=True)]

    theta = _evaluate(f, [pack(parts)])
    bias = np.zeros(theta.shape)
    var = np.zeros(theta.shape)
    replicates = []
    for d, (nb, size) in enumerate(blocks):
        reps = np.empty((nb, *theta.shape))
        for i in range(nb):
            sets = list(parts)
            sets[d] = [_drop(a, i * size, (i + 1) * size) for a in parts[d]]
            reps[i] = _evaluate(f, [pack(sets)], theta.shape)
        bias_d, var_d = _jackknife_moments(theta, reps)
        bias += bias_d
        var += var_d
        replicates.append(reps)
    return JackknifeResult(
        value=_unwrap(theta - bias),
        error=_unwrap(np.sqrt(var)),
        bias=_unwrap(bias),
        naive=_unwrap(theta),
        replicates=tuple(replicates),
    )


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: int | np.random.Generator | None = None
) -> np.ndarray:
    """Resampling indices of the stationary bootstrap (Politis and Romano 1994).

    Blocks start at uniformly random positions, have geometrically distributed
    lengths with mean ``mean_block`` and wrap around the end of the series.

    Parameters
    ----------
    n:
        Length of the series (and of the resample).
    mean_block:
        Mean block length ``1/p >= 1``; ``1`` gives the i.i.d. bootstrap.
    rng:
        Seed or ``numpy.random.Generator``.

    Returns
    -------
    numpy.ndarray
        ``n`` integer indices into the series.
    """
    n = _as_int(n, "n", 1)
    mean_block = _mean_block(mean_block, "mean_block")
    gen = _as_generator(rng)
    p = 1.0 / mean_block
    # block lengths ~ Geometric(p) on {1, 2, ...}: a new block starts at each position
    # with probability p; draw enough of them to cover n positions
    lengths = gen.geometric(p, size=int(n * p + 5.0 * np.sqrt(n * p)) + 10)
    while lengths.sum() < n:
        lengths = np.concatenate((lengths, gen.geometric(p, size=int(n * p) + 10)))
    ends = np.cumsum(lengths)
    k = int(np.searchsorted(ends, n)) + 1  # blocks needed to cover n positions
    first = np.concatenate(([0], ends[: k - 1]))  # position of each block in the resample
    shift = gen.integers(0, n, size=k) - first  # block start in the data minus its position
    idx = np.repeat(shift, lengths[:k])[:n] + np.arange(n)
    idx[idx >= n] -= n  # wrap around the end (idx < 2n)
    return idx


def _scheme(n: int, block_length: Any, stationary: bool) -> tuple[str, Any, int]:
    """Validated resampling scheme: kind, block parameter and resampling unit."""
    if stationary:
        if block_length is None:
            raise ValueError("the stationary bootstrap needs block_length (the mean block length)")
        mean_block = _mean_block(block_length, "block_length")
        if mean_block > n:
            raise ValueError(f"block_length={mean_block} exceeds the number of samples ({n})")
        return "stationary", mean_block, max(1, round(mean_block))
    if block_length is None:
        return "iid", None, 1
    ell = _as_int(block_length, "block_length", 1)
    if ell > n:
        raise ValueError(f"block_length={ell} exceeds the number of samples ({n})")
    return "blocks", ell, ell


def _indices(
    kind: str, param: Any, n: int, n_boot: int, gen: np.random.Generator
) -> Iterator[np.ndarray]:
    """Resampling indices of ``n_boot`` bootstrap samples."""
    if kind == "iid":
        rows = max(1, min(n_boot, _IID_BATCH // n))  # batches amortise the Generator overhead
        for start in range(0, n_boot, rows):
            yield from gen.integers(0, n, size=(min(rows, n_boot - start), n))
    elif kind == "blocks":
        offsets = np.arange(param)
        for _ in range(n_boot):
            starts = gen.integers(0, n, size=-(-n // param))
            idx = (starts[:, None] + offsets).ravel()[:n]
            idx[idx >= n] -= n  # circular blocks wrap around the end (idx < 2n)
            yield idx
    else:
        for _ in range(n_boot):
            yield stationary_bootstrap_indices(n, param, gen)


def _quantiles(sorted_reps: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per-component quantiles (linear interpolation) at ``levels`` of shape ``(2, ...)``."""
    pos = np.clip(levels, 0.0, 1.0) * (sorted_reps.shape[0] - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, sorted_reps.shape[0] - 1)
    w = pos - lo
    return (1.0 - w) * np.take_along_axis(sorted_reps, lo, axis=0) + w * np.take_along_axis(
        sorted_reps, hi, axis=0
    )


def _acceleration(
    f: Callable[..., Any], data: list[np.ndarray], unit: int, shape: tuple[int, ...]
) -> np.ndarray:
    """BCa acceleration from a jackknife that deletes resampling units.

    Units are samples (i.i.d.) or blocks of ``unit`` samples.  Beyond
    ``_BCA_MAX_GROUPS`` units, consecutive units are pooled into that many groups;
    the acceleration is unaffected to leading order, the cost stays bounded.
    """
    n = data[0].shape[0]
    n_units = n // unit
    if n_units < 2:
        return np.zeros(shape)
    n_groups = min(n_units, _BCA_MAX_GROUPS)
    cuts = np.rint(np.linspace(0, n_units, n_groups + 1)).astype(np.intp)
    edges = (n - n_units * unit) + unit * cuts
    jk = np.empty((n_groups, *shape))
    for g in range(n_groups):
        jk[g] = _evaluate(f, [_drop(a, edges[g], edges[g + 1]) for a in data], shape)
    u = jk.mean(axis=0) - jk
    num = np.sum(u**3, axis=0)
    den = 6.0 * np.sum(u**2, axis=0) ** 1.5
    return np.divide(num, den, out=np.zeros(shape), where=den > 0)


def _interval(
    method: str,
    theta: np.ndarray,
    reps: np.ndarray,
    error: np.ndarray,
    alpha: float,
    f: Callable[..., Any],
    data: list[np.ndarray],
    unit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "normal":
        z = ndtri(1.0 - alpha / 2)
        return theta - z * error, theta + z * error
    tails = np.array([alpha / 2, 1.0 - alpha / 2]).reshape((2,) + (1,) * theta.ndim)
    if method == "bca":
        b = reps.shape[0]
        # bias correction: fraction of replicates below the estimate (ties count half)
        below = (np.sum(reps < theta, axis=0) + 0.5 * np.sum(reps == theta, axis=0)) / b
        z0 = ndtri(np.clip(below, 0.5 / b, 1.0 - 0.5 / b))
        acc = _acceleration(f, data, unit, theta.shape)
        zt = z0 + ndtri(tails)
        levels = ndtr(z0 + zt / (1.0 - acc * zt))
    else:
        levels = np.broadcast_to(tails, (2, *theta.shape))
    q = _quantiles(np.sort(reps, axis=0), levels)
    if method == "basic":
        return 2.0 * theta - q[1], 2.0 * theta - q[0]
    return q[0], q[1]


def bootstrap(
    f: Callable[..., Any],
    *arrays: ArrayLike,
    n_boot: int = 1000,
    rng: int | np.random.Generator | None = None,
    block_length: float | None = None,
    stationary: bool = False,
    method: str = "percentile",
    confidence: float = 0.6827,
) -> BootstrapResult:
    """Bootstrap error and confidence interval of an arbitrary estimator.

    Parameters
    ----------
    f:
        ``f(*arrays)`` returning a scalar or an array.
    *arrays:
        Data arrays sharing axis 0; they are resampled jointly (same indices).
    n_boot:
        Number of bootstrap replicates.
    rng:
        Seed or ``numpy.random.Generator``.
    block_length:
        ``None``: i.i.d. resampling of samples.  Otherwise circular moving blocks
        of this (integer) length or, with ``stationary=True``, geometric blocks
        of this mean length.  Choose it much longer than the autocorrelation time.
    stationary:
        Use the stationary bootstrap of Politis and Romano.
    method:
        ``"percentile"``, ``"basic"`` (reflected percentile), ``"normal"``
        (``value -/+ z error``) or ``"bca"`` (bias-corrected and accelerated; the
        acceleration comes from a jackknife over the resampling units, i.e.
        samples or blocks).
    confidence:
        Coverage of the interval, default one standard deviation.

    Returns
    -------
    BootstrapResult
    """
    data, n = _as_arrays(arrays)
    n_boot = _as_int(n_boot, "n_boot", 2)
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {_METHODS}")
    confidence = _as_real(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    gen = _as_generator(rng)
    kind, param, unit = _scheme(n, block_length, bool(stationary))

    theta = _evaluate(f, data)
    reps = np.empty((n_boot, *theta.shape))
    for b, idx in enumerate(_indices(kind, param, n, n_boot, gen)):
        reps[b] = _evaluate(f, [a[idx] for a in data], theta.shape)
    error = reps.std(axis=0, ddof=1)
    lo, hi = _interval(method, theta, reps, error, 1.0 - confidence, f, data, unit)
    return BootstrapResult(
        value=_unwrap(theta),
        error=_unwrap(error),
        ci=(_unwrap(lo), _unwrap(hi)),
        replicates=reps,
        method=method,
        confidence=confidence,
    )
