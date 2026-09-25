"""Blocking analysis of the error of the mean of a correlated time series.

Averaging adjacent pairs of a stationary series leaves its mean unchanged and
weakens its correlations.  Once the blocks are much longer than the
autocorrelation time the blocked values are independent, and the naive
standard error of the blocked series is the true error of the mean.

* :func:`blocking_levels` tabulates the naive error ``sqrt(var_k / (n_k - 1))``
  and its uncertainty ``error / sqrt(2 (n_k - 1))`` at every blocking level
  ``k``, following H. Flyvbjerg and H. G. Petersen, "Error estimates on averages
  of correlated data", J. Chem. Phys. 91, 461 (1989).  The error grows with
  ``k`` and levels off on a plateau.
* :func:`blocking_error` locates the plateau automatically with the hypothesis
  test of M. Jonsson, "Standard error estimation by an automated blocking
  method", Phys. Rev. E 98, 043304 (2018): the first level from which on the
  lag-one autocovariances of all blocked series are compatible with zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2

from .autocorr import _as_real, _as_series

__all__ = ["BlockingLevels", "BlockingResult", "blocking_levels", "blocking_error"]

_MIN_SAMPLES = 2**4  # minimum length for the automated method
# M_j is bounded on the last levels (M_{d-1} = 1/8 identically), so they always "pass";
# a level with fewer blocks than this is not evidence of a plateau
_MIN_RELIABLE_BLOCKS = 2**5


@dataclass(frozen=True, eq=False)
class BlockingLevels:
    """Naive error of the mean at each blocking level (Flyvbjerg-Petersen table).

    Attributes
    ----------
    level:
        Blocking level ``k``; a block averages about ``2**k`` original samples.
    n_blocks:
        Length ``n_k`` of the blocked series.
    error:
        ``sqrt(var_k / (n_k - 1))`` with ``var_k`` the biased variance of the
        blocked series.
    error_error:
        Uncertainty of ``error``, ``error / sqrt(2 (n_k - 1))``.
    """

    level: np.ndarray
    n_blocks: np.ndarray
    error: np.ndarray
    error_error: np.ndarray


@dataclass(frozen=True, eq=False)
class BlockingResult:
    """Error of the mean from :func:`blocking_error`.

    Attributes
    ----------
    mean:
        Mean of the analysed series (the most recent ``2**d`` samples).
    error:
        Standard error of ``mean``, ``sqrt(var_j / n_j)`` at the selected level.
    level:
        Selected blocking level ``j``.
    levels:
        Flyvbjerg-Petersen table of the analysed series.
    reliable:
        ``False`` if no level passed the test (the deepest level is then used) or
        the selected level keeps fewer than 32 blocks.  The test has almost no
        power on the last levels, which therefore always pass; selecting one of
        them typically means that ``N`` is not much larger than ``tau_int``.
    """

    mean: float
    error: float
    level: int
    levels: BlockingLevels
    reliable: bool


def _variance(y: np.ndarray) -> float:
    """Biased variance, exactly zero for a constant series."""
    return float(np.var(y)) if np.ptp(y) > 0 else 0.0


def _halve(y: np.ndarray) -> np.ndarray:
    """One blocking step; for odd length the oldest sample is discarded."""
    y = y[y.size % 2 :]
    return 0.5 * (y[0::2] + y[1::2])


def _table(n_blocks: np.ndarray, var: np.ndarray) -> BlockingLevels:
    dof = n_blocks - 1.0
    error = np.sqrt(var / dof)
    return BlockingLevels(
        level=np.arange(n_blocks.size),
        n_blocks=n_blocks,
        error=error,
        error_error=error / np.sqrt(2.0 * dof),
    )


def blocking_levels(x: ArrayLike) -> BlockingLevels:
    """Flyvbjerg-Petersen blocking table of a 1-D time series.

    The series is repeatedly replaced by the averages of adjacent pairs (the
    oldest sample is dropped when the length is odd) until fewer than two
    blocks remain.

    Parameters
    ----------
    x:
        1-D time series with at least 2 samples.

    Returns
    -------
    BlockingLevels
    """
    y = _as_series(x)
    sizes, variances = [], []
    while y.size >= 2:
        sizes.append(y.size)
        variances.append(_variance(y))
        y = _halve(y)
    return _table(np.array(sizes), np.array(variances))


def blocking_error(x: ArrayLike, alpha: float = 0.01) -> BlockingResult:
    """Error of the mean with Jonsson's automated blocking method.

    ``x`` is truncated to its most recent ``2**d`` samples (the oldest ones are
    dropped).  For every level ``k = 0..d-1`` the blocked series of length
    ``n_k = 2**(d-k)`` has biased variance ``var_k`` and lag-one autocovariance
    ``gamma_k`` (about the overall mean).  The statistic

        M_j = sum_{k=j}^{d-1} n_k ((n_k - 1) var_k / n_k^2 + gamma_k)^2 / var_k^2

    is asymptotically ``chi^2(d - j)`` distributed if the blocked series are
    uncorrelated from level ``j`` on, and the smallest ``j`` with
    ``M_j < chi^2_{1-alpha}(d - j)`` is selected.

    Parameters
    ----------
    x:
        1-D time series with at least ``2**4`` samples.
    alpha:
        Significance level of the test.

    Returns
    -------
    BlockingResult
    """
    y = _as_series(x, min_len=_MIN_SAMPLES)
    alpha = _as_real(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    d = y.size.bit_length() - 1
    y = y[y.size - 2**d :]  # keep the most recent 2**d samples
    mu = float(y.mean()) if np.ptp(y) > 0 else float(y[0])
    n_k = 2.0 ** (d - np.arange(d))
    var_k = np.empty(d)
    gam_k = np.empty(d)
    for k in range(d):
        dy = y - mu
        var_k[k] = _variance(y)
        gam_k[k] = np.dot(dy[:-1], dy[1:]) / y.size
        y = _halve(y)
    # (n_k - 1) var_k / n_k^2 removes the bias of gamma_k caused by the subtracted mean;
    # each term is then asymptotically chi^2(1) for uncorrelated blocks
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = n_k * ((n_k - 1.0) * var_k / n_k**2 + gam_k) ** 2 / var_k**2
    terms[var_k == 0] = 0.0  # constant blocked series: nothing left to test
    m = np.cumsum(terms[::-1])[::-1]
    passed = np.flatnonzero(m < chi2.ppf(1.0 - alpha, d - np.arange(d)))
    j = int(passed[0]) if passed.size else d - 1
    return BlockingResult(
        mean=mu,
        error=float(np.sqrt(var_k[j] / n_k[j])),
        level=j,
        levels=_table(n_k.astype(np.int64), var_k),
        reliable=bool(passed.size) and n_k[j] >= _MIN_RELIABLE_BLOCKS,
    )
