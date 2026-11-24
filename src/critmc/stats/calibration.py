"""Calibration of estimates and error bars against a known answer.

One run compared with an exact value gives a single pull
``z = (estimate - exact) / error``, which can only reveal a bias of the order of
the error bar itself.  Repeating the run with ``n`` independent random streams
separates the two things a pull mixes together:

* the *estimator*: the mean of the ``n`` estimates has an error that is
  measured from their scatter, without reference to the quoted error bars, and
  resolves a bias ``sqrt(n)`` times smaller than one run can;
* the *error bar*: the scatter of the estimates is the true statistical error
  of one run, to be compared with the error the analysis quoted.

Two summaries of the error bar are reported because they answer different
questions.  ``error_ratio`` is the true scatter divided by the r.m.s. quoted
error: it says whether the errors have the right size on average.  The width
of the pull distribution and its coverage say whether the quoted interval of an
individual run contains the truth as often as claimed.  They differ when the
estimate and its error are correlated across runs - a run that fluctuates low
then also quotes a small error and lands many of its own sigmas from the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri, stdtr

__all__ = ["PullStatistics", "Calibration", "pull_statistics", "calibration"]

COVERAGE_1 = 0.6826894921370859  # P(|z| < 1) for a standard normal variable
COVERAGE_2 = 0.9544997361036416  # P(|z| < 2)
TAIL_3 = 0.0026997960632601866  # P(|z| > 3)


@dataclass(frozen=True)
class PullStatistics:
    """Moments and coverage of pulls, which are standard normal if all is well.

    For pulls of shape ``(n, ...)`` every field has the trailing shape ``(...)``.

    Attributes
    ----------
    n:
        Number of pulls per entry.
    mean, mean_error:
        Mean pull and its standard error (expected: 0).
    sd, sd_error:
        Standard deviation of the pulls and its standard error from the fourth
        moment, valid for non-normal pulls (expected: 1).
    within_1, within_2:
        Fractions of pulls with ``|z| < 1`` and ``|z| < 2`` (expected: 0.683, 0.954).
    beyond_3:
        Number of pulls with ``|z| > 3`` (expected: ``0.0027 n``).
    """

    n: int
    mean: np.ndarray
    mean_error: np.ndarray
    sd: np.ndarray
    sd_error: np.ndarray
    within_1: np.ndarray
    within_2: np.ndarray
    beyond_3: np.ndarray


@dataclass(frozen=True)
class Calibration:
    """Result of :func:`calibration`.

    For ``values`` of shape ``(n, ...)`` every field has the trailing shape ``(...)``.

    Attributes
    ----------
    n:
        Number of independent runs.
    mean, mean_error:
        Mean of the estimates and its standard error from their scatter.
    bias, bias_z, bias_p:
        ``mean - reference``, the same in units of ``mean_error``, and its
        two-sided p-value from Student's t distribution with ``n - 1`` degrees
        of freedom.
    scatter:
        Standard deviation of the estimates: the true error of a single run.
    rms_error:
        Root mean square of the quoted errors.
    error_ratio, error_ratio_error:
        ``scatter / rms_error`` and its standard error (expected: 1; above 1 the
        quoted errors are too small).
    correlation:
        Correlation coefficient between an estimate and its quoted error.
    pulls:
        :class:`PullStatistics` of ``(values - reference) / errors``.
    """

    n: int
    mean: np.ndarray
    mean_error: np.ndarray
    bias: np.ndarray
    bias_z: np.ndarray
    bias_p: np.ndarray
    scatter: np.ndarray
    rms_error: np.ndarray
    error_ratio: np.ndarray
    error_ratio_error: np.ndarray
    correlation: np.ndarray
    pulls: PullStatistics

    @property
    def bias_normal(self) -> np.ndarray:
        """``bias_z`` mapped to the standard normal variable with the same p-value.

        Sums of squares of these are chi-square distributed whatever ``n`` is,
        which sums of squares of ``bias_z`` (Student t) are not.
        """
        return np.sign(self.bias_z) * -ndtri(0.5 * np.clip(self.bias_p, 1e-300, 1.0))


def _as_runs(a: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim < 1 or a.shape[0] < 2:
        raise ValueError(f"{name} needs at least two runs along axis 0, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite entries")
    return a


def pull_statistics(pulls: ArrayLike) -> PullStatistics:
    """Moments and coverage of independent pulls along axis 0."""
    z = _as_runs(pulls, "pulls")
    n = z.shape[0]
    mean = z.mean(axis=0)
    dev = z - mean
    var = np.sum(dev**2, axis=0) / (n - 1)
    sd = np.sqrt(var)
    m4 = np.mean(dev**4, axis=0)
    # delta method: var(s^2) = (m4 - s^4) / n, hence var(s) = var(s^2) / (4 s^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_error = np.where(sd > 0, np.sqrt(np.clip(m4 - var**2, 0.0, None) / n) / (2 * sd), 0.0)
    a = np.abs(z)
    return PullStatistics(
        n=n,
        mean=mean,
        mean_error=sd / np.sqrt(n),
        sd=sd,
        sd_error=sd_error,
        within_1=np.mean(a < 1, axis=0),
        within_2=np.mean(a < 2, axis=0),
        beyond_3=np.sum(a > 3, axis=0),
    )


def calibration(values: ArrayLike, errors: ArrayLike, reference: ArrayLike) -> Calibration:
    """Compare ``n`` independent estimates and their quoted errors with a known answer.

    Parameters
    ----------
    values, errors:
        Estimates and their quoted one-sigma errors, shape ``(n,)`` or
        ``(n, ...)`` with the independent runs along axis 0.  Trailing axes
        (temperatures, observables) are treated separately.
    reference:
        The exact answer, broadcastable to the trailing shape.

    Returns
    -------
    Calibration

    Examples
    --------
    Estimates with honest errors of size 0.1::

        rng = np.random.default_rng(1)
        cal = calibration(rng.normal(2.0, 0.1, 500), np.full(500, 0.1), 2.0)
        cal.bias_z, cal.error_ratio, cal.pulls.sd   # all compatible with 0, 1, 1
    """
    x = _as_runs(values, "values")
    err = _as_runs(errors, "errors")
    if err.shape != x.shape:
        raise ValueError(f"values and errors differ in shape: {x.shape} and {err.shape}")
    if np.any(err <= 0):
        raise ValueError("errors must be positive")
    ref = np.broadcast_to(np.asarray(reference, dtype=np.float64), x.shape[1:])
    n = x.shape[0]

    mean = x.mean(axis=0)
    dev = x - mean
    var = np.sum(dev**2, axis=0) / (n - 1)
    scatter = np.sqrt(var)
    mean_error = scatter / np.sqrt(n)
    bias = mean - ref
    with np.errstate(divide="ignore", invalid="ignore"):
        bias_z = np.where(mean_error > 0, bias / mean_error, 0.0)
    bias_p = 2.0 * stdtr(n - 1, -np.abs(bias_z))

    # ratio of two means, R = <(x - xbar)^2> / <err^2>, by the delta method
    a = dev**2 * (n / (n - 1))
    b = err**2
    b_mean = b.mean(axis=0)
    ratio2 = var / b_mean
    ratio = np.sqrt(ratio2)
    resid = a - ratio2 * b
    ratio2_error = np.sqrt(np.sum((resid - resid.mean(axis=0)) ** 2, axis=0) / (n - 1) / n) / b_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_error = np.where(ratio > 0, ratio2_error / (2 * ratio), 0.0)

    err_dev = err - err.mean(axis=0)
    norm = np.sqrt(np.sum(dev**2, axis=0) * np.sum(err_dev**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(norm > 0, np.sum(dev * err_dev, axis=0) / norm, 0.0)

    return Calibration(
        n=n,
        mean=mean,
        mean_error=mean_error,
        bias=bias,
        bias_z=bias_z,
        bias_p=bias_p,
        scatter=scatter,
        rms_error=np.sqrt(b_mean),
        error_ratio=ratio,
        error_ratio_error=ratio_error,
        correlation=corr,
        pulls=pull_statistics((x - ref) / err),
    )
