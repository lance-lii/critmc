"""Tests for :mod:`critmc.stats.calibration`.

Synthetic runs with a known truth: every statistic must report an honest set
of runs as honest, and each kind of defect (biased estimator, error bars of
the wrong size, error bars that follow the estimate) must show up in the
statistic meant for it.  Fixed seeds make the tolerances deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from critmc.stats import Calibration, PullStatistics, calibration, pull_statistics

TRUTH = 2.0
SIGMA = 0.1
N = 4000


def honest(seed: int, n: int = N, shape: tuple = ()):
    rng = np.random.default_rng(seed)
    return rng.normal(TRUTH, SIGMA, size=(n, *shape)), np.full((n, *shape), SIGMA)


def test_honest_runs_pass_every_check():
    x, err = honest(1)
    cal = calibration(x, err, TRUTH)
    assert isinstance(cal, Calibration) and isinstance(cal.pulls, PullStatistics)
    assert cal.n == N
    assert abs(cal.bias_z) < 3
    assert cal.bias_p > 0.003
    assert cal.mean_error == pytest.approx(SIGMA / np.sqrt(N), rel=0.05)
    assert cal.rms_error == pytest.approx(SIGMA)
    assert abs(cal.error_ratio - 1) < 3 * cal.error_ratio_error
    assert abs(cal.pulls.mean) < 3 * cal.pulls.mean_error
    assert abs(cal.pulls.sd - 1) < 3 * cal.pulls.sd_error
    assert cal.pulls.within_1 == pytest.approx(0.6827, abs=0.03)
    assert cal.pulls.within_2 == pytest.approx(0.9545, abs=0.015)
    assert cal.pulls.beyond_3 < 30  # 10.8 expected


def test_small_bias_is_resolved_by_many_runs():
    # a tenth of an error bar: invisible in one run, 6 sigma in the mean of 4000
    x, err = honest(2)
    cal = calibration(x + 0.1 * SIGMA, err, TRUTH)
    assert cal.bias == pytest.approx(0.1 * SIGMA, abs=4 * cal.mean_error)
    assert cal.bias_z > 4
    assert cal.bias_p < 1e-4
    assert cal.pulls.mean == pytest.approx(0.1, abs=4 * cal.pulls.mean_error)
    assert abs(cal.error_ratio - 1) < 3 * cal.error_ratio_error  # the errors are still right


def test_errors_of_the_wrong_size_are_measured():
    x, err = honest(3)
    cal = calibration(x, 0.8 * err, TRUTH)
    assert cal.error_ratio == pytest.approx(1.25, abs=4 * cal.error_ratio_error)
    assert cal.error_ratio - 1 > 10 * cal.error_ratio_error
    assert cal.pulls.sd == pytest.approx(1.25, abs=4 * cal.pulls.sd_error)
    assert abs(cal.bias_z) < 3  # the estimator is still unbiased


def test_error_that_follows_the_estimate_keeps_its_size_but_loses_coverage():
    # the quoted error grows with the estimate, as for a variance-like observable;
    # its mean square is right, yet low estimates sit many of their own sigmas away
    rng = np.random.default_rng(4)
    g = rng.standard_normal(N)
    k = 0.3
    err = SIGMA * np.exp(k * g - k * k)  # <err^2> = SIGMA^2 exactly
    cal = calibration(TRUTH + SIGMA * g, err, TRUTH)
    assert abs(cal.bias_z) < 3
    assert abs(cal.error_ratio - 1) < 3 * cal.error_ratio_error
    assert cal.correlation > 0.9
    assert cal.pulls.mean < -5 * cal.pulls.mean_error
    assert cal.pulls.sd - 1 > 5 * cal.pulls.sd_error


@pytest.mark.parametrize("case", ["normal", "student-t", "error-follows-deviation"])
def test_quoted_errors_of_the_statistics_match_their_scatter(case):
    # 600 repetitions of a 200-run study; every "+-" must equal the actual spread
    rng = np.random.default_rng(5)
    shape = (200, 600)
    g = rng.standard_t(8, size=shape) if case == "student-t" else rng.standard_normal(shape)
    if case == "error-follows-deviation":
        err = SIGMA * np.sqrt(0.2 + 0.8 * g * g)  # runs far from the truth quote large errors
    else:
        err = SIGMA * np.exp(0.2 * rng.standard_normal(shape))
    cal = calibration(TRUTH + SIGMA * g, err, TRUTH)
    for value, quoted in [
        (cal.mean, cal.mean_error),
        (cal.error_ratio, cal.error_ratio_error),
        (cal.pulls.mean, cal.pulls.mean_error),
        (cal.pulls.sd, cal.pulls.sd_error),
    ]:
        spread = value.std(ddof=1)
        assert np.sqrt(np.mean(quoted**2)) == pytest.approx(spread, rel=0.1)


def test_trailing_axes_are_independent_columns():
    x, err = honest(6, n=300, shape=(4, 3))
    scale = np.linspace(0.5, 3.0, 12).reshape(4, 3)  # every column has errors of its own size
    x, err = TRUTH + (x - TRUTH) * scale, err * scale
    ref = np.full((4, 3), TRUTH)
    ref[1, 2] = TRUTH + 0.5 * SIGMA * scale[1, 2]
    cal = calibration(x, err, ref)
    assert cal.bias_z.shape == (4, 3) and cal.pulls.sd.shape == (4, 3)
    one = calibration(x[:, 1, 2], err[:, 1, 2], ref[1, 2])
    names = ("mean", "bias_z", "bias_p", "rms_error", "error_ratio", "error_ratio_error")
    for name in (*names, "correlation"):
        assert getattr(cal, name)[1, 2] == pytest.approx(float(getattr(one, name)))
    for name in ("mean", "sd", "sd_error", "within_1", "within_2", "beyond_3"):
        assert getattr(cal.pulls, name)[1, 2] == pytest.approx(float(getattr(one.pulls, name)))
    assert cal.bias_z[1, 2] < -5 and np.abs(np.delete(cal.bias_z.ravel(), 5)).max() < 3.5


def test_scalar_reference_broadcasts():
    x, err = honest(7, n=50, shape=(5,))
    a = calibration(x, err, TRUTH)
    b = calibration(x, err, np.full(5, TRUTH))
    np.testing.assert_allclose(a.bias_z, b.bias_z)


def test_bias_normal_is_chi_square_for_few_runs():
    # with 5 runs bias_z follows Student's t with 4 degrees of freedom (variance 2)
    x, err = honest(8, n=5, shape=(40_000,))
    cal = calibration(x, err, TRUTH)
    assert np.mean(cal.bias_z**2) == pytest.approx(2.0, rel=0.1)
    assert np.mean(cal.bias_normal**2) == pytest.approx(1.0, rel=0.03)
    assert np.all(np.sign(cal.bias_normal) == np.sign(cal.bias_z))
    # many runs: the two agree
    x, err = honest(9, n=2000, shape=(50,))
    cal = calibration(x, err, TRUTH)
    np.testing.assert_allclose(cal.bias_normal, cal.bias_z, atol=0.01)


def test_pull_statistics_counts():
    z = np.array([-3.5, -1.5, -0.5, 0.0, 0.5, 0.9, 1.1, 2.5, 3.2, 0.2])
    p = pull_statistics(z)
    assert p.n == 10
    assert p.mean == pytest.approx(z.mean())
    assert p.sd == pytest.approx(z.std(ddof=1))
    assert p.mean_error == pytest.approx(z.std(ddof=1) / np.sqrt(10))
    assert p.within_1 == pytest.approx(0.5)
    assert p.within_2 == pytest.approx(0.7)
    assert p.beyond_3 == 2


def test_identical_values_do_not_divide_by_zero():
    cal = calibration(np.full(10, 1.0), np.full(10, 0.1), 1.0)
    assert cal.scatter == 0 and cal.bias_z == 0 and cal.error_ratio == 0
    assert cal.correlation == 0 and cal.pulls.sd == 0 and cal.pulls.sd_error == 0


INVALID = [
    pytest.param(lambda: calibration([1.0], [0.1], 1.0), id="one-run"),
    pytest.param(lambda: calibration(1.0, 0.1, 1.0), id="scalar"),
    pytest.param(lambda: calibration([1.0, 2.0], [0.1, 0.1, 0.1], 1.0), id="shape-mismatch"),
    pytest.param(lambda: calibration([1.0, 2.0], [0.1, 0.0], 1.0), id="zero-error"),
    pytest.param(lambda: calibration([1.0, 2.0], [0.1, -0.1], 1.0), id="negative-error"),
    pytest.param(lambda: calibration([1.0, np.nan], [0.1, 0.1], 1.0), id="nan-value"),
    pytest.param(lambda: calibration([1.0, 2.0], [0.1, np.inf], 1.0), id="inf-error"),
    pytest.param(lambda: calibration(np.ones((3, 2)), np.ones((3, 2)), np.ones(3)), id="ref-shape"),
    pytest.param(lambda: pull_statistics([0.3]), id="one-pull"),
    pytest.param(lambda: pull_statistics([0.3, np.nan]), id="nan-pull"),
]


@pytest.mark.parametrize("call", INVALID)
def test_invalid_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()
