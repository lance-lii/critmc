"""Tests for :mod:`critmc.stats`: autocorrelation times, Gamma-method, blocking, resampling.

The reference process is the stationary AR(1) chain
``x_t = phi x_{t-1} + sqrt(1 - phi^2) eps_t`` with unit variance, ``rho(t) = phi^|t|``,
``tau_int = (1 + phi) / (2 (1 - phi))`` and ``var(mean) ~ 2 tau_int / N``.  All random
numbers come from fixed seeds, so the statistical tolerances below (typically 3-5 sigma)
are checked deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.special import ndtri

from critmc.stats import (
    AutocorrTime,
    BlockingResult,
    BootstrapResult,
    GammaResult,
    JackknifeResult,
    autocorrelation,
    autocovariance,
    block_means,
    blocking_error,
    blocking_levels,
    bootstrap,
    effective_sample_size,
    gamma_method,
    jackknife,
    jackknife_independent,
    jackknife_means,
    stationary_bootstrap_indices,
    tau_int_sokal,
)

PHI = 0.9
TAU_AR1 = (1 + PHI) / (2 * (1 - PHI))  # = 9.5
N_LONG = 2**18
SIGMA_MEAN = np.sqrt(2 * TAU_AR1 / N_LONG)  # error of the mean of the long chain
CONF = 0.6827


def ar1(n: int, phi: float, rng: np.random.Generator, n_series: int | None = None) -> np.ndarray:
    """Stationary unit-variance AR(1) series, shape ``(n,)`` or ``(n_series, n)``."""
    shape = (n,) if n_series is None else (n_series, n)
    eps = rng.standard_normal(shape)
    # x_0 = phi x_{-1} + sqrt(1 - phi^2) eps_0 with x_{-1} ~ N(0, 1): stationary from the start
    zi = phi * rng.standard_normal(shape[:-1] + (1,))
    x, _ = lfilter([np.sqrt(1 - phi**2)], [1.0, -phi], eps, axis=-1, zi=zi)
    return x


@pytest.fixture(scope="module")
def ar1_long() -> np.ndarray:
    return ar1(N_LONG, PHI, np.random.default_rng(20260924))


# --------------------------------------------------------------------------------------
# autocovariance and autocorrelation
# --------------------------------------------------------------------------------------


def test_autocovariance_matches_direct_sum():
    x = np.random.default_rng(1).standard_normal(257)
    dx = x - x.mean()
    direct = np.array([np.dot(dx[: x.size - t], dx[t:]) / (x.size - t) for t in range(41)])
    np.testing.assert_allclose(autocovariance(x, max_lag=40), direct, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(autocorrelation(x, 40), direct / direct[0], rtol=1e-10, atol=1e-13)
    assert autocovariance(x).shape == (257 // 2 + 1,)
    assert autocovariance(x, max_lag=256).shape == (257,)


def test_autocovariance_treats_columns_independently():
    x = np.random.default_rng(2).standard_normal((101, 3))
    gamma = autocovariance(x)
    assert gamma.shape == (51, 3)
    for k in range(3):
        np.testing.assert_allclose(gamma[:, k], autocovariance(x[:, k]), rtol=1e-12)
    np.testing.assert_array_equal(autocovariance(np.full(20, 0.1), 3), np.zeros(4))


def test_autocorrelation_of_ar1(ar1_long):
    rho = autocorrelation(ar1_long, max_lag=10)
    assert rho[0] == 1.0
    np.testing.assert_allclose(rho, PHI ** np.arange(11), atol=0.03)


# --------------------------------------------------------------------------------------
# AR(1) benchmarks against exact results
# --------------------------------------------------------------------------------------


def test_tau_int_sokal_ar1(ar1_long):
    res = tau_int_sokal(ar1_long)
    assert isinstance(res, AutocorrTime) and res.converged
    assert res.tau == pytest.approx(TAU_AR1, rel=0.10)
    # the window is the smallest W with W >= 6 tau_int(W)
    assert res.window >= 6.0 * res.tau
    assert np.all(np.arange(res.window) < 6.0 * res.tau_curve[: res.window])
    assert res.tau_curve[res.window] == res.tau
    assert res.dtau == pytest.approx(res.tau * np.sqrt(2 * (2 * res.window + 1) / N_LONG))
    assert res.rho.shape == res.tau_curve.shape == (N_LONG // 2 + 1,)


def test_gamma_method_ar1(ar1_long):
    res = gamma_method(ar1_long)
    assert isinstance(res, GammaResult)
    assert res.converged and res.n == N_LONG and res.n_replicas == 1 and res.q_value is None
    assert res.value == pytest.approx(ar1_long.mean(), abs=1e-12)
    assert res.tau_int == pytest.approx(TAU_AR1, rel=0.10)
    assert res.error == pytest.approx(SIGMA_MEAN, rel=0.10)
    # the reported uncertainties cover the deviations from the exact values
    assert abs(res.tau_int - TAU_AR1) < 4 * res.dtau_int
    assert abs(res.error - SIGMA_MEAN) < 4 * res.error_of_error
    assert res.rho[0] == 1.0 and res.rho.size == 2 * res.window + 1
    assert res.tau_curve[res.window] == pytest.approx(res.tau_int)


def test_effective_sample_size_ar1(ar1_long):
    assert effective_sample_size(ar1_long) == pytest.approx(N_LONG / (2 * TAU_AR1), rel=0.10)


def test_blocking_error_ar1(ar1_long):
    res = blocking_error(ar1_long)
    assert isinstance(res, BlockingResult) and res.reliable
    assert res.level > 0 and 2**res.level > TAU_AR1
    assert res.error == pytest.approx(SIGMA_MEAN, rel=0.15)
    assert res.mean == pytest.approx(ar1_long.mean())  # N is a power of two: nothing dropped
    assert res.levels.n_blocks[0] == N_LONG and res.levels.n_blocks[-1] == 2
    # the naive error (level 0) is too small by sqrt(2 tau_int)
    assert res.levels.error[0] == pytest.approx(SIGMA_MEAN / np.sqrt(2 * TAU_AR1), rel=0.05)


def test_block_bootstraps_ar1(ar1_long):
    stat = bootstrap(np.mean, ar1_long, n_boot=200, rng=11, block_length=200, stationary=True)
    mbb = bootstrap(np.mean, ar1_long, n_boot=200, rng=12, block_length=200)
    assert stat.error == pytest.approx(SIGMA_MEAN, rel=0.25)
    assert mbb.error == pytest.approx(SIGMA_MEAN, rel=0.25)
    # i.i.d. resampling ignores the autocorrelations
    iid = bootstrap(np.mean, ar1_long, n_boot=100, rng=13)
    assert iid.error == pytest.approx(SIGMA_MEAN / np.sqrt(2 * TAU_AR1), rel=0.25)


def test_blocked_jackknife_ar1(ar1_long):
    res = jackknife(np.mean, ar1_long, n_blocks=64)
    assert res.error == pytest.approx(SIGMA_MEAN, rel=0.30)
    assert res.value == pytest.approx(ar1_long.mean())


def test_white_noise_tau_is_one_half():
    x = np.random.default_rng(3).standard_normal(2**16)
    naive = x.std(ddof=1) / np.sqrt(x.size)
    sokal = tau_int_sokal(x)
    gam = gamma_method(x)
    assert sokal.converged and abs(sokal.tau - 0.5) < 0.05
    assert gam.converged and abs(gam.tau_int - 0.5) < 0.05
    assert gam.error == pytest.approx(naive, rel=0.05)
    assert effective_sample_size(x) == pytest.approx(x.size, rel=0.10)
    blk = blocking_error(x)
    assert blk.reliable and blk.level == 0 and blk.error == pytest.approx(naive, rel=1e-3)


# --------------------------------------------------------------------------------------
# Gamma-method: derived quantities, replicas, special cases
# --------------------------------------------------------------------------------------

N_REP, LEN_REP = 200, 4096
# a = 2 + x, b = 1 + 0.3 (0.6 x + 0.8 y) with independent AR(1) chains x, y.  For F = <a>/<b>
# the projected fluctuation is x - 2 * 0.3 (0.6 x + 0.8 y) = 0.64 x - 0.48 y, again AR(1)
# with the same tau_int and variance 0.64**2 + 0.48**2 = 0.64 (delta method).
RATIO_SIGMA = np.sqrt(2 * TAU_AR1 * 0.64 / LEN_REP)


def ratio(means: np.ndarray) -> float:
    return means[0] / means[1]


def square(means: np.ndarray) -> float:
    return means[0] ** 2


@pytest.fixture(scope="module")
def ratio_replicas() -> list[np.ndarray]:
    rng = np.random.default_rng(5)
    x = ar1(LEN_REP, PHI, rng, N_REP)
    y = ar1(LEN_REP, PHI, rng, N_REP)
    a, b = 2.0 + x, 1.0 + 0.3 * (0.6 * x + 0.8 * y)
    return [np.column_stack((a[k], b[k])) for k in range(N_REP)]


def test_gamma_method_ratio_matches_replica_spread(ratio_replicas):
    results = [gamma_method(r, ratio) for r in ratio_replicas]
    values = np.array([r.value for r in results])
    errors = np.array([r.error for r in results])
    assert errors.mean() == pytest.approx(values.std(ddof=1), rel=0.20)
    assert errors.mean() == pytest.approx(RATIO_SIGMA, rel=0.10)
    assert np.mean([r.tau_int for r in results]) == pytest.approx(TAU_AR1, rel=0.10)


def test_gamma_method_multi_replica(ratio_replicas):
    res = gamma_method(ratio_replicas, ratio, names=("a", "b"))
    assert res.n_replicas == N_REP and res.n == N_REP * LEN_REP and res.names == ("a", "b")
    assert res.error == pytest.approx(RATIO_SIGMA / np.sqrt(N_REP), rel=0.10)
    assert res.tau_int == pytest.approx(TAU_AR1, rel=0.10)
    assert 0.0 < res.q_value < 1.0
    grand = np.concatenate(ratio_replicas).mean(axis=0)
    assert res.value == pytest.approx(ratio(grand), abs=0.1 * res.error)


def test_gamma_method_unequal_replicas_agree_with_single_chain(ar1_long):
    x = ar1_long[:100_000]
    single = gamma_method(x)
    split = gamma_method([x[:20_000], x[20_000:55_000], x[55_000:]])
    assert split.n == x.size and split.n_replicas == 3
    assert split.value == pytest.approx(x.mean())  # linear f: no replica bias
    assert split.error == pytest.approx(single.error, rel=0.05)
    assert split.tau_int == pytest.approx(single.tau_int, rel=0.05)


def uwerr_reference(replicas, f, S=1.5):
    """Loop-based transcription of Wolff's UWerr.m (gradient step sqrt(var / N))."""
    lengths = np.array([len(r) for r in replicas])
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    data = np.concatenate(replicas)
    n, n_rep = len(data), len(replicas)
    abar = data.mean(axis=0)
    h = np.sqrt(data.var(axis=0) / n)
    grad = np.empty(abar.size)
    for a in range(abar.size):
        step = h[a] * np.eye(abar.size)[a]
        grad[a] = (f(abar + step) - f(abar - step)) / (2 * h[a])
    delta = (data - abar) @ grad
    gamma = [np.mean(delta**2)]
    g_int, w_opt, t = 0.0, None, 1
    while w_opt is None and t <= lengths.min() // 2:
        s = 0.0
        for i, m in zip(starts, lengths, strict=True):
            s += delta[i : i + m - t] @ delta[i + t : i + m]
        gamma.append(s / (n - n_rep * t))
        g_int += gamma[t] / gamma[0]
        tau_w = np.finfo(float).eps if g_int <= 0 else S / np.log((g_int + 1) / g_int)
        if np.exp(-t / tau_w) - tau_w / np.sqrt(t * n) < 0:
            w_opt = t
        t += 1
    gamma = np.array(gamma)
    c_f = gamma[0] + 2 * gamma[1:].sum()
    gamma += c_f / n
    c_f = gamma[0] + 2 * gamma[1:].sum()
    value = f(abar)
    if n_rep > 1:
        f_rep = [f(data[i : i + m].mean(axis=0)) for i, m in zip(starts, lengths, strict=True)]
        value -= (lengths @ f_rep / n - value) / (n_rep - 1)
    return value, np.sqrt(c_f / n), c_f / (2 * gamma[0]), w_opt


def test_gamma_method_matches_uwerr_reference():
    rng = np.random.default_rng(27)
    x, y = ar1(30_000, 0.8, rng), ar1(30_000, 0.8, rng)
    data = np.column_stack((2.0 + x, 1.0 + 0.3 * (0.6 * x + 0.8 * y)))

    def f(m):
        return m[0] / m[1] + 0.1 * np.log(m[1])

    for replicas in ([data], [data[:7000], data[7000:18000], data[18000:]]):
        value, error, tau, window = uwerr_reference(replicas, f)
        res = gamma_method(replicas, f)
        assert window is not None and res.window == window
        assert (res.value, res.error, res.tau_int) == pytest.approx((value, error, tau), rel=1e-10)


def test_gamma_method_replica_bias_correction_is_exact():
    rng = np.random.default_rng(6)
    reps = [rng.normal(1.0, 1.0, size=m) for m in (50, 80, 120)]
    res = gamma_method(reps, square)
    x = np.concatenate(reps)
    f_all = x.mean() ** 2
    f_rep = sum(r.size * r.mean() ** 2 for r in reps) / x.size
    assert res.value == pytest.approx(f_all - (f_rep - f_all) / (len(reps) - 1), rel=1e-12)


def test_gamma_method_constant_data():
    res = gamma_method(np.full(100, 0.1))
    assert (res.value, res.error, res.tau_int, res.window) == (0.1, 0.0, 0.5, 0)
    assert res.converged and res.error_of_error == 0.0 and res.dtau_int == 0.0
    # a constant primary next to a fluctuating one contributes no error
    y = np.random.default_rng(7).standard_normal(1000)
    two = gamma_method(np.column_stack((y, np.full(1000, 2.0))), lambda m: m[0] * m[1])
    assert two.value == pytest.approx(2.0 * y.mean())
    assert two.error == pytest.approx(2.0 * gamma_method(y).error, rel=1e-9)


def test_gamma_method_window_options(ar1_long):
    x = ar1_long[:4096]
    capped = gamma_method(x, max_window=5)
    assert not capped.converged and capped.window == 5
    assert capped.tau_int < 0.6 * TAU_AR1  # truncated far too early
    naive = gamma_method(x, S=0)
    assert naive.window == 0 and naive.tau_int == pytest.approx(0.5)
    assert naive.error == pytest.approx(x.std() / np.sqrt(x.size), rel=1e-3)
    assert not tau_int_sokal(x, max_lag=5).converged


def test_gamma_method_accepts_plain_sequences():
    res = gamma_method([1.0, 2.0, 4.0, 3.0, 5.0, 4.0])
    assert res.n == 6 and res.n_replicas == 1 and res.value == pytest.approx(19.0 / 6.0)


# --------------------------------------------------------------------------------------
# blocking
# --------------------------------------------------------------------------------------


def test_blocking_levels_formulae():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0])
    lv = blocking_levels(x)
    # odd length: the oldest sample is dropped, [1,4,1,5,9,2,6,5] -> [2.5,3,5.5,5.5] -> ...
    series = [x, np.array([2.5, 3.0, 5.5, 5.5]), np.array([2.75, 5.5])]
    np.testing.assert_array_equal(lv.level, [0, 1, 2])
    np.testing.assert_array_equal(lv.n_blocks, [9, 4, 2])
    expected = np.array([np.sqrt(s.var() / (s.size - 1)) for s in series])
    np.testing.assert_allclose(lv.error, expected)
    np.testing.assert_allclose(lv.error_error, expected / np.sqrt(2.0 * (lv.n_blocks - 1)))


def test_blocking_error_keeps_most_recent_power_of_two():
    x = np.random.default_rng(8).standard_normal(1000)
    res = blocking_error(x)
    assert res.levels.n_blocks[0] == 512
    assert res.mean == pytest.approx(x[-512:].mean())
    constant = blocking_error(np.full(64, 0.1))
    assert constant.error == 0.0 and constant.mean == 0.1 and constant.level == 0


def test_blocking_error_flags_insufficient_data():
    # tau_int ~ 1000 >> N: only the last, powerless levels pass the test
    res = blocking_error(ar1(64, 0.999, np.random.default_rng(9)))
    assert not res.reliable and res.levels.n_blocks[res.level] < 32


# --------------------------------------------------------------------------------------
# jackknife
# --------------------------------------------------------------------------------------


def test_jackknife_mean_equals_textbook_error():
    x = np.random.default_rng(10).exponential(size=50)
    res = jackknife(np.mean, x, n_blocks=x.size)
    assert isinstance(res, JackknifeResult)
    assert res.error == pytest.approx(x.std(ddof=1) / np.sqrt(x.size), rel=1e-12)
    assert res.value == pytest.approx(x.mean(), rel=1e-12)
    assert res.bias == pytest.approx(0.0, abs=1e-14)
    assert res.replicates.shape == (50,)
    # default number of blocks: min(N, 64)
    assert jackknife(np.mean, x).replicates.shape == (50,)
    assert jackknife(np.mean, np.arange(1000.0)).replicates.shape == (64,)


def test_jackknife_removes_bias_of_mean_squared():
    rng = np.random.default_rng(11)
    x = rng.normal(0.5, 1.0, size=30)
    res = jackknife(lambda v: np.mean(v) ** 2, x, n_blocks=x.size)
    # exact identity: the corrected estimator is the unbiased xbar^2 - s^2 / n
    assert res.value == pytest.approx(x.mean() ** 2 - x.var(ddof=1) / x.size, rel=1e-10)
    assert res.naive == pytest.approx(x.mean() ** 2, rel=1e-12)
    # over many samples (blocks of two), the naive estimate is biased by sigma^2 / n = 0.1
    samples = rng.normal(0.5, 1.0, size=(2000, 10))
    runs = [jackknife(lambda v: np.mean(v) ** 2, s, n_blocks=5) for s in samples]
    assert np.mean([r.naive for r in runs]) - 0.25 == pytest.approx(0.1, abs=0.03)
    assert np.mean([r.value for r in runs]) == pytest.approx(0.25, abs=0.03)


def test_jackknife_vector_valued():
    x = np.random.default_rng(12).standard_normal(200)

    def moments(v):
        return np.array([v.mean(), (v**2).mean()])

    vec = jackknife(moments, x, n_blocks=20)
    first = jackknife(np.mean, x, n_blocks=20)
    second = jackknife(lambda v: (v**2).mean(), x, n_blocks=20)
    assert vec.replicates.shape == (20, 2)
    for field in ("value", "error", "bias", "naive"):
        expected = [getattr(first, field), getattr(second, field)]
        np.testing.assert_allclose(getattr(vec, field), expected, rtol=1e-12, atol=1e-15)


def test_jackknife_means_agrees_with_jackknife():
    rng = np.random.default_rng(13)
    a = 1.0 + rng.standard_normal(1003)  # not a multiple of the number of blocks
    b = 2.0 + 0.5 * a + rng.standard_normal(1003)
    ref = jackknife(lambda u, v: u.mean() / v.mean(), a, b, n_blocks=20)
    fast = jackknife_means(lambda mu, mv: mu / mv, block_means(a, 20), block_means(b, 20))
    for field in ("value", "error", "bias", "naive"):
        assert getattr(fast, field) == pytest.approx(getattr(ref, field), rel=1e-8)
    np.testing.assert_allclose(fast.replicates, ref.replicates, rtol=1e-12)
    # multi-column block means and a vector-valued f
    ab = np.column_stack((a, b))
    vec = jackknife_means(lambda m: np.array([m[0] / m[1], m[0] * m[1]]), block_means(ab, 20))
    assert vec.value[0] == pytest.approx(ref.value, rel=1e-8)
    assert vec.error.shape == (2,)


def test_jackknife_independent_adds_variances_and_biases():
    rng = np.random.default_rng(14)
    x1 = rng.normal(0.0, 1.0, size=40)
    x2 = rng.normal(2.0, 3.0, size=60)
    e1 = x1.std(ddof=1) / np.sqrt(x1.size)
    e2 = x2.std(ddof=1) / np.sqrt(x2.size)
    res = jackknife_independent(
        lambda ds: ds[0][0].mean() - ds[1][0].mean(), [(x1,), (x2,)], n_blocks=[40, 60]
    )
    assert res.error == pytest.approx(np.hypot(e1, e2), rel=1e-12)
    assert res.value == pytest.approx(x1.mean() - x2.mean(), rel=1e-12)
    assert [r.shape for r in res.replicates] == [(40,), (60,)]
    # biases of independent data sets add: mean1^2 + mean2^2 is corrected exactly
    sq = jackknife_independent(
        lambda ds: ds[0][0].mean() ** 2 + ds[1][0].mean() ** 2, [(x1,), (x2,)], n_blocks=[40, 60]
    )
    expected = x1.mean() ** 2 - e1**2 + x2.mean() ** 2 - e2**2
    assert sq.value == pytest.approx(expected, rel=1e-10)
    # bare arrays; default n_blocks = min(N, 64) = N here
    bare = jackknife_independent(lambda ds: ds[0].mean() - ds[1].mean(), [x1, x2])
    assert bare.error == pytest.approx(res.error, rel=1e-12)


def test_block_means_drops_oldest_samples():
    np.testing.assert_allclose(block_means(np.arange(10.0), 3), [2.0, 5.0, 8.0])
    y = np.arange(20.0).reshape(10, 2)
    np.testing.assert_allclose(block_means(y, 3), [[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]])
    np.testing.assert_allclose(block_means(np.arange(6), 6), np.arange(6.0))


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def test_bootstrap_ci_coverage_exponential_mean():
    # expected coverage ~0.65 for both methods (skewed data, n = 40); sd of the estimate 0.014
    rng = np.random.default_rng(15)
    n, n_rep, n_boot = 40, 1200, 200
    hits = {"percentile": 0, "bca": 0}
    for _ in range(n_rep):
        x = rng.exponential(size=n)  # true mean 1
        seed = int(rng.integers(2**32))
        for method in hits:
            lo, hi = bootstrap(np.mean, x, n_boot=n_boot, rng=seed, method=method).ci
            hits[method] += lo <= 1.0 <= hi
    for method, h in hits.items():
        assert 0.60 <= h / n_rep <= 0.76, (method, h / n_rep)


def test_bootstrap_normal_error_and_interval():
    x = np.random.default_rng(16).standard_normal(400)
    res = bootstrap(np.mean, x, n_boot=2000, rng=17, method="normal")
    assert isinstance(res, BootstrapResult) and res.method == "normal" and res.confidence == CONF
    assert res.value == x.mean() and res.replicates.shape == (2000,)
    assert res.error == pytest.approx(x.std(ddof=1) / np.sqrt(x.size), rel=0.08)
    z = ndtri(0.5 + CONF / 2)
    assert res.ci == pytest.approx((res.value - z * res.error, res.value + z * res.error))


def test_bootstrap_percentile_and_basic_intervals():
    x = np.random.default_rng(18).exponential(size=50)
    pct = bootstrap(np.mean, x, n_boot=500, rng=19)
    basic = bootstrap(np.mean, x, n_boot=500, rng=19, method="basic")
    np.testing.assert_array_equal(basic.replicates, pct.replicates)  # same seed, same resamples
    q = np.quantile(pct.replicates, [0.5 - CONF / 2, 0.5 + CONF / 2])
    assert pct.ci == pytest.approx(tuple(q))
    assert basic.ci == pytest.approx((2 * pct.value - pct.ci[1], 2 * pct.value - pct.ci[0]))


def test_bootstrap_vector_valued_bca_and_block_schemes():
    x = np.random.default_rng(20).exponential(size=60)

    def mean_and_std(v):
        return np.array([v.mean(), v.std()])

    vec = bootstrap(mean_and_std, x, n_boot=300, rng=np.random.default_rng(21), method="bca")
    scalar = bootstrap(np.mean, x, n_boot=300, rng=np.random.default_rng(21), method="bca")
    lo, hi = vec.ci
    assert vec.replicates.shape == (300, 2) and lo.shape == hi.shape == (2,)
    assert np.all(lo < vec.value) and np.all(vec.value < hi)
    assert (lo[0], hi[0]) == pytest.approx(scalar.ci)
    for kwargs in ({"block_length": 5}, {"block_length": 5.0, "stationary": True}):
        res = bootstrap(np.mean, x, n_boot=200, rng=22, method="bca", **kwargs)
        assert res.ci[0] < res.value < res.ci[1]


def test_bootstrap_resamples_arrays_jointly():
    a = np.random.default_rng(23).exponential(size=100)
    for kwargs in ({}, {"block_length": 7}, {"block_length": 7.0, "stationary": True}):
        res = bootstrap(lambda u, v: u.mean() / v.mean(), a, a.copy(), n_boot=50, rng=24, **kwargs)
        np.testing.assert_allclose(res.replicates, 1.0)
        assert res.replicates.std() < 1e-14


def test_stationary_bootstrap_indices():
    n = 100_000
    idx = stationary_bootstrap_indices(n, 10.0, np.random.default_rng(25))
    assert idx.shape == (n,) and idx.min() >= 0 and idx.max() < n
    n_blocks = 1 + np.count_nonzero(idx[1:] != (idx[:-1] + 1) % n)
    assert n / n_blocks == pytest.approx(10.0, rel=0.05)
    np.testing.assert_array_equal(idx, stationary_bootstrap_indices(n, 10.0, 25))
    iid = stationary_bootstrap_indices(1000, 1.0, 26)  # mean block 1: no runs beyond chance
    assert np.count_nonzero(iid[1:] == iid[:-1] + 1) < 10


# --------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------

_X = np.arange(10.0)

INVALID_CALLS = [
    pytest.param(lambda: autocovariance([]), id="autocov-empty"),
    pytest.param(lambda: autocovariance([1.0]), id="autocov-one-sample"),
    pytest.param(lambda: autocovariance([1.0, np.nan, 2.0]), id="autocov-nan"),
    pytest.param(lambda: autocovariance(["a", "b"]), id="autocov-strings"),
    pytest.param(lambda: autocovariance(_X, max_lag=10), id="autocov-lag-too-large"),
    pytest.param(lambda: autocovariance(_X, max_lag=-1), id="autocov-negative-lag"),
    pytest.param(lambda: autocovariance(_X, max_lag=2.5), id="autocov-float-lag"),
    pytest.param(lambda: autocorrelation(np.ones(10)), id="autocorr-constant"),
    pytest.param(lambda: tau_int_sokal(np.ones((10, 2))), id="sokal-2d"),
    pytest.param(lambda: tau_int_sokal(np.ones(10)), id="sokal-constant"),
    pytest.param(lambda: tau_int_sokal(_X, c=0.0), id="sokal-c"),
    pytest.param(lambda: gamma_method([]), id="gamma-empty"),
    pytest.param(lambda: gamma_method(np.ones((10, 2))), id="gamma-f-required"),
    pytest.param(lambda: gamma_method(_X, f="mean"), id="gamma-f-not-callable"),
    pytest.param(lambda: gamma_method(_X, f=lambda a: np.ones(2)), id="gamma-f-vector"),
    pytest.param(lambda: gamma_method(_X, f=lambda a: np.nan), id="gamma-f-nan"),
    pytest.param(lambda: gamma_method(_X, S=-1.0), id="gamma-S"),
    pytest.param(lambda: gamma_method(_X, names=("a", "b")), id="gamma-names"),
    pytest.param(lambda: gamma_method(_X, max_window=0), id="gamma-max-window"),
    pytest.param(lambda: gamma_method(np.ones((4, 2, 2)), ratio), id="gamma-3d"),
    pytest.param(
        lambda: gamma_method([np.ones((10, 2)), np.ones((10, 3))], ratio), id="gamma-mixed-width"
    ),
    pytest.param(lambda: gamma_method([_X, np.arange(1.0)]), id="gamma-short-replica"),
    pytest.param(lambda: effective_sample_size([np.nan, 1.0, 2.0]), id="ess-nan"),
    pytest.param(lambda: blocking_levels(np.ones((4, 4))), id="blocking-2d"),
    pytest.param(lambda: blocking_error(np.arange(15.0)), id="blocking-too-short"),
    pytest.param(lambda: blocking_error(np.arange(64.0), alpha=0.0), id="blocking-alpha"),
    pytest.param(lambda: block_means(np.arange(5.0), 6), id="block-means-too-many"),
    pytest.param(lambda: block_means(np.arange(5.0), 0), id="block-means-zero"),
    pytest.param(lambda: block_means(3.0, 1), id="block-means-scalar"),
    pytest.param(lambda: jackknife(np.mean), id="jk-no-data"),
    pytest.param(lambda: jackknife(np.mean, _X, n_blocks=1), id="jk-one-block"),
    pytest.param(lambda: jackknife(np.mean, _X, n_blocks=11), id="jk-too-many-blocks"),
    pytest.param(lambda: jackknife(np.mean, _X, np.arange(9.0)), id="jk-length-mismatch"),
    pytest.param(lambda: jackknife(lambda v: "x", _X), id="jk-non-numeric-f"),
    pytest.param(lambda: jackknife_means(np.mean, np.arange(1.0)), id="jkm-one-block"),
    pytest.param(lambda: jackknife_independent(ratio, []), id="jki-empty"),
    pytest.param(lambda: jackknife_independent(ratio, [5.0]), id="jki-bad-dataset"),
    pytest.param(lambda: jackknife_independent(ratio, [(_X,)], n_blocks=[2, 3]), id="jki-n-blocks"),
    pytest.param(lambda: bootstrap(np.mean, _X, method="studentized"), id="boot-method"),
    pytest.param(lambda: bootstrap(np.mean, _X, confidence=1.5), id="boot-confidence"),
    pytest.param(lambda: bootstrap(np.mean, _X, n_boot=1), id="boot-n-boot"),
    pytest.param(lambda: bootstrap(np.mean, _X, stationary=True), id="boot-stationary-no-block"),
    pytest.param(lambda: bootstrap(np.mean, _X, block_length=0), id="boot-block-zero"),
    pytest.param(lambda: bootstrap(np.mean, _X, block_length=11), id="boot-block-too-long"),
    pytest.param(lambda: bootstrap(np.mean, _X, block_length=2.5), id="boot-block-float"),
    pytest.param(lambda: bootstrap(np.mean, _X, rng="seed"), id="boot-rng"),
    pytest.param(lambda: stationary_bootstrap_indices(10, 0.5, 1), id="sb-mean-block"),
    pytest.param(lambda: stationary_bootstrap_indices(0, 2.0, 1), id="sb-n"),
]


@pytest.mark.parametrize("call", INVALID_CALLS)
def test_invalid_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()
