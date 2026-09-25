import numpy as np
import pytest

from critmc import Ising, Simulation, square
from critmc.exact import onsager
from critmc.fss import (
    Estimate,
    SizeData,
    binder_crossing,
    collapse_quality,
    data_collapse,
    fit_log,
    fit_power_law,
    fit_shift,
    format_value,
    peak,
)


def test_format_value():
    assert format_value(1.23456, 0.00321) == "1.2346(32)"
    assert format_value(0.5, 0.05) == "0.500(50)"
    assert format_value(1234.5, 21.0) == "1234(21)"
    assert format_value(2.0, 0.0) == "2"
    assert str(Estimate(1.0, 0.25)) == "1.00(25)"


def test_power_law_fit_recovers_exponent():
    rng = np.random.default_rng(0)
    L = np.array([8, 12, 16, 24, 32, 48, 64], dtype=float)
    y = 0.9 * L**1.75 * (1 + 0.3 * L**-1.0)
    dy = 0.002 * y
    y = y + rng.normal(0, dy)
    plain = fit_power_law(L[3:], y[3:], dy[3:])
    corr = fit_power_law(L, y, dy, omega=1.0)
    assert corr["exponent"].value == pytest.approx(1.75, abs=3 * corr["exponent"].error)
    assert corr.p_value > 0.001
    assert plain["exponent"].error > 0
    assert "chi2/dof" in str(corr)


def test_log_and_shift_fits():
    L = np.array([8, 16, 32, 64, 128], dtype=float)
    c = 0.4945 * np.log(L) + 0.14
    res = fit_log(L, c, np.full_like(c, 1e-3))
    assert res["a"].value == pytest.approx(0.4945, abs=1e-6)
    b = 0.44 + 0.5 * L**-1.0
    s = fit_shift(L, b, np.full_like(b, 1e-6))
    assert s["beta_c"].value == pytest.approx(0.44, abs=1e-6)
    assert s["x"].value == pytest.approx(1.0, abs=1e-4)
    fixed = fit_shift(L, b, np.full_like(b, 1e-6), exponent=1.0)
    assert fixed["beta_c"].value == pytest.approx(0.44, abs=1e-8)


def test_data_collapse_recovers_parameters():
    rng = np.random.default_rng(1)
    bc, inv_nu, kappa = 0.44, 1.0, 1.75

    def scaling(x):
        return 1.0 / (1.0 + x**2)

    curves = {}
    for L in (16, 32, 64):
        betas = bc + np.linspace(-1.5, 1.5, 25) / L**inv_nu
        y = L**kappa * scaling((betas - bc) * L**inv_nu)
        dy = 0.01 * y
        curves[L] = (betas, y + rng.normal(0, dy), dy)
    good = collapse_quality(curves, bc, inv_nu, kappa)
    bad = collapse_quality(curves, bc + 0.01, inv_nu, kappa)
    assert good < 3 < bad
    res = data_collapse(curves, bc + 0.003, 0.9, 1.6)
    assert res.beta_c == pytest.approx(bc, abs=2e-3)
    assert res.inv_nu == pytest.approx(inv_nu, abs=0.05)
    assert res.kappa == pytest.approx(kappa, abs=0.05)


@pytest.fixture(scope="module")
def ising_sizes():
    data = {}
    bc = onsager.BETA_C
    for L in (8, 16):
        runs = []
        for k, b in enumerate((bc - 0.02, bc, bc + 0.02)):
            sim = Simulation(Ising(square(L)), b, "wolff", seed=100 * L + k)
            sim.thermalize(500)
            runs.append(sim.run(40_000))
        data[L] = SizeData.from_runs(runs, n_blocks=20)
    return data


def test_binder_crossing_near_exact_critical_point(ising_sizes):
    res = binder_crossing(ising_sizes[8], ising_sizes[16], (0.42, 0.46))
    beta, err = res["beta"]
    assert err > 0
    assert abs(beta - onsager.BETA_C) < 0.01
    assert 0.55 < res["U"].value < 0.66


def test_observables_and_peak(ising_sizes):
    d = ising_sizes[16]
    lo, hi = d.reliable_window()
    # the window covers the runs and ends where the effective sample size collapses
    assert lo < d.betas.min() and hi > d.betas.max()
    assert d.full.ess(hi + 0.05) < 0.05 * d.full.ess(onsager.BETA_C)
    obs = d.at(onsager.BETA_C, ["chi0", "U", "dU", "c"])
    assert all(v.error > 0 for v in obs.values())
    assert obs["dU"].value > 0
    p = peak(d, "c", (0.40, 0.46))
    assert 0.40 < p["beta"].value < 0.46
    assert p["c"].value > obs["c"].value - 3 * obs["c"].error
    vals, errs, reps = d.curve(np.array([0.43, 0.44]), "U")
    assert reps.shape == (20, 2) and np.all(errs > 0)
