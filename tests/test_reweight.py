"""Reweighting tested on i.i.d. samples drawn from exact energy distributions."""

import numpy as np
import pytest

from critmc.exact import dos, kaufman
from critmc.reweight import MultiHistogram, Reweighter, canonical_observables, kish_ess


@pytest.fixture(scope="module")
def dos8():
    return dos.ising_dos(8)


def draw(d, beta, n, rng):
    p = d.energy_distribution(beta)
    return rng.choice(d.energies, size=n, p=p)


def test_single_histogram_recovers_exact_energy(dos8):
    rng = np.random.default_rng(1)
    b0 = 0.42
    e = draw(dos8, b0, 400_000, rng)
    rw = Reweighter(e, b0)
    n = 64
    for b in (0.40, 0.42, 0.44):
        u = rw.mean(e, b) / n
        exact = kaufman.energy(b, 8)
        # error of the reweighted mean grows as the effective sample size shrinks
        err = np.sqrt(np.var(e) / rw.ess(b)) / n
        assert abs(u - exact) < 5 * err
    lo, hi = rw.reliable_range(0.1)
    assert lo < b0 < hi


def test_reweighting_derivative_is_specific_heat(dos8):
    rng = np.random.default_rng(2)
    b0 = 0.44
    e = draw(dos8, b0, 400_000, rng)
    rw = Reweighter(e, b0)
    c = -(b0**2) * rw.derivative(e, b0) / 64
    assert c == pytest.approx(kaufman.specific_heat(b0, 8), rel=0.02)


def test_canonical_observables_at_beta0_are_plain_averages():
    rng = np.random.default_rng(3)
    e = rng.normal(-100, 5, 1000)
    m = np.abs(rng.normal(0.5, 0.1, 1000))
    obs = canonical_observables(e, m, m**2, m**4, 0.3, 0.3, 64)
    assert obs["absm"] == pytest.approx(m.mean())
    assert obs["U"] == pytest.approx(1 - np.mean(m**4) / (3 * np.mean(m**2) ** 2))
    assert obs["c"] == pytest.approx(0.09 * e.var() / 64)


def test_kish_ess():
    assert kish_ess(np.zeros(100)) == pytest.approx(100)
    assert kish_ess(np.array([0.0, -np.inf, -np.inf])) == pytest.approx(1)


def test_multihistogram_free_energies_match_exact(dos8):
    rng = np.random.default_rng(4)
    betas = [0.36, 0.40, 0.44, 0.48, 0.52]
    energies = [draw(dos8, b, 100_000, rng) for b in betas]
    mh = MultiHistogram(energies, betas)
    assert mh.converged
    exact = kaufman.log_partition(np.array(betas), 8)
    assert np.allclose(-mh.f, exact - exact[0], atol=0.02)
    # interpolation between the runs
    grid = np.linspace(0.37, 0.51, 8)
    u = mh.canonical(grid, 64)["u"]
    assert np.allclose(u, kaufman.energy(grid, 8), atol=4e-3)
    lz = mh.log_z(grid)
    ex = kaufman.log_partition(grid, 8) - kaufman.log_partition(betas[0], 8)
    assert np.allclose(lz, ex, atol=0.02)


def test_multihistogram_with_inefficiencies_and_observables(dos8):
    rng = np.random.default_rng(5)
    betas = [0.40, 0.46]
    energies = [draw(dos8, b, 50_000, rng) for b in betas]
    obs = [{"x": e**2} for e in energies]
    mh = MultiHistogram(energies, betas, obs, inefficiency=[2.0, 3.0])
    b = 0.43
    mean_e = mh.mean(mh.energy, b)
    assert mean_e / 64 == pytest.approx(kaufman.energy(b, 8), abs=5e-3)
    assert mh.mean("x", b) > mean_e**2
    with pytest.raises(ValueError):
        MultiHistogram(energies, betas, inefficiency=[0.5, 1.0])
