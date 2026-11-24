import numpy as np
import pytest

from critmc import cubic, square
from critmc.exact.enumeration import ea_energies
from critmc.glass import (
    DisorderAverage,
    EdwardsAnderson,
    ParallelTempering,
    feedback_temperatures,
    geometric_temperatures,
)


@pytest.fixture(scope="module")
def small_sample():
    lat = square(4)
    ea = EdwardsAnderson(lat, seed=11)
    energies = ea_energies(lat, ea.couplings)
    n = lat.n_sites
    states = np.arange(1 << n)
    spins = (2 * ((states[:, None] >> np.arange(n)) & 1) - 1).astype(np.float64)
    return lat, ea, energies, spins


def test_neighbour_couplings_consistent(small_sample):
    lat, ea, energies, spins = small_sample
    s = spins[12345].astype(np.int8)
    e_bonds = ea.energy(s)
    e_sites = -0.5 * sum(
        ea.jn[i, k] * s[i] * s[lat.nbr[i, k]] for i in range(lat.n_sites) for k in range(lat.z)
    )
    assert e_bonds == pytest.approx(e_sites)
    assert e_bonds == pytest.approx(energies[12345])


def test_parallel_tempering_matches_enumeration(small_sample):
    lat, ea, energies, spins = small_sample
    temps = geometric_temperatures(0.4, 2.5, 8)
    run = ParallelTempering(ea, temps, seed=5).run(16)
    eq = run.equilibrium()
    n = lat.n_sites
    b = lat.bonds
    for k, T in enumerate(run.temperatures):
        w = np.exp(-(energies - energies.min()) / T)
        w /= w.sum()
        corr = (spins * w[:, None]).T @ spins
        assert eq["e"][k] == pytest.approx(w @ energies / n, abs=0.01)
        assert eq["q2"][k] == pytest.approx(np.sum(corr**2) / n**2, abs=0.01)
        assert eq["ql"][k] == pytest.approx(np.mean(corr[b[:, 0], b[:, 1]] ** 2), abs=0.01)
    assert np.all(run.swap_rate > 0.3)
    assert run.trips.sum() > 100


def test_gaussian_identity_and_disorder_average():
    lat = cubic(4)
    temps = geometric_temperatures(0.8, 2.0, 8)
    runs = [
        ParallelTempering(EdwardsAnderson(lat, seed=s), temps, seed=100 + s).run(14)
        for s in range(20)
    ]
    avg = DisorderAverage(runs, L=4)
    eq = avg.equilibration()
    # in equilibrium the disorder averages of e and -(z/2)(1 - q_l)/T agree for Gaussian
    # couplings; single samples scatter, so compare within the sample-to-sample error
    z = (eq["e"][-1] - eq["e_from_ql"][-1]) / eq["residual_error"][-1]
    assert np.all(np.abs(z) < 4.0)
    assert avg.identity_residuals().shape == (14, 20, 8)
    obs = avg.observables(n_boot=50)
    g, dg = obs["g"]
    assert g[-1] > g[0]  # temperatures run hot -> cold; g grows towards low T
    assert np.all(dg >= 0)
    assert np.all(obs["xi_L"][0] >= 0)


def test_feedback_temperatures_keep_endpoints():
    temps = np.linspace(0.5, 2.0, 6)
    hist = np.array([[10, 90], [30, 70], [50, 50], [60, 40], [80, 20], [95, 5]])
    new = feedback_temperatures(temps, hist)
    assert new[0] == pytest.approx(0.5) and new[-1] == pytest.approx(2.0)
    assert np.all(np.diff(new) > 0)


def _label_hist(f):
    """Histogram of replica labels with fraction ``f`` coming from the hottest temperature."""
    up = np.round(1e6 * np.asarray(f)).astype(np.int64)
    return np.column_stack([up, 1_000_000 - up])


def test_feedback_temperatures_leave_an_optimal_set_unchanged():
    # f falling by the same amount across every interval is the optimum of
    # Katzgraber et al., whatever the spacing of the temperatures
    temps = geometric_temperatures(0.5, 2.0, 9)
    new = feedback_temperatures(temps, _label_hist(np.linspace(0.0, 1.0, 9)))
    np.testing.assert_allclose(new, temps, rtol=1e-12)


def test_feedback_temperatures_move_towards_a_bottleneck():
    # replicas diffuse freely except across the interval [1.0, 1.25], where f jumps
    temps = np.linspace(0.5, 2.0, 7)
    f = np.array([0.0, 0.05, 0.1, 0.9, 0.95, 1.0, 1.0])
    new = feedback_temperatures(temps, _label_hist(f))
    inside = lambda t: int(np.sum((t > 1.0) & (t < 1.25)))  # noqa: E731
    assert inside(temps) == 0 and inside(new) >= 2
    assert np.all(np.diff(new) > 0)


def test_feedback_temperatures_accept_any_order():
    temps = geometric_temperatures(0.5, 2.0, 9)
    hist = _label_hist(np.linspace(0.0, 1.0, 9))
    np.testing.assert_allclose(
        feedback_temperatures(temps[::-1], hist[::-1]), feedback_temperatures(temps, hist)
    )
