"""Statistical correctness of every update algorithm against exact results.

Each chain is seeded, so the tests are deterministic; the tolerance is
|z| < 4.5 with batch-means errors, i.e. a correct sampler essentially never
fails while a bias of a few standard errors is caught.
"""

import numpy as np
import pytest
from conftest import assert_consistent

from critmc import ON, Ising, Potts, Simulation, chain, square, triangular
from critmc.exact import chains, enumeration
from critmc.kernels import ising as k_ising
from critmc.kernels import on as k_on
from critmc.kernels import potts as k_potts

ISING_UPDATES = ["metropolis", "metropolis-random", "heatbath", "wolff", "sw", "metropolis+wolff"]


@pytest.fixture(scope="module")
def ising_3x3():
    lat = square(3)
    return lat, enumeration.ising_joint_dos(lat)


@pytest.mark.parametrize("updates", ISING_UPDATES)
def test_ising_matches_enumeration(ising_3x3, updates):
    lat, jd = ising_3x3
    beta = 0.35
    n = lat.n_sites
    sim = Simulation(Ising(lat), beta, updates, seed=12)
    sim.thermalize(2000)
    ts = sim.run(200_000)
    assert_consistent(ts["E"], jd.average(beta, lambda e, m: e))
    assert_consistent(ts["m2"], jd.average(beta, lambda e, m: (m / n) ** 2))
    assert_consistent(ts["m4"], jd.average(beta, lambda e, m: (m / n) ** 4))
    # incremental bookkeeping agrees with a recomputation
    assert sim.running["e_b"] == k_ising.bond_energy(sim.state, lat.bonds)
    assert sim.running["m"] == sim.state.sum()


@pytest.mark.parametrize("updates", ["metropolis", "heatbath", "wolff", "sw"])
def test_triangular_ising_matches_enumeration(updates):
    lat = triangular(3)
    jd = enumeration.ising_joint_dos(lat)
    beta = 0.25
    sim = Simulation(Ising(lat), beta, updates, seed=3)
    sim.thermalize(1000)
    ts = sim.run(150_000)
    assert_consistent(ts["E"], jd.average(beta, lambda e, m: e))
    assert_consistent(ts["m2"], jd.average(beta, lambda e, m: (m / lat.n_sites) ** 2))


@pytest.mark.parametrize("updates", ["metropolis", "heatbath"])
def test_ising_in_a_field(updates):
    lat = square(3)
    beta, h = 0.3, 0.4
    # exact averages with a field from the joint density of states
    jd = enumeration.ising_joint_dos(lat)
    e_b, m = np.meshgrid(jd.energies, jd.orders, indexing="ij")
    with np.errstate(divide="ignore"):
        logw = np.log(jd.counts) - beta * (e_b - h * m)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    sim = Simulation(Ising(lat, h=h), beta, updates, seed=8)
    sim.thermalize(1000)
    ts = sim.run(150_000)
    assert_consistent(ts["E"], float(np.sum(w * (e_b - h * m))))
    assert_consistent(ts["m"], float(np.sum(w * m)) / lat.n_sites)


def test_cluster_updates_reject_fields():
    with pytest.raises(ValueError):
        Simulation(Ising(square(4), h=0.1), 0.4, "wolff")


@pytest.mark.parametrize("updates", ["metropolis", "metropolis-random", "heatbath", "wolff", "sw"])
def test_potts_matches_enumeration(updates):
    lat = square(3)
    q = 3
    jd = enumeration.potts_joint_dos(lat, q)
    beta = 0.8
    n = lat.n_sites
    sim = Simulation(Potts(lat, q), beta, updates, seed=21)
    sim.thermalize(2000)
    ts = sim.run(150_000)
    assert_consistent(ts["E"], jd.average(beta, lambda e, s: e))
    m2_exact = jd.average(beta, lambda e, s: (s / n**2 - 1.0) / (q - 1.0))
    assert_consistent(ts["m2"], m2_exact)
    assert sim.running["energy"] == k_potts.bond_energy(sim.state, lat.bonds)
    assert np.array_equal(sim.running["counts"], np.bincount(sim.state, minlength=q))


@pytest.mark.parametrize("n_comp", [2, 3])
@pytest.mark.parametrize(
    "updates", ["metropolis", "heatbath", "wolff", "sw", "metropolis+3*overrelax"]
)
def test_on_ring_matches_transfer_matrix(n_comp, updates):
    n_sites = 8
    beta = 1.1
    sim = Simulation(ON(chain(n_sites), n_comp), beta, updates, seed=5)
    sim.thermalize(2000)
    ts = sim.run(100_000)
    assert_consistent(ts["e"], chains.on_ring_energy(beta, n_sites, n_comp))
    e = ts["E"]
    c_exact = chains.on_ring_specific_heat(beta, n_sites, n_comp)
    c_mc = beta**2 * e.var() / n_sites
    assert c_mc == pytest.approx(c_exact, rel=0.06)


def test_on_state_stays_normalised_and_energy_consistent():
    lat = square(6)
    sim = Simulation(ON(lat, 3), 0.7, "metropolis+2*overrelax+wolff+sw", seed=1)
    sim.thermalize(200)
    ts = sim.run(200)
    assert np.allclose(np.linalg.norm(sim.state, axis=1), 1.0)
    assert ts["E"][-1] == pytest.approx(k_on.energy(sim.state, lat.bonds))


def test_wolff_cluster_energy_change_is_exact():
    lat = square(8)
    rng = np.random.default_rng(0)
    s = (2 * rng.integers(0, 2, lat.n_sites) - 1).astype(np.int8)
    queue = np.empty(lat.n_sites, dtype=np.int64)
    mark = np.full(lat.n_sites, -1, dtype=np.int64)
    for tag in range(200):
        e0 = k_ising.bond_energy(s, lat.bonds)
        m0 = s.sum()
        d_e, d_m, _ = k_ising.wolff_update(s, lat.nbr, 0.6, rng, queue, mark, tag)
        assert k_ising.bond_energy(s, lat.bonds) - e0 == d_e
        assert s.sum() - m0 == d_m


def test_reproducible_given_seed():
    runs = []
    for _ in range(2):
        sim = Simulation(Potts(square(6), 4), 1.0, "wolff+metropolis", seed=99)
        sim.thermalize(50)
        runs.append(sim.run(100)["E"])
    assert np.array_equal(runs[0], runs[1])


def test_schedule_parsing_and_tuning():
    sim = Simulation(ON(square(8), 2), 1.0, "metropolis+wolff", seed=2)
    info = sim.thermalize(500)
    assert info["wolff_per_meas"] >= 1
    assert 0.001 <= info["step"] <= 10
    with pytest.raises(ValueError):
        Simulation(Ising(square(4)), 0.4, "overrelax")
    with pytest.raises(ValueError):
        Simulation(Ising(square(4)), 0.4, "0*wolff")
    with pytest.raises(ValueError):
        Simulation(Ising(square(4)), 0.4, "gibbs")


def test_wolff_tuning_is_bounded_after_a_hot_start():
    """Regression: tuning from a disordered start must not overshoot the cluster count.

    Clusters of a random configuration are tiny; setting N / <|C|> directly would
    schedule thousands of clusters per measurement, each huge once the system orders.
    """
    sim = Simulation(Ising(square(64)), 0.4406868, "wolff", seed=3, start="random")
    info = sim.thermalize(400)
    history = [1] + info["wolff_history"]
    ratios = np.array(history[1:]) / np.array(history[:-1])
    assert np.all(ratios <= 4.0) and np.all(ratios >= 0.25)
    # at beta_c clusters span a sizeable fraction of the lattice: a few flips per sweep
    assert info["wolff_per_meas"] <= 16
    ts = sim.run(200)
    assert 0.3 < ts.meta["sweeps_per_meas"] < 3.0
