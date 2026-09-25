import numpy as np
import pytest

from critmc import Ising, Potts, square
from critmc.exact import dos, enumeration, kaufman, potts
from critmc.flathist import EnergyLevels, Multicanonical, WangLandau


def test_energy_levels_mapping():
    lv = EnergyLevels.of(Ising(square(4)))
    assert lv.size == 17
    assert lv.energies[0] == -32 and lv.energies[-1] == 32
    assert lv.index(-24) == 2
    pv = EnergyLevels.of(Potts(square(4), 3))
    assert pv.size == 33 and pv.index(-32) == 0
    assert lv.window(-2.0, 0.0) == (0, 8)


@pytest.mark.parametrize("one_over_t", [False, True])
def test_wang_landau_ising_against_exact_dos(one_over_t):
    L = 6
    wl = WangLandau(Ising(square(L)), seed=3, criterion="flat")
    res = wl.run(1e-6, one_over_t=one_over_t)
    exact = dos.ising_dos(L).log_g
    got = res.normalized()
    v = res.visited
    assert np.array_equal(v, np.isfinite(exact))
    err = np.abs(got[v] - exact[v])
    # plain Wang-Landau saturates at a finite error; the 1/t variant keeps converging
    assert err.mean() < (0.02 if one_over_t else 0.08)
    th = res.canonical([0.3, 0.44, 0.6])
    tol = 0.01 if one_over_t else 0.03
    assert np.allclose(th["u"], kaufman.energy(np.array([0.3, 0.44, 0.6]), L), atol=tol)


def test_wang_landau_checkpoints_and_stages():
    wl = WangLandau(Ising(square(4)), seed=1)
    res = wl.run(1e-5, checkpoints=[10_000, 100_000, 1_000_000])
    assert [c[0] for c in res.checkpoints] == [10_000, 100_000, 1_000_000]
    assert res.stages and res.stages[0]["ln_f"] == 1.0


def test_wang_landau_potts_window():
    lat = square(4, 3)
    jd = enumeration.potts_joint_dos(lat, 3)
    exact = dict(zip(jd.energies, np.log(jd.counts.sum(axis=1))))
    wl = WangLandau(Potts(lat, 3), e_window=(-2.0, -1.0), seed=4)
    res = wl.run(1e-7)
    got = res.normalized()
    lv = res.levels.energies
    ref = [exact[e] for e in lv[res.visited]]
    # window normalisation fixes the lowest level; compare differences
    diff = got[res.visited] - np.array(ref)
    assert np.ptp(diff) < 0.02
    assert lv[res.visited].max() <= -12


def test_multicanonical_reweighting_to_canonical():
    L = 6
    exact_lng = dos.ising_dos(L).log_g
    mu = Multicanonical(Ising(square(L)), exact_lng, seed=8)
    out = mu.run(200_000, sweeps_per_meas=1, n_therm=100)
    visited = out.hist > 0
    assert visited.sum() == np.isfinite(exact_lng).sum()
    for b in (0.3, 0.44, 0.55):
        assert out.canonical(b)["u"][0] == pytest.approx(kaufman.energy(b, L), abs=0.01)


def test_multicanonical_potts_double_peak():
    model = Potts(square(8), 10)
    wl = WangLandau(model, e_window=(-1.95, -0.6), seed=9)
    res = wl.run(1e-5)
    mu = Multicanonical(model, res.ln_g, e_window=(-1.95, -0.6), seed=10)
    out = mu.run(100_000)
    bc = potts.beta_c(10)
    b = out.equal_height_beta((bc - 0.05, bc + 0.05), split=-1.3)
    dp = out.double_peak(b, split=-1.3)
    assert dp["p_ordered"] == pytest.approx(dp["p_disordered"], rel=1e-6)
    assert dp["e_ordered"] < -1.5 < -1.1 < dp["e_disordered"]
    assert abs(b - bc) < 0.05
