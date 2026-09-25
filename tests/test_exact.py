import mpmath as mp
import numpy as np
import pytest

from critmc.exact import chains, dos, enumeration, kaufman, onsager, potts, zeros
from critmc.lattice import hypercubic, square


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 4), (3, 4), (4, 3), (2, 6), (4, 5)])
def test_kaufman_matches_enumeration(shape):
    lat = hypercubic(shape)
    jd = enumeration.ising_joint_dos(lat)
    n = lat.n_sites
    for beta in (0.05, 0.3, onsager.BETA_C, 0.7, 1.5):
        th = kaufman.thermodynamics(beta, *shape)
        u = jd.average(beta, lambda e, m: e) / n
        e2 = jd.average(beta, lambda e, m: e * e)
        c = beta**2 * (e2 - (u * n) ** 2) / n
        assert th["lnZ"][0] == pytest.approx(jd.log_z(beta), abs=1e-11)
        assert th["u"][0] == pytest.approx(u, abs=1e-11)
        assert th["c"][0] == pytest.approx(c, abs=1e-10)


def test_kaufman_at_the_critical_point_is_finite():
    th = kaufman.thermodynamics(onsager.BETA_C, 64)
    assert np.all(np.isfinite([th["u"][0], th["c"][0], th["lnZ"][0]]))
    # finite-size specific heat at K_c grows like (8 K_c^2 / pi) ln L
    c1 = kaufman.specific_heat(onsager.BETA_C, 512)
    c2 = kaufman.specific_heat(onsager.BETA_C, 1024)
    assert (c2 - c1) / np.log(2) == pytest.approx(onsager.heat_capacity_amplitude(), rel=1e-3)


@pytest.mark.parametrize("T", [1.5, 2.0, 2.2, 2.4, 3.0, 5.0])
def test_kaufman_converges_to_onsager(T):
    beta = 1.0 / T
    th = kaufman.thermodynamics(beta, 400)
    assert th["u"][0] == pytest.approx(onsager.energy(T), abs=2e-9)
    if abs(T - onsager.T_C) > 0.1:
        assert th["c"][0] == pytest.approx(onsager.specific_heat(T), abs=1e-6)
    # below T_c the torus has a two-fold degenerate ground state: f_L = f - T ln2 / N
    shift = T * np.log(2) / 400**2 if T < onsager.T_C else 0.0
    assert th["f"][0] == pytest.approx(onsager.free_energy(T) - shift, abs=1e-9)


def test_onsager_thermodynamic_consistency():
    for T in (1.7, 2.1, 2.5, 3.5):
        b, h = 1.0 / T, 1e-5
        bf = lambda x: x * onsager.free_energy(1.0 / x)  # noqa: E731
        assert (bf(b + h) - bf(b - h)) / (2 * h) == pytest.approx(onsager.energy(T), abs=1e-7)
        du = (onsager.energy(1 / (b + h)) - onsager.energy(1 / (b - h))) / (2 * h)
        assert -(b**2) * du == pytest.approx(onsager.specific_heat(T), rel=1e-6)
    assert onsager.magnetization(2.0) == pytest.approx(0.9113, abs=1e-4)
    assert onsager.magnetization(2.5) == 0.0
    assert onsager.T_C == pytest.approx(2.269185314213022)


def test_exact_dos_small_lattice_matches_enumeration():
    d = dos.ising_dos(4)
    jd = enumeration.ising_joint_dos(square(4))
    by_energy = dict(zip(jd.energies.astype(int), jd.counts.sum(axis=1)))
    for e, g in zip(d.energies.astype(int), d.g):
        assert g == by_energy.get(e, 0)


@pytest.mark.parametrize("L", [6, 8, 12])
def test_exact_dos_reproduces_kaufman(L):
    d = dos.ising_dos(L)
    assert sum(d.g) == 2 ** (L * L)
    betas = np.array([0.1, 0.35, onsager.BETA_C, 0.6])
    c = d.canonical(betas)
    th = kaufman.thermodynamics(betas, L)
    assert np.allclose(c["u"], th["u"], atol=1e-12)
    assert np.allclose(c["c"], th["c"], atol=1e-10)
    assert np.allclose(c["lnZ"], th["lnZ"], atol=1e-10)


def test_exact_dos_rectangular_and_cache(tmp_path):
    d = dos.ising_dos(4, 6, cache_dir=tmp_path)
    assert (tmp_path / "ising_dos_4x6.txt").exists()
    again = dos.ising_dos(4, 6, cache_dir=tmp_path)
    assert again.g == d.g
    lat = hypercubic((4, 6))
    jd = enumeration.ising_joint_dos(lat)
    assert d.log_z(0.3)[0] == pytest.approx(jd.log_z(0.3), abs=1e-12)
    with pytest.raises(ValueError):
        dos.ising_dos(5)


def test_polynomial_value_matches_real_kaufman():
    with mp.workdps(30):
        L, beta = 6, 0.37
        p = dos.partition_polynomial_value(mp.exp(-4 * beta), L)
        ln_z = float(mp.log(p.real)) + 2 * L * L * beta
    assert ln_z == pytest.approx(kaufman.log_partition(beta, L), abs=1e-12)


def test_fisher_zeros_small_lattice():
    z = zeros.fisher_zeros(dos.ising_dos(4))
    assert len(z) == 2 * 16
    # zeros come in +- pairs and complex-conjugate pairs
    assert np.allclose(np.sort_complex(z), np.sort_complex(-z), atol=1e-8)
    assert np.allclose(np.sort_complex(z), np.sort_complex(np.conj(z)), atol=1e-8)


def test_leading_fisher_zeros_give_nu_equal_one():
    lz = zeros.leading_zeros([8, 16, 32, 64, 128])
    L = np.array([z.L for z in lz], dtype=float)
    d = np.array([z.distance for z in lz])
    slope = np.polyfit(np.log(L[-3:]), np.log(d[-3:]), 1)[0]
    assert -slope == pytest.approx(1.0, abs=2e-3)
    # the leading zero approaches u_c = sqrt(2) - 1 from the upper half plane
    assert all(z.u.imag > 0 for z in lz)
    assert lz[-1].distance < 6e-3


def test_potts_exact_values():
    assert potts.beta_c(2) == pytest.approx(np.log(1 + np.sqrt(2)))
    assert potts.latent_heat(10) == pytest.approx(0.696049, abs=1e-6)
    assert potts.latent_heat(4) == 0.0
    eo, ed = potts.phase_energies(10)
    assert eo == pytest.approx(-1.664253, abs=1e-6)
    assert ed == pytest.approx(-0.968203, abs=1e-6)
    # the 2-state Potts model is the Ising model with K_Ising = K_Potts / 2
    assert potts.beta_c(2) / 2 == pytest.approx(onsager.BETA_C)


def test_on_chain_limits():
    from scipy.special import iv

    for b in (0.4, 1.0, 2.5):
        assert chains.on_ring_energy(b, 400, 2) == pytest.approx(-iv(1, b) / iv(0, b), rel=1e-10)
        langevin = 1 / np.tanh(b) - 1 / b
        assert chains.on_ring_energy(b, 400, 3) == pytest.approx(-langevin, rel=1e-10)
    # specific heat is the beta-derivative of the energy
    b, h = 0.9, 1e-5
    du = (chains.on_ring_energy(b + h, 6, 3) - chains.on_ring_energy(b - h, 6, 3)) / (2 * h)
    assert -(b**2) * du == pytest.approx(chains.on_ring_specific_heat(b, 6, 3), rel=1e-6)


def test_potts_enumeration_consistency():
    lat = square(3)
    jd = enumeration.potts_joint_dos(lat, 2)
    ji = enumeration.ising_joint_dos(lat)
    # q = 2 Potts with coupling 2K equals Ising with coupling K: E_potts = (E_ising - B) / 2
    beta = 0.3
    u_p = jd.average(2 * beta, lambda e, s: e)
    u_i = ji.average(beta, lambda e, m: e)
    assert u_p == pytest.approx((u_i - lat.n_bonds) / 2, abs=1e-12)
