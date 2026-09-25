import numpy as np
import pytest

from critmc.lattice import chain, cubic, hypercubic, make_lattice, square, triangular
from critmc.rng import make_rng, spawn_rngs


@pytest.mark.parametrize(
    "lat",
    [chain(5), square(4), square(3, 5), cubic(3), triangular(4), triangular(3, 5), square(2)],
    ids=repr,
)
def test_neighbour_table_is_symmetric(lat):
    n, z = lat.nbr.shape
    for i in range(n):
        for k in range(z):
            j = lat.nbr[i, k]
            # the opposite column points back
            assert lat.nbr[j, k ^ 1] == i


@pytest.mark.parametrize("lat", [chain(6), square(4), cubic(4), triangular(3)], ids=repr)
def test_every_bond_once(lat):
    assert lat.n_bonds == lat.n_sites * lat.z // 2
    # bond multiset equals the neighbour pairs counted from both ends
    from collections import Counter

    pairs = Counter(tuple(sorted(b)) for b in lat.bonds.tolist())
    nb = Counter(tuple(sorted((i, int(j)))) for i in range(lat.n_sites) for j in lat.nbr[i])
    assert all(nb[p] == 2 * c for p, c in pairs.items())
    assert sum(pairs.values()) == lat.n_bonds


def test_bond_axis_and_geometry():
    lat = square(5)
    dx = lat.coords[lat.bonds[:, 1]] - lat.coords[lat.bonds[:, 0]]
    dx = (dx + 1) % 5 - 1
    for a in range(2):
        sel = lat.bond_axis == a
        assert np.all(dx[sel, a] == 1)
        assert np.all(dx[sel, 1 - a] == 0)


def test_triangular_neighbours_are_at_unit_distance():
    lat = triangular(6)
    for i in range(lat.n_sites):
        for j in lat.nbr[i]:
            d = lat.positions[j] - lat.positions[i]
            # account for periodic images
            best = min(
                np.linalg.norm(d + a * 6 * lat.primitive[0] + b * 6 * lat.primitive[1])
                for a in (-1, 0, 1)
                for b in (-1, 0, 1)
            )
            assert best == pytest.approx(1.0)


def test_sublattice_colouring():
    lat = square(4)
    assert lat.bipartite
    s = lat.sublattice
    assert np.all(s[lat.bonds[:, 0]] != s[lat.bonds[:, 1]])
    assert not square(3).bipartite
    assert not triangular(4).bipartite


def test_site_index_and_kmin():
    lat = cubic(4)
    assert lat.site(1, 2, 3) == 1 + 4 * (2 + 4 * 3)
    assert lat.site(-1, 0, 0) == lat.site(3, 0, 0)
    assert np.allclose(lat.kmin(0), [np.pi / 2, 0, 0])
    tri = triangular(8)
    k = tri.kmin(0)
    assert k @ tri.primitive[0] == pytest.approx(2 * np.pi / 8)
    assert k @ tri.primitive[1] == pytest.approx(0.0, abs=1e-12)


def test_invalid_lattices():
    with pytest.raises(ValueError):
        hypercubic((1, 4))
    with pytest.raises(ValueError):
        make_lattice("kagome", 4)


def test_rng_streams_are_reproducible_and_distinct():
    a = make_rng(42, "fss", 16).random(5)
    b = make_rng(42, "fss", 16).random(5)
    c = make_rng(42, "fss", 32).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    streams = spawn_rngs(7, 3)
    draws = [r.random() for r in streams]
    assert len(set(draws)) == 3
    g = make_rng(1)
    assert make_rng(g) is g
    with pytest.raises(ValueError):
        make_rng(g, "key")
