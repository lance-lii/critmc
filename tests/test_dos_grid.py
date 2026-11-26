import pytest

from critmc.exact import dos, enumeration
from critmc.lattice import hypercubic


@pytest.mark.parametrize("shape", [(2, 2), (2, 4), (4, 4), (2, 8), (4, 8), (2, 6), (4, 6)])
def test_counts_including_the_folded_end_levels(shape):
    L, M = shape
    d = dos.ising_dos(L, M)
    assert len(d.g) == L * M + 1
    assert d.g[0] == d.g[-1] == 2
    assert sum(d.g) == 2 ** (L * M)
    if L * M <= 16:
        joint = enumeration.ising_joint_dos(hypercubic((L, M)))
        by_energy = dict(zip(joint.energies.astype(int), joint.counts.sum(axis=1)))
        assert [by_energy.get(int(e), 0) for e in d.energies] == list(d.g)


@pytest.mark.parametrize("shape, points", [((4, 4), 8), ((8, 8), 32), ((4, 6), 16)])
def test_number_of_evaluations(monkeypatch, shape, points):
    calls = []
    evaluate = dos.partition_polynomial_value
    monkeypatch.setattr(
        dos, "partition_polynomial_value", lambda *a: calls.append(1) or evaluate(*a)
    )
    dos.ising_dos(*shape)
    assert len(calls) == points  # half the grid; the rest are complex conjugates
