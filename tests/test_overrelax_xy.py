import numpy as np
from numba import njit

from critmc import XY, cubic, square
from critmc.kernels import on


@njit
def general_sweep(s, nbr, field):
    """Over-relaxation with loops over the components, as used for n > 2."""
    n_comp = s.shape[1]
    for i in range(nbr.shape[0]):
        for c in range(n_comp):
            field[c] = 0.0
        for k in range(nbr.shape[1]):
            j = nbr[i, k]
            for c in range(n_comp):
                field[c] += s[j, c]
        hh = 0.0
        sh = 0.0
        for c in range(n_comp):
            hh += field[c] * field[c]
            sh += s[i, c] * field[c]
        if hh == 0.0:
            continue
        f = 2.0 * sh / hh
        for c in range(n_comp):
            s[i, c] = f * field[c] - s[i, c]


def test_xy_sweep_is_bit_identical_to_the_general_loop():
    for lattice in (square(2), square(15), cubic(6)):
        for start in ("random", "ordered"):
            spins = XY(lattice).initial_state(start, np.random.default_rng(5))
            reference = spins.copy()
            for _ in range(50):
                on.overrelax_sweep(spins, lattice.nbr, np.empty(2))
                general_sweep(reference, lattice.nbr, np.empty(2))
                assert spins.tobytes() == reference.tobytes()


def test_zero_local_field_leaves_the_spin():
    lattice = square(4)
    spins = np.zeros((16, 2))
    spins[0] = [1.0, 0.0]
    reference = spins.copy()
    on.overrelax_sweep(spins, lattice.nbr, np.empty(2))
    general_sweep(reference, lattice.nbr, np.empty(2))
    assert spins.tobytes() == reference.tobytes()
