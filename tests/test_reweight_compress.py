import numpy as np
import pytest

from critmc.reweight import MultiHistogram


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("inefficiency", [None, [1.0, 3.0, 3.0]])
def test_groups_match_unique_rows(discrete, inefficiency):
    rng = np.random.default_rng(2)
    energies = []
    for k in range(3):
        if discrete:
            e = rng.integers(-15 + 4 * k, 6 + 4 * k, 500) * 4.0
            e[:3] = [0.0, -0.0, 0.0]  # signed zeros belong to one level
        else:
            e = rng.normal(-3.0 + k, 2.0, 300)
        energies.append(e)
    mh = MultiHistogram(energies, [0.30, 0.28, 0.26], inefficiency=inefficiency)

    rows = np.stack([mh.energy, mh.sample_logw], axis=1)
    uniq, inverse, count = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(mh._u_energy, uniq[:, 0] - mh._shift)
    np.testing.assert_array_equal(mh._u_logmult, np.log(count) + uniq[:, 1])
    np.testing.assert_array_equal(mh._inverse, inverse.ravel())
