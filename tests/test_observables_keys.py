import numpy as np
import pytest

from critmc.reweight import observables_from_weights


def inputs():
    rng = np.random.default_rng(3)
    n = 4001
    w = rng.random(n)
    w /= w.sum()
    energy = rng.integers(-100, 0, n) * 4.0
    absm = rng.random(n)
    return w, energy, absm, absm**2, absm**4, 0.44, 64


def test_each_key_alone_equals_the_full_set_exactly():
    full = observables_from_weights(*inputs())
    assert list(full) == [
        "e", "c", "absm", "m2", "m4", "chi", "chi0", "U", "dU", "dlogm", "dlogm2",
    ]
    for key, value in full.items():
        assert observables_from_weights(*inputs(), keys=(key,)) == {key: value}


def test_unknown_key():
    with pytest.raises(KeyError):
        observables_from_weights(*inputs(), keys=("nope",))
