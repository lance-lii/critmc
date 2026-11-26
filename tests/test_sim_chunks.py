import numpy as np
import pytest

from critmc import XY, Ising, Potts, Simulation, square


@pytest.mark.parametrize("model", [Ising(square(6)), Potts(square(6), 3), XY(square(6))])
@pytest.mark.parametrize("chunk", [1, 7, 16, 50, 1000])
def test_streams_do_not_depend_on_the_chunk_size(model, chunk):
    whole = Simulation(model, 0.6, "metropolis", seed=4).run(50)
    parts = Simulation(model, 0.6, "metropolis", seed=4).run(50, chunk=chunk)
    assert list(parts.raw) == list(whole.raw)
    for key, value in whole.raw.items():
        assert parts.raw[key].dtype == value.dtype
        assert parts.raw[key].shape == value.shape
        np.testing.assert_array_equal(parts.raw[key], value)
