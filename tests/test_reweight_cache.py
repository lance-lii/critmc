import numpy as np
from scipy.special import logsumexp

from critmc.reweight import MultiHistogram


def build():
    rng = np.random.default_rng(1)
    energies = [rng.integers(-200 + 20 * k, -100 + 20 * k, 400) * 4.0 for k in range(3)]
    return MultiHistogram(energies, [0.46, 0.44, 0.42], inefficiency=[1.0, 2.5, 4.0])


def uncached_weights(mh, beta):
    ld = mh._log_denominator(mh._fs)
    lw = (-beta * mh._u_energy - ld)[mh._inverse] + mh.sample_logw
    w = np.exp(lw - lw.max())
    return w / w.sum()


def test_cached_denominator_gives_the_same_bytes():
    mh = build()
    for beta in (0.40, 0.43, 0.43, 0.47):
        assert mh.weights(beta).tobytes() == uncached_weights(mh, beta).tobytes()
        ld = mh._log_denominator(mh._fs)
        direct = logsumexp(mh._u_logmult - beta * mh._u_energy - ld)
        assert mh.log_z(beta)[0] == direct - (beta - mh.betas[0]) * mh._shift


def test_solving_again_refreshes_the_denominator():
    mh = build()
    before = mh.weights(0.43)
    mh.betas = mh.betas + np.array([0.0, 0.003, -0.002])
    mh.solve()
    after = mh.weights(0.43)
    assert after.tobytes() == uncached_weights(mh, 0.43).tobytes()
    assert after.tobytes() != before.tobytes()
