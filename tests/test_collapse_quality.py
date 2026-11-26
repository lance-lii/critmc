import numpy as np
import pytest

from critmc.fss import collapse_quality


def reference(curves, beta_c, inv_nu, kappa):
    """The definition written point by point."""
    scaled = []
    for L, (b, y, dy) in sorted(curves.items()):
        x = (np.asarray(b) - beta_c) * L**inv_nu
        s = float(L) ** (-kappa)
        order = np.argsort(x)
        scaled.append((x[order], np.asarray(y)[order] * s, np.asarray(dy)[order] * s))
    total, count = 0.0, 0
    for p, (xp, yp, dyp) in enumerate(scaled):
        for i, xi in enumerate(xp):
            px, py, pw = [], [], []
            for q, (xq, yq, dyq) in enumerate(scaled):
                if q == p or xi < xq[0] or xi > xq[-1]:
                    continue
                j = int(np.clip(np.searchsorted(xq, xi) - 1, 0, len(xq) - 2))
                for k in (j, j + 1):
                    px.append(xq[k])
                    py.append(yq[k])
                    pw.append(1.0 / dyq[k] ** 2)
            if len(px) < 2:
                continue
            px, py, pw = map(np.asarray, (px, py, pw))
            k0, kx, ky = pw.sum(), pw @ px, pw @ py
            kxx, kxy = pw @ (px * px), pw @ (px * py)
            det = k0 * kxx - kx * kx
            if det <= 0:
                continue
            yfit = (kxx * ky - kx * kxy) / det + xi * (k0 * kxy - kx * ky) / det
            var_fit = (kxx - 2 * xi * kx + xi * xi * k0) / det
            total += (yp[i] - yfit) ** 2 / (dyp[i] ** 2 + var_fit)
            count += 1
    return total / count if count else np.inf


@pytest.mark.parametrize("seed", range(5))
def test_matches_the_pointwise_definition_exactly(seed):
    rng = np.random.default_rng(seed)
    curves = {}
    for L in (8, 16, 32, 64):
        n = int(rng.integers(2, 30))
        beta = rng.uniform(0.40, 0.48, n)
        if n > 3:
            beta[1] = beta[0]  # tied abscissae sit on a cell edge
        curves[L] = (beta, rng.random(n), rng.uniform(0.001, 0.02, n))
    for params in [(0.4407, 1.0, 0.0), (0.44, 0.8, 1.75), (0.9, 1.0, 0.0), (0.44, 0.0, 0.0)]:
        assert collapse_quality(curves, *params) == reference(curves, *params)
