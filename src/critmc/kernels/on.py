"""Compiled kernels for the classical O(n) vector model.

    H = -sum_<ij> S_i . S_j,      S_i in R^n,  |S_i| = 1.

``n = 2`` is the XY (plane-rotator) model, ``n = 3`` the classical Heisenberg
model.  The state is an ``(N, n)`` float64 array of unit vectors.

Cluster updates follow Wolff's embedding [U. Wolff, PRL 62, 361 (1989)]: a
random unit vector ``r`` defines the reflection ``S -> S - 2 (S.r) r`` and an
embedded Ising model with couplings ``J_ij = (S_i.r)(S_j.r)``; bonds are
activated with probability ``1 - exp(-2 beta J_ij)`` when ``J_ij > 0``.

Because the energy is a float it is recomputed from scratch at every
measurement instead of being accumulated.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .common import (
    HEATBATH,
    METROPOLIS,
    METROPOLIS_RANDOM,
    OVERRELAX,
    SWENDSEN_WANG,
    WOLFF,
    find_root,
    randbelow,
    union,
)


@njit(cache=True)
def energy(s, bonds):
    n_comp = s.shape[1]
    e = 0.0
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        dot = 0.0
        for c in range(n_comp):
            dot += s[i, c] * s[j, c]
        e -= dot
    return e


@njit(cache=True)
def magnetization(s, out):
    n_comp = s.shape[1]
    for c in range(n_comp):
        out[c] = 0.0
    for i in range(s.shape[0]):
        for c in range(n_comp):
            out[c] += s[i, c]


@njit(cache=True)
def stiffness_sums(s, bonds, bond_axis, c_out, s_out):
    """Per-axis sums for the spin stiffness of rotations in the (0, 1) plane.

    ``c_out[a] = sum_{bonds along a} (S_i^0 S_j^0 + S_i^1 S_j^1)`` and
    ``s_out[a] = sum_{bonds along a} (S_i^0 S_j^1 - S_i^1 S_j^0)``; for XY spins
    these are ``sum cos(theta_j - theta_i)`` and ``sum sin(theta_j - theta_i)``.
    """
    for a in range(c_out.shape[0]):
        c_out[a] = 0.0
        s_out[a] = 0.0
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        a = bond_axis[b]
        c_out[a] += s[i, 0] * s[j, 0] + s[i, 1] * s[j, 1]
        s_out[a] += s[i, 0] * s[j, 1] - s[i, 1] * s[j, 0]


@njit(cache=True)
def normalize(s):
    n_comp = s.shape[1]
    for i in range(s.shape[0]):
        norm = 0.0
        for c in range(n_comp):
            norm += s[i, c] * s[i, c]
        norm = np.sqrt(norm)
        for c in range(n_comp):
            s[i, c] /= norm


@njit(cache=True, inline="always")
def _local_field(s, nbr, i, field):
    n_comp = s.shape[1]
    for c in range(n_comp):
        field[c] = 0.0
    for k in range(nbr.shape[1]):
        j = nbr[i, k]
        for c in range(n_comp):
            field[c] += s[j, c]


@njit(cache=True, inline="always")
def _random_unit(rng, out):
    norm = 0.0
    for c in range(out.shape[0]):
        out[c] = rng.standard_normal()
        norm += out[c] * out[c]
    norm = np.sqrt(norm)
    for c in range(out.shape[0]):
        out[c] /= norm


@njit(cache=True)
def metropolis_sweep(s, nbr, beta, step, rng, random_order, trial, field):
    """Propose ``S' = (S + step * g) / |S + step * g|`` with Gaussian ``g``.

    The proposal density depends only on ``S . S'``, hence it is symmetric.
    """
    n, _ = nbr.shape
    n_comp = s.shape[1]
    accepted = 0
    for t in range(n):
        i = randbelow(rng, n) if random_order else t
        _local_field(s, nbr, i, field)
        norm = 0.0
        for c in range(n_comp):
            trial[c] = s[i, c] + step * rng.standard_normal()
            norm += trial[c] * trial[c]
        norm = np.sqrt(norm)
        d_e = 0.0
        for c in range(n_comp):
            trial[c] /= norm
            d_e -= (trial[c] - s[i, c]) * field[c]
        if d_e <= 0.0 or rng.random() < np.exp(-beta * d_e):
            for c in range(n_comp):
                s[i, c] = trial[c]
            accepted += 1
    return accepted


@njit(cache=True)
def overrelax_sweep(s, nbr, field):
    """Reflect every spin about its local field (energy conserving)."""
    n, _ = nbr.shape
    n_comp = s.shape[1]
    for i in range(n):
        _local_field(s, nbr, i, field)
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


@njit(cache=True)
def _von_mises(rng, kappa):
    """Best & Fisher (1979) rejection sampler for the von Mises law on (-pi, pi]."""
    if kappa < 1e-8:
        return np.pi * (2.0 * rng.random() - 1.0)
    root = np.sqrt(1.0 + 4.0 * kappa * kappa)
    tau = 1.0 + root
    # rho = (tau - sqrt(2 tau)) / (2 kappa), rearranged to avoid cancellation at small kappa
    rho = 2.0 * kappa * tau / ((root + 1.0) * (tau + np.sqrt(2.0 * tau)))
    r = (1.0 + rho * rho) / (2.0 * rho)
    while True:
        z = np.cos(np.pi * rng.random())
        f = (1.0 + r * z) / (r + z)
        c = kappa * (r - f)
        u2 = rng.random()
        if c * (2.0 - c) - u2 > 0.0 or np.log(c / u2) + 1.0 - c >= 0.0:
            break
    theta = np.arccos(min(1.0, max(-1.0, f)))
    return theta if rng.random() < 0.5 else -theta


@njit(cache=True)
def heatbath_sweep(s, nbr, beta, rng, field, e1, e2):
    """Exact conditional sampling for n = 2 (von Mises) and n = 3 (exponential in cos)."""
    n, _ = nbr.shape
    n_comp = s.shape[1]
    for i in range(n):
        _local_field(s, nbr, i, field)
        hn = 0.0
        for c in range(n_comp):
            hn += field[c] * field[c]
        hn = np.sqrt(hn)
        kappa = beta * hn
        if n_comp == 2:
            phi0 = np.arctan2(field[1], field[0]) if hn > 0.0 else 0.0
            phi = phi0 + _von_mises(rng, kappa)
            s[i, 0] = np.cos(phi)
            s[i, 1] = np.sin(phi)
        else:
            if hn == 0.0:
                _random_unit(rng, e1)
                for c in range(3):
                    s[i, c] = e1[c]
                continue
            r = rng.random()
            if kappa < 1e-8:
                u = 2.0 * r - 1.0
            else:
                u = 1.0 + np.log(r + (1.0 - r) * np.exp(-2.0 * kappa)) / kappa
            u = min(1.0, max(-1.0, u))
            hx = field[0] / hn
            hy = field[1] / hn
            hz = field[2] / hn
            # orthonormal frame (h, e1, e2)
            if abs(hx) < 0.9:
                ax, ay, az = 1.0, 0.0, 0.0
            else:
                ax, ay, az = 0.0, 1.0, 0.0
            proj = ax * hx + ay * hy + az * hz
            e1[0] = ax - proj * hx
            e1[1] = ay - proj * hy
            e1[2] = az - proj * hz
            nrm = np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            e1[0] /= nrm
            e1[1] /= nrm
            e1[2] /= nrm
            e2[0] = hy * e1[2] - hz * e1[1]
            e2[1] = hz * e1[0] - hx * e1[2]
            e2[2] = hx * e1[1] - hy * e1[0]
            phi = 2.0 * np.pi * rng.random()
            w = np.sqrt(max(0.0, 1.0 - u * u))
            cp = np.cos(phi)
            sp = np.sin(phi)
            s[i, 0] = u * hx + w * (cp * e1[0] + sp * e2[0])
            s[i, 1] = u * hy + w * (cp * e1[1] + sp * e2[1])
            s[i, 2] = u * hz + w * (cp * e1[2] + sp * e2[2])


@njit(cache=True)
def wolff_update(s, nbr, beta, rng, queue, mark, tag, proj, rvec):
    """One Wolff cluster flip; returns ``(dE, cluster size)``."""
    n, z = nbr.shape
    n_comp = s.shape[1]
    _random_unit(rng, rvec)
    seed = randbelow(rng, n)
    p0 = 0.0
    for c in range(n_comp):
        p0 += s[seed, c] * rvec[c]
    proj[seed] = p0
    for c in range(n_comp):
        s[seed, c] -= 2.0 * p0 * rvec[c]
    mark[seed] = tag
    queue[0] = seed
    head = 0
    tail = 1
    while head < tail:
        k = queue[head]
        head += 1
        pk = proj[k]  # projection before the flip
        for cc in range(z):
            j = nbr[k, cc]
            if mark[j] == tag:
                continue
            pj = 0.0
            for c in range(n_comp):
                pj += s[j, c] * rvec[c]
            x = pk * pj
            if x > 0.0 and rng.random() < 1.0 - np.exp(-2.0 * beta * x):
                proj[j] = pj
                for c in range(n_comp):
                    s[j, c] -= 2.0 * pj * rvec[c]
                mark[j] = tag
                queue[tail] = j
                tail += 1

    d_e = 0.0
    for t in range(tail):
        k = queue[t]
        pk = proj[k]
        for cc in range(z):
            j = nbr[k, cc]
            if mark[j] != tag:
                pj = 0.0
                for c in range(n_comp):
                    pj += s[j, c] * rvec[c]
                d_e += 2.0 * pk * pj
    return d_e, tail


@njit(cache=True)
def sw_update(s, bonds, beta, rng, parent, size, flag, proj, rvec):
    """Swendsen-Wang update of the embedded Ising model along a random axis."""
    n = s.shape[0]
    n_comp = s.shape[1]
    _random_unit(rng, rvec)
    for i in range(n):
        parent[i] = i
        size[i] = 1
        p = 0.0
        for c in range(n_comp):
            p += s[i, c] * rvec[c]
        proj[i] = p
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        x = proj[i] * proj[j]
        if x > 0.0 and rng.random() < 1.0 - np.exp(-2.0 * beta * x):
            union(parent, size, i, j)
    c2 = 0
    n_clusters = 0
    for i in range(n):
        if parent[i] == i:
            flag[i] = 1 if rng.random() < 0.5 else 0
            c2 += size[i] * size[i]
            n_clusters += 1
    for i in range(n):
        if flag[find_root(parent, i)]:
            for c in range(n_comp):
                s[i, c] -= 2.0 * proj[i] * rvec[c]
    return c2, n_clusters


@njit(cache=True)
def run(
    s,
    nbr,
    bonds,
    bond_axis,
    beta,
    step,
    rng,
    schedule,
    n_meas,
    e_out,
    m_out,
    stiff_c_out,
    stiff_s_out,
    aux_out,
    queue,
    mark,
    tag,
    parent,
    size,
    flag,
    proj,
):
    """Execute ``schedule`` ``n_meas`` times, measuring E, M and stiffness sums."""
    n_comp = s.shape[1]
    rvec = np.empty(n_comp)
    trial = np.empty(n_comp)
    field = np.empty(n_comp)
    e1 = np.empty(n_comp)
    e2 = np.empty(n_comp)
    mvec = np.empty(n_comp)
    n_axes = stiff_c_out.shape[1]
    c_tmp = np.empty(n_axes)
    s_tmp = np.empty(n_axes)
    for meas in range(n_meas):
        accepted = 0
        wolff_size = 0
        c2 = 0
        n_clusters = 0
        for row in range(schedule.shape[0]):
            code = schedule[row, 0]
            for _ in range(schedule[row, 1]):
                if code == METROPOLIS or code == METROPOLIS_RANDOM:
                    accepted += metropolis_sweep(
                        s, nbr, beta, step, rng, code == METROPOLIS_RANDOM, trial, field
                    )
                elif code == HEATBATH:
                    heatbath_sweep(s, nbr, beta, rng, field, e1, e2)
                elif code == OVERRELAX:
                    overrelax_sweep(s, nbr, field)
                elif code == WOLFF:
                    tag += 1
                    _, c = wolff_update(s, nbr, beta, rng, queue, mark, tag, proj, rvec)
                    wolff_size += c
                elif code == SWENDSEN_WANG:
                    c2, n_clusters = sw_update(s, bonds, beta, rng, parent, size, flag, proj, rvec)
        normalize(s)
        e_out[meas] = energy(s, bonds)
        magnetization(s, mvec)
        for c in range(n_comp):
            m_out[meas, c] = mvec[c]
        stiffness_sums(s, bonds, bond_axis, c_tmp, s_tmp)
        for a in range(n_axes):
            stiff_c_out[meas, a] = c_tmp[a]
            stiff_s_out[meas, a] = s_tmp[a]
        aux_out[meas, 0] = accepted
        aux_out[meas, 1] = wolff_size
        aux_out[meas, 2] = c2
        aux_out[meas, 3] = n_clusters
    return tag
