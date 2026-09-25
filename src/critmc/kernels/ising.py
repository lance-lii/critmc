"""Compiled kernels for the Ising model.

    H = -sum_<ij> s_i s_j - h sum_i s_i,      s_i in {-1, +1}.

The state is an ``int8`` array.  Kernels track the integer *bond energy*
``E_b = -sum_<ij> s_i s_j`` and the magnetisation ``M = sum_i s_i`` exactly, so
the total energy ``E = E_b - h M`` never accumulates rounding error.

Cluster updates (Wolff, Swendsen-Wang) use the Fortuin-Kasteleyn bond
probability ``p = 1 - exp(-2 beta)`` and are only valid at ``h = 0``; the Python
layer enforces this.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .common import (
    HEATBATH,
    METROPOLIS,
    METROPOLIS_RANDOM,
    SWENDSEN_WANG,
    WOLFF,
    find_root,
    randbelow,
    union,
)


@njit(cache=True)
def bond_energy(s, bonds):
    e = 0
    for b in range(bonds.shape[0]):
        e -= s[bonds[b, 0]] * s[bonds[b, 1]]
    return e


@njit(cache=True)
def magnetization(s):
    m = 0
    for i in range(s.shape[0]):
        m += s[i]
    return m


@njit(cache=True)
def metropolis_table(beta, h, z):
    """Acceptance probabilities indexed by ``[s_i * sum_j s_j + z, (s_i + 1) // 2]``."""
    table = np.empty((2 * z + 1, 2))
    for a in range(-z, z + 1):
        for t in range(2):
            spin = 2 * t - 1
            d_e = 2.0 * a + 2.0 * h * spin
            table[a + z, t] = 1.0 if d_e <= 0.0 else np.exp(-beta * d_e)
    return table


@njit(cache=True)
def heatbath_table(beta, h, z):
    """Probability that the new spin is +1, indexed by ``sum_j s_j + z``."""
    table = np.empty(2 * z + 1)
    for a in range(-z, z + 1):
        table[a + z] = 1.0 / (1.0 + np.exp(-2.0 * beta * (a + h)))
    return table


@njit(cache=True)
def metropolis_sweep(s, nbr, table, rng, random_order):
    n, z = nbr.shape
    d_eb = 0
    d_m = 0
    accepted = 0
    for step in range(n):
        i = randbelow(rng, n) if random_order else step
        si = s[i]
        field = 0
        for k in range(z):
            field += s[nbr[i, k]]
        a = si * field
        p = table[a + z, (si + 1) >> 1]
        if p >= 1.0 or rng.random() < p:
            s[i] = -si
            d_eb += 2 * a
            d_m -= 2 * si
            accepted += 1
    return d_eb, d_m, accepted


@njit(cache=True)
def heatbath_sweep(s, nbr, p_up, rng):
    n, z = nbr.shape
    d_eb = 0
    d_m = 0
    changed = 0
    for i in range(n):
        field = 0
        for k in range(z):
            field += s[nbr[i, k]]
        new = 1 if rng.random() < p_up[field + z] else -1
        old = s[i]
        if new != old:
            d_eb += 2 * old * field
            d_m += 2 * new
            s[i] = new
            changed += 1
    return d_eb, d_m, changed


@njit(cache=True)
def wolff_update(s, nbr, p_add, rng, queue, mark, tag):
    """Grow and flip one Wolff cluster.

    Cluster members are flipped as soon as they are added, so the test
    ``s[j] == s0`` automatically excludes them.  ``mark`` records membership
    (by the unique ``tag`` of this cluster) for the boundary-energy pass.
    """
    n, z = nbr.shape
    seed = randbelow(rng, n)
    s0 = s[seed]
    s[seed] = -s0
    mark[seed] = tag
    queue[0] = seed
    head = 0
    tail = 1
    while head < tail:
        k = queue[head]
        head += 1
        for c in range(z):
            j = nbr[k, c]
            if s[j] == s0 and rng.random() < p_add:
                s[j] = -s0
                mark[j] = tag
                queue[tail] = j
                tail += 1

    # only bonds crossing the cluster boundary change energy
    d_eb = 0
    for t in range(tail):
        k = queue[t]
        for c in range(z):
            j = nbr[k, c]
            if mark[j] != tag:
                d_eb += 2 * s0 * s[j]
    return d_eb, -2 * s0 * tail, tail


@njit(cache=True)
def sw_update(s, bonds, p_add, rng, parent, size, flag):
    """One Swendsen-Wang update; returns ``(sum_C |C|^2, number of clusters)``."""
    n = s.shape[0]
    for i in range(n):
        parent[i] = i
        size[i] = 1
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        if s[i] == s[j] and rng.random() < p_add:
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
            s[i] = -s[i]
    return c2, n_clusters


@njit(cache=True)
def run(
    s,
    nbr,
    bonds,
    beta,
    h,
    rng,
    schedule,
    n_meas,
    e_b,
    m,
    e_out,
    m_out,
    aux_out,
    queue,
    mark,
    tag,
    parent,
    size,
    flag,
):
    """Execute ``schedule`` ``n_meas`` times, recording ``E_b`` and ``M`` after each.

    Returns the updated ``(E_b, M, tag)``.
    """
    z = nbr.shape[1]
    table = metropolis_table(beta, h, z)
    p_up = heatbath_table(beta, h, z)
    p_add = 1.0 - np.exp(-2.0 * beta)
    for meas in range(n_meas):
        accepted = 0
        wolff_size = 0
        c2 = 0
        n_clusters = 0
        for row in range(schedule.shape[0]):
            code = schedule[row, 0]
            for _ in range(schedule[row, 1]):
                if code == METROPOLIS or code == METROPOLIS_RANDOM:
                    d_eb, d_m, acc = metropolis_sweep(s, nbr, table, rng, code == METROPOLIS_RANDOM)
                    e_b += d_eb
                    m += d_m
                    accepted += acc
                elif code == HEATBATH:
                    d_eb, d_m, acc = heatbath_sweep(s, nbr, p_up, rng)
                    e_b += d_eb
                    m += d_m
                    accepted += acc
                elif code == WOLFF:
                    tag += 1
                    d_eb, d_m, c = wolff_update(s, nbr, p_add, rng, queue, mark, tag)
                    e_b += d_eb
                    m += d_m
                    wolff_size += c
                elif code == SWENDSEN_WANG:
                    c2, n_clusters = sw_update(s, bonds, p_add, rng, parent, size, flag)
                    e_b = bond_energy(s, bonds)
                    m = magnetization(s)
        e_out[meas] = e_b
        m_out[meas] = m
        aux_out[meas, 0] = accepted
        aux_out[meas, 1] = wolff_size
        aux_out[meas, 2] = c2
        aux_out[meas, 3] = n_clusters
    return e_b, m, tag
