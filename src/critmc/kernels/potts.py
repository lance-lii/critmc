"""Compiled kernels for the q-state Potts model.

    H = -sum_<ij> delta(s_i, s_j),      s_i in {0, ..., q-1}.

The integer energy ``E = -(number of satisfied bonds)`` and the occupation
numbers ``counts[k] = #{i : s_i = k}`` are tracked exactly.  The
Fortuin-Kasteleyn bond probability is ``p = 1 - exp(-beta)``.
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
        if s[bonds[b, 0]] == s[bonds[b, 1]]:
            e -= 1
    return e


@njit(cache=True)
def state_counts(s, q, counts):
    for k in range(q):
        counts[k] = 0
    for i in range(s.shape[0]):
        counts[s[i]] += 1


@njit(cache=True)
def metropolis_sweep(s, nbr, q, boltz, rng, random_order, counts):
    """Propose a uniformly chosen *different* state; ``boltz[d] = exp(-beta d)``."""
    n, z = nbr.shape
    d_e = 0
    accepted = 0
    for step in range(n):
        i = randbelow(rng, n) if random_order else step
        old = s[i]
        new = randbelow(rng, q - 1)
        if new >= old:
            new += 1
        n_old = 0
        n_new = 0
        for k in range(z):
            v = s[nbr[i, k]]
            if v == old:
                n_old += 1
            elif v == new:
                n_new += 1
        delta = n_old - n_new
        if delta <= 0 or rng.random() < boltz[delta]:
            s[i] = new
            counts[old] -= 1
            counts[new] += 1
            d_e += delta
            accepted += 1
    return d_e, accepted


@njit(cache=True)
def heatbath_sweep(s, nbr, q, weight, rng, counts, nk):
    """Sample each site from its conditional distribution; ``weight[n] = exp(beta n)``."""
    n, z = nbr.shape
    d_e = 0
    changed = 0
    for i in range(n):
        for k in range(q):
            nk[k] = 0
        for c in range(z):
            nk[s[nbr[i, c]]] += 1
        total = 0.0
        for k in range(q):
            total += weight[nk[k]]
        r = rng.random() * total
        new = q - 1
        cum = 0.0
        for k in range(q):
            cum += weight[nk[k]]
            if r < cum:
                new = k
                break
        old = s[i]
        if new != old:
            d_e += nk[old] - nk[new]
            counts[old] -= 1
            counts[new] += 1
            s[i] = new
            changed += 1
    return d_e, changed


@njit(cache=True)
def wolff_update(s, nbr, q, p_add, rng, queue, mark, tag, counts):
    """Grow a cluster of state ``a`` and recolour it with a random ``b != a``."""
    n, z = nbr.shape
    seed = randbelow(rng, n)
    a = s[seed]
    b = randbelow(rng, q - 1)
    if b >= a:
        b += 1
    s[seed] = b
    mark[seed] = tag
    queue[0] = seed
    head = 0
    tail = 1
    while head < tail:
        k = queue[head]
        head += 1
        for c in range(z):
            j = nbr[k, c]
            if s[j] == a and rng.random() < p_add:
                s[j] = b
                mark[j] = tag
                queue[tail] = j
                tail += 1

    d_e = 0
    for t in range(tail):
        k = queue[t]
        for c in range(z):
            j = nbr[k, c]
            if mark[j] != tag:
                v = s[j]
                if v == a:
                    d_e += 1  # (a, a) satisfied -> (b, a) broken
                elif v == b:
                    d_e -= 1  # (a, b) broken -> (b, b) satisfied
    counts[a] -= tail
    counts[b] += tail
    return d_e, tail


@njit(cache=True)
def sw_update(s, bonds, q, p_add, rng, parent, size, flag):
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
            flag[i] = randbelow(rng, q)
            c2 += size[i] * size[i]
            n_clusters += 1
    for i in range(n):
        s[i] = flag[find_root(parent, i)]
    return c2, n_clusters


@njit(cache=True)
def run(
    s,
    nbr,
    bonds,
    q,
    beta,
    rng,
    schedule,
    n_meas,
    energy,
    counts,
    e_out,
    counts_out,
    aux_out,
    queue,
    mark,
    tag,
    parent,
    size,
    flag,
):
    """Execute ``schedule`` ``n_meas`` times recording the energy and occupations."""
    z = nbr.shape[1]
    boltz = np.empty(z + 1)
    weight = np.empty(z + 1)
    for d in range(z + 1):
        boltz[d] = np.exp(-beta * d)
        weight[d] = np.exp(beta * d)
    nk = np.empty(q, dtype=np.int64)
    p_add = 1.0 - np.exp(-beta)
    for meas in range(n_meas):
        accepted = 0
        wolff_size = 0
        c2 = 0
        n_clusters = 0
        for row in range(schedule.shape[0]):
            code = schedule[row, 0]
            for _ in range(schedule[row, 1]):
                if code == METROPOLIS or code == METROPOLIS_RANDOM:
                    d_e, acc = metropolis_sweep(
                        s, nbr, q, boltz, rng, code == METROPOLIS_RANDOM, counts
                    )
                    energy += d_e
                    accepted += acc
                elif code == HEATBATH:
                    d_e, acc = heatbath_sweep(s, nbr, q, weight, rng, counts, nk)
                    energy += d_e
                    accepted += acc
                elif code == WOLFF:
                    tag += 1
                    d_e, c = wolff_update(s, nbr, q, p_add, rng, queue, mark, tag, counts)
                    energy += d_e
                    wolff_size += c
                elif code == SWENDSEN_WANG:
                    c2, n_clusters = sw_update(s, bonds, q, p_add, rng, parent, size, flag)
                    energy = bond_energy(s, bonds)
                    state_counts(s, q, counts)
        e_out[meas] = energy
        for k in range(q):
            counts_out[meas, k] = counts[k]
        aux_out[meas, 0] = accepted
        aux_out[meas, 1] = wolff_size
        aux_out[meas, 2] = c2
        aux_out[meas, 3] = n_clusters
    return energy, tag
