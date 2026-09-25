"""Compiled parallel tempering for Ising spin glasses.

``H = -sum_<ij> J_ij s_i s_j`` with quenched couplings.  ``jn[i, k]`` is the
coupling of the bond between ``i`` and ``nbr[i, k]``.

Layout: ``spins[r, c, :]`` is configuration ``c`` of replica set ``r``
(``r = 0, 1`` give the two independent copies needed for overlaps), and
``perm[r, k]`` is the configuration currently at temperature index ``k``.
Replica-exchange moves only permute ``perm``, never copy spins.

Measurements are accumulated in logarithmic bins of PT steps: bin ``b``
covers steps ``[2^b - 1, 2^(b+1) - 1)``.  Comparing successive bins is the
standard equilibration test, and the last bin (the second half of the run)
provides the equilibrium thermal averages.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# columns of the accumulator acc[bin, temperature, :]
OBS_E = 0  # energy per spin (average of both replica sets)
OBS_QL = 1  # link overlap
OBS_Q = 2  # |q|
OBS_Q2 = 3  # q^2
OBS_Q4 = 4  # q^4
OBS_QK2 = 5  # |q(k_min)|^2 averaged over lattice directions
OBS_COUNT = 6  # number of measurements
N_OBS = 7


@njit(cache=True)
def energy(s, bonds, jb):
    e = 0.0
    for b in range(bonds.shape[0]):
        e -= jb[b] * s[bonds[b, 0]] * s[bonds[b, 1]]
    return e


@njit(cache=True)
def metropolis_sweep(s, nbr, jn, beta, rng):
    n, z = nbr.shape
    d_total = 0.0
    for i in range(n):
        h = 0.0
        for k in range(z):
            h += jn[i, k] * s[nbr[i, k]]
        d_e = 2.0 * s[i] * h
        if d_e <= 0.0 or rng.random() < np.exp(-beta * d_e):
            s[i] = -s[i]
            d_total += d_e
    return d_total


@njit(cache=True)
def _measure(sa, sb, bonds, cos_phase, sin_phase):
    n = sa.shape[0]
    q = 0
    for i in range(n):
        q += sa[i] * sb[i]
    ql = 0
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        ql += sa[i] * sa[j] * sb[i] * sb[j]
    n_dir = cos_phase.shape[0]
    qk2 = 0.0
    for d in range(n_dir):
        re = 0.0
        im = 0.0
        for i in range(n):
            qi = sa[i] * sb[i]
            re += qi * cos_phase[d, i]
            im += qi * sin_phase[d, i]
        qk2 += (re * re + im * im) / (n * n)
    return q / n, ql / bonds.shape[0], qk2 / n_dir


@njit(cache=True)
def pt_run(
    spins,
    energies,
    perm,
    labels,
    nbr,
    jn,
    bonds,
    jb,
    betas,
    rng,
    n_bins,
    sweeps_per_step,
    cos_phase,
    sin_phase,
    acc,
    swap_acc,
    swap_try,
    label_hist,
    trips,
):
    """Run ``2^n_bins - 1`` parallel-tempering steps.

    ``labels[r, c]`` is 1 if configuration ``c`` last visited the hottest
    temperature (index 0) and 2 if it last visited the coldest one;
    ``label_hist[k, 0/1]`` counts labels 1/2 seen at temperature ``k`` (input
    to feedback optimisation) and ``trips[r]`` counts one-way hot<->cold passages.
    """
    n_rep = spins.shape[0]
    m = betas.shape[0]
    n = spins.shape[2]
    total = (1 << n_bins) - 1
    b = 0
    next_edge = 1
    for t in range(total):
        if t >= next_edge:
            b += 1
            next_edge = (1 << (b + 1)) - 1
            # remove accumulated rounding error of the incremental energies
            for r in range(n_rep):
                for c in range(m):
                    energies[r, c] = energy(spins[r, c], bonds, jb)
        for r in range(n_rep):
            for k in range(m):
                c = perm[r, k]
                for _ in range(sweeps_per_step):
                    energies[r, c] += metropolis_sweep(spins[r, c], nbr, jn, betas[k], rng)
            # replica exchange between neighbouring temperatures
            for k in range(m - 1):
                c1 = perm[r, k]
                c2 = perm[r, k + 1]
                x = (betas[k] - betas[k + 1]) * (energies[r, c1] - energies[r, c2])
                swap_try[k] += 1
                if x >= 0.0 or rng.random() < np.exp(x):
                    perm[r, k] = c2
                    perm[r, k + 1] = c1
                    swap_acc[k] += 1
            # round-trip bookkeeping
            hot = perm[r, 0]
            cold = perm[r, m - 1]
            if labels[r, hot] == 2:
                trips[r] += 1
            labels[r, hot] = 1
            if labels[r, cold] == 1:
                trips[r] += 1
            labels[r, cold] = 2
            for k in range(m):
                lab = labels[r, perm[r, k]]
                if lab == 1:
                    label_hist[k, 0] += 1
                elif lab == 2:
                    label_hist[k, 1] += 1
        for k in range(m):
            ca = perm[0, k]
            cb = perm[1, k]
            q, ql, qk2 = _measure(spins[0, ca], spins[1, cb], bonds, cos_phase, sin_phase)
            acc[b, k, OBS_E] += 0.5 * (energies[0, ca] + energies[1, cb]) / n
            acc[b, k, OBS_QL] += ql
            acc[b, k, OBS_Q] += abs(q)
            acc[b, k, OBS_Q2] += q * q
            acc[b, k, OBS_Q4] += q * q * q * q
            acc[b, k, OBS_QK2] += qk2
            acc[b, k, OBS_COUNT] += 1.0
