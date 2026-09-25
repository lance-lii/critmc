"""Compiled random walks in energy space (Wang-Landau and multicanonical).

Energy levels are addressed by an integer index:

* Ising (even coordination): ``idx = (E_b + B) / 4``, because every spin flip
  changes the bond energy by a multiple of four;
* Potts: ``idx = E + B``.

Moves that would leave the window ``[lo, hi]`` are rejected, and the current
level is updated as for a rejected move, which keeps the walk unbiased at the
window edges [Schulz, Binder, Mueller & Landau, PRE 67, 067102 (2003)].
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .common import randbelow


@njit(cache=True, inline="always")
def _accept(ln_w_old, ln_w_new, rng):
    # accept with min(1, exp(ln_w_new - ln_w_old)) where ln_w = -ln g (weights)
    return ln_w_new >= ln_w_old or rng.random() < np.exp(ln_w_new - ln_w_old)


@njit(cache=True)
def wl_ising(s, nbr, idx, ln_g, hist, ln_f, n_moves, lo, hi, rng, t0, n_levels, one_over_t):
    """Wang-Landau walk; ``one_over_t`` sets ``ln f = n_levels / t`` (Belardinelli-Pereyra)."""
    n, z = nbr.shape
    for t in range(n_moves):
        if one_over_t:
            ln_f = n_levels / (t0 + t + 1.0)
        i = randbelow(rng, n)
        si = s[i]
        field = 0
        for k in range(z):
            field += s[nbr[i, k]]
        new = idx + (si * field) // 2  # dE_b = 2 s_i h_i, divided by 4
        if lo <= new <= hi and _accept(-ln_g[idx], -ln_g[new], rng):
            s[i] = -si
            idx = new
        ln_g[idx] += ln_f
        hist[idx] += 1
    return idx


@njit(cache=True)
def wl_potts(s, nbr, q, idx, ln_g, hist, ln_f, n_moves, lo, hi, rng, t0, n_levels, one_over_t):
    n, z = nbr.shape
    for t in range(n_moves):
        if one_over_t:
            ln_f = n_levels / (t0 + t + 1.0)
        i = randbelow(rng, n)
        old = s[i]
        new_state = randbelow(rng, q - 1)
        if new_state >= old:
            new_state += 1
        n_old = 0
        n_new = 0
        for k in range(z):
            v = s[nbr[i, k]]
            if v == old:
                n_old += 1
            elif v == new_state:
                n_new += 1
        new = idx + n_old - n_new
        if lo <= new <= hi and _accept(-ln_g[idx], -ln_g[new], rng):
            s[i] = new_state
            idx = new
        ln_g[idx] += ln_f
        hist[idx] += 1
    return idx


@njit(cache=True)
def muca_ising(s, nbr, idx, m, ln_w, hist, n_meas, moves_per_meas, lo, hi, rng, idx_out, m_out):
    """Multicanonical walk with fixed weights ``ln_w``; records level and magnetisation."""
    n, z = nbr.shape
    for meas in range(n_meas):
        for _ in range(moves_per_meas):
            i = randbelow(rng, n)
            si = s[i]
            field = 0
            for k in range(z):
                field += s[nbr[i, k]]
            new = idx + (si * field) // 2
            if lo <= new <= hi and _accept(ln_w[idx], ln_w[new], rng):
                s[i] = -si
                idx = new
                m -= 2 * si
            hist[idx] += 1
        idx_out[meas] = idx
        m_out[meas] = m
    return idx, m


@njit(cache=True)
def muca_potts(
    s, nbr, q, idx, counts, ln_w, hist, n_meas, moves_per_meas, lo, hi, rng, idx_out, sq_out
):
    """Multicanonical Potts walk; records level and ``sum_k n_k^2``."""
    n, z = nbr.shape
    for meas in range(n_meas):
        for _ in range(moves_per_meas):
            i = randbelow(rng, n)
            old = s[i]
            new_state = randbelow(rng, q - 1)
            if new_state >= old:
                new_state += 1
            n_old = 0
            n_new = 0
            for k in range(z):
                v = s[nbr[i, k]]
                if v == old:
                    n_old += 1
                elif v == new_state:
                    n_new += 1
            new = idx + n_old - n_new
            if lo <= new <= hi and _accept(ln_w[idx], ln_w[new], rng):
                s[i] = new_state
                counts[old] -= 1
                counts[new_state] += 1
                idx = new
            hist[idx] += 1
        idx_out[meas] = idx
        sq = 0
        for k in range(q):
            sq += counts[k] * counts[k]
        sq_out[meas] = sq
    return idx
