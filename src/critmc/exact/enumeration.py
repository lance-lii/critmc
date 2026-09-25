"""Brute-force enumeration of small lattices.

The joint density of states ``g(E, M)`` of every configuration is the
ultimate ground truth for testing: from it any canonical average at any
temperature follows exactly.  Enumeration is vectorised over chunks of
configurations encoded as integers and is practical up to ``N ~ 25`` Ising
spins or ``q^N ~ 10^7`` Potts states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ..lattice import Lattice

__all__ = ["JointDOS", "ising_joint_dos", "potts_joint_dos", "ea_energies"]

_CHUNK = 1 << 18


@dataclass(frozen=True)
class JointDOS:
    """Exact joint density of states.

    ``counts[a, b]`` is the number of configurations with energy
    ``energies[a]`` and order parameter value ``orders[b]``.  For the Ising
    model the order parameter is the magnetisation ``M``; for the Potts model
    it is ``q sum_k n_k^2`` (from which ``|m|^2`` follows).
    """

    energies: np.ndarray
    orders: np.ndarray
    counts: np.ndarray
    n_sites: int

    def _weights(self, beta: float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            log_w = np.log(self.counts) - beta * self.energies[:, None]
        return np.exp(log_w - logsumexp(log_w))

    def log_z(self, beta: float) -> float:
        with np.errstate(divide="ignore"):
            log_w = np.log(self.counts) - beta * self.energies[:, None]
        return float(logsumexp(log_w))

    def average(self, beta: float, fn) -> float:
        """Exact ``<fn(E, O)>`` with ``E`` the energy and ``O`` the order variable."""
        w = self._weights(beta)
        e, o = np.meshgrid(self.energies, self.orders, indexing="ij")
        return float(np.sum(w * fn(e, o)))

    def energy_distribution(self, beta: float) -> tuple[np.ndarray, np.ndarray]:
        p = self._weights(beta).sum(axis=1)
        return self.energies, p


def _ising_chunk(states: np.ndarray, n: int, bonds: np.ndarray):
    bits = ((states[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int8)
    s = 2 * bits - 1
    e = -np.sum(s[:, bonds[:, 0]] * s[:, bonds[:, 1]], axis=1, dtype=np.int64)
    m = s.sum(axis=1, dtype=np.int64)
    return e, m


def ising_joint_dos(lattice: Lattice) -> JointDOS:
    """Exact ``g(E_b, M)`` of the zero-field Ising model on ``lattice``."""
    n = lattice.n_sites
    if n > 26:
        raise ValueError(f"enumeration of 2^{n} states is not practical")
    n_b = lattice.n_bonds
    counts = np.zeros((2 * n_b + 1, n + 1), dtype=np.int64)
    total = 1 << n
    for start in range(0, total, _CHUNK):
        states = np.arange(start, min(total, start + _CHUNK), dtype=np.int64)
        e, m = _ising_chunk(states, n, lattice.bonds)
        np.add.at(counts, (e + n_b, (m + n) // 2), 1)
    energies = np.arange(-n_b, n_b + 1)
    orders = np.arange(-n, n + 1, 2)
    keep = counts.sum(axis=1) > 0
    return JointDOS(energies[keep].astype(float), orders.astype(float), counts[keep], n)


def potts_joint_dos(lattice: Lattice, q: int) -> JointDOS:
    """Exact ``g(E, S)`` of the Potts model with ``S = q * sum_k n_k^2``.

    ``|m|^2 = (S / N^2 - 1) / (q - 1)`` in the simplex normalisation used by
    :func:`critmc.models.derive`.
    """
    n = lattice.n_sites
    if q**n > 3 * 10**7:
        raise ValueError(f"enumeration of {q}^{n} states is not practical")
    n_b = lattice.n_bonds
    total = q**n
    hist: dict[tuple[int, int], int] = {}
    powers = q ** np.arange(n, dtype=np.int64)
    for start in range(0, total, _CHUNK):
        states = np.arange(start, min(total, start + _CHUNK), dtype=np.int64)
        s = (states[:, None] // powers) % q
        e = -np.sum(s[:, lattice.bonds[:, 0]] == s[:, lattice.bonds[:, 1]], axis=1)
        occ = np.stack([(s == k).sum(axis=1) for k in range(q)], axis=1)
        order = q * np.sum(occ * occ, axis=1)
        keys, cnt = np.unique(np.stack([e, order], axis=1), axis=0, return_counts=True)
        for (ee, oo), c in zip(keys, cnt):
            hist[(int(ee), int(oo))] = hist.get((int(ee), int(oo)), 0) + int(c)
    energies = np.arange(-n_b, 1)
    orders = np.array(sorted({o for _, o in hist}), dtype=np.int64)
    counts = np.zeros((len(energies), len(orders)), dtype=np.int64)
    o_index = {o: j for j, o in enumerate(orders)}
    for (ee, oo), c in hist.items():
        counts[ee + n_b, o_index[oo]] += c
    keep = counts.sum(axis=1) > 0
    return JointDOS(energies[keep].astype(float), orders.astype(float), counts[keep], n)


def ea_energies(lattice: Lattice, couplings: np.ndarray) -> np.ndarray:
    """Energies of all ``2^N`` configurations of an Ising model with bond couplings.

    ``couplings[b]`` multiplies bond ``lattice.bonds[b]``:
    ``E = -sum_b J_b s_i s_j``.  Used to test the spin-glass machinery.
    """
    n = lattice.n_sites
    if n > 24:
        raise ValueError(f"enumeration of 2^{n} states is not practical")
    total = 1 << n
    out = np.empty(total)
    for start in range(0, total, _CHUNK):
        states = np.arange(start, min(total, start + _CHUNK), dtype=np.int64)
        bits = ((states[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int8)
        s = (2 * bits - 1).astype(np.float64)
        out[start : start + len(states)] = (
            -(s[:, lattice.bonds[:, 0]] * s[:, lattice.bonds[:, 1]]) @ couplings
        )
    return out
