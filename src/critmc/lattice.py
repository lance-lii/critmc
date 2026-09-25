"""Periodic lattice geometries.

Every lattice is described by a dense neighbour table ``nbr`` of shape
``(N, z)`` and a bond list ``bonds`` of shape ``(B, 2)`` in which every
nearest-neighbour bond appears exactly once.  The numba kernels only ever
see these two integer arrays, which keeps them independent of geometry.

For the hypercubic and triangular lattices the neighbour columns come in
opposite pairs: ``nbr[:, 2*a]`` is the site displaced by ``+e_a`` and
``nbr[:, 2*a + 1]`` the site displaced by ``-e_a``.  Column ``k ^ 1`` therefore
points in the direction opposite to column ``k``, and the bond list is built
from the "forward" columns so that ``bond_axis`` identifies the lattice
direction of each bond (needed e.g. for the helicity modulus).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

__all__ = ["Lattice", "hypercubic", "chain", "square", "cubic", "triangular", "make_lattice"]


@dataclass(frozen=True, eq=False)
class Lattice:
    """Immutable description of a periodic lattice.

    Attributes
    ----------
    name:
        Geometry label (``"chain"``, ``"square"``, ``"cubic"``, ``"triangular"``...).
    shape:
        Linear extent along each primitive direction.
    nbr:
        ``(N, z)`` neighbour table.
    bonds:
        ``(B, 2)`` list of bonds ``(i, j)``; ``j`` is the forward neighbour of ``i``.
    bond_axis:
        ``(B,)`` primitive direction of each bond.
    positions:
        ``(N, d)`` Cartesian coordinates, used for Fourier transforms.
    coords:
        ``(N, d)`` integer coordinates along the primitive directions.
    primitive:
        ``(d, d)`` primitive vectors (rows) in Cartesian coordinates.
    sublattice:
        ``(N,)`` two-colouring for bipartite lattices, otherwise ``None``.
    """

    name: str
    shape: tuple[int, ...]
    nbr: np.ndarray
    bonds: np.ndarray
    bond_axis: np.ndarray
    positions: np.ndarray
    coords: np.ndarray
    primitive: np.ndarray
    sublattice: np.ndarray | None

    @property
    def n_sites(self) -> int:
        return int(self.nbr.shape[0])

    @property
    def z(self) -> int:
        """Coordination number."""
        return int(self.nbr.shape[1])

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def n_bonds(self) -> int:
        return int(self.bonds.shape[0])

    @property
    def L(self) -> int:
        """Linear size (first extent); all built-in lattices are isotropic."""
        return int(self.shape[0])

    @property
    def bipartite(self) -> bool:
        return self.sublattice is not None

    def site(self, *x: int) -> int:
        """Index of the site with integer coordinates ``x`` (periodic)."""
        if len(x) != self.dim:
            raise ValueError(f"expected {self.dim} coordinates, got {len(x)}")
        wrapped = [int(c) % n for c, n in zip(x, self.shape)]
        return int(np.ravel_multi_index(wrapped, self.shape, order="F"))

    def reciprocal(self) -> np.ndarray:
        """Reciprocal primitive vectors ``b_i`` with ``a_i . b_j = 2 pi delta_ij``."""
        return 2.0 * np.pi * np.linalg.inv(self.primitive).T

    def kmin(self, axis: int = 0) -> np.ndarray:
        """Smallest non-zero wave vector compatible with the boundary conditions."""
        return self.reciprocal()[axis] / self.shape[axis]

    def __repr__(self) -> str:
        dims = "x".join(str(n) for n in self.shape)
        return f"Lattice({self.name}, {dims}, N={self.n_sites}, z={self.z})"


def _check_extent(shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(n) for n in shape)
    if not shape:
        raise ValueError("lattice needs at least one dimension")
    if any(n < 2 for n in shape):
        # extent 1 would create self-bonds, which the single-spin kernels do not handle
        raise ValueError(f"every linear extent must be >= 2, got {shape}")
    return shape


def hypercubic(shape: tuple[int, ...] | list[int], name: str | None = None) -> Lattice:
    """Simple hypercubic lattice with periodic boundaries.

    Sites are numbered with the first coordinate running fastest,
    ``i = x0 + L0 * (x1 + L1 * (x2 + ...))``.
    """
    shape = _check_extent(tuple(shape))
    d = len(shape)
    n = prod(shape)
    coords = np.stack(np.unravel_index(np.arange(n), shape, order="F"), axis=1).astype(np.int64)

    nbr = np.empty((n, 2 * d), dtype=np.int64)
    for a in range(d):
        for step, col in ((1, 2 * a), (-1, 2 * a + 1)):
            shifted = coords.copy()
            shifted[:, a] = (shifted[:, a] + step) % shape[a]
            nbr[:, col] = np.ravel_multi_index(shifted.T, shape, order="F")

    sites = np.arange(n, dtype=np.int64)
    bonds = np.concatenate([np.stack([sites, nbr[:, 2 * a]], axis=1) for a in range(d)])
    bond_axis = np.repeat(np.arange(d, dtype=np.int64), n)

    even = all(extent % 2 == 0 for extent in shape)
    sublattice = (coords.sum(axis=1) % 2).astype(np.int8) if even else None

    if name is None:
        name = {1: "chain", 2: "square", 3: "cubic"}.get(d, f"hypercubic{d}d")
    return Lattice(
        name=name,
        shape=shape,
        nbr=nbr,
        bonds=bonds,
        bond_axis=bond_axis,
        positions=coords.astype(np.float64),
        coords=coords,
        primitive=np.eye(d),
        sublattice=sublattice,
    )


def chain(L: int) -> Lattice:
    """Periodic ring of ``L`` sites."""
    return hypercubic((L,), name="chain")


def square(L: int, M: int | None = None) -> Lattice:
    """``L x M`` square lattice (``M = L`` by default) on a torus."""
    return hypercubic((L, L if M is None else M), name="square")


def cubic(L: int) -> Lattice:
    """``L^3`` simple cubic lattice on a 3-torus."""
    return hypercubic((L, L, L), name="cubic")


def triangular(L: int, M: int | None = None) -> Lattice:
    """Triangular lattice on an ``L x M`` rhombic torus (coordination 6).

    The three bond directions are ``a1``, ``a2`` and ``a1 - a2`` with
    ``a1 = (1, 0)`` and ``a2 = (1/2, sqrt(3)/2)``.
    """
    shape = _check_extent((L, L if M is None else M))
    n = prod(shape)
    coords = np.stack(np.unravel_index(np.arange(n), shape, order="F"), axis=1).astype(np.int64)
    directions = ((1, 0), (0, 1), (1, -1))

    nbr = np.empty((n, 6), dtype=np.int64)
    for a, (dx, dy) in enumerate(directions):
        for sign, col in ((1, 2 * a), (-1, 2 * a + 1)):
            x = (coords[:, 0] + sign * dx) % shape[0]
            y = (coords[:, 1] + sign * dy) % shape[1]
            nbr[:, col] = np.ravel_multi_index((x, y), shape, order="F")

    sites = np.arange(n, dtype=np.int64)
    bonds = np.concatenate([np.stack([sites, nbr[:, 2 * a]], axis=1) for a in range(3)])
    bond_axis = np.repeat(np.arange(3, dtype=np.int64), n)
    primitive = np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    return Lattice(
        name="triangular",
        shape=shape,
        nbr=nbr,
        bonds=bonds,
        bond_axis=bond_axis,
        positions=coords @ primitive,
        coords=coords,
        primitive=primitive,
        sublattice=None,
    )


_BUILDERS = {
    "chain": lambda L: chain(L),
    "square": lambda L: square(L),
    "cubic": lambda L: cubic(L),
    "triangular": lambda L: triangular(L),
}


def make_lattice(name: str, L: int) -> Lattice:
    """Build a lattice by name (``chain``, ``square``, ``cubic``, ``triangular``)."""
    try:
        return _BUILDERS[name](L)
    except KeyError:
        raise ValueError(f"unknown lattice {name!r}; choose from {sorted(_BUILDERS)}") from None
