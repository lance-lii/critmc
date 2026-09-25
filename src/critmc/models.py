"""Spin models: Hamiltonians on a lattice bound to their compiled kernels.

A model knows how to create states, which update moves it supports, how to
advance a Markov chain through its ``run`` kernel, and how to turn the raw
measurement streams into per-site observables.  Models are cheap, stateless
descriptions; the evolving chain lives in :class:`critmc.sim.Simulation`.

Conventions (``J = k_B = 1`` throughout):

* Ising   ``H = -sum_<ij> s_i s_j - h sum_i s_i``
* Potts   ``H = -sum_<ij> delta(s_i, s_j)``
* O(n)    ``H = -sum_<ij> S_i . S_j``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .kernels import ising as k_ising
from .kernels import on as k_on
from .kernels import potts as k_potts
from .kernels.common import N_AUX, Workspace
from .lattice import Lattice, make_lattice

__all__ = ["Model", "Ising", "Potts", "ON", "XY", "Heisenberg", "build_model", "derive"]


class Model(ABC):
    """Base class of all lattice spin models."""

    kind: str = ""
    updates: tuple[str, ...] = ()

    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    @property
    def n_sites(self) -> int:
        return self.lattice.n_sites

    # -- states --------------------------------------------------------------
    @abstractmethod
    def initial_state(self, start: str, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def energy(self, state: np.ndarray) -> float: ...

    # -- chain evolution -----------------------------------------------------
    @abstractmethod
    def begin(self, state: np.ndarray) -> dict[str, Any]:
        """Running quantities tracked incrementally by the kernels."""

    @abstractmethod
    def advance(
        self,
        state: np.ndarray,
        running: dict[str, Any],
        beta: float,
        rng: np.random.Generator,
        schedule: np.ndarray,
        n_meas: int,
        ws: Workspace,
    ) -> dict[str, np.ndarray]:
        """Run ``n_meas`` schedule applications and return the raw measurements."""

    def check_updates(self, names: list[str]) -> None:
        bad = [n for n in names if n not in self.updates]
        if bad:
            raise ValueError(
                f"{type(self).__name__} does not support updates {bad}; "
                f"available: {list(self.updates)}"
            )

    # -- bookkeeping ----------------------------------------------------------
    def spec(self) -> dict[str, Any]:
        return {"model": self.kind, "lattice": self.lattice.name, "L": self.lattice.L}

    def meta(self) -> dict[str, Any]:
        """Metadata needed to interpret raw measurements without the model object."""
        return {
            **self.spec(),
            "N": self.n_sites,
            "dim": self.lattice.dim,
            "n_bonds": self.lattice.n_bonds,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lattice!r})"


class Ising(Model):
    """Ising model with optional uniform field ``h``."""

    kind = "ising"
    updates = ("metropolis", "metropolis-random", "heatbath", "wolff", "sw", "swendsen-wang")

    def __init__(self, lattice: Lattice, h: float = 0.0):
        super().__init__(lattice)
        self.h = float(h)

    def check_updates(self, names: list[str]) -> None:
        super().check_updates(names)
        cluster = {"wolff", "sw", "swendsen-wang"}
        if self.h != 0.0 and cluster.intersection(names):
            raise ValueError("cluster updates require zero field h")

    def initial_state(self, start, rng):
        n = self.n_sites
        if start == "random":
            return (2 * rng.integers(0, 2, size=n) - 1).astype(np.int8)
        if start in ("ordered", "up"):
            return np.ones(n, dtype=np.int8)
        if start == "down":
            return -np.ones(n, dtype=np.int8)
        raise ValueError(f"unknown start {start!r}")

    def energy(self, state):
        return float(k_ising.bond_energy(state, self.lattice.bonds)) - self.h * float(state.sum())

    def begin(self, state):
        return {
            "e_b": int(k_ising.bond_energy(state, self.lattice.bonds)),
            "m": int(k_ising.magnetization(state)),
        }

    def advance(self, state, running, beta, rng, schedule, n_meas, ws):
        e_b = np.empty(n_meas, dtype=np.int64)
        m = np.empty(n_meas, dtype=np.int64)
        aux = np.zeros((n_meas, N_AUX), dtype=np.int64)
        lat = self.lattice
        running["e_b"], running["m"], ws.tag = k_ising.run(
            state,
            lat.nbr,
            lat.bonds,
            float(beta),
            self.h,
            rng,
            schedule,
            n_meas,
            running["e_b"],
            running["m"],
            e_b,
            m,
            aux,
            ws.queue,
            ws.mark,
            ws.tag,
            ws.parent,
            ws.size,
            ws.flag,
        )
        energy = e_b - self.h * m if self.h != 0.0 else e_b
        return {"E": energy, "Eb": e_b, "M": m, "aux": aux}

    def spec(self):
        return {**super().spec(), "h": self.h}


class Potts(Model):
    """q-state Potts model."""

    kind = "potts"
    updates = ("metropolis", "metropolis-random", "heatbath", "wolff", "sw", "swendsen-wang")

    def __init__(self, lattice: Lattice, q: int):
        super().__init__(lattice)
        if q < 2 or q > 127:
            raise ValueError("q must be in [2, 127]")
        self.q = int(q)

    def initial_state(self, start, rng):
        n = self.n_sites
        if start == "random":
            return rng.integers(0, self.q, size=n).astype(np.int8)
        if start in ("ordered", "up"):
            return np.zeros(n, dtype=np.int8)
        raise ValueError(f"unknown start {start!r}")

    def energy(self, state):
        return float(k_potts.bond_energy(state, self.lattice.bonds))

    def begin(self, state):
        counts = np.zeros(self.q, dtype=np.int64)
        k_potts.state_counts(state, self.q, counts)
        return {"energy": int(k_potts.bond_energy(state, self.lattice.bonds)), "counts": counts}

    def advance(self, state, running, beta, rng, schedule, n_meas, ws):
        e = np.empty(n_meas, dtype=np.int64)
        counts = np.empty((n_meas, self.q), dtype=np.int64)
        aux = np.zeros((n_meas, N_AUX), dtype=np.int64)
        lat = self.lattice
        running["energy"], ws.tag = k_potts.run(
            state,
            lat.nbr,
            lat.bonds,
            self.q,
            float(beta),
            rng,
            schedule,
            n_meas,
            running["energy"],
            running["counts"],
            e,
            counts,
            aux,
            ws.queue,
            ws.mark,
            ws.tag,
            ws.parent,
            ws.size,
            ws.flag,
        )
        return {"E": e, "counts": counts, "aux": aux}

    def spec(self):
        return {**super().spec(), "q": self.q}


class ON(Model):
    """Classical O(n) vector model with unit spins."""

    kind = "on"
    updates = (
        "metropolis",
        "metropolis-random",
        "heatbath",
        "overrelax",
        "wolff",
        "sw",
        "swendsen-wang",
    )

    def __init__(self, lattice: Lattice, n: int):
        super().__init__(lattice)
        if n < 2:
            raise ValueError("use the Ising model for n = 1")
        if n > 3:
            # the heat-bath kernel is specialised to n = 2, 3; everything else is generic
            self.updates = tuple(u for u in self.updates if u != "heatbath")
        self.n = int(n)

    def initial_state(self, start, rng):
        if start == "random":
            s = rng.standard_normal((self.n_sites, self.n))
            s /= np.linalg.norm(s, axis=1, keepdims=True)
            return s
        if start in ("ordered", "up"):
            s = np.zeros((self.n_sites, self.n))
            s[:, 0] = 1.0
            return s
        raise ValueError(f"unknown start {start!r}")

    def energy(self, state):
        return float(k_on.energy(state, self.lattice.bonds))

    def begin(self, state):
        return {"step": 1.0}

    def advance(self, state, running, beta, rng, schedule, n_meas, ws):
        lat = self.lattice
        n_axes = int(lat.bond_axis.max()) + 1
        e = np.empty(n_meas)
        m = np.empty((n_meas, self.n))
        sc = np.empty((n_meas, n_axes))
        ss = np.empty((n_meas, n_axes))
        aux = np.zeros((n_meas, N_AUX), dtype=np.int64)
        ws.tag = k_on.run(
            state,
            lat.nbr,
            lat.bonds,
            lat.bond_axis,
            float(beta),
            float(running["step"]),
            rng,
            schedule,
            n_meas,
            e,
            m,
            sc,
            ss,
            aux,
            ws.queue,
            ws.mark,
            ws.tag,
            ws.parent,
            ws.size,
            ws.flag,
            ws.proj,
        )
        return {"E": e, "M": m, "stiff_c": sc, "stiff_s": ss, "aux": aux}

    def spec(self):
        return {**super().spec(), "n": self.n}


def XY(lattice: Lattice) -> ON:
    """Plane-rotator (XY) model, O(2)."""
    return ON(lattice, 2)


def Heisenberg(lattice: Lattice) -> ON:
    """Classical Heisenberg model, O(3)."""
    return ON(lattice, 3)


def build_model(spec: dict[str, Any]) -> Model:
    """Inverse of :meth:`Model.spec`."""
    lattice = make_lattice(spec["lattice"], int(spec["L"]))
    kind = spec["model"]
    if kind == "ising":
        return Ising(lattice, h=float(spec.get("h", 0.0)))
    if kind == "potts":
        return Potts(lattice, int(spec["q"]))
    if kind == "on":
        return ON(lattice, int(spec["n"]))
    raise ValueError(f"unknown model kind {kind!r}")


def derive(raw: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-site observables from raw measurement streams.

    Returns ``e`` (energy per site), ``absm``, ``m2``, ``m4`` (powers of the order
    parameter per site) and, for the Ising model, the signed ``m``.
    """
    n = float(meta["N"])
    kind = meta["model"]
    out: dict[str, np.ndarray] = {"e": np.asarray(raw["E"], dtype=np.float64) / n}
    if kind == "ising":
        m = raw["M"] / n
        out["m"] = m
        out["m2"] = m * m
    elif kind == "potts":
        q = int(meta["q"])
        rho = raw["counts"] / n
        # simplex order parameter: |m|^2 = (q sum_k rho_k^2 - 1) / (q - 1)
        out["m2"] = np.clip((q * np.sum(rho * rho, axis=1) - 1.0) / (q - 1.0), 0.0, None)
        out["mmax"] = (q * rho.max(axis=1) - 1.0) / (q - 1.0)
    elif kind == "on":
        mv = raw["M"] / n
        out["m2"] = np.sum(mv * mv, axis=1)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    out["absm"] = np.sqrt(out["m2"])
    out["m4"] = out["m2"] ** 2
    return out
