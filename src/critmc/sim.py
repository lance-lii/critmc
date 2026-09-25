"""Canonical Markov-chain Monte Carlo driver.

Example
-------
>>> from critmc import Simulation, Ising, square
>>> sim = Simulation(Ising(square(32)), beta=0.44, updates="wolff", seed=1)
>>> sim.thermalize(2_000)
>>> ts = sim.run(20_000)
>>> ts["e"].mean(), ts["absm"].mean()

An *update schedule* is a ``+``-separated list of moves applied once per
measurement, e.g. ``"metropolis"``, ``"sw"``, ``"heatbath+4*overrelax"`` or
``"metropolis+wolff"``.  A ``wolff`` term without an explicit count is tuned
during thermalisation so that one measurement flips about ``N`` spins on
average (one "sweep equivalent").  For continuous spins the Metropolis step
size is tuned towards 50 % acceptance.  Tuning only happens during
:meth:`Simulation.thermalize`, so production chains are time-homogeneous.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_npz, save_npz
from .kernels.common import (
    AUX_ACCEPTED,
    AUX_WOLFF_SIZE,
    HEATBATH,
    METROPOLIS,
    METROPOLIS_RANDOM,
    OVERRELAX,
    SWENDSEN_WANG,
    UPDATE_CODES,
    WOLFF,
    Workspace,
)
from .models import Model, derive
from .rng import make_rng

__all__ = ["TimeSeries", "Simulation", "parse_updates"]


@dataclass
class TimeSeries:
    """Raw measurement streams of one Markov chain plus its metadata.

    Indexing returns a raw stream (``"E"``, ``"M"``, ``"counts"``...) or a
    derived per-site observable (``"e"``, ``"absm"``, ``"m2"``, ``"m4"``...).
    """

    raw: dict[str, np.ndarray]
    meta: dict[str, Any]
    _derived: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return int(len(self.raw["E"]))

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self.raw:
            return self.raw[key]
        if not self._derived:
            self._derived = derive(self.raw, self.meta)
        try:
            return self._derived[key]
        except KeyError:
            raise KeyError(f"no observable {key!r}; have {sorted(self.keys())}") from None

    def keys(self) -> list[str]:
        if not self._derived:
            self._derived = derive(self.raw, self.meta)
        return sorted(set(self.raw) | set(self._derived))

    @property
    def beta(self) -> float:
        return float(self.meta["beta"])

    @property
    def n_sites(self) -> int:
        return int(self.meta["N"])

    @property
    def L(self) -> int:
        return int(self.meta["L"])

    def slice(self, start: int | None = None, stop: int | None = None) -> TimeSeries:
        raw = {k: v[start:stop] for k, v in self.raw.items()}
        return TimeSeries(raw, dict(self.meta))

    @staticmethod
    def concatenate(parts: list[TimeSeries]) -> TimeSeries:
        if not parts:
            raise ValueError("nothing to concatenate")
        raw = {k: np.concatenate([p.raw[k] for p in parts]) for k in parts[0].raw}
        meta = dict(parts[0].meta)
        meta["n_meas"] = sum(len(p) for p in parts)
        return TimeSeries(raw, meta)

    def save(self, path: str | Path) -> Path:
        return save_npz(path, self.raw, self.meta)

    @classmethod
    def load(cls, path: str | Path) -> TimeSeries:
        raw, meta = load_npz(path)
        return cls(raw, meta)


def parse_updates(updates: str) -> list[tuple[str, int | None]]:
    """Parse ``"heatbath+4*overrelax+wolff"`` into ``[(name, count or None), ...]``."""
    rows: list[tuple[str, int | None]] = []
    for token in updates.replace(" ", "").lower().split("+"):
        if not token:
            continue
        if "*" in token:
            count, name = token.split("*", 1)
            k = int(count)
            if k < 1:
                raise ValueError(f"repetition count must be positive in {token!r}")
            rows.append((name, k))
        else:
            rows.append((token, None))
    if not rows:
        raise ValueError("empty update schedule")
    unknown = [name for name, _ in rows if name not in UPDATE_CODES]
    if unknown:
        raise ValueError(f"unknown updates {unknown}; choose from {sorted(UPDATE_CODES)}")
    return rows


class Simulation:
    """A single Markov chain at fixed inverse temperature ``beta``.

    Parameters
    ----------
    model:
        The spin model (e.g. ``Ising(square(32))``).
    beta:
        Inverse temperature.
    updates:
        Update schedule applied once per measurement (see module docstring).
    seed:
        Integer seed, ``SeedSequence`` or ``Generator``.
    start:
        ``"random"`` (hot) or ``"ordered"`` (cold) initial state.
    state:
        Explicit initial configuration (overrides ``start``); it is copied.
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        updates: str = "metropolis",
        *,
        seed: Any = None,
        start: str = "random",
        state: np.ndarray | None = None,
    ):
        rows = parse_updates(updates)
        model.check_updates([name for name, _ in rows])
        self.model = model
        self.beta = float(beta)
        self.updates = updates
        self.seed = seed if isinstance(seed, (int, np.integer)) else None
        self.rng = make_rng(seed)
        self.schedule = np.array(
            [[UPDATE_CODES[name], 1 if k is None else k] for name, k in rows], dtype=np.int64
        )
        self._auto_wolff = [r for r, (name, k) in enumerate(rows) if name == "wolff" and k is None]
        self.state = model.initial_state(start, self.rng) if state is None else state.copy()
        self.start = "given" if state is not None else start
        self.ws = Workspace(model.n_sites)
        self.running = model.begin(self.state)
        self.n_therm = 0
        self.n_meas = 0
        self._mean_cluster = float("nan")
        self._wolff_history: list[int] = []

    # ------------------------------------------------------------------
    def _count(self, *codes: int) -> int:
        mask = np.isin(self.schedule[:, 0], codes)
        return int(self.schedule[mask, 1].sum())

    def _advance(self, n_meas: int) -> dict[str, np.ndarray]:
        return self.model.advance(
            self.state, self.running, self.beta, self.rng, self.schedule, n_meas, self.ws
        )

    def thermalize(self, n_meas: int, *, tune: bool = True, n_chunks: int = 10) -> dict[str, Any]:
        """Discard ``n_meas`` measurements, tuning adaptive parameters on the way."""
        n_chunks = max(1, min(n_chunks, n_meas))
        sizes = np.diff(np.linspace(0, n_meas, n_chunks + 1).astype(int))
        n_metro = self._count(METROPOLIS, METROPOLIS_RANDOM)
        n_wolff = self._count(WOLFF)
        for chunk in sizes:
            if chunk == 0:
                continue
            raw = self._advance(int(chunk))
            if not tune:
                continue
            aux = raw["aux"]
            if n_wolff:
                # the later half of the chunk is the better-equilibrated one
                late = aux[len(aux) // 2 :, AUX_WOLFF_SIZE]
                mean_c = late.sum() / (n_wolff * len(late))
                self._mean_cluster = float(mean_c)
                if self._auto_wolff and mean_c > 0:
                    target = max(1, int(round(self.model.n_sites / mean_c)))
                    for row in self._auto_wolff:
                        current = int(self.schedule[row, 1])
                        # Change the count by at most a factor 4 per chunk.  A hot start has
                        # tiny clusters; jumping straight to N / <|C|> would make the next
                        # chunk (thousands of large clusters per measurement) enormous once
                        # the system orders.
                        self.schedule[row, 1] = int(
                            np.clip(target, max(1, current // 4), 4 * current)
                        )
                    self._wolff_history.append(int(self.schedule[self._auto_wolff[0], 1]))
                    n_wolff = self._count(WOLFF)
            if "step" in self.running and n_metro:
                rate = aux[:, AUX_ACCEPTED].sum() / (n_metro * self.model.n_sites * len(aux))
                factor = float(np.clip(rate / 0.5, 0.5, 2.0))
                self.running["step"] = float(np.clip(self.running["step"] * factor, 1e-3, 10.0))
        self.n_therm += int(n_meas)
        return self.tuning()

    def tuning(self) -> dict[str, Any]:
        info: dict[str, Any] = {"schedule": self.schedule.tolist()}
        if self._count(WOLFF):
            info["wolff_per_meas"] = self._count(WOLFF)
            info["mean_cluster"] = self._mean_cluster
            if self._wolff_history:
                info["wolff_history"] = list(self._wolff_history)
        if "step" in self.running:
            info["step"] = self.running["step"]
        return info

    def run(self, n_meas: int, *, chunk: int = 1 << 16) -> TimeSeries:
        """Record ``n_meas`` measurements (in chunks to bound memory churn)."""
        if n_meas < 1:
            raise ValueError("n_meas must be positive")
        t0 = time.perf_counter()
        parts: list[dict[str, np.ndarray]] = []
        done = 0
        while done < n_meas:
            k = min(chunk, n_meas - done)
            parts.append(self._advance(k))
            done += k
        wall = time.perf_counter() - t0
        raw = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        self.n_meas += n_meas

        aux = raw["aux"]
        n = self.model.n_sites
        n_wolff = self._count(WOLFF)
        mean_cluster = (
            float(aux[:, AUX_WOLFF_SIZE].sum() / (n_wolff * n_meas)) if n_wolff else float("nan")
        )
        sweeps = self._count(METROPOLIS, METROPOLIS_RANDOM, HEATBATH, OVERRELAX, SWENDSEN_WANG) + (
            n_wolff * mean_cluster / n if n_wolff else 0.0
        )
        n_single = self._count(METROPOLIS, METROPOLIS_RANDOM)
        meta = {
            **self.model.meta(),
            "beta": self.beta,
            "updates": self.updates,
            "schedule": self.schedule.tolist(),
            "seed": self.seed,
            "start": self.start,
            "n_therm": self.n_therm,
            "n_meas": n_meas,
            "sweeps_per_meas": sweeps,
            "mean_cluster": mean_cluster,
            "acceptance": (
                float(aux[:, AUX_ACCEPTED].sum() / (n_single * n * n_meas)) if n_single else None
            ),
            "step": self.running.get("step"),
            "wall_time": wall,
            "site_updates_per_s": sweeps * n * n_meas / wall if wall > 0 else None,
        }
        return TimeSeries(raw, meta)
