"""Flat-histogram Monte Carlo: Wang-Landau and multicanonical sampling.

Wang-Landau [F. Wang & D. P. Landau, PRL 86, 2050 (2001)] performs a random
walk in energy space with acceptance ``min(1, g(E)/g(E'))`` while updating
``ln g(E) += ln f`` at every step; ``ln f`` is reduced whenever the energy
histogram is "flat".  Halving ``ln f`` geometrically makes the error saturate;
the 1/t refinement [R. E. Belardinelli & V. D. Pereyra, PRE 75, 046701 (2007)]
switches to ``ln f(t) = N_levels / t`` once ``ln f`` falls below that curve,
after which the error decays like ``t^{-1/2}``.

A converged ``ln g`` then serves as the weight of a multicanonical production
run [B. A. Berg & T. Neuhaus, PRL 68, 9 (1992)] whose samples are reweighted
to any temperature.  This is what makes strongly first-order transitions,
with their exponentially suppressed mixed-phase configurations, tractable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .kernels import flathist as kf
from .kernels import ising as k_ising
from .kernels import potts as k_potts
from .models import Ising, Model, Potts
from .rng import make_rng

__all__ = ["EnergyLevels", "WangLandau", "WLResult", "Multicanonical", "MucaResult"]


@dataclass(frozen=True)
class EnergyLevels:
    """Mapping between level indices and (bond) energies for a discrete model."""

    kind: str
    n_sites: int
    n_bonds: int
    q: int = 2

    @classmethod
    def of(cls, model: Model) -> EnergyLevels:
        lat = model.lattice
        if isinstance(model, Ising):
            if model.h != 0.0:
                raise ValueError("flat-histogram sampling is implemented for h = 0")
            if lat.z % 2:
                raise ValueError("Ising energy levels require even coordination")
            return cls("ising", lat.n_sites, lat.n_bonds, 2)
        if isinstance(model, Potts):
            return cls("potts", lat.n_sites, lat.n_bonds, model.q)
        raise TypeError("flat-histogram methods support Ising and Potts models")

    @property
    def size(self) -> int:
        return self.n_bonds // 2 + 1 if self.kind == "ising" else self.n_bonds + 1

    @property
    def energies(self) -> np.ndarray:
        idx = np.arange(self.size)
        return 4.0 * idx - self.n_bonds if self.kind == "ising" else idx - float(self.n_bonds)

    def index(self, energy: float) -> int:
        e = float(energy)
        return (
            int(round((e + self.n_bonds) / 4))
            if self.kind == "ising"
            else int(round(e + self.n_bonds))
        )

    def window(self, e_min: float | None, e_max: float | None) -> tuple[int, int]:
        """Level range covering per-site energies ``[e_min, e_max]`` (``None`` = open)."""
        n = self.n_sites
        lo = 0 if e_min is None else max(0, int(np.ceil(self._fractional(e_min * n))))
        hi = (
            self.size - 1
            if e_max is None
            else min(self.size - 1, int(np.floor(self._fractional(e_max * n))))
        )
        if lo >= hi:
            raise ValueError("empty energy window")
        return lo, hi

    def _fractional(self, energy: float) -> float:
        return (energy + self.n_bonds) / 4 if self.kind == "ising" else energy + self.n_bonds

    @property
    def log_total(self) -> float:
        """``ln`` of the number of configurations (normalisation of ``g``)."""
        return self.n_sites * np.log(self.q)


def _current_level(model: Model, state: np.ndarray, levels: EnergyLevels) -> int:
    return levels.index(model.energy(state))


def _walk_into_window(
    model: Model,
    state: np.ndarray,
    levels: EnergyLevels,
    lo: int,
    hi: int,
    rng: np.random.Generator,
    max_steps: int = 10**7,
) -> int:
    """Greedy random walk until the configuration's level lies inside ``[lo, hi]``."""
    nbr = model.lattice.nbr
    idx = _current_level(model, state, levels)
    n = model.n_sites
    for _ in range(max_steps):
        if lo <= idx <= hi:
            return idx
        i = int(rng.integers(n))
        if levels.kind == "ising":
            new = idx + int(state[i] * state[nbr[i]].sum()) // 2
            new_val = -state[i]
        else:
            q = levels.q
            new_val = int(rng.integers(q - 1))
            new_val += new_val >= state[i]
            neigh = state[nbr[i]]
            new = idx + int(np.sum(neigh == state[i])) - int(np.sum(neigh == new_val))
        if (idx < lo and new >= idx) or (idx > hi and new <= idx):
            state[i] = new_val
            idx = new
    raise RuntimeError("could not reach the energy window")


@dataclass
class WLResult:
    """Outcome of a Wang-Landau run."""

    levels: EnergyLevels
    ln_g: np.ndarray
    visited: np.ndarray
    window: tuple[int, int]
    trials: int
    ln_f: float
    wall_time: float
    stages: list[dict] = field(default_factory=list)
    checkpoints: list[tuple[int, float, np.ndarray]] = field(default_factory=list)

    @property
    def energies(self) -> np.ndarray:
        return self.levels.energies

    def normalized(self, ln_g: np.ndarray | None = None) -> np.ndarray:
        """``ln g`` normalised to the number of states (full window) or to 0 at the lowest level."""
        ln_g = self.ln_g if ln_g is None else ln_g
        out = np.full_like(ln_g, -np.inf)
        v = self.visited
        full = self.window == (0, self.levels.size - 1)
        shift = self.levels.log_total - logsumexp(ln_g[v]) if full else -ln_g[v][0]
        out[v] = ln_g[v] + shift
        return out

    def canonical(self, beta) -> dict[str, np.ndarray]:
        """Energy and specific heat per site from ``ln g`` (restricted to the window)."""
        beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        v = self.visited
        e = self.energies[v]
        a = self.normalized()[v][None, :] - beta[:, None] * e[None, :]
        lz = logsumexp(a, axis=1)
        p = np.exp(a - lz[:, None])
        mean = p @ e
        var = p @ (e * e) - mean**2
        n = self.levels.n_sites
        return {"lnZ": lz, "u": mean / n, "c": beta**2 * var / n}


class WangLandau:
    """Wang-Landau sampler for Ising and Potts models.

    Parameters
    ----------
    model:
        Zero-field Ising or Potts model.
    e_window:
        Optional ``(e_min, e_max)`` per-site energy window.
    flatness:
        Histogram flatness required to reduce ``ln f`` (``min H >= flatness * mean H``
        over levels seen so far).  With ``criterion="visit"`` every seen level
        merely has to be visited again, the criterion of Belardinelli & Pereyra.
    seed:
        Seed for the random stream.
    """

    def __init__(
        self,
        model: Model,
        *,
        e_window: tuple[float | None, float | None] | None = None,
        flatness: float = 0.8,
        criterion: str = "flat",
        seed=None,
    ):
        if criterion not in ("flat", "visit"):
            raise ValueError("criterion must be 'flat' or 'visit'")
        self.model = model
        self.levels = EnergyLevels.of(model)
        self.window = self.levels.window(*e_window) if e_window else (0, self.levels.size - 1)
        self.flatness = float(flatness)
        self.criterion = criterion
        self.rng = make_rng(seed)
        self.state = model.initial_state("random", self.rng)
        self.idx = _walk_into_window(model, self.state, self.levels, *self.window, self.rng)

    def _kernel(self, ln_g, hist, ln_f, moves, t0, n_levels, one_over_t):
        lat = self.model.lattice
        lo, hi = self.window
        if self.levels.kind == "ising":
            return kf.wl_ising(
                self.state,
                lat.nbr,
                self.idx,
                ln_g,
                hist,
                ln_f,
                moves,
                lo,
                hi,
                self.rng,
                float(t0),
                float(n_levels),
                one_over_t,
            )
        return kf.wl_potts(
            self.state,
            lat.nbr,
            self.levels.q,
            self.idx,
            ln_g,
            hist,
            ln_f,
            moves,
            lo,
            hi,
            self.rng,
            float(t0),
            float(n_levels),
            one_over_t,
        )

    def run(
        self,
        ln_f_final: float = 1e-6,
        *,
        one_over_t: bool = True,
        check_sweeps: int = 100,
        max_trials: float = 1e12,
        checkpoints=None,
    ) -> WLResult:
        """Iterate until ``ln f < ln_f_final``.

        ``checkpoints`` is an increasing sequence of trial counts at which a
        copy of ``ln g`` is stored (used to study convergence).
        """
        size = self.levels.size
        ln_g = np.zeros(size)
        hist = np.zeros(size, dtype=np.int64)
        seen = np.zeros(size, dtype=bool)
        cps = (
            list(np.asarray(sorted(checkpoints), dtype=np.int64)) if checkpoints is not None else []
        )
        saved: list[tuple[int, float, np.ndarray]] = []
        stages: list[dict] = []
        n = self.model.n_sites
        check_moves = max(1, int(check_sweeps) * n)
        ln_f = 1.0
        mode_1t = False
        n_levels = 0
        t = 0
        t_start = time.perf_counter()
        while True:
            block = check_moves
            if cps and cps[0] - t < block:
                block = int(cps[0] - t)
            if block > 0:
                self.idx = self._kernel(ln_g, hist, ln_f, block, t, n_levels, mode_1t)
                t += block
            seen |= hist > 0
            if mode_1t:
                ln_f = n_levels / t
            while cps and cps[0] <= t:
                saved.append((int(cps.pop(0)), float(ln_f), ln_g.copy()))
            if not mode_1t:
                h = hist[seen]
                ok = (
                    h.min() > 0
                    if self.criterion == "visit"
                    else h.min() >= self.flatness * h.mean()
                )
                if ok:
                    stages.append({"ln_f": ln_f, "trials": t})
                    ln_f /= 2.0
                    hist[:] = 0
                    if one_over_t and ln_f < seen.sum() / t:
                        mode_1t = True
                        n_levels = int(seen.sum())
                        ln_f = n_levels / t
            if ln_f < ln_f_final or t >= max_trials:
                break
        while cps:  # checkpoints beyond the end of the run
            saved.append((int(cps.pop(0)), float(ln_f), ln_g.copy()))
        return WLResult(
            self.levels,
            ln_g,
            seen.copy(),
            self.window,
            t,
            ln_f,
            time.perf_counter() - t_start,
            stages,
            saved,
        )


@dataclass
class MucaResult:
    """Multicanonical time series plus the weights they were generated with."""

    levels: EnergyLevels
    ln_g: np.ndarray  # ln g used for the weights w(E) = 1/g(E)
    level_series: np.ndarray
    order_series: np.ndarray  # Ising: M; Potts: sum_k n_k^2
    window: tuple[int, int]
    hist: np.ndarray

    @property
    def energy(self) -> np.ndarray:
        return self.levels.energies[self.level_series]

    def order_parameter_m2(self) -> np.ndarray:
        """Squared order parameter per site for every measurement."""
        n = self.levels.n_sites
        if self.levels.kind == "ising":
            return (self.order_series / n) ** 2
        q = self.levels.q
        return np.clip((q * self.order_series / n**2 - 1.0) / (q - 1.0), 0.0, None)

    def log_weights(self, beta: float) -> np.ndarray:
        """Reweighting factors ``ln[exp(-beta E) g(E)]`` of each sample (unnormalised)."""
        return -beta * self.energy + self.ln_g[self.level_series]

    def average(self, beta: float, obs: np.ndarray) -> float:
        lw = self.log_weights(beta)
        w = np.exp(lw - lw.max())
        return float(np.sum(w * obs) / np.sum(w))

    def canonical(self, beta) -> dict[str, np.ndarray]:
        beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        n = self.levels.n_sites
        e = self.energy
        m2 = self.order_parameter_m2()
        out = {k: np.empty_like(beta) for k in ("u", "c", "m2", "m4")}
        for i, b in enumerate(beta):
            lw = self.log_weights(b)
            w = np.exp(lw - lw.max())
            w /= w.sum()
            mean = w @ e
            out["u"][i] = mean / n
            out["c"][i] = b * b * (w @ (e * e) - mean**2) / n
            out["m2"][i] = w @ m2
            out["m4"][i] = w @ (m2 * m2)
        return out

    def energy_distribution(
        self, beta: float, smooth: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Canonical ``P_beta(e)`` over levels, from the multicanonical histogram.

        ``smooth`` is the width (in energy per site) of a Gaussian kernel applied
        to the histogram of visits before reweighting.  A width well below the
        peak widths suppresses the level-to-level noise that makes arg-max peak
        positions and valley depths erratic, at negligible bias.
        """
        hist = self.hist.astype(np.float64)
        v = hist > 0
        if smooth > 0:
            sigma = smooth * self.levels.n_sites / (4.0 if self.levels.kind == "ising" else 1.0)
            half = int(np.ceil(3 * sigma))
            if half >= 1:
                k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
                num = np.convolve(np.where(v, hist, 0.0), k, mode="same")
                den = np.convolve(v.astype(float), k, mode="same")
                hist = np.where(v, num / np.maximum(den, 1e-300), 0.0)
        a = np.full(hist.shape, -np.inf)
        a[v] = np.log(hist[v]) + self.ln_g[v] - beta * self.levels.energies[v]
        p = np.exp(a - logsumexp(a[v]))
        return self.levels.energies / self.levels.n_sites, p

    def double_peak(
        self, beta: float, split: float | None = None, smooth: float = 0.0
    ) -> dict[str, float]:
        """Peak positions/heights of a bimodal ``P_beta(e)`` separated at ``split``."""
        e, p = self.energy_distribution(beta, smooth)
        mask = p > 0
        e, p = e[mask], p[mask]
        if split is None:
            split = float(np.sum(e * p))
        low, high = e < split, e >= split
        i_lo = np.argmax(np.where(low, p, -1))
        i_hi = np.argmax(np.where(high, p, -1))
        between = (e > e[i_lo]) & (e < e[i_hi])
        p_min = p[between].min() if between.any() else np.nan
        return {
            "e_ordered": float(e[i_lo]),
            "e_disordered": float(e[i_hi]),
            "p_ordered": float(p[i_lo]),
            "p_disordered": float(p[i_hi]),
            "p_min": float(p_min),
            "weight_ordered": float(p[low].sum()),
        }

    def phase_means(self, beta: float, split: float) -> dict[str, float]:
        """Mean energy per site of each phase (samples below/above ``split``) at ``beta``.

        Conditional means are far less noisy than peak positions and converge
        to the same coexisting energies in the thermodynamic limit.
        """
        e = self.energy / self.levels.n_sites
        lw = self.log_weights(beta)
        w = np.exp(lw - lw.max())
        low = e < split
        w_lo, w_hi = w[low].sum(), w[~low].sum()
        return {
            "e_ordered": float(w[low] @ e[low] / w_lo),
            "e_disordered": float(w[~low] @ e[~low] / w_hi),
            "weight_ordered": float(w_lo / (w_lo + w_hi)),
        }

    def equal_height_beta(
        self, bracket: tuple[float, float], split: float | None = None, smooth: float = 0.0
    ) -> float:
        """Inverse temperature at which both peaks of ``P_beta(e)`` have equal height."""

        def f(b):
            dp = self.double_peak(b, split, smooth)
            return np.log(dp["p_ordered"]) - np.log(dp["p_disordered"])

        return float(brentq(f, *bracket, xtol=1e-10))

    def equal_weight_beta(self, bracket: tuple[float, float], split: float) -> float:
        """Inverse temperature at which both phases carry equal probability."""
        return float(
            brentq(
                lambda b: self.phase_means(b, split)["weight_ordered"] - 0.5, *bracket, xtol=1e-10
            )
        )


class Multicanonical:
    """Multicanonical production run with weights ``1/g(E)`` from Wang-Landau."""

    def __init__(
        self,
        model: Model,
        ln_g: np.ndarray,
        *,
        e_window: tuple[float | None, float | None] | None = None,
        seed=None,
    ):
        self.model = model
        self.levels = EnergyLevels.of(model)
        if len(ln_g) != self.levels.size:
            raise ValueError("ln_g does not match the model's energy levels")
        self.window = self.levels.window(*e_window) if e_window else (0, self.levels.size - 1)
        lo, hi = self.window
        ln_g = np.asarray(ln_g, dtype=np.float64).copy()
        inside = ln_g[lo : hi + 1]
        finite = np.isfinite(inside)
        if not finite.all():
            # never-visited levels inside the window (e.g. forbidden energies): forbid them
            inside[~finite] = np.inf
        self.ln_g = ln_g
        self.ln_w = np.where(np.isfinite(ln_g), -ln_g, -np.inf)
        self.rng = make_rng(seed)
        self.state = model.initial_state("random", self.rng)
        self.idx = _walk_into_window(model, self.state, self.levels, lo, hi, self.rng)
        self.hist = np.zeros(self.levels.size, dtype=np.int64)

    def run(self, n_meas: int, sweeps_per_meas: int = 1, n_therm: int = 0) -> MucaResult:
        lat = self.model.lattice
        n = self.model.n_sites
        lo, hi = self.window
        moves = int(sweeps_per_meas) * n
        idx_out = np.empty(n_meas, dtype=np.int64)
        ord_out = np.empty(n_meas, dtype=np.int64)
        for phase, count in (("therm", n_therm), ("prod", n_meas)):
            if count == 0:
                continue
            idx_buf = idx_out if phase == "prod" else np.empty(count, dtype=np.int64)
            ord_buf = ord_out if phase == "prod" else np.empty(count, dtype=np.int64)
            hist = self.hist if phase == "prod" else np.zeros_like(self.hist)
            if self.levels.kind == "ising":
                m = int(k_ising.magnetization(self.state))
                self.idx, _ = kf.muca_ising(
                    self.state,
                    lat.nbr,
                    self.idx,
                    m,
                    self.ln_w,
                    hist,
                    count,
                    moves,
                    lo,
                    hi,
                    self.rng,
                    idx_buf,
                    ord_buf,
                )
            else:
                counts = np.zeros(self.levels.q, dtype=np.int64)
                k_potts.state_counts(self.state, self.levels.q, counts)
                self.idx = kf.muca_potts(
                    self.state,
                    lat.nbr,
                    self.levels.q,
                    self.idx,
                    counts,
                    self.ln_w,
                    hist,
                    count,
                    moves,
                    lo,
                    hi,
                    self.rng,
                    idx_buf,
                    ord_buf,
                )
        return MucaResult(self.levels, self.ln_g, idx_out, ord_out, self.window, self.hist.copy())
