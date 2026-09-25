r"""Edwards-Anderson spin glasses simulated with parallel tempering.

``H = -sum_<ij> J_ij s_i s_j`` with independent couplings drawn from a unit
Gaussian (or ``+-1``).  Two independent replica sets run at every temperature
and supply the spin overlap ``q = N^-1 sum_i s_i^a s_i^b``, the link overlap
``q_l`` and the Fourier overlap ``q(k)`` at the smallest wave vector.

Disorder-averaged observables (square brackets ``[.]``):

* Binder ratio ``g = (3 - [<q^4>] / [<q^2>]^2) / 2``;
* spin-glass susceptibility ``chi_SG(k) = N [<|q(k)|^2>]``;
* finite-size correlation length
  ``xi_L = sqrt(chi_SG(0) / chi_SG(k_min) - 1) / (2 sin(k_min / 2))``.

``g`` and ``xi_L / L`` are dimensionless, so curves for different ``L``
cross at ``T_c``.

For Gaussian disorder an exact identity [H. G. Katzgraber, M. Palassini &
A. P. Young, PRB 63, 184422 (2001)] links energy and link overlap in
equilibrium, ``[<e>] = -(z/2) (1 - [<q_l>]) / T``; the two sides approach each
other from opposite directions as the simulation equilibrates.  This makes a
parameter-free equilibration test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernels import glass as kg
from .lattice import Lattice
from .rng import make_rng

__all__ = [
    "EdwardsAnderson",
    "TemperingRun",
    "ParallelTempering",
    "DisorderAverage",
    "feedback_temperatures",
    "geometric_temperatures",
]


class EdwardsAnderson:
    """One disorder realisation of the Edwards-Anderson model."""

    def __init__(
        self,
        lattice: Lattice,
        couplings: np.ndarray | None = None,
        *,
        disorder: str = "gaussian",
        seed=None,
    ):
        self.lattice = lattice
        self.disorder = disorder
        if couplings is None:
            rng = make_rng(seed)
            if disorder == "gaussian":
                couplings = rng.standard_normal(lattice.n_bonds)
            elif disorder == "bimodal":
                couplings = rng.choice([-1.0, 1.0], size=lattice.n_bonds)
            else:
                raise ValueError("disorder must be 'gaussian' or 'bimodal'")
        couplings = np.asarray(couplings, dtype=np.float64)
        if couplings.shape != (lattice.n_bonds,):
            raise ValueError("need one coupling per bond")
        self.couplings = couplings
        self.jn = self._neighbour_couplings()

    def _neighbour_couplings(self) -> np.ndarray:
        lat = self.lattice
        jn = np.full(lat.nbr.shape, np.nan)
        # bond b = (i, j) is the forward neighbour in column 2*axis of i and,
        # seen from j, the backward neighbour in column 2*axis + 1
        for b, (i, j) in enumerate(lat.bonds):
            a = lat.bond_axis[b]
            jn[i, 2 * a] = self.couplings[b]
            jn[j, 2 * a + 1] = self.couplings[b]
        if np.isnan(jn).any():
            raise ValueError("lattice neighbour table is not in forward/backward pair form")
        return jn

    @property
    def n_sites(self) -> int:
        return self.lattice.n_sites

    def energy(self, state: np.ndarray) -> float:
        return float(kg.energy(state, self.lattice.bonds, self.couplings))


def geometric_temperatures(t_min: float, t_max: float, n: int) -> np.ndarray:
    """``n`` temperatures in geometric progression (a common PT starting point)."""
    return t_min * (t_max / t_min) ** (np.arange(n) / (n - 1))


@dataclass
class TemperingRun:
    """Accumulated measurements of one disorder sample."""

    temperatures: np.ndarray
    n_sites: int
    z: int
    acc: np.ndarray  # (bins, temperatures, N_OBS)
    swap_rate: np.ndarray
    label_hist: np.ndarray
    trips: np.ndarray
    wall_time: float = 0.0
    meta: dict = field(default_factory=dict)

    def bin_means(self) -> dict[str, np.ndarray]:
        """Thermal averages per logarithmic bin, shape ``(bins, temperatures)``."""
        cnt = self.acc[:, :, kg.OBS_COUNT]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {
                name: self.acc[:, :, col] / cnt
                for name, col in (
                    ("e", kg.OBS_E),
                    ("ql", kg.OBS_QL),
                    ("absq", kg.OBS_Q),
                    ("q2", kg.OBS_Q2),
                    ("q4", kg.OBS_Q4),
                    ("qk2", kg.OBS_QK2),
                )
            }
        return out

    def equilibrium(self) -> dict[str, np.ndarray]:
        """Averages over the last bin (second half of the run)."""
        return {k: v[-1] for k, v in self.bin_means().items()}


class ParallelTempering:
    """Parallel tempering for an Edwards-Anderson sample with two replica sets."""

    def __init__(
        self, model: EdwardsAnderson, temperatures, *, seed=None, sweeps_per_step: int = 1
    ):
        temps = np.sort(np.asarray(temperatures, dtype=np.float64))[::-1]  # hot -> cold
        self.model = model
        self.temperatures = temps
        self.betas = 1.0 / temps
        self.sweeps_per_step = int(sweeps_per_step)
        self.rng = make_rng(seed)
        m = len(temps)
        n = model.n_sites
        self.spins = (2 * self.rng.integers(0, 2, size=(2, m, n)) - 1).astype(np.int8)
        lat = model.lattice
        self.energies = np.array(
            [
                [kg.energy(self.spins[r, c], lat.bonds, model.couplings) for c in range(m)]
                for r in range(2)
            ]
        )
        self.perm = np.tile(np.arange(m, dtype=np.int64), (2, 1))
        self.labels = np.zeros((2, m), dtype=np.int64)
        kvec = np.array([lat.kmin(a) for a in range(lat.dim)])
        phase = lat.positions @ kvec.T  # (N, dim)
        self.cos_phase = np.ascontiguousarray(np.cos(phase).T)
        self.sin_phase = np.ascontiguousarray(np.sin(phase).T)

    def run(self, n_bins: int) -> TemperingRun:
        """Run ``2^n_bins - 1`` PT steps, accumulating log-binned measurements."""
        import time

        m = len(self.betas)
        lat = self.model.lattice
        acc = np.zeros((n_bins, m, kg.N_OBS))
        swap_acc = np.zeros(m - 1, dtype=np.int64)
        swap_try = np.zeros(m - 1, dtype=np.int64)
        label_hist = np.zeros((m, 2), dtype=np.int64)
        trips = np.zeros(2, dtype=np.int64)
        t0 = time.perf_counter()
        kg.pt_run(
            self.spins,
            self.energies,
            self.perm,
            self.labels,
            lat.nbr,
            self.model.jn,
            lat.bonds,
            self.model.couplings,
            self.betas,
            self.rng,
            n_bins,
            self.sweeps_per_step,
            self.cos_phase,
            self.sin_phase,
            acc,
            swap_acc,
            swap_try,
            label_hist,
            trips,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = swap_acc / swap_try
        return TemperingRun(
            self.temperatures.copy(),
            self.model.n_sites,
            lat.z,
            acc,
            rate,
            label_hist,
            trips,
            time.perf_counter() - t0,
            {"n_bins": n_bins, "sweeps_per_step": self.sweeps_per_step},
        )


def feedback_temperatures(temperatures: np.ndarray, label_hist: np.ndarray) -> np.ndarray:
    """One iteration of feedback-optimised temperatures.

    H. G. Katzgraber, S. Trebst, D. A. Huse & M. Troyer, J. Stat. Mech. P03018
    (2006): with ``f(T)`` the fraction of replicas at ``T`` that last visited
    the hottest temperature, the optimal density of temperatures is
    ``eta(T) ~ sqrt(df/dT / dT)``; new temperatures are placed at equal
    quantiles of ``eta``.  End points are kept fixed.
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    order = np.argsort(temps)
    t = temps[order]
    up = label_hist[order, 0].astype(float)
    down = label_hist[order, 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = up / (up + down)
    f = np.nan_to_num(f, nan=0.5)
    # enforce monotonicity (f rises from ~0 at T_min to ~1 at T_max)
    f = np.maximum.accumulate(f)
    df = np.maximum(np.diff(f), 1e-6)
    dt = np.diff(t)
    eta = np.sqrt(df / dt)
    cum = np.concatenate([[0.0], np.cumsum(eta * dt)])
    cum /= cum[-1]
    targets = np.linspace(0.0, 1.0, len(t))
    new = np.interp(targets, cum, t)
    new[0], new[-1] = t[0], t[-1]
    return new


class DisorderAverage:
    """Disorder averages of many :class:`TemperingRun` results at common temperatures."""

    def __init__(self, runs: list[TemperingRun], L: int):
        if not runs:
            raise ValueError("no runs")
        self.runs = runs
        self.L = int(L)
        self.temperatures = runs[0].temperatures
        self.n_sites = runs[0].n_sites
        self.z = runs[0].z
        eq = [r.equilibrium() for r in runs]
        self.samples = {k: np.array([e[k] for e in eq]) for k in eq[0]}  # (samples, temps)

    @staticmethod
    def _observables(s: dict[str, np.ndarray], n_sites: int, L: int) -> dict[str, np.ndarray]:
        q2 = s["q2"].mean(axis=0)
        q4 = s["q4"].mean(axis=0)
        qk2 = s["qk2"].mean(axis=0)
        kmin = 2.0 * np.pi / L
        with np.errstate(invalid="ignore"):
            xi = np.sqrt(np.clip(q2 / qk2 - 1.0, 0.0, None)) / (2.0 * np.sin(kmin / 2.0))
        return {
            "e": s["e"].mean(axis=0),
            "ql": s["ql"].mean(axis=0),
            "g": 0.5 * (3.0 - q4 / q2**2),
            "chi_sg": n_sites * q2,
            "xi_L": xi / L,
        }

    def observables(
        self, n_boot: int = 500, seed: int = 0
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Disorder averages with bootstrap errors (resampling disorder realisations)."""
        rng = make_rng(seed)
        full = self._observables(self.samples, self.n_sites, self.L)
        n = len(self.runs)
        boots = {k: [] for k in full}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            b = self._observables(
                {k: v[idx] for k, v in self.samples.items()}, self.n_sites, self.L
            )
            for k in full:
                boots[k].append(b[k])
        return {k: (full[k], np.std(np.array(boots[k]), axis=0, ddof=1)) for k in full}

    def identity_residuals(self) -> np.ndarray:
        """Per-sample ``(bins, samples, temperatures)`` residuals ``e + (z/2)(1 - q_l)/T``.

        Only their disorder average vanishes in equilibrium; single samples
        fluctuate by ``O(1)`` per bond.
        """
        e = np.array([r.bin_means()["e"] for r in self.runs])
        ql = np.array([r.bin_means()["ql"] for r in self.runs])
        res = e + (self.z / 2.0) * (1.0 - ql) / self.temperatures[None, None, :]
        return np.transpose(res, (1, 0, 2))

    def equilibration(self) -> dict[str, np.ndarray]:
        """Per-bin disorder averages of ``e`` and of ``-(z/2)(1 - q_l)/T`` (Gaussian identity).

        Also returns the standard error of their difference over samples.
        """
        e = np.array([r.bin_means()["e"] for r in self.runs]).mean(axis=0)
        ql = np.array([r.bin_means()["ql"] for r in self.runs]).mean(axis=0)
        e_from_ql = -(self.z / 2.0) * (1.0 - ql) / self.temperatures[None, :]
        res = self.identity_residuals()
        err = res.std(axis=1, ddof=1) / np.sqrt(res.shape[1])
        return {"e": e, "e_from_ql": e_from_ql, "residual_error": err}
