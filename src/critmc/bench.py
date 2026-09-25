"""Throughput benchmarks of the compiled kernels.

Throughput is reported as *site updates per second*: single-site algorithms
visit ``N`` sites per sweep, cluster algorithms are credited with the number
of spins they flip (Wolff) or ``N`` per update (Swendsen-Wang).  Compilation
is excluded by a warm-up run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .exact.onsager import BETA_C
from .lattice import cubic, square
from .models import ON, Ising, Potts
from .sim import Simulation

__all__ = ["BenchResult", "benchmark", "default_suite"]


@dataclass(frozen=True)
class BenchResult:
    label: str
    updates: str
    n_sites: int
    seconds: float
    site_updates_per_s: float


def benchmark(
    label: str, model, beta: float, updates: str, n_meas: int = 200, seed: int = 0
) -> BenchResult:
    sim = Simulation(model, beta, updates, seed=seed)
    sim.thermalize(max(20, n_meas // 5))  # compiles and tunes
    t0 = time.perf_counter()
    ts = sim.run(n_meas)
    dt = time.perf_counter() - t0
    rate = ts.meta["sweeps_per_meas"] * model.n_sites * n_meas / dt
    return BenchResult(label, updates, model.n_sites, dt, rate)


def default_suite(L: int = 64, n_meas: int = 200) -> list[BenchResult]:
    runs = [
        ("Ising 2D", Ising(square(L)), BETA_C, "metropolis"),
        ("Ising 2D", Ising(square(L)), BETA_C, "metropolis-random"),
        ("Ising 2D", Ising(square(L)), BETA_C, "heatbath"),
        ("Ising 2D", Ising(square(L)), BETA_C, "wolff"),
        ("Ising 2D", Ising(square(L)), BETA_C, "sw"),
        ("Ising 3D", Ising(cubic(max(8, L // 4))), 0.2216546, "wolff"),
        ("Potts q=3 2D", Potts(square(L), 3), 1.0050525, "metropolis"),
        ("Potts q=3 2D", Potts(square(L), 3), 1.0050525, "wolff"),
        ("XY 2D", ON(square(L), 2), 1.12, "metropolis"),
        ("XY 2D", ON(square(L), 2), 1.12, "overrelax"),
        ("XY 2D", ON(square(L), 2), 1.12, "wolff"),
        ("Heisenberg 3D", ON(cubic(max(8, L // 4)), 3), 0.693, "heatbath"),
        ("Heisenberg 3D", ON(cubic(max(8, L // 4)), 3), 0.693, "wolff"),
    ]
    return [benchmark(label, model, beta, upd, n_meas) for label, model, beta, upd in runs]
