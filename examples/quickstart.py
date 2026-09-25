"""Simulate the 2D Ising model and compare with the exact solution.

python examples/quickstart.py
"""

import numpy as np

from critmc import Ising, Simulation, square
from critmc.exact import kaufman, onsager
from critmc.stats import gamma_method

L = 32
beta = onsager.BETA_C  # exact critical coupling, ln(1 + sqrt 2) / 2

sim = Simulation(Ising(square(L)), beta, updates="wolff", seed=42)
sim.thermalize(2_000)  # also tunes the number of clusters per measurement
ts = sim.run(50_000)

# energy per site with a Gamma-method error bar
e = gamma_method(ts["e"])
print(
    f"<e>  = {e.value:.5f} +- {e.error:.5f}   exact {kaufman.energy(beta, L):.5f}"
    f"   (tau_int = {e.tau_int:.2f} measurements)"
)

# specific heat: a derived quantity, error from the projected fluctuations
E = ts["E"].astype(float)
c = gamma_method(np.column_stack([E, E**2]), lambda a: beta**2 * (a[1] - a[0] ** 2) / L**2)
print(f"c    = {c.value:.4f} +- {c.error:.4f}     exact {kaufman.specific_heat(beta, L):.4f}")

# Binder cumulant (no exact finite-L value; U* = 0.61069 as L -> infinity)
u = gamma_method(np.column_stack([ts["m2"], ts["m4"]]), lambda a: 1 - a[1] / (3 * a[0] ** 2))
print(f"U4   = {u.value:.4f} +- {u.error:.4f}")
print(f"throughput: {ts.meta['site_updates_per_s']:.2e} site updates per second")
