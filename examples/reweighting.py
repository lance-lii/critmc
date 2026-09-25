"""One simulation, a whole curve: single-histogram reweighting near beta_c.

python examples/reweighting.py
"""

import numpy as np

from critmc import Ising, Simulation, square
from critmc.exact import kaufman, onsager
from critmc.reweight import Reweighter

L = 24
beta0 = onsager.BETA_C
sim = Simulation(Ising(square(L)), beta0, "wolff", seed=7)
sim.thermalize(2_000)
ts = sim.run(100_000)

rw = Reweighter(ts["E"], beta0, {"m2": ts["m2"], "m4": ts["m4"]})
lo, hi = rw.reliable_range(min_fraction=0.1)
print(f"reliable reweighting range: beta in [{lo:.4f}, {hi:.4f}]")
print(f"{'beta':>8} {'u (MC)':>10} {'u exact':>10} {'U4':>8} {'ESS':>8}")
for beta in np.linspace(lo, hi, 7):
    u = rw.mean(ts["E"].astype(float), beta) / L**2
    m2, m4 = rw.mean("m2", beta), rw.mean("m4", beta)
    print(
        f"{beta:8.4f} {u:10.5f} {kaufman.energy(beta, L):10.5f} "
        f"{1 - m4 / (3 * m2**2):8.4f} {rw.ess(beta):8.0f}"
    )
