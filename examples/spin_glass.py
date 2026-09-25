"""Parallel tempering for one 3D Edwards-Anderson sample, with the equilibration test.

python examples/spin_glass.py
"""

import numpy as np

from critmc import cubic
from critmc.glass import EdwardsAnderson, ParallelTempering, geometric_temperatures

model = EdwardsAnderson(cubic(6), seed=3)  # Gaussian couplings on a 6^3 torus
pt = ParallelTempering(model, geometric_temperatures(0.6, 2.0, 16), seed=4)
run = pt.run(n_bins=14)  # 2^14 - 1 parallel-tempering steps

print("swap acceptance:", np.round(run.swap_rate, 2))
print("round trips per replica set:", run.trips // 2)
bins = run.bin_means()
T = run.temperatures
k = int(np.argmin(T))
print(f"\nequilibration at T = {T[k]:.2f} (one sample: the identity only holds on average)")
print(f"{'bin':>4} {'e':>10} {'-3(1-q_l)/T':>13} {'<q^2>':>8}")
for b in range(len(bins["e"])):
    print(
        f"{b:4d} {bins['e'][b, k]:10.4f} {-3 * (1 - bins['ql'][b, k]) / T[k]:13.4f} {bins['q2'][b, k]:8.4f}"
    )
