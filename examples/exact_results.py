"""Exact finite-size results: density of states, Fisher zeros, thermodynamics.

python examples/exact_results.py
"""

import numpy as np

from critmc.exact import dos, kaufman, zeros

# exact integer density of states of the 16 x 16 torus (2^256 states in total)
d = dos.ising_dos(16)
j = int(np.argmax(d.g))
print(
    f"L=16: {sum(1 for g in d.g if g)} occupied levels; most degenerate level E = "
    f"{d.energies[j]:.0f} with g = {d.g[j]:.4e}"
)
print(f"sum of g = 2^{np.log2(float(sum(d.g))):.0f} (exact integer arithmetic)")

# thermodynamics from g(E) agrees with Kaufman's closed form
betas = np.array([0.3, 0.4406868, 0.5])
print("u from g(E): ", np.round(d.canonical(betas)["u"], 12))
print("u (Kaufman): ", np.round(kaufman.energy(betas, 16), 12))

# leading Fisher zeros: |u1 - u_c| ~ L^(-1/nu) with nu = 1
for z in zeros.leading_zeros([8, 16, 32, 64, 128]):
    print(f"L={z.L:4d}  u1 = {z.u.real:.8f} + {z.u.imag:.8f}i   |u1 - u_c| = {z.distance:.6e}")
