"""Reference values used to validate simulations.

Exact values are marked ``exact``; the rest are high-precision literature
estimates, quoted with their uncertainty and source.  They are only used for
comparison in reports and plots, never as inputs to an analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Reference", "REFERENCES", "reference"]


@dataclass(frozen=True)
class Reference:
    value: float
    error: float
    source: str

    def __iter__(self):
        yield self.value
        yield self.error


_KC_2D = 0.5 * np.log1p(np.sqrt(2.0))

REFERENCES: dict[str, dict[str, Reference]] = {
    "ising2d": {
        "beta_c": Reference(_KC_2D, 0.0, "exact (Kramers-Wannier / Onsager)"),
        "T_c": Reference(1.0 / _KC_2D, 0.0, "exact"),
        "nu": Reference(1.0, 0.0, "exact"),
        "gamma/nu": Reference(1.75, 0.0, "exact"),
        "beta/nu": Reference(0.125, 0.0, "exact"),
        "alpha/nu": Reference(0.0, 0.0, "exact (logarithmic)"),
        # Q* = <m^2>^2/<m^4> = 0.856216(1) on the square torus
        "binder_U4": Reference(
            1.0 - 1.0 / (3.0 * 0.856216), 1e-6, "Kamieniarz & Bloete, J. Phys. A 26, 201 (1993)"
        ),
        "c_amplitude": Reference(8.0 * _KC_2D**2 / np.pi, 0.0, "exact (Ferdinand & Fisher 1969)"),
        "z_metropolis": Reference(2.1667, 0.0005, "Nightingale & Bloete, PRL 76, 4548 (1996)"),
        "z_wolff": Reference(0.25, 0.01, "Coddington & Baillie, PRL 68, 962 (1992)"),
        "z_sw": Reference(0.25, 0.01, "Coddington & Baillie, PRL 68, 962 (1992)"),
    },
    "ising3d": {
        "beta_c": Reference(0.221654626, 5e-9, "Ferrenberg, Xu & Landau, PRE 97, 043301 (2018)"),
        "nu": Reference(0.629971, 4e-6, "Kos, Poland, Simmons-Duffin & Vichi, JHEP 08 (2016) 036"),
        "gamma/nu": Reference(2.0 - 0.036298, 2e-6, "Kos et al. (2016), eta = 0.036298(2)"),
        "beta/nu": Reference((1.0 + 0.036298) / 2.0, 1e-6, "scaling relation with eta"),
        "omega": Reference(0.8297, 0.0002, "Kos et al. (2016)"),
        "binder_ratio": Reference(1.6036, 0.0001, "Hasenbusch, PRB 82, 174433 (2010)"),
    },
    "xy2d": {
        "T_bkt": Reference(0.89294, 0.00008, "Hasenbusch, J. Phys. A 38, 5869 (2005)"),
        "stiffness_jump": Reference(2.0 / np.pi, 0.0, "exact (Nelson & Kosterlitz 1977)"),
        "eta": Reference(0.25, 0.0, "exact at T_BKT"),
    },
    "heisenberg3d": {
        "beta_c": Reference(0.693002, 2e-6, "Campostrini et al., PRB 65, 144520 (2002)"),
        "nu": Reference(0.7112, 0.0005, "Campostrini et al. (2002)"),
        "gamma/nu": Reference(2.0 - 0.0375, 0.0005, "Campostrini et al. (2002)"),
    },
    "potts2d": {
        # exact values follow from critmc.exact.potts
    },
    "ea3d_gauss": {
        "T_c": Reference(0.951, 0.009, "Katzgraber, Koerner & Young, PRB 73, 224432 (2006)"),
        "nu": Reference(2.44, 0.09, "Katzgraber, Koerner & Young (2006)"),
    },
}


def reference(system: str, quantity: str) -> Reference:
    try:
        return REFERENCES[system][quantity]
    except KeyError:
        raise KeyError(f"no reference value for {system}/{quantity}") from None
