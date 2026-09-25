"""critmc: Monte Carlo simulation and statistical analysis of critical phenomena.

The package is organised in layers:

* :mod:`critmc.lattice`, :mod:`critmc.models`, :mod:`critmc.kernels` - geometry,
  Hamiltonians and numba-compiled update kernels;
* :mod:`critmc.sim`, :mod:`critmc.flathist`, :mod:`critmc.tempering`,
  :mod:`critmc.glass` - Markov-chain drivers (canonical, Wang-Landau /
  multicanonical, replica exchange, spin glasses);
* :mod:`critmc.exact` - exact and high-precision reference results;
* :mod:`critmc.stats`, :mod:`critmc.reweight`, :mod:`critmc.fss` - error
  analysis, histogram reweighting and finite-size scaling.
"""

from .lattice import Lattice, chain, cubic, hypercubic, make_lattice, square, triangular
from .models import ON, XY, Heisenberg, Ising, Model, Potts, build_model
from .rng import make_rng, spawn_rngs
from .sim import Simulation, TimeSeries

__version__ = "1.0.0"

__all__ = [
    "Lattice",
    "chain",
    "square",
    "cubic",
    "triangular",
    "hypercubic",
    "make_lattice",
    "Model",
    "Ising",
    "Potts",
    "ON",
    "XY",
    "Heisenberg",
    "build_model",
    "Simulation",
    "TimeSeries",
    "make_rng",
    "spawn_rngs",
    "__version__",
]
