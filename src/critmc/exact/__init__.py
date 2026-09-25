"""Exact solutions and high-precision reference values.

* :mod:`~critmc.exact.onsager` - infinite square-lattice Ising model;
* :mod:`~critmc.exact.kaufman` - finite ``L x M`` Ising torus, any size;
* :mod:`~critmc.exact.dos` - exact integer density of states via
  complex-temperature evaluation of Kaufman's formula;
* :mod:`~critmc.exact.zeros` - Fisher zeros of the partition function;
* :mod:`~critmc.exact.enumeration` - brute force for tiny lattices;
* :mod:`~critmc.exact.chains` - O(n) rings via transfer operators;
* :mod:`~critmc.exact.potts` - self-dual Potts critical point and latent heat;
* :mod:`~critmc.exact.constants` - literature reference values.
"""

from . import chains, constants, enumeration, kaufman, onsager, potts

__all__ = ["chains", "constants", "enumeration", "kaufman", "onsager", "potts"]
