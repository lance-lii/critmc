"""Reproducible random-number streams.

All randomness flows through :class:`numpy.random.Generator` objects backed by
``PCG64DXSM``.  Numba can consume these generators inside ``@njit`` kernels
and advances the very same bit-generator state, so a simulation is fully
determined by its seed no matter how the work is split between Python and
compiled code.

Independent streams for batches of simulations are derived with
:class:`numpy.random.SeedSequence` *spawn keys*: a stream is identified by a
root seed plus a tuple of keys such as ``("fss", L, replica)``.  The mapping is
deterministic, so any single run of a large batch can be reproduced in
isolation.
"""

from __future__ import annotations

import zlib
from collections.abc import Hashable

import numpy as np

__all__ = ["seed_sequence", "make_rng", "spawn_rngs"]

SeedLike = int | np.random.SeedSequence | np.random.Generator | None


def _key_to_int(key: Hashable) -> int:
    if isinstance(key, (bool, np.bool_)):
        return int(key)
    if isinstance(key, (int, np.integer)):
        if key < 0:
            raise ValueError("integer stream keys must be non-negative")
        return int(key)
    # strings, floats, tuples: stable 32-bit hash of the repr (not Python's salted hash())
    text = key if isinstance(key, str) else repr(key)
    return zlib.crc32(text.encode("utf-8"))


def seed_sequence(seed: int | None, *keys: Hashable) -> np.random.SeedSequence:
    """Seed sequence for the stream ``(seed, *keys)``.

    ``seed=None`` draws fresh OS entropy (non-reproducible).
    """
    spawn_key = tuple(_key_to_int(k) for k in keys)
    return np.random.SeedSequence(entropy=seed, spawn_key=spawn_key)


def make_rng(seed: SeedLike = None, *keys: Hashable) -> np.random.Generator:
    """Return a ``Generator`` for the given seed and optional stream keys.

    An existing ``Generator`` is passed through unchanged (keys are then not
    allowed, because they would silently be ignored).
    """
    if isinstance(seed, np.random.Generator):
        if keys:
            raise ValueError("stream keys cannot be applied to an existing Generator")
        return seed
    if isinstance(seed, np.random.SeedSequence):
        ss = seed
        if keys:
            extra = tuple(_key_to_int(k) for k in keys)
            ss = np.random.SeedSequence(entropy=ss.entropy, spawn_key=ss.spawn_key + extra)
    else:
        ss = seed_sequence(seed, *keys)
    return np.random.Generator(np.random.PCG64DXSM(ss))


def spawn_rngs(seed: int | None, n: int, *keys: Hashable) -> list[np.random.Generator]:
    """``n`` statistically independent generators below the stream ``(seed, *keys)``."""
    return [make_rng(seed, *keys, i) for i in range(n)]
