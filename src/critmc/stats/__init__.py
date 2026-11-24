"""Statistical error analysis of Monte Carlo time series.

* :mod:`.autocorr` - autocorrelation functions, integrated autocorrelation
  times (Madras-Sokal windowing) and Wolff's Gamma-method with error
  propagation to derived quantities and replica combination;
* :mod:`.blocking` - Flyvbjerg-Petersen blocking and Jonsson's automated
  plateau detection;
* :mod:`.calibration` - bias, error-bar size and coverage of repeated runs
  compared with a known answer;
* :mod:`.resample` - blocked jackknife (including independent data sets) and
  i.i.d., moving-block and stationary bootstrap with percentile, basic, normal
  and BCa intervals.

All functions take samples along axis 0 (the Markov-chain time).
"""

from __future__ import annotations

from .autocorr import (
    AutocorrTime,
    GammaResult,
    autocorrelation,
    autocovariance,
    effective_sample_size,
    gamma_method,
    tau_int_sokal,
)
from .blocking import BlockingLevels, BlockingResult, blocking_error, blocking_levels
from .calibration import Calibration, PullStatistics, calibration, pull_statistics
from .resample import (
    BootstrapResult,
    JackknifeResult,
    block_means,
    bootstrap,
    jackknife,
    jackknife_independent,
    jackknife_means,
    stationary_bootstrap_indices,
)

__all__ = [
    # autocorrelation
    "AutocorrTime",
    "GammaResult",
    "autocovariance",
    "autocorrelation",
    "tau_int_sokal",
    "gamma_method",
    "effective_sample_size",
    # blocking
    "BlockingLevels",
    "BlockingResult",
    "blocking_levels",
    "blocking_error",
    # calibration against a known answer
    "Calibration",
    "PullStatistics",
    "calibration",
    "pull_statistics",
    # resampling
    "JackknifeResult",
    "BootstrapResult",
    "block_means",
    "jackknife",
    "jackknife_means",
    "jackknife_independent",
    "bootstrap",
    "stationary_bootstrap_indices",
]
