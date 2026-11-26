import subprocess
import sys

import numpy as np
from scipy import stats
from scipy.special import chdtrc, gammaincinv


def test_scipy_stats_is_not_imported():
    code = "import sys, critmc.stats, critmc.fss; sys.exit('scipy.stats' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_special_functions_equal_the_chi2_distribution():
    df = np.arange(1, 65)
    for alpha in (1e-12, 1e-4, 0.01, 0.05, 0.5, 0.999):
        np.testing.assert_array_equal(
            2.0 * gammaincinv(df / 2, 1.0 - alpha), stats.chi2.ppf(1.0 - alpha, df)
        )
    for x in (0.0, 1e-8, 0.5, 3.0, 40.0, 1e4, np.inf):
        np.testing.assert_array_equal(chdtrc(df, x), stats.chi2.sf(x, df))
