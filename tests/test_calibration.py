"""Prespecified Monte Carlo calibration checks with binomial acceptance tolerances."""

import numpy as np
import pandas as pd
from scipy.stats import binom

from cgraph.estimation import AIPW
from cgraph.testing import GaussianPartialCorrelation


def test_gaussian_null_rejection_calibration():
    rng = np.random.default_rng(7391)
    repetitions, rejected = 400, 0
    test = GaussianPartialCorrelation()
    for _ in range(repetitions):
        z = rng.normal(size=300)
        frame = pd.DataFrame(
            {"z": z, "x": 0.7 * z + rng.normal(size=300), "y": -0.3 * z + rng.normal(size=300)}
        )
        result = test.test(
            frame, "x", "y", ("z",), dict.fromkeys(frame.columns, "continuous_gaussian")
        )
        rejected += result["p_value"] < 0.05
    assert binom.ppf(0.0005, repetitions, 0.05) <= rejected <= binom.ppf(0.9995, repetitions, 0.05)


def test_aipw_interval_coverage_in_supported_linear_scm():
    rng = np.random.default_rng(8181)
    repetitions, covered, errors = 100, 0, []
    for _ in range(repetitions):
        z = rng.normal(size=800)
        t = rng.binomial(1, 1 / (1 + np.exp(-0.4 * z)))
        y = 1.7 * t + z + rng.normal(size=800)
        fitted = AIPW(adjustment=("z",), cv=3).fit(pd.DataFrame({"z": z, "t": t}), y, treatment="t")
        lo, hi = fitted.estimate_.confidence_interval
        covered += lo <= 1.7 <= hi
        errors.append(fitted.estimate_.value - 1.7)
    assert covered >= binom.ppf(0.0005, repetitions, 0.95)
    assert abs(np.mean(errors)) < 0.04
