"""Estimator correctness, held-out scoring, scale validation, and exact sensitivity fixtures."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from cgraph.estimation import AIPW
from cgraph.sensitivity import RiskRatio, evalue, rosenbaum_pairs, rosenbaum_sign


def cohort(seed=40, n=4000):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-0.4 * z)))
    y = 2 * t + 1.5 * z + rng.normal(size=n)
    return pd.DataFrame({"z": z, "t": t}), pd.Series(y)


def test_aipw_recovers_effect_and_holdout_does_not_mutate():
    X, y = cohort()
    estimator = AIPW(adjustment=("z",), random_state=8).fit(X, y, treatment="t")
    assert abs(estimator.estimate_.value - 2) < 0.15
    assert (
        estimator.estimate_.confidence_interval[0] < 2 < estimator.estimate_.confidence_interval[1]
    )
    before = estimator.estimate_
    X2, y2 = cohort(seed=41)
    estimate, diagnostics = estimator.evaluate(X2, y2)
    assert abs(estimate.value - 2) < 0.15
    assert estimator.estimate_ == before
    assert diagnostics["extreme_propensities"] == 0
    assert not hasattr(clone(estimator), "models_")
    with pytest.raises(ValueError):
        estimator.evaluate(X2, y2, treated=0, control=1)
    with pytest.raises(ValueError):
        AIPW(adjustment=("z",)).fit(X, y.iloc[::-1], treatment="t")


def test_evalue_validation_and_protective_symmetry():
    kwargs = {"outcome_event": "event=1", "contrast": "treated/control"}
    assert evalue(RiskRatio(2, **kwargs))["point"] == pytest.approx(2 + np.sqrt(2))
    assert evalue(RiskRatio(0.5, **kwargs))["point"] == evalue(RiskRatio(2, **kwargs))["point"]
    assert evalue(RiskRatio(1.2, (0.9, 1.6), **kwargs))["confidence_limit"] == 1
    with pytest.raises(TypeError):
        evalue(2)
    with pytest.raises(ValueError):
        RiskRatio(2, scope="marginal", **kwargs)


def test_rosenbaum_exact_sign_and_pair_validation():
    result = rosenbaum_sign([1] * 10)
    assert result["bounds"][0]["upper_p_value"] == pytest.approx(2**-10)
    assert [b["upper_p_value"] for b in result["bounds"]] == sorted(
        b["upper_p_value"] for b in result["bounds"]
    )
    paired = rosenbaum_pairs([3, 1, 1, 2], [1, 0, 0, 1], ["a", "a", "b", "b"])
    assert paired["n_pairs"] == 2
    with pytest.raises(ValueError):
        rosenbaum_pairs([1, 2], [1, 1], ["a", "a"])


def test_categorical_estimation_and_balance_are_invariant_to_numeric_labels():
    rng = np.random.default_rng(39)
    category = rng.integers(0, 3, size=800)
    treatment = rng.binomial(1, np.array([0.3, 0.5, 0.7])[category])
    outcome = 2 * treatment + (category == 1) * 4 - (category == 2) * 2 + rng.normal(size=800)
    frame = pd.DataFrame({"category": category, "t": treatment})
    fitted = AIPW(adjustment=("category",), categorical=("category",)).fit(
        frame, outcome, treatment="t"
    )
    assert "category" not in fitted.diagnostics_["weighted_standardized_differences"]
    assert set(fitted.diagnostics_["weighted_category_differences"]["category"]) == {
        "0.0",
        "1.0",
        "2.0",
    }
    remapped = frame.copy()
    remapped["category"] = frame["category"].map({0: 90, 1: -7, 2: 12})
    second = AIPW(adjustment=("category",), categorical=("category",)).fit(
        remapped, outcome, treatment="t"
    )
    assert fitted.estimate_.value == pytest.approx(second.estimate_.value, abs=1e-8)
