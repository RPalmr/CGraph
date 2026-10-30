"""Cross-fitted binary-treatment AIPW with explicit support and inference diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class EffectEstimate:
    """Mean-difference estimate and uncertainty under IID AIPW regularity assumptions."""

    value: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n: int
    scale: str = "mean_difference"
    method: str = "cross_fitted_aipw"

    def __post_init__(self):
        lo, hi = self.confidence_interval
        if (
            not np.isfinite([self.value, self.standard_error, lo, hi]).all()
            or self.standard_error < 0
            or not lo <= self.value <= hi
            or not 0 < self.confidence_level < 1
            or self.n < 2
        ):
            raise ValueError("Invalid or nonfinite effect estimate")
        object.__setattr__(self, "confidence_interval", (lo, hi))

    def to_dict(self):
        """Return JSON-compatible point and uncertainty information."""
        return asdict(self)


@runtime_checkable
class CausalEstimator(Protocol):
    """Protocol required of causal-estimation integrations; predictive coefficients do not qualify."""

    def fit(self, X, y, *, treatment, control=0, treated=1):
        """Fit the explicitly specified treatment contrast."""
        ...

    def evaluate(self, X, y, *, treatment, control=0, treated=1):
        """Evaluate new independent units without fitting their outcomes."""
        ...


def validate_frame(X, y):
    """Validate an explicit feature frame and an aligned, finite scalar outcome."""
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
        raise ValueError("X must be a DataFrame with unique columns")
    if any(not isinstance(n, str) or not n for n in X.columns):
        raise ValueError("Feature names must be nonempty strings")
    if X.index.has_duplicates:
        raise ValueError("Duplicate unit indices are unsupported for IID analysis")
    if isinstance(y, (pd.Series, pd.DataFrame)) and not X.index.equals(y.index):
        raise ValueError("X and y indices must align exactly")
    outcome = np.asarray(y, dtype=float)
    if outcome.ndim != 1 or len(outcome) != len(X) or not np.isfinite(outcome).all():
        raise ValueError("y must be a finite scalar outcome aligned with X")
    if len(X) < 4:
        raise ValueError("At least four independent observations are required")
    return outcome


class AIPW(BaseEstimator):
    """Cross-fitted ATE for binary treatment, complete numeric covariates, and IID rows.

    Default nuisance learners use standardized logistic regression and linear
    outcome regression. Preprocessing is fit separately within every training
    fold. Neither weights nor propensities are silently clipped. Valid confidence
    intervals additionally require causal identification, overlap, moment, and
    nuisance convergence assumptions that this estimator cannot prove.
    """

    def __init__(
        self,
        adjustment=(),
        cv=5,
        propensity_model=None,
        outcome_model=None,
        confidence_level=0.95,
        overlap_threshold=0.01,
        random_state=0,
    ):
        self.adjustment = adjustment
        self.cv = cv
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.confidence_level = confidence_level
        self.overlap_threshold = overlap_threshold
        self.random_state = random_state

    def _reset_fitted(self):
        for key in tuple(self.__dict__):
            if key.endswith("_") and not key.startswith("__"):
                delattr(self, key)

    def set_params(self, **params):
        """Invalidate fitted evidence when estimator parameters change."""
        result = super().set_params(**params)
        if params:
            self._reset_fitted()
        return result

    def _inputs(self, X, y, treatment, control, treated):
        outcome = validate_frame(X, y)
        if self.adjustment is None:
            raise ValueError("Resolve an explicit adjustment set before fitting")
        names = tuple(self.adjustment)
        if len(set(names)) != len(names) or treatment in names:
            raise ValueError("Invalid adjustment variables")
        if treatment not in X or not set(names) <= set(X.columns):
            raise ValueError("Missing treatment or adjustment columns")
        levels = X[treatment].to_numpy()
        if control == treated or not np.isin(levels, [control, treated]).all():
            raise ValueError("Treatment must use exactly the declared two levels")
        t = (levels == treated).astype(int)
        if len(np.unique(t)) != 2:
            raise ValueError("Both treatment arms are required")
        covariates = X[list(names)].to_numpy(dtype=float)
        if not np.isfinite(covariates).all():
            raise ValueError("Missing/nonfinite covariates require an explicit supported strategy")
        if not names:
            covariates = np.zeros((len(X), 1))
        return covariates, t, outcome

    def _models(self):
        if not self.adjustment:
            return DummyClassifier(strategy="prior"), DummyRegressor(strategy="mean")
        propensity = (
            self.propensity_model
            if self.propensity_model is not None
            else make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, random_state=self.random_state)
            )
        )
        outcome = (
            self.outcome_model
            if self.outcome_model is not None
            else make_pipeline(StandardScaler(), LinearRegression())
        )
        return clone(propensity), clone(outcome)

    def _fit_models(self, c, t, y):
        propensity, outcome = self._models()
        propensity.fit(c, t)
        m0, m1 = clone(outcome), clone(outcome)
        m0.fit(c[t == 0], y[t == 0])
        m1.fit(c[t == 1], y[t == 1])
        return propensity, m0, m1

    @staticmethod
    def _predict(models, c):
        propensity, m0, m1 = models
        index = list(propensity.classes_).index(1)
        return propensity.predict_proba(c)[:, index], m0.predict(c), m1.predict(c)

    def _summarize(self, c, t, y, e, m0, m1, *, method):
        if any(np.shape(a) != np.shape(y) for a in (e, m0, m1)):
            raise ValueError("Nuisance learners must return one prediction per unit")
        if not all(np.isfinite(a).all() for a in (e, m0, m1)):
            raise ValueError("Nuisance learner produced nonfinite predictions")
        if np.any((e <= 1e-10) | (e >= 1 - 1e-10)):
            raise ValueError("Numerically unsupported propensity scores; no clipping was applied")
        scores = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
        if not np.isfinite(scores).all():
            raise ValueError("Numerical overflow in AIPW scores")
        value, se = float(np.mean(scores)), float(np.std(scores, ddof=1) / np.sqrt(len(y)))
        radius = float(norm.ppf((1 + self.confidence_level) / 2) * se)
        estimate = EffectEstimate(
            value,
            se,
            (value - radius, value + radius),
            self.confidence_level,
            len(y),
            method=method,
        )
        weights = np.where(t == 1, 1 / e, 1 / (1 - e))
        diagnostics = {
            "overlap_threshold": self.overlap_threshold,
            "extreme_propensities": int(
                np.sum((e < self.overlap_threshold) | (e > 1 - self.overlap_threshold))
            ),
            "maximum_weight": float(weights.max()),
            "arms": {},
            "weighted_standardized_differences": {},
        }
        for arm in (0, 1):
            mask = t == arm
            w = weights[mask]
            diagnostics["arms"][str(arm)] = {
                "n": int(mask.sum()),
                "effective_sample_size": float(w.sum() ** 2 / np.dot(w, w)),
                "propensity_quantiles": np.quantile(e[mask], [0, 0.05, 0.5, 0.95, 1]).tolist(),
            }
        for i, name in enumerate(self.adjustment):
            denominator = np.sqrt((np.var(c[t == 0, i]) + np.var(c[t == 1, i])) / 2)
            delta = np.average(c[t == 1, i], weights=weights[t == 1]) - np.average(
                c[t == 0, i], weights=weights[t == 0]
            )
            diagnostics["weighted_standardized_differences"][name] = (
                float(delta / denominator) if denominator > 0 else None
            )
        diagnostics["outcome_rmse"] = float(np.sqrt(np.mean((y - np.where(t == 1, m1, m0)) ** 2)))
        diagnostics["propensity_brier_score"] = float(np.mean((t - e) ** 2))
        return estimate, diagnostics

    def fit(self, X, y, *, treatment, control=0, treated=1):
        """Cross-fit nuisance learners, estimate ATE, and fit final models for holdout scoring."""
        self._reset_fitted()
        if (
            not isinstance(self.cv, int)
            or self.cv < 2
            or not 0 < self.confidence_level < 1
            or not 0 < self.overlap_threshold < 0.5
        ):
            raise ValueError("Invalid fold, confidence, or overlap configuration")
        c, t, outcome = self._inputs(X, y, treatment, control, treated)
        if min(np.bincount(t)) < self.cv:
            raise ValueError("Each treatment arm must contain at least cv observations")
        splitter = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        e, m0, m1 = (np.empty(len(t)) for _ in range(3))
        folds = np.empty(len(t), dtype=int)
        for fold, (train, test) in enumerate(splitter.split(c, t)):
            models = self._fit_models(c[train], t[train], outcome[train])
            e[test], m0[test], m1[test] = self._predict(models, c[test])
            folds[test] = fold
        estimate, diagnostics = self._summarize(
            c, t, outcome, e, m0, m1, method="cross_fitted_aipw"
        )
        final_models = self._fit_models(c, t, outcome)
        self.models_, self.estimate_, self.diagnostics_ = final_models, estimate, diagnostics
        self.treatment_, self.control_, self.treated_ = treatment, control, treated
        self.fold_assignments_ = folds
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def evaluate(self, X, y, *, treatment=None, control=None, treated=None):
        """Score independent holdout units with frozen nuisance models; never refit."""
        check_is_fitted(self, "models_")
        treatment = self.treatment_ if treatment is None else treatment
        control = self.control_ if control is None else control
        treated = self.treated_ if treated is None else treated
        if (treatment, control, treated) != (self.treatment_, self.control_, self.treated_):
            raise ValueError("Holdout treatment contrast must match the fitted estimand")
        c, t, outcome = self._inputs(X, y, treatment, control, treated)
        return self._summarize(
            c, t, outcome, *self._predict(self.models_, c), method="frozen_nuisance_aipw"
        )
