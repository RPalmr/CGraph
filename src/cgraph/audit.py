"""Scikit-learn-style orchestration of graph, data, estimation, and evidence checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .estimation import AIPW, validate_frame
from .graph import DAG, find_adjustment, validate_adjustment
from .refutation import stress_tests
from .reports import AuditReport, CheckResult, freeze, thaw
from .symbolic import identify
from .testing import audit_implications


@dataclass(frozen=True)
class AuditConfig:
    """Prespecified test and stress budgets; statistical execution is IID-only."""

    ci_test: object = None
    alpha: float = 0.05
    max_ci_tests: int = 200
    identification_timeout: float = 10.0
    max_identification_calls: int = 10000
    placebo_repetitions: int = 0
    subset_repetitions: int = 0
    subset_fraction: float = 0.8

    def __post_init__(self):
        if (
            not 0 < self.alpha < 1
            or not isinstance(self.max_ci_tests, int)
            or self.max_ci_tests < 1
        ):
            raise ValueError("Invalid CI configuration")
        if (
            not np.isfinite(self.identification_timeout)
            or self.identification_timeout <= 0
            or self.max_identification_calls < 1
        ):
            raise ValueError("Identification budgets must be positive")
        if (
            any(
                not isinstance(n, int) or not 0 <= n <= 1000
                for n in (self.placebo_repetitions, self.subset_repetitions)
            )
            or not 0 < self.subset_fraction < 1
        ):
            raise ValueError("Invalid stress-test configuration")


def infer_schema(frame):
    """Infer measurement classes conservatively without assuming Gaussianity."""
    return {
        n: "binary"
        if frame[n].nunique() <= 2
        else "continuous"
        if pd.api.types.is_numeric_dtype(frame[n])
        else "categorical"
        for n in frame
    }


def _data_fingerprint(frame):
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


class CausalAudit(BaseEstimator):
    """Audit a fixed graph and query with supported statistical methods.

    The constructor follows scikit-learn parameter conventions. ``fit`` records
    training evidence; ``audit`` creates a separate held-out report. A supplied
    graph is an assumption, not a conclusion of the empirical checks. AIPW is
    currently the supported built-in causal estimation backend.
    """

    def __init__(
        self, graph, query, estimator=None, config=None, random_state=0, study_design="iid"
    ):
        self.graph = graph
        self.query = query
        self.estimator = estimator
        self.config = config
        self.random_state = random_state
        self.study_design = study_design

    def _frame(self, X, y):
        outcome = validate_frame(X, y)
        if len(self.query.outcome) != 1:
            raise ValueError(
                "Statistical audits require a scalar outcome; use identify for joint queries"
            )
        name = self.query.outcome[0]
        if name in X:
            raise ValueError("Outcome must be supplied only through y")
        frame = X.copy()
        frame[name] = outcome
        unknown = set(frame.columns) - set(self.graph.nodes)
        if unknown:
            raise ValueError(f"Columns absent from the causal graph: {sorted(unknown)}")
        return frame, outcome

    def _checks(self, frame):
        return audit_implications(
            self.graph_,
            frame,
            self.schema_,
            test=self.config_.ci_test,
            alpha=self.config_.alpha,
            max_tests=self.config_.max_ci_tests,
        )

    def _report(self, checks, estimate, *, dataset_id, phase):
        versions = {
            n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pandas", "scikit-learn")
        }
        provenance = {
            "package_version": "0.1.0",
            "python": sys.version.split()[0],
            "dependencies": versions,
            "dataset_id": dataset_id,
            "dataset_id_verified": False,
            "phase": phase,
            "graph_fingerprint": self.graph_.fingerprint,
            "random_state": self.random_state,
            "assumptions": thaw(self.assumptions_),
            "schema": dict(self.schema_),
            "study_design": self.study_design,
            "target_population": self.query_.target_population,
            "configuration": {
                "alpha": self.config_.alpha,
                "max_ci_tests": self.config_.max_ci_tests,
                "ci_test": type(self.config_.ci_test).__name__
                if self.config_.ci_test
                else "GaussianPartialCorrelation",
                "placebo_repetitions": self.config_.placebo_repetitions,
                "subset_repetitions": self.config_.subset_repetitions,
            },
        }
        if self.estimator_ is not None:
            provenance["adjustment"] = list(self.estimator_.adjustment)
            provenance["fold_assignment_hash"] = hashlib.sha256(
                self.estimator_.fold_assignments_.tobytes()
            ).hexdigest()
            provenance["estimator"] = {
                k: repr(v) for k, v in self.estimator_.get_params(deep=False).items()
            }
        return AuditReport(
            self.graph_.to_dict(),
            asdict(self.query_),
            self.identification_.to_dict(),
            tuple(checks),
            estimate.to_dict() if estimate else None,
            provenance,
        )

    @staticmethod
    def _overlap(diagnostics):
        bad = diagnostics["extreme_propensities"] > 0
        return CheckResult(
            "overlap",
            "completed",
            "estimator_diagnostic",
            "Weak empirical overlap detected"
            if bad
            else "No extreme empirical propensity scores detected; population positivity remains assumed",
            "violation_detected" if bad else "no_violation_detected",
            diagnostics,
        )

    def fit(self, X, y, *, schema=None, assumptions=(), dataset_id="training"):
        """Fit supported nuisance models and preserve a training audit snapshot."""
        if self.study_design != "iid":
            raise ValueError(
                "Statistical auditing currently requires independent observational units"
            )
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("Use a nonnegative integer random_state for reproducible audits")
        self.graph.validate_nodes(self.query.treatment + self.query.outcome)
        frame, outcome = self._frame(X, y)
        self.graph_, self.query_ = self.graph, self.query
        self.config_ = self.config or AuditConfig()
        self.assumptions_ = freeze(assumptions)
        self.schema_ = dict(schema) if schema is not None else infer_schema(frame)
        if set(self.schema_) - set(self.graph.nodes) or any(
            v not in ("continuous", "continuous_gaussian", "binary", "categorical")
            for v in self.schema_.values()
        ):
            raise ValueError("Invalid variable schema")
        self.identification_ = identify(
            self.graph_,
            self.query_,
            max_calls=self.config_.max_identification_calls,
            timeout=self.config_.identification_timeout,
        )
        checks = list(self._checks(frame))
        self.estimator_, estimate = None, None
        reason = None
        if self.identification_.status != "identified":
            reason = (
                "Identification did not complete successfully; causal estimation was not executed"
            )
        elif not isinstance(self.graph_, DAG) or self.query_.kind != "ate":
            reason = "Numerical estimation requires a DAG and scalar binary-treatment ATE"
        elif self.estimator is not None and not isinstance(self.estimator, AIPW):
            reason = "This release supports the AIPW estimator contract only"
        else:
            estimator = (
                clone(self.estimator)
                if self.estimator is not None
                else AIPW(adjustment=None, random_state=self.random_state)
            )
            try:
                if estimator.adjustment is None:
                    estimator.set_params(
                        adjustment=find_adjustment(
                            self.graph_, self.query_.treatment, self.query_.outcome
                        )
                    )
                if estimator.adjustment is None:
                    reason = "No supported observed adjustment set exists"
                elif not validate_adjustment(
                    self.graph_, self.query_.treatment, self.query_.outcome, estimator.adjustment
                ).valid:
                    reason = "Proposed adjustment set fails the DAG total-effect criterion"
                else:
                    estimator.fit(
                        X,
                        outcome,
                        treatment=self.query_.treatment[0],
                        control=self.query_.control,
                        treated=self.query_.treated,
                    )
                    self.estimator_, estimate = estimator, estimator.estimate_
                    checks.append(self._overlap(estimator.diagnostics_))
                    checks.extend(
                        stress_tests(
                            estimator,
                            X,
                            outcome,
                            treatment=self.query_.treatment[0],
                            control=self.query_.control,
                            treated=self.query_.treated,
                            placebo_repetitions=self.config_.placebo_repetitions,
                            subset_repetitions=self.config_.subset_repetitions,
                            subset_fraction=self.config_.subset_fraction,
                            random_state=self.random_state,
                        )
                    )
            except (ValueError, TimeoutError) as exc:
                checks.append(
                    CheckResult(
                        "estimation",
                        "resource_limited"
                        if isinstance(exc, TimeoutError)
                        else "insufficient_data",
                        "estimator_diagnostic",
                        str(exc),
                    )
                )
        if reason:
            checks.append(CheckResult("estimation", "unsupported", "estimator_diagnostic", reason))
        self.training_fingerprint_ = _data_fingerprint(frame)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.report_ = self._report(
            checks, estimate, dataset_id=dataset_id, phase="training_cross_fitted"
        )
        return self

    def audit_report(self):
        """Return the existing immutable training report without executing new work."""
        check_is_fitted(self, "report_")
        return self.report_

    def audit(self, X, y, *, dataset_id="holdout"):
        """Audit new independent data using frozen models and the fitted graph/query."""
        check_is_fitted(self, "report_")
        if self.graph != self.graph_ or self.query != self.query_:
            raise ValueError("Graph/query parameters changed; refit before auditing")
        frame, outcome = self._frame(X, y)
        if _data_fingerprint(frame) == self.training_fingerprint_:
            raise ValueError("The training dataset cannot be labeled as an independent holdout")
        checks, estimate = list(self._checks(frame)), None
        if self.estimator_ is not None:
            try:
                estimate, diagnostics = self.estimator_.evaluate(X, outcome)
                checks.append(self._overlap(diagnostics))
            except ValueError as exc:
                checks.append(
                    CheckResult("estimation", "insufficient_data", "estimator_diagnostic", str(exc))
                )
        else:
            checks.append(
                CheckResult(
                    "estimation",
                    "unsupported",
                    "estimator_diagnostic",
                    "No supported fitted estimator is available",
                )
            )
        return self._report(
            checks, estimate, dataset_id=dataset_id, phase="heldout_frozen_nuisance"
        )
